use super::node::{BranchNodeCompact, BranchNodeCompact as Node};
use reth_db::{
    cursor::{DbCursorRO, DbCursorRW},
    tables, Error as DbError,
};
use reth_primitives::H256;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct CursorSubNode {
    key: Vec<u8>,
    node: Option<Node>,
    nibble: i8,
}

impl Default for CursorSubNode {
    fn default() -> Self {
        Self::new(vec![], None)
    }
}

impl CursorSubNode {
    fn new(key: Vec<u8>, node: Option<Node>) -> Self {
        // Find the first nibble that is set in the state mask of the node.
        let nibble = match &node {
            Some(n) if n.root_hash.is_none() => {
                (0i8..16).find(|i| n.state_mask & (1u16 << i) != 0).unwrap()
            }
            _ => -1,
        };

        let res = CursorSubNode { key, node, nibble };
        res
    }

    fn full_key(&self) -> Vec<u8> {
        let mut out = self.key.clone();
        if self.nibble >= 0 {
            out.push(self.nibble as u8)
        }
        out
    }

    fn state_flag(&self) -> bool {
        if let Some(node) = &self.node {
            if self.nibble >= 0 {
                return node.state_mask & (1u16 << self.nibble) != 0
            }
        }
        true
    }

    fn tree_flag(&self) -> bool {
        if let Some(node) = &self.node {
            if self.nibble >= 0 {
                return node.tree_mask & (1u16 << self.nibble) != 0
            }
        }
        true
    }

    fn hash_flag(&self) -> bool {
        match &self.node {
            Some(node) => match self.nibble {
                // This guy has it
                -1 => node.root_hash.is_some(),
                // Or get it from the children
                _ => node.hash_mask & (1u16 << self.nibble) != 0,
            },
            None => false,
        }
    }

    pub fn hash(&self) -> Option<H256> {
        if self.hash_flag() {
            let node = self.node.as_ref().unwrap();
            match self.nibble {
                -1 => node.root_hash,
                _ => Some(node.hash_for_nibble(self.nibble)),
            }
        } else {
            None
        }
    }
}

pub struct AccountsCursor<'a, C> {
    pub cursor: &'a mut C,
    pub stack: Vec<CursorSubNode>,
    pub can_skip_state: bool,
}

#[derive(Error, Debug)]
pub enum AccountsCursorError {
    #[error(transparent)]
    DbError(#[from] DbError),
}

type Result<T> = std::result::Result<T, AccountsCursorError>;

impl<'a, 'cursor, C> AccountsCursor<'a, C>
where
    C: DbCursorRO<'cursor, tables::AccountsTrie2> + DbCursorRW<'cursor, tables::AccountsTrie2>,
{
    pub fn new(cursor: &'a mut C) -> Self {
        // Initialize the cursor with a single empty stack element.
        Self { cursor, can_skip_state: false, stack: vec![CursorSubNode::default()] }
    }

    pub fn print(&self) {
        tracing::trace!("====================== STACK ======================");
        for node in &self.stack {
            dbg!(&node.node);
            tracing::trace!(
                "key: {:?}, node: {}, nibble: {}, state: {}, tree: {}, hash: {}",
                node.key,
                node.node.is_some(),
                node.nibble,
                node.state_flag(),
                node.tree_flag(),
                node.hash_flag()
            );
        }
        tracing::trace!("====================== END STACK ======================\n");
    }

    #[tracing::instrument(skip(self))]
    pub fn next(&mut self) -> Result<Option<Vec<u8>>> {
        if let Some(last) = self.stack.last() {
            // tracing::trace!("Can skip state? {}", self.can_skip_state);
            // tracing::trace!("Children in trie? {}", self.children_are_in_trie());
            if !self.can_skip_state && self.children_are_in_trie() {
                // tracing::trace!("Last nibble: {}", last.nibble);
                match last.nibble {
                    // 0xFF -> move to the next sibling since we're done
                    -1 => self.move_to_next_sibling(true)?,
                    _ => self.consume_node()?,
                }
            } else {
                self.move_to_next_sibling(false)?;
            }
            self.update_skip_state();
        }

        Ok(self.key())
    }

    /// Reads the current root node from the DB.
    fn node(&mut self) -> Result<Option<(Vec<u8>, BranchNodeCompact)>> {
        // Seek to the intermediate node that matches the current key, or the next one if it's not
        // found for the provided key.
        let Some((key, value)) = self.cursor.seek(self.key().expect("key must exist"))? else {
            return Ok(None);
        };
        tracing::trace!(
            "Found intermediate node at at: {:?}, value: {:?}",
            hex::encode(&key),
            hex::encode(&value)
        );

        // TODO: Handle, but it seems like it should always work?
        let node = Node::unmarshal(&value).expect("node must be unmarshalled");
        assert_ne!(node.state_mask, 0);

        Ok(Some((key, node)))
    }

    #[tracing::instrument(skip(self), fields(key = hex::encode(&self.key().unwrap())))]
    fn consume_node(&mut self) -> Result<()> {
        let Some((key, node)) = self.node()? else {
            tracing::trace!("No entry found, clearing stack & returning");
            self.stack.clear();
            return Ok(());
        };

        // TODO: Why is this needed?
        if !key.is_empty() && !self.stack.is_empty() {
            tracing::trace!("Overriding stack nibble from {} to {}", self.stack[0].nibble, key[0]);
            self.stack[0].nibble = key[0] as i8;
        }

        let subnode = CursorSubNode::new(key, Some(node));
        let nibble = subnode.nibble;
        self.stack.push(subnode);
        self.update_skip_state();

        // TODO: Can we remove this conditional?
        if !self.can_skip_state || nibble != -1 {
            tracing::trace!(nibble, "Can't skip state, or nibble is not -1, deleting current");
            self.cursor.delete_current()?;
        }

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    fn move_to_next_sibling(
        &mut self,
        allow_root_to_child_nibble_within_subnode: bool,
    ) -> Result<()> {
        let Some(sn) = self.stack.last_mut() else {
            tracing::trace!("empty stack, returning");
            return Ok(());
        };

        tracing::trace!("nibble: {}", sn.nibble);
        if sn.nibble >= 15 || (sn.nibble < 0 && !allow_root_to_child_nibble_within_subnode) {
            tracing::trace!("pop and restart");
            self.stack.pop();
            self.move_to_next_sibling(false)?;
            return Ok(())
        }

        sn.nibble += 1;

        if sn.node.is_none() {
            tracing::trace!("node is none, consume node");
            return self.consume_node()
        }

        // TODO: What is this for?
        while sn.nibble < 16 {
            if sn.state_flag() {
                return Ok(())
            }
            sn.nibble += 1;
        }

        self.stack.pop();
        self.move_to_next_sibling(false)?;

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    fn key(&self) -> Option<Vec<u8>> {
        self.stack.last().map(|n| n.full_key())
    }

    #[tracing::instrument(skip(self))]
    fn hash(&self) -> Option<H256> {
        self.stack.last().and_then(|n| n.hash())
    }

    #[tracing::instrument(skip(self))]
    fn children_are_in_trie(&self) -> bool {
        self.stack.last().map_or(false, |n| n.tree_flag())
    }

    #[tracing::instrument(skip(self))]
    fn update_skip_state(&mut self) {
        self.can_skip_state = if let Some(key) = self.key() {
            tracing::trace!("Key: {:?}", hex::encode(&key));
            // let s = [self.prefix.as_slice(), key.as_slice()].concat();
            // tracing::trace!("Checking if prefix exists {:?}", hex::encode(&s));

            let contains_prefix = false; // !self.changed.contains(s.as_slice());
            let hash_flag = self.stack.last().unwrap().hash_flag();
            let val = !contains_prefix && self.stack.last().unwrap().hash_flag();
            tracing::trace!(
                "contains_prefix: {}, hash_flag: {}, val: {}",
                contains_prefix,
                hash_flag,
                val
            );

            val
        } else {
            false
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Transaction;
    use reth_db::{mdbx::test_utils::create_test_rw_db, tables, transaction::DbTxMut};

    #[test]
    fn test_intermediate_hashes_cursor_traversal_1() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            // .with_env_filter(EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .try_init();

        let db = create_test_rw_db();
        let tx = Transaction::new(db.as_ref()).unwrap();
        let mut trie = tx.cursor_write::<tables::AccountsTrie2>().unwrap();

        // Create 3 nodes with a common pre-fix 0x1. We store the nodes with their nibbles as key
        let inputs = vec![
            // State Mask: 0b0000_0000_0000_1011: 0, 1, 3 idxs to be hashed
            // Tree Mask: 0b0000_0000_0000_1001: 0, 3 idxs to be pulled from the tree?
            (vec![0x1u8], BranchNodeCompact::new(0b1011, 0b1001, 0, vec![], None)),
            // State Mask: 0b0000_0000_0000_1010: 1, 3 idxs to be hashed
            // No data to pull from tree
            (vec![0x1u8, 0x0, 0xB], BranchNodeCompact::new(0b1010, 0, 0, vec![], None)),
            // State Mask: 0b0000_0000_0000_1110: 1, 2, 3 idxs to be hashed
            // No data to pull from tree
            (vec![0x1u8, 0x3], BranchNodeCompact::new(0b1110, 0, 0, vec![], None)),
        ];

        for (k, v) in &inputs {
            trie.upsert(k.to_vec(), v.marshal()).unwrap();
        }

        // let mut changed = PrefixSet::new();
        let mut cursor = AccountsCursor::new(&mut trie);
        assert!(cursor.key().unwrap().is_empty());

        // We're traversing the path in lexigraphical order.
        for expected in vec![
            vec![0x1, 0x0],
            vec![0x1, 0x0, 0xB, 0x1],
            vec![0x1, 0x0, 0xB, 0x3],
            vec![0x1, 0x1],
            vec![0x1, 0x3],
            vec![0x1, 0x3, 0x1],
            vec![0x1, 0x3, 0x2],
            vec![0x1, 0x3, 0x3],
        ] {
            let got = cursor.next().unwrap().unwrap();
            assert_eq!(got, expected);
        }

        // There should be 8 paths traversed in total from 3 branches.
        let got = cursor.next().unwrap();
        assert!(got.is_none());
    }
}
