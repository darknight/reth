use super::node::{BranchNodeCompact, BranchNodeCompact as Node};
use reth_db::{
    cursor::{DbCursorRO, DbCursorRW, DbDupCursorRO, DbDupCursorRW},
    tables,
    transaction::{DbTx, DbTxMut},
    Error as DbError,
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
        dbg!(&res);
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

    fn hash(&self) -> Option<H256> {
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
}

#[derive(Error, Debug)]
enum AccountsCursorError {
    #[error(transparent)]
    DbError(#[from] DbError),
}

type Result<T> = std::result::Result<T, AccountsCursorError>;

impl<'a, 'cursor, C> AccountsCursor<'a, C>
where
    C: DbCursorRW<'cursor, tables::AccountsTrie2>,
{
    pub fn new(cursor: &'a mut C) -> Self {
        // Initialize the cursor with a single empty stack element.
        Self { cursor, stack: vec![CursorSubNode::default()] }
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
    fn next(&mut self) -> Result<Option<Vec<u8>>> {
        if let Some(last) = self.stack.last() {
            // tracing::trace!("Can skip state? {}", self.can_skip_state);
            // tracing::trace!("Children in trie? {}", self.children_are_in_trie());
            // if !self.can_skip_state && self.children_are_in_trie() {
            //     tracing::trace!("Last nibble: {}", last.nibble);
            //     match last.nibble {
            //         // 0xFF -> move to the next sibling since we're done
            //         -1 => self.move_to_next_sibling(true)?,
            //         _ => self.consume_node(&self.key().unwrap(), false)?,
            //     }
            // } else {
            //     self.move_to_next_sibling(false)?;
            // }
            // self.update_skip_state();
        }

        Ok(self.key())
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Transaction;
    use proptest::{prelude::ProptestConfig, proptest};
    use reth_db::{mdbx::test_utils::create_test_rw_db, tables, transaction::DbTxMut};
    use reth_primitives::{keccak256, proofs::KeccakHasher, Account, Address, H256, U256};
    use reth_rlp::encode_fixed_size;
    use std::{collections::BTreeMap, ops::DerefMut};

    #[test]
    fn test_intermediate_hashes_cursor_traversal_1() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            // .with_env_filter(EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .try_init();

        let db = create_test_rw_db();
        let mut tx = Transaction::new(db.as_ref()).unwrap();
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

        for expected in vec![
            vec![0x1, 0x0],
            // vec![0x1, 0x0, 0xB, 0x1],
            // vec![0x1, 0x0, 0xB, 0x3],
            // vec![0x1, 0x1],
            // vec![0x1, 0x3],
            // vec![0x1, 0x3, 0x1],
            // vec![0x1, 0x3, 0x2],
            // vec![0x1, 0x3, 0x3],
        ] {
            let got = cursor.next().unwrap().unwrap();
            assert_eq!(got, expected);
        }

        // // There should be 8 paths traversed in total from 3 branches.
        // let got = cursor.next().unwrap();
        // assert!(got.is_none());
    }
}
