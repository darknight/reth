use super::{
    nibbles::Nibbles,
    node::{BranchNodeCompact, BranchNodeCompact as Node},
    prefix_set::PrefixSet,
};
use reth_db::{
    cursor::{DbCursorRO, DbCursorRW, DbDupCursorRO, DbDupCursorRW},
    table::Key,
    tables, Error as DbError,
};
use reth_primitives::{StoredNibbles, StoredNibblesSubKey, H256};
use std::marker::PhantomData;
use thiserror::Error;

fn increment_key(unpacked: &[u8]) -> Option<Vec<u8>> {
    let mut out = unpacked.to_vec();

    for i in (0..out.len()).rev() {
        let nibble = out[i];
        assert!(nibble < 0x10);
        if nibble < 0xf {
            out[i] += 1;
            return Some(out)
        } else {
            out[i] = 0;
        }
    }

    None
}

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

pub struct TrieWalker<'a, K, C> {
    pub cursor: &'a mut C,
    pub stack: Vec<CursorSubNode>,
    pub can_skip_state: bool,
    pub changes: PrefixSet,
    __phantom: PhantomData<K>,
}

#[derive(Error, Debug)]
pub enum AccountsCursorError {
    #[error(transparent)]
    DbError(#[from] DbError),
}

pub trait TrieCursor<K: Key> {
    fn seek_exact(&mut self, key: K) -> Result<Option<(Vec<u8>, Vec<u8>)>>;

    fn seek(&mut self, key: K) -> Result<Option<(Vec<u8>, Vec<u8>)>>;
    fn delete_current(&mut self) -> Result<()>;
    fn upsert(&mut self, key: K, value: Vec<u8>) -> Result<()>;
}

pub struct AccountTrieCursor<C>(pub C);

impl<'a, C> TrieCursor<StoredNibbles> for AccountTrieCursor<C>
where
    C: DbCursorRO<'a, tables::AccountsTrie2> + DbCursorRW<'a, tables::AccountsTrie2>,
{
    fn seek_exact(&mut self, key: StoredNibbles) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        Ok(self.0.seek_exact(key.into())?.map(|value| (value.0.inner.to_vec(), value.1)))
    }

    fn seek(&mut self, key: StoredNibbles) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        Ok(self.0.seek(key.into())?.map(|value| (value.0.inner.to_vec(), value.1)))
    }

    fn delete_current(&mut self) -> Result<()> {
        Ok(self.0.delete_current()?)
    }

    fn upsert(&mut self, key: StoredNibbles, value: Vec<u8>) -> Result<()> {
        Ok(self.0.upsert(key, value)?)
    }
}

pub struct StorageTrieCursor<C> {
    cursor: C,
    hashed_address: H256,
}

impl<C> StorageTrieCursor<C> {
    pub fn new(cursor: C, hashed_address: H256) -> Self {
        Self { cursor, hashed_address }
    }
}

impl<'a, C> TrieCursor<StoredNibblesSubKey> for StorageTrieCursor<C>
where
    C: DbDupCursorRO<'a, tables::StoragesTrie2>
        + DbDupCursorRW<'a, tables::StoragesTrie2>
        + DbCursorRO<'a, tables::StoragesTrie2>
        + DbCursorRW<'a, tables::StoragesTrie2>,
{
    fn seek_exact(&mut self, key: StoredNibblesSubKey) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        Ok(self
            .cursor
            .seek_by_key_subkey(self.hashed_address, key.clone())?
            .filter(|e| e.nibbles == key)
            .map(|value| (value.nibbles.inner.to_vec(), value.node)))
    }

    fn seek(&mut self, key: StoredNibblesSubKey) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        Ok(self
            .cursor
            .seek_by_key_subkey(self.hashed_address, key.clone())?
            .map(|value| (value.nibbles.inner.to_vec(), value.node)))
    }

    fn delete_current(&mut self) -> Result<()> {
        Ok(self.cursor.delete_current()?)
    }

    fn upsert(&mut self, key: StoredNibblesSubKey, value: Vec<u8>) -> Result<()> {
        if let Some(entry) = self.cursor.seek_by_key_subkey(self.hashed_address, key.clone())? {
            // "seek exact"
            if entry.nibbles == key {
                self.cursor.delete_current()?;
            }
        }

        self.cursor.upsert(
            self.hashed_address,
            reth_primitives::StorageTrieEntry2 { nibbles: key, node: value },
        )?;

        Ok(())
    }
}

type Result<T> = std::result::Result<T, AccountsCursorError>;

impl<'a, K: Key + From<Vec<u8>>, C: TrieCursor<K>> TrieWalker<'a, K, C> {
    pub fn new(cursor: &'a mut C, changes: PrefixSet) -> Self {
        // Initialize the cursor with a single empty stack element.
        let mut this = Self {
            cursor,
            can_skip_state: false,
            stack: vec![CursorSubNode::default()],
            changes,
            __phantom: PhantomData::default(),
        };

        // When instantiating the walker, we first need to check if there's
        // a node at the root of the trie. If there is, we need to push it.
        let root = this.node(true).unwrap();
        if let Some((key, value)) = root {
            this.stack[0] = CursorSubNode::new(key, Some(value));
        }
        this.update_skip_state();

        this
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

    /// Reads the current root node from the DB. If `exact` is passed as true, then we're looking
    /// for the root node during walker instantiation.
    fn node(&mut self, exact: bool) -> Result<Option<(Vec<u8>, BranchNodeCompact)>> {
        // Seek to the intermediate node that matches the current key, or the next one if it's not
        // found for the provided key.
        tracing::trace!("Seeking to key: {:?}", hex::encode(&self.key().unwrap()));
        let entry = if exact {
            self.cursor.seek_exact(self.key().expect("key must exist").into())?
        } else {
            self.cursor.seek(self.key().expect("key must exist").into())?
        };
        let Some((key, value)) = entry else {
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
        let Some((key, node)) = self.node(false)? else {
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
    pub fn key(&self) -> Option<Vec<u8>> {
        self.stack.last().map(|n| n.full_key())
    }

    #[tracing::instrument(skip(self))]
    pub fn hash(&self) -> Option<H256> {
        self.stack.last().and_then(|n| n.hash())
    }

    #[tracing::instrument(skip(self))]
    pub fn children_are_in_trie(&self) -> bool {
        self.stack.last().map_or(false, |n| n.tree_flag())
    }

    pub fn first_uncovered_prefix(&self) -> Option<Vec<u8>> {
        self.key().as_ref().and_then(|key| {
            if self.can_skip_state {
                increment_key(key).map(|k| Nibbles::from(k).pack())
            } else {
                Some(Nibbles::from(key.as_slice()).pack())
            }
        })
    }

    #[tracing::instrument(skip(self))]
    fn update_skip_state(&mut self) {
        self.can_skip_state = if let Some(key) = self.key() {
            tracing::trace!("Key: {:?}", hex::encode(&key));
            let contains_prefix = self.changes.contains(key.as_slice());
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
    use hex_literal::hex;
    use reth_db::{mdbx::test_utils::create_test_rw_db, tables, transaction::DbTxMut};

    // tests that upsert and seek match on the storagetrie cursor
    #[test]
    fn test_storage_cursor_abstraction() {
        let db = create_test_rw_db();
        let tx = Transaction::new(db.as_ref()).unwrap();
        let cursor = tx.cursor_dup_write::<tables::StoragesTrie2>().unwrap();

        let mut cursor = StorageTrieCursor::new(cursor, H256::random());

        let key = vec![0x2, 0x3];
        let value = vec![0x04, 0x05, 0x06];

        cursor.upsert(key.clone().into(), value.clone()).unwrap();
        // We are not able to find the key for some reason. Probably related
        // to the key/subkey encoding or how we are using the dupcursor.
        assert_eq!(cursor.seek(key.clone().into()).unwrap().unwrap().1, value);
    }

    // Create 3 nodes with a common pre-fix 0x1. We store the nodes with their nibbles as key
    fn inputs1() -> Vec<(Vec<u8>, BranchNodeCompact)> {
        vec![
            // State Mask: 0b0000_0000_0000_1011: 0, 1, 3 idxs to be hashed
            // Tree Mask: 0b0000_0000_0000_1001: 0, 3 to be used from the tree (?)
            (vec![0x1u8], BranchNodeCompact::new(0b1011, 0b1001, 0, vec![], None)),
            // State Mask: 0b0000_0000_0000_1010: 1, 3 idxs to be hashed
            // No data to pull from tree
            (vec![0x1u8, 0x0, 0xB], BranchNodeCompact::new(0b1010, 0, 0, vec![], None)),
            // State Mask: 0b0000_0000_0000_1110: 1, 2, 3 idxs to be hashed
            // No data to pull from tree
            (vec![0x1u8, 0x3], BranchNodeCompact::new(0b1110, 0, 0, vec![], None)),
        ]
    }

    fn expected1() -> Vec<Vec<u8>> {
        vec![
            vec![0x1, 0x0],
            // The [0x1, 0x0] prefix is shared by the first 2 nodes, however:
            // 1. 0x0 for the first node points to the child node path
            // 2. 0x0 for the second node is a key.
            // So to proceed to add 1 and 3, we need to push the sibling first (0xB).
            vec![0x1, 0x0, 0xB, 0x1],
            vec![0x1, 0x0, 0xB, 0x3],
            vec![0x1, 0x1],
            vec![0x1, 0x3],
            vec![0x1, 0x3, 0x1],
            vec![0x1, 0x3, 0x2],
            vec![0x1, 0x3, 0x3],
        ]
    }

    #[test]
    fn test_accounts_cursor_1() {
        let inputs = inputs1();
        let expected = expected1();

        // Both of the `test_cursor` tests below should have the exact same execution paths.
        // One simply uses the `AccountTrieCursor` and the other uses the `StorageTrieCursor`.
        // This is made possible by the `TrieCursor` trait which makes the `StorageTrieCursor`
        // behave like the `AccountTrieCursor` for a specific hashed address.
        let db = create_test_rw_db();
        let tx = Transaction::new(db.as_ref()).unwrap();
        let trie = AccountTrieCursor(tx.cursor_write::<tables::AccountsTrie2>().unwrap());
        test_cursor(trie, inputs.clone().into_iter(), expected);
    }

    #[test]
    fn test_storage_cursor_1() {
        let inputs = inputs1();
        let expected = expected1();

        let db = create_test_rw_db();
        let tx = Transaction::new(db.as_ref()).unwrap();
        let trie = StorageTrieCursor::new(
            tx.cursor_dup_write::<tables::StoragesTrie2>().unwrap(),
            H256::random(),
        );
        test_cursor(trie, inputs.into_iter(), expected);
    }

    fn inputs2() -> Vec<(Vec<u8>, BranchNodeCompact)> {
        vec![
            (
                vec![0x4u8],
                Node::new(
                    // 2 and 4
                    0b10100,
                    0,
                    0b00100,
                    vec![H256::from(hex!(
                        "0384e6e2c2b33c4eb911a08a7ff57f83dc3eb86d8d0c92ec112f3b416d6685a9"
                    ))],
                    None,
                ),
            ),
            (
                vec![0x6u8],
                Node::new(
                    // 1 and 4
                    0b10010,
                    0,
                    0b00010,
                    vec![H256::from(hex!(
                        "7f9a58b00625a6e725559acf327baf88d90e4a5b65a2003acd24f110c0441df1"
                    ))],
                    None,
                ),
            ),
        ]
    }

    fn expected2() -> Vec<Vec<u8>> {
        vec![vec![0x4, 0x2], vec![0x4, 0x4], vec![0x6, 0x1], vec![0x6, 0x4]]
    }

    #[test]
    fn test_accounts_cursor_2() {
        let inputs = inputs2();
        let expected = expected2();
        let db = create_test_rw_db();
        let tx = Transaction::new(db.as_ref()).unwrap();
        let trie = AccountTrieCursor(tx.cursor_write::<tables::AccountsTrie2>().unwrap());
        test_cursor(trie, inputs.clone().into_iter(), expected);
    }

    #[test]
    fn test_storage_cursor_2() {
        let inputs = inputs2();
        let expected = expected2();

        let db = create_test_rw_db();
        let tx = Transaction::new(db.as_ref()).unwrap();
        let trie = StorageTrieCursor::new(
            tx.cursor_dup_write::<tables::StoragesTrie2>().unwrap(),
            H256::random(),
        );
        test_cursor(trie, inputs.into_iter(), expected);
    }

    fn test_cursor<K, T, I>(mut trie: T, inputs: I, expected: Vec<Vec<u8>>)
    where
        K: Key + From<Vec<u8>>,
        T: TrieCursor<K>,
        I: Iterator<Item = (Vec<u8>, BranchNodeCompact)>,
    {
        for (k, v) in inputs {
            trie.upsert(k.into(), v.marshal()).unwrap();
        }

        let mut walker = TrieWalker::new(&mut trie, Default::default());
        assert!(walker.key().unwrap().is_empty());

        // We're traversing the path in lexigraphical order.
        for expected in expected {
            let got = walker.next().unwrap();
            assert_eq!(got.unwrap(), expected);
        }

        // There should be 8 paths traversed in total from 3 branches.
        let got = walker.next().unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn cursor_rootnode_with_changesets() {
        let db = create_test_rw_db();
        let tx = Transaction::new(db.as_ref()).unwrap();

        let mut trie = StorageTrieCursor::new(
            tx.cursor_dup_write::<tables::StoragesTrie2>().unwrap(),
            H256::random(),
        );

        for (k, v) in vec![
            (
                vec![],
                Node::new(
                    // 2 and 4 are set
                    0b10100,
                    0b00100,
                    0,
                    vec![],
                    Some(
                        hex!("c570b66136e99d07c6c6360769de1d9397805849879dd7c79cf0b8e6694bfb0e")
                            .into(),
                    ),
                ),
            ),
            (
                vec![0x2],
                Node::new(
                    // 1 is set
                    0b00010,
                    0,
                    0b00010,
                    vec![hex!("6fc81f58df057a25ca6b687a6db54aaa12fbea1baf03aa3db44d499fb8a7af65")
                        .into()],
                    None,
                ),
            ),
        ] {
            trie.upsert(k.into(), v.marshal()).unwrap();
        }

        // No changes
        let mut cursor = TrieWalker::new(&mut trie, Default::default());
        assert_eq!(cursor.key(), Some(vec![])); // root
        assert!(cursor.can_skip_state); // due to root_hash
        cursor.next().unwrap(); // skips to end of trie
        assert_eq!(cursor.key(), None);

        // Some changes
        let mut changed = PrefixSet::new();
        // We insert somethign that's not part of the existing trie/prefix.
        changed.insert(&[0xD, 0x5]);
        let mut cursor = TrieWalker::new(&mut trie, changed);

        // root node
        assert_eq!(cursor.key(), Some(vec![]));
        // should not be able to skip state due to the changed values
        assert!(!cursor.can_skip_state);
        cursor.next().unwrap();
        assert_eq!(cursor.key(), Some(vec![0x2]));
        cursor.next().unwrap();
        assert_eq!(cursor.key(), Some(vec![0x2, 0x1]));
        cursor.next().unwrap();
        assert_eq!(cursor.key(), Some(vec![0x4]));

        cursor.next().unwrap();
        assert_eq!(cursor.key(), None); // end of trie
    }
}
