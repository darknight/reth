use super::node::BranchNodeCompact;
use reth_db::{
    cursor::{DbCursorRO, DbCursorRW, DbDupCursorRO, DbDupCursorRW},
    tables,
    transaction::{DbTx, DbTxMut},
    Error as DbError,
};

// type AccountsTrieCursor<'tx, TX> =
//     Arc<Mutex<<TX as DbTxMutGAT<'tx>>::CursorMut<tables::AccountsTrie>>>;

pub struct AccountsCursor<'a, C> {
    pub cursor: &'a mut C,
}

impl<'a, 'cursor, C> AccountsCursor<'a, C>
where
    C: DbCursorRW<'cursor, tables::AccountsTrie2>,
{
    pub fn new(cursor: &'a mut C) -> Self {
        Self { cursor }
    }

    pub fn key(&self) -> &[u8] {
        self.cursor.key()
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
        let db = create_test_rw_db();
        let mut tx = Transaction::new(db.as_ref()).unwrap();
        let mut trie = tx.cursor_write::<tables::AccountsTrie2>().unwrap();

        // Create 3 nodes with a common pre-fix 0x1. We store the nodes with their nibbles as key
        let inputs = vec![
            // State Mask: 0b0000_0000_0000_1011: 0, 1, 3 idxs to be hashed
            // Tree Mask: 0b0000_0000_0000_1001: 0, 3 idxs to be pulled from the tree?
            (vec![0x1u8], BranchNodeCompact::new(0b1011, 0b1001, 0, vec![], None)),
            (vec![0x1u8, 0x0, 0xB], BranchNodeCompact::new(0b1010, 0, 0, vec![], None)),
            (vec![0x1u8, 0x3], BranchNodeCompact::new(0b1110, 0, 0, vec![], None)),
        ];

        for (k, v) in &inputs {
            trie.upsert(k.to_vec(), v.marshal()).unwrap();
        }

        // let mut changed = PrefixSet::new();
        let mut cursor = AccountsCursor::new(&mut trie);

        // assert!(cursor.key().unwrap().is_empty());

        // cursor.next().unwrap();
        // assert_eq!(cursor.key().unwrap(), vec![0x1, 0x0]);

        // cursor.next().unwrap();
        // assert_eq!(cursor.key().unwrap(), vec![0x1, 0x0, 0xB, 0x1]);

        // cursor.next().unwrap();
        // assert_eq!(cursor.key().unwrap(), vec![0x1, 0x0, 0xB, 0x3]);

        // cursor.next().unwrap();
        // assert_eq!(cursor.key().unwrap(), vec![0x1, 0x1]);

        // cursor.next().unwrap();
        // assert_eq!(cursor.key().unwrap(), vec![0x1, 0x3]);

        // cursor.next().unwrap();
        // assert_eq!(cursor.key().unwrap(), vec![0x1, 0x3, 0x1]);

        // cursor.next().unwrap();
        // assert_eq!(cursor.key().unwrap(), vec![0x1, 0x3, 0x2]);

        // cursor.next().unwrap();
        // assert_eq!(cursor.key().unwrap(), vec![0x1, 0x3, 0x3]);

        // cursor.next().unwrap();
        // assert!(cursor.key().is_none());
    }
}
