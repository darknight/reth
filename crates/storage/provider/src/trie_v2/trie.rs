// 1. Calculate Root given entire presorted hashed thing
// Must work with:
// 1. Storage Trie Cursor
// 2. Accounts Trie Cursor
// 2. Update root given a list of updates
// Be able to calculate incremental state root without taking a write lock

use super::{account::EthAccount, nibbles::Nibbles};
use crate::trie_v2::hash_builder::HashBuilder;
use reth_db::{
    cursor::{DbCursorRO, DbDupCursorRO},
    tables,
    transaction::{DbTx, DbTxMut},
    Error as DbError,
};
use reth_primitives::{
    keccak256, proofs::EMPTY_ROOT, Address, StorageEntry, StorageTrieEntry2, H256,
};
use reth_rlp::Encodable;
use thiserror::Error;
use tokio::sync::mpsc;

pub struct StateRoot<'a, TX> {
    pub tx: &'a TX,
}

impl<'a, TX> StateRoot<'a, TX> {
    pub fn new(tx: &'a TX) -> Self {
        Self { tx }
    }
}

#[derive(Error, Debug)]
pub enum StateRootError {
    #[error(transparent)]
    DB(#[from] DbError),
    #[error(transparent)]
    StorageRootError(#[from] StorageRootError),
}

impl<'a, 'tx, TX: DbTx<'tx> + DbTxMut<'tx>> StateRoot<'a, TX> {
    /// Walks the entire hashed storage table entry for the given address and calculates the storage
    /// root
    #[tracing::instrument(skip(self))]
    pub fn root(&self) -> Result<H256, StateRootError> {
        tracing::debug!(target: "loader", "calculating state root");
        // Instantiate the walker
        let mut cursor = self.tx.cursor_read::<tables::HashedAccount>()?;
        let mut walker = cursor.walk(None)?;

        let (branch_node_tx, mut branch_node_rx) = mpsc::unbounded_channel();
        let mut hash_builder = HashBuilder::default().with_store_tx(branch_node_tx);

        let mut progress = 0;
        while let Some(item) = walker.next() {
            let (hashed_address, account) = item?;
            tracing::trace!(target: "loader", ?hashed_address, "merklizing account");

            let storage_root = if account.has_bytecode() {
                StorageRoot::new_hashed(self.tx, hashed_address).root()?
            } else {
                EMPTY_ROOT
            };

            let account = EthAccount::from(account).with_storage_root(storage_root);
            let mut account_rlp = Vec::with_capacity(account.length());
            account.encode(&mut account_rlp);

            let nibbles = Nibbles::unpack(hashed_address);
            hash_builder.add_leaf(nibbles, &account_rlp);

            while let Ok((key, branch_node)) = branch_node_rx.try_recv() {
                self.tx.put::<tables::AccountsTrie2>(key.into(), branch_node.marshal())?;
            }

            progress += 1;
            // there's about 50M accounts hashed, so we want to report every 1%
            // 2 / 100M
            if progress % 500_000 == 0 {
                println!("Accounts Merklized so far: {}", progress);
            }
        }

        let root = hash_builder.root();

        Ok(root)
    }
}

// Create a walker at a specific prefix on the database.
// Pass a cursor to the hashed storage table (or hashed account later)
// For each element:
// 1. If its value is 0, delete it from the Hashed table
// 2. Nibble(key) & Some(rlp(value)) or None
pub struct StorageRoot<'a, TX> {
    pub tx: &'a TX,
    pub hashed_address: H256,
}

impl<'a, TX> StorageRoot<'a, TX> {
    /// Creates a new storage root calculator given an address
    pub fn new(tx: &'a TX, address: Address) -> Self {
        Self::new_hashed(tx, keccak256(&address))
    }

    /// Creates a new storage root calculator given an address
    pub fn new_hashed(tx: &'a TX, hashed_address: H256) -> Self {
        Self { tx, hashed_address }
    }
}

#[derive(Error, Debug)]
pub enum StorageRootError {
    #[error(transparent)]
    DB(#[from] DbError),
}

impl<'a, 'tx, TX: DbTx<'tx> + DbTxMut<'tx>> StorageRoot<'a, TX> {
    /// Walks the entire hashed storage table entry for the given address and calculates the storage
    /// root
    #[tracing::instrument(skip(self))]
    pub fn root(&self) -> Result<H256, StorageRootError> {
        tracing::debug!(target: "loader", hashed_address = ?self.hashed_address, "calculating storage root");

        // Instantiate the walker
        let mut cursor = self.tx.cursor_dup_read::<tables::HashedStorage>()?;
        let mut entry = cursor.seek_by_key_subkey(self.hashed_address, H256::zero())?;

        let (branch_node_tx, mut branch_node_rx) = mpsc::unbounded_channel();
        let mut hash_builder = HashBuilder::default().with_store_tx(branch_node_tx);

        while let Some(StorageEntry { key: hashed_slot, value }) = entry {
            let nibbles = Nibbles::unpack(hashed_slot);
            hash_builder.add_leaf(nibbles, reth_rlp::encode_fixed_size(&value).as_ref());

            while let Ok((key, branch_node)) = branch_node_rx.try_recv() {
                self.tx.put::<tables::StoragesTrie2>(
                    self.hashed_address,
                    StorageTrieEntry2 { nibbles: key.into(), node: branch_node.marshal() },
                )?;
            }

            // Should be able to use walk_dup, but any call to next() causes an assert fail in
            // mdbx.c
            entry = cursor.next_dup()?.map(|(_, v)| v);
        }

        let root = hash_builder.root();

        Ok(root)
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

    fn insert_account<'a, TX: DbTxMut<'a>>(
        tx: &mut TX,
        address: Address,
        account: Account,
        storage: &BTreeMap<H256, U256>,
    ) {
        let hashed_address = keccak256(address);
        tx.put::<tables::HashedAccount>(hashed_address, account).unwrap();

        for (k, v) in storage {
            tx.put::<tables::HashedStorage>(
                hashed_address,
                StorageEntry { key: keccak256(k), value: *v },
            )
            .unwrap();
        }
    }

    fn state_root<I, S>(accounts: I) -> H256
    where
        I: Iterator<Item = (Address, (Account, S))>,
        S: IntoIterator<Item = (H256, U256)>,
    {
        let encoded_accounts = accounts.into_iter().map(|(address, (account, storage))| {
            let storage_root =
                if account.has_bytecode() { storage_root(storage.into_iter()) } else { EMPTY_ROOT };
            let mut out = Vec::new();
            EthAccount::from(account).with_storage_root(storage_root).encode(&mut out);
            (address, out)
        });

        triehash::sec_trie_root::<KeccakHasher, _, _, _>(encoded_accounts)
    }

    fn storage_root<I: Iterator<Item = (H256, U256)>>(storage: I) -> H256 {
        let encoded_storage = storage.map(|(k, v)| (k, encode_fixed_size(&v).to_vec()));

        H256(triehash::sec_trie_root::<KeccakHasher, _, _, _>(encoded_storage).0)
    }

    #[test]
    fn arbitrary_storage_root() {
        proptest!(ProptestConfig::with_cases(10), |(item: (Address, std::collections::BTreeMap<H256, U256>))| {
            let (address, storage) = item;

            let hashed_address = keccak256(address);
            let db = create_test_rw_db();
            let mut tx = Transaction::new(db.as_ref()).unwrap();
            for (key, value) in &storage {
                tx.put::<tables::HashedStorage>(
                    hashed_address,
                    StorageEntry { key: keccak256(key), value: *value },
                )
                .unwrap();
            }
            tx.commit().unwrap();

            let got = StorageRoot::new(tx.deref_mut(), address).root().unwrap();
            let expected = storage_root(storage.into_iter());
            assert_eq!(expected, got);
        });
    }

    #[test]
    // This ensures that the walker goes over all the storage slots
    fn test_storage_root() {
        let db = create_test_rw_db();
        let mut tx = Transaction::new(db.as_ref()).unwrap();

        let address = Address::random();
        let storage = BTreeMap::from([
            (H256::zero(), U256::from(3)),
            (H256::from_low_u64_be(2), U256::from(1)),
        ]);

        let code = "el buen fla";
        let account = Account {
            nonce: 155,
            balance: U256::from(414241124u32),
            bytecode_hash: Some(keccak256(code)),
        };

        insert_account(&mut *tx, address, account, &storage);
        tx.commit().unwrap();

        let got = StorageRoot::new(tx.deref_mut(), address).root().unwrap();

        assert_eq!(storage_root(storage.into_iter()), got);
    }

    type State = BTreeMap<Address, (Account, BTreeMap<H256, U256>)>;

    #[test]
    fn arbitrary_state_root() {
        proptest!(
            ProptestConfig::with_cases(10), | (state: State) | {
                // set the bytecodehash for the accounts so that storage root is computed
                // this is needed because proptest will generate accs with empty bytecodehash
                // but non-empty storage, which is obviously invalid
                let state = state
                    .into_iter()
                    .map(|(addr, (mut acc, storage))| {
                        if !storage.is_empty() {
                            acc.bytecode_hash = Some(H256::random());
                        }
                        (addr, (acc, storage))
                    })
                    .collect::<BTreeMap<_, _>>();
                test_state_root_with_state(state)
            }
        );
    }

    fn test_state_root_with_state(state: State) {
        let db = create_test_rw_db();
        let mut tx = Transaction::new(db.as_ref()).unwrap();

        for (address, (account, storage)) in &state {
            insert_account(&mut *tx, *address, *account, storage)
        }
        tx.commit().unwrap();
        let expected = state_root(state.into_iter());

        let got = StateRoot::new(tx.deref_mut()).root().unwrap();
        assert_eq!(expected, got);
    }
}
