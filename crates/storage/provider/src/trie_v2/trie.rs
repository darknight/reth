// 1. Calculate Root given entire presorted hashed thing
// Must work with:
// 1. Storage Trie Cursor
// 2. Accounts Trie Cursor
// 2. Update root given a list of updates
// Be able to calculate incremental state root without taking a write lock

use super::{
    account::EthAccount,
    cursor::{AccountTrieCursor, StorageTrieCursor, TrieWalker},
    hash_builder::HashBuilder,
    nibbles::Nibbles,
    prefix_set::PrefixSet,
};
use reth_db::{
    cursor::{DbCursorRO, DbDupCursorRO},
    tables,
    transaction::{DbTx, DbTxMut},
    Error as DbError,
};
use reth_primitives::{keccak256, proofs::EMPTY_ROOT, Address, StorageEntry, H256};
use reth_rlp::Encodable;
use thiserror::Error;

pub struct StateRoot<'a, TX> {
    pub tx: &'a TX,
    pub account_changes: PrefixSet,
    pub storage_changes: PrefixSet,
}

impl<'a, TX> StateRoot<'a, TX> {
    pub fn new(tx: &'a TX) -> Self {
        Self { tx, account_changes: PrefixSet::default(), storage_changes: PrefixSet::default() }
    }

    pub fn with_account_changes(mut self, changes: PrefixSet) -> Self {
        self.account_changes = changes;
        self
    }

    pub fn with_storage_changes(mut self, changes: PrefixSet) -> Self {
        self.storage_changes = changes;
        self
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
    // /// Walks the entire hashed storage table entry for the given address and calculates the
    // storage /// root
    // #[tracing::instrument(skip(self))]
    // pub fn root(&self) -> Result<H256, StateRootError> {
    //     tracing::debug!(target: "loader", "calculating state root");
    //     // Instantiate the walker
    //     let mut cursor = self.tx.cursor_read::<tables::HashedAccount>()?;
    //     let mut walker = cursor.walk(None)?;

    //     // let (branch_node_tx, mut branch_node_rx) = mpsc::unbounded_channel();
    //     let mut hash_builder = HashBuilder::default(); // .with_store_tx(branch_node_tx);

    //     let mut progress = 0;
    //     while let Some(item) = walker.next() {
    //         let (hashed_address, account) = item?;
    //         tracing::trace!(target: "loader", ?hashed_address, "merklizing account");

    //         if account.is_empty() {
    //             continue
    //         }

    //         let storage_root = if account.has_bytecode() {
    //             StorageRoot::new_hashed(self.tx, hashed_address).root()?
    //         } else {
    //             EMPTY_ROOT
    //         };

    //         let account = EthAccount::from(account).with_storage_root(storage_root);
    //         let mut account_rlp = Vec::with_capacity(account.length());
    //         account.encode(&mut account_rlp);

    //         let nibbles = Nibbles::unpack(hashed_address);
    //         hash_builder.add_leaf(nibbles, &account_rlp);

    //         // while let Ok((key, branch_node)) = branch_node_rx.try_recv() {
    //         //     self.tx.put::<tables::AccountsTrie2>(key.hex_data.into(),
    //         // branch_node.marshal())?; }

    //         progress += 1;
    //         // there's about 50M accounts hashed, so we want to report every 1%
    //         // 2 / 100M
    //         if progress % 500_000 == 0 {
    //             println!("Accounts Merklized so far: {}", progress);
    //         }
    //     }

    //     let root = hash_builder.root();

    //     Ok(root)
    // }

    pub fn root(&self) -> Result<H256, StateRootError> {
        let mut hashed_account_cursor = self.tx.cursor_read::<tables::HashedAccount>()?;
        let mut trie_cursor = AccountTrieCursor(self.tx.cursor_write::<tables::AccountsTrie2>()?);
        let mut walker = TrieWalker::new(&mut trie_cursor, PrefixSet::default() /* TODO: */);

        let mut hash_builder = HashBuilder::default();

        while let Some(key) = walker.key() {
            if walker.can_skip_state {
                hash_builder.add_branch_from_db(
                    Nibbles::unpack(key),
                    walker.hash().clone().unwrap(),
                    walker.children_are_in_trie(),
                );
            }

            let seek_key = match walker.first_uncovered_prefix() {
                Some(mut uncovered) => {
                    uncovered.resize(32, 0);
                    H256::from_slice(uncovered.as_slice())
                }
                None => break,
            };

            walker.next().unwrap(); // TODO: handle

            let mut acc = hashed_account_cursor.seek(seek_key)?;
            let trie_key = walker.key();

            while let Some((hashed_address, account)) = acc {
                let packed_key = hashed_address.as_bytes();
                let unpacked_key = Nibbles::unpack(packed_key);

                if let Some(ref key) = trie_key {
                    if Nibbles::from(key.as_slice()) < unpacked_key {
                        break
                    }
                }

                // TODO: fix this
                // let storage_root =
                //     self.calculate_storage_root(address.as_bytes(), storage_changes)?;
                let storage_root = if account.has_bytecode() {
                    StorageRoot::new_hashed(self.tx, hashed_address).root()?
                } else {
                    EMPTY_ROOT
                };

                let account = EthAccount::from(account).with_storage_root(storage_root);
                let mut account_rlp = Vec::with_capacity(account.length());
                account.encode(&mut account_rlp);

                hash_builder.add_leaf(unpacked_key, &account_rlp);

                acc = hashed_account_cursor.next()?;
            }
        }

        Ok(hash_builder.root())
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
    // /// Walks the entire hashed storage table entry for the given address and calculates the
    // storage /// root
    // #[tracing::instrument(skip(self))]
    // pub fn root(&self) -> Result<H256, StorageRootError> {
    //     tracing::debug!(target: "loader", hashed_address = ?self.hashed_address, "calculating
    // storage root");

    //     // Instantiate the cursor
    //     let mut cursor = self.tx.cursor_dup_read::<tables::HashedStorage>()?;
    //     let mut entry = cursor.seek_by_key_subkey(self.hashed_address, H256::zero())?;

    //     // let (branch_node_tx, mut branch_node_rx) = mpsc::unbounded_channel();
    //     let mut hash_builder = HashBuilder::default(); // .with_store_tx(branch_node_tx);
    //     while let Some(StorageEntry { key: hashed_slot, value }) = entry {
    //         let nibbles = Nibbles::unpack(hashed_slot);
    //         hash_builder.add_leaf(nibbles, reth_rlp::encode_fixed_size(&value).as_ref());

    //         // while let Ok((key, branch_node)) = branch_node_rx.try_recv() {
    //         //     self.tx.put::<tables::StoragesTrie2>(
    //         //         self.hashed_address,
    //         //         StorageTrieEntry2 { nibbles: key.hex_data.into(), node:
    // branch_node.marshal()         // },     )?;
    //         // }

    //         // Should be able to use walk_dup, but any call to next() causes an assert fail in
    //         // mdbx.c
    //         entry = cursor.next_dup()?.map(|(_, v)| v);
    //     }

    //     let root = hash_builder.root();

    //     Ok(root)
    // }

    pub fn root(&self) -> Result<H256, StorageRootError> {
        let mut hashed_storage_cursor = self.tx.cursor_dup_read::<tables::HashedStorage>()?;
        let mut trie_cursor = StorageTrieCursor::new(
            self.tx.cursor_dup_write::<tables::StoragesTrie2>()?,
            self.hashed_address,
        );
        let mut walker = TrieWalker::new(&mut trie_cursor, PrefixSet::default() /* TODO: */);

        let mut hash_builder = HashBuilder::default();

        while let Some(key) = walker.key() {
            if walker.can_skip_state {
                hash_builder.add_branch_from_db(
                    Nibbles::unpack(key),
                    walker.hash().clone().unwrap(),
                    walker.children_are_in_trie(),
                );
            }

            let seek_key = match walker.first_uncovered_prefix() {
                Some(mut uncovered) => {
                    uncovered.resize(32, 0);
                    uncovered
                }
                None => break,
            };

            walker.next().unwrap(); // TODO:

            // TODO:
            let mut storage = hashed_storage_cursor
                .seek_by_key_subkey(self.hashed_address, H256::from_slice(seek_key.as_slice()))?;
            let trie_key = walker.key();

            while let Some(StorageEntry { key: hashed_key, value }) = storage {
                let unpacked_loc = Nibbles::unpack(hashed_key.as_bytes());
                if let Some(ref key) = trie_key {
                    if Nibbles::from(key.as_slice()) < unpacked_loc {
                        break
                    }
                }
                hash_builder.add_leaf(unpacked_loc, reth_rlp::encode_fixed_size(&value).as_ref());
                storage = hashed_storage_cursor.next_dup()?.map(|(_, v)| v);
            }
        }

        Ok(hash_builder.root())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Transaction;
    use proptest::{prelude::ProptestConfig, proptest};
    use reth_db::{
        cursor::DbCursorRW, mdbx::test_utils::create_test_rw_db, models::AccountBeforeTx, tables,
        transaction::DbTxMut,
    };
    use reth_primitives::{keccak256, proofs::KeccakHasher, Account, Address, H256, U256};
    use reth_rlp::encode_fixed_size;
    use std::{
        collections::BTreeMap,
        ops::{Deref, DerefMut, Mul},
        str::FromStr,
    };

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
        let encoded_accounts = accounts.into_iter().filter_map(|(address, (account, storage))| {
            if account.is_empty() {
                return None
            }
            let storage_root =
                if account.has_bytecode() { storage_root(storage.into_iter()) } else { EMPTY_ROOT };
            let mut out = Vec::new();
            EthAccount::from(account).with_storage_root(storage_root).encode(&mut out);
            Some((address, out))
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

    // TODO:
    // #[test]
    // This ensures we dont add empty accounts to the trie
    fn test_empty_account() {
        let state: State = BTreeMap::from([
            (
                Address::random(),
                (
                    Account { nonce: 0, balance: U256::from(0), bytecode_hash: None },
                    BTreeMap::default(),
                ),
            ),
            (
                Address::random(),
                (
                    Account {
                        nonce: 155,
                        balance: U256::from(414241124u32),
                        bytecode_hash: Some(keccak256("test")),
                    },
                    BTreeMap::from([
                        (H256::zero(), U256::from(3)),
                        (H256::from_low_u64_be(2), U256::from(1)),
                    ]),
                ),
            ),
        ]);
        test_state_root_with_state(state);
    }

    #[test]
    // This ensures we return an empty root when there are no storage entries
    fn test_empty_storage_root() {
        let db = create_test_rw_db();
        let mut tx = Transaction::new(db.as_ref()).unwrap();

        let address = Address::random();
        let code = "el buen fla";
        let account = Account {
            nonce: 155,
            balance: U256::from(414241124u32),
            bytecode_hash: Some(keccak256(code)),
        };
        insert_account(&mut *tx, address, account, &Default::default());
        tx.commit().unwrap();

        let got = StorageRoot::new(tx.deref_mut(), address).root().unwrap();
        assert_eq!(got, EMPTY_ROOT);
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

    fn encode_account(account: Account, storage_root: Option<H256>) -> Vec<u8> {
        let mut account = EthAccount::from(account);
        if let Some(storage_root) = storage_root {
            account = account.with_storage_root(storage_root);
        }
        let mut account_rlp = Vec::with_capacity(account.length());
        account.encode(&mut account_rlp);
        account_rlp
    }

    #[test]
    fn account_and_storage_trie() {
        let ether = U256::from(1e18);
        let storage = BTreeMap::from(
            [
                ("1200000000000000000000000000000000000000000000000000000000000000", 0x42_u128),
                ("1400000000000000000000000000000000000000000000000000000000000000", 0x01_u128),
                ("3000000000000000000000000000000000000000000000000000000000E00000", 0x127a89_u128),
                ("3000000000000000000000000000000000000000000000000000000000E00001", 0x05_u128),
            ]
            .map(|(slot, val)| (H256::from_str(slot).unwrap(), U256::from(val))),
        );

        let db = create_test_rw_db();
        let mut tx = Transaction::new(db.as_ref()).unwrap();

        // TODO: let mut hashed_accounts = txn.cursor(tables::HashedAccount).unwrap();
        let mut hashed_account_cursor = tx.cursor_write::<tables::HashedAccount>().unwrap();
        let mut hashed_storage_cursor = tx.cursor_dup_write::<tables::HashedStorage>().unwrap();

        // TODO: let mut hb = HashBuilder::new(None);
        let mut hash_builder = HashBuilder::default();

        // Insert first account
        let key1 =
            H256::from_str("b000000000000000000000000000000000000000000000000000000000000000")
                .unwrap();
        let account1 = Account { nonce: 0, balance: U256::from(3).mul(ether), bytecode_hash: None };
        hashed_account_cursor.upsert(key1, account1).unwrap();
        hash_builder.add_leaf(Nibbles::unpack(key1), &encode_account(account1, None));

        // Some address whose hash starts with 0xB040
        let address2 = Address::from_str("7db3e81b72d2695e19764583f6d219dbee0f35ca").unwrap();
        let key2 = keccak256(address2);
        assert_eq!(key2[0], 0xB0);
        assert_eq!(key2[1], 0x40);
        let account2 = Account { nonce: 0, balance: ether.clone(), ..Default::default() };
        hashed_account_cursor.upsert(key2, account2).unwrap();
        hash_builder.add_leaf(Nibbles::unpack(key2), &encode_account(account2, None));

        // Some address whose hash starts with 0xB041
        let address3 = Address::from_str("16b07afd1c635f77172e842a000ead9a2a222459").unwrap();
        let key3 = keccak256(address3);
        assert_eq!(key3[0], 0xB0);
        assert_eq!(key3[1], 0x41);
        let code_hash =
            H256::from_str("5be74cad16203c4905c068b012a2e9fb6d19d036c410f16fd177f337541440dd")
                .unwrap();
        let account3 =
            Account { nonce: 0, balance: U256::from(2).mul(ether), bytecode_hash: Some(code_hash) };
        hashed_account_cursor.upsert(key3, account3).unwrap();
        for (hashed_slot, value) in storage {
            if hashed_storage_cursor
                .seek_by_key_subkey(key3, hashed_slot)
                .unwrap()
                .filter(|e| e.key == hashed_slot)
                .is_some()
            {
                hashed_storage_cursor.delete_current().unwrap();
            }
            hashed_storage_cursor.upsert(key3, StorageEntry { key: hashed_slot, value }).unwrap();
        }
        let account3_storage_root = StorageRoot::new(tx.deref_mut(), address3).root().unwrap();
        hash_builder.add_leaf(
            Nibbles::unpack(key3),
            &encode_account(account3, Some(account3_storage_root)),
        );

        let key4a =
            H256::from_str("B1A0000000000000000000000000000000000000000000000000000000000000")
                .unwrap();
        let account4a =
            Account { nonce: 0, balance: U256::from(4).mul(ether), ..Default::default() };
        hashed_account_cursor.upsert(key4a, account4a).unwrap();
        hash_builder.add_leaf(Nibbles::unpack(key4a), &encode_account(account4a, None));

        let key5 =
            H256::from_str("B310000000000000000000000000000000000000000000000000000000000000")
                .unwrap();
        let account5 =
            Account { nonce: 0, balance: U256::from(8).mul(ether), ..Default::default() };
        hashed_account_cursor.upsert(key5, account5).unwrap();
        hash_builder.add_leaf(Nibbles::unpack(key5), &encode_account(account5, None));

        let key6 =
            H256::from_str("B340000000000000000000000000000000000000000000000000000000000000")
                .unwrap();
        let account6 =
            Account { nonce: 0, balance: U256::from(1).mul(ether), ..Default::default() };
        hashed_account_cursor.upsert(key6, account6).unwrap();
        hash_builder.add_leaf(Nibbles::unpack(key6), &encode_account(account6, None));

        // ----------------------------------------------------------------
        // Populate account & storage trie DB tables
        // ----------------------------------------------------------------

        let expected_root =
            H256::from_str("72861041bc90cd2f93777956f058a545412b56de79af5eb6b8075fe2eabbe015")
                .unwrap();
        let computed_expected_root: H256 = triehash::trie_root::<KeccakHasher, _, _, _>([
            (key1, encode_account(account1, None)),
            (key2, encode_account(account2, None)),
            (key3, encode_account(account3, Some(account3_storage_root))),
            (key4a, encode_account(account4a, None)),
            (key5, encode_account(account5, None)),
            (key6, encode_account(account6, None)),
        ]);
        // Check computed trie root to ensure correctness
        assert_eq!(computed_expected_root, expected_root);

        // Check hash builder root
        assert_eq!(hash_builder.root(), computed_expected_root);

        // Check state root calculation from scratch
        assert_eq!(StateRoot::new(tx.deref()).root().unwrap(), computed_expected_root);

        // ----------------------------------------------------------------
        // Check account trie
        // ----------------------------------------------------------------

        // let node_map = read_all_nodes(txn.cursor(tables::TrieAccount).unwrap());
        // assert_eq!(node_map.len(), 2);

        // let node1a = &node_map[&vec![0xB]];

        // assert_eq!(node1a.state_mask, 0b1011);
        // assert_eq!(node1a.tree_mask, 0b0001);
        // assert_eq!(node1a.hash_mask, 0b1001);

        // assert_eq!(node1a.root_hash, None);
        // assert_eq!(node1a.hashes.len(), 2);

        // let node2a = &node_map[&vec![0xB, 0x0]];

        // assert_eq!(node2a.state_mask, 0b10001);
        // assert_eq!(node2a.tree_mask, 0b00000);
        // assert_eq!(node2a.hash_mask, 0b10000);

        // assert_eq!(node2a.root_hash, None);
        // assert_eq!(node2a.hashes.len(), 1);

        // ----------------------------------------------------------------
        // Check storage trie
        // ----------------------------------------------------------------
        // let node_map = read_all_nodes(txn.cursor(tables::TrieStorage).unwrap());
        // assert_eq!(node_map.len(), 1);

        // let node3 = &node_map[&key3.0.to_vec()];

        // assert_eq!(node3.state_mask, 0b1010);
        // assert_eq!(node3.tree_mask, 0b0000);
        // assert_eq!(node3.hash_mask, 0b0010);

        // assert_eq!(node3.root_hash, Some(storage_root));
        // assert_eq!(node3.hashes.len(), 1);

        // ----------------------------------------------------------------
        // Add an account
        // ----------------------------------------------------------------

        // Some address whose hash starts with 0xB1
        let address4b = Address::from_str("4f61f2d5ebd991b85aa1677db97307caf5215c91").unwrap();
        let key4b = keccak256(address4b);
        assert_eq!(key4b.0[0], key4a.0[0]);
        let account4b =
            Account { nonce: 0, balance: U256::from(5).mul(ether), bytecode_hash: None };
        hashed_account_cursor.upsert(key4b, account4b).unwrap();

        let changeset = AccountBeforeTx { address: address4b, info: None };

        // let mut account_change_table = txn.cursor(tables::AccountChangeSet).unwrap();
        // account_change_table
        //     .upsert(BlockNumber(1), tables::AccountChange { address: address4b, account: None })
        //     .unwrap();

        // increment_intermediate_hashes(&txn, &temp_dir, BlockNumber(0), None).unwrap();

        // let node_map = read_all_nodes(txn.cursor(tables::TrieAccount).unwrap());
        // assert_eq!(node_map.len(), 2);

        // let node1b = &node_map[&vec![0xB]];
        // assert_eq!(node1b.state_mask, 0b1011);
        // assert_eq!(node1b.tree_mask, 0b0001);
        // assert_eq!(node1b.hash_mask, 0b1011);

        // assert_eq!(node1b.root_hash, None);

        // assert_eq!(node1b.hashes.len(), 3);
        // assert_eq!(node1a.hashes[0], node1b.hashes[0]);
        // assert_eq!(node1a.hashes[1], node1b.hashes[2]);

        // let node2b = &node_map[&vec![0xB, 0x0]];
        // assert_eq!(node2a, node2b);

        // drop(hashed_accounts);
        // drop(account_change_table);
        // txn.commit().unwrap();

        // --------------------------
        // // Delete an account
        // {
        //     let txn = db.begin_mutable().unwrap();
        //     let mut hashed_accounts = txn.cursor(tables::HashedAccount).unwrap();
        //     let account_trie = txn.cursor(tables::TrieAccount).unwrap();
        //     let mut account_change_table = txn.cursor(tables::AccountChangeSet).unwrap();
        //     {
        //         let account = hashed_accounts.seek_exact(key2).unwrap().unwrap().1;
        //         hashed_accounts.delete_current().unwrap();
        //         account_change_table
        //             .upsert(
        //                 BlockNumber(2),
        //                 tables::AccountChange { address: address2, account: Some(account) },
        //             )
        //             .unwrap();
        //     }

        //     increment_intermediate_hashes(&txn, &temp_dir, BlockNumber(1), None).unwrap();

        //     let node_map = read_all_nodes(account_trie);
        //     assert_eq!(node_map.len(), 1);

        //     let node1c = &node_map[&vec![0xB]];
        //     assert_eq!(node1c.state_mask, 0b1011);
        //     assert_eq!(node1c.tree_mask, 0b0000);
        //     assert_eq!(node1c.hash_mask, 0b1011);

        //     assert_eq!(node1c.root_hash, None);

        //     assert_eq!(node1c.hashes.len(), 3);
        //     assert_ne!(node1b.hashes[0], node1c.hashes[0]);
        //     assert_eq!(node1b.hashes[1], node1c.hashes[1]);
        //     assert_eq!(node1b.hashes[2], node1c.hashes[2]);
        // }

        // --------------------------
        // // Delete several accounts
        // {
        //     let txn = db.begin_mutable().unwrap();
        //     let mut hashed_accounts = txn.cursor(tables::HashedAccount).unwrap();
        //     let account_trie = txn.cursor(tables::TrieAccount).unwrap();
        //     let mut account_change_table = txn.cursor(tables::AccountChangeSet).unwrap();
        //     for (key, address) in [(key2, address2), (key3, address3)] {
        //         let account = hashed_accounts.seek_exact(key).unwrap().unwrap().1;
        //         hashed_accounts.delete_current().unwrap();
        //         account_change_table
        //             .upsert(
        //                 BlockNumber(2),
        //                 tables::AccountChange { address, account: Some(account) },
        //             )
        //             .unwrap();
        //     }

        //     increment_intermediate_hashes(&txn, &temp_dir, BlockNumber(1), None).unwrap();

        //     assert_eq!(
        //         read_all_nodes(account_trie),
        //         hashmap! {
        //             vec![0xB] => Node::new(0b1011, 0b0000, 0b1010, vec![node1b.hashes[1],
        // node1b.hashes[2]], None)         }
        //     );
        // }
    }
}
