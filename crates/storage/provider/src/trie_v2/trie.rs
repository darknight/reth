// 1. Calculate Root given entire presorted hashed thing
// Must work with:
// 1. Storage Trie Cursor
// 2. Accounts Trie Cursor
// 2. Update root given a list of updates
// Be able to calculate incremental state root without taking a write lock

use std::{collections::HashMap, ops::Range, path::Prefix};

use super::{
    account::EthAccount,
    cursor::{AccountTrieCursor, StorageTrieCursor, TrieWalker},
    hash_builder::{BranchNodeSender, HashBuilder},
    nibbles::Nibbles,
    node::BranchNodeCompact,
    prefix_set::PrefixSet,
};
use reth_db::{
    cursor::{DbCursorRO, DbDupCursorRO},
    models::{AccountBeforeTx, TransitionIdAddress},
    tables,
    transaction::{DbTx, DbTxMut},
    Error as DbError,
};
use reth_primitives::{keccak256, proofs::EMPTY_ROOT, Address, StorageEntry, TransitionId, H256};
use reth_rlp::Encodable;
use thiserror::Error;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

pub type BranchNodeUpdateSender = UnboundedSender<BranchNodeUpdate>;

pub enum BranchNodeUpdate {
    Account(Nibbles, BranchNodeCompact),
    Storage(H256, Nibbles, BranchNodeCompact),
}

pub struct StateRoot<'a, TX> {
    pub tx: &'a TX,
    pub branch_node_update_sender: Option<BranchNodeUpdateSender>,
    pub account_changes: PrefixSet,
    pub storage_changes: HashMap<H256, PrefixSet>,
}

impl<'a, TX> StateRoot<'a, TX> {
    pub fn new(tx: &'a TX) -> Self {
        Self {
            tx,
            branch_node_update_sender: None,
            account_changes: PrefixSet::default(),
            storage_changes: PrefixSet::default(),
        }
    }

    pub fn with_branch_node_update_sender(mut self, sender: BranchNodeUpdateSender) -> Self {
        self.branch_node_update_sender = Some(sender);
        self
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
    /// Walks the entire hashed storage table entry for the given address and calculates the storage
    /// root
    pub async fn root(&self) -> Result<H256, StateRootError> {
        tracing::debug!(target: "loader", "calculating state root");

        let mut hashed_account_cursor = self.tx.cursor_read::<tables::HashedAccount>()?;
        let mut trie_cursor = AccountTrieCursor(self.tx.cursor_write::<tables::AccountsTrie2>()?);
        let mut walker = TrieWalker::new(&mut trie_cursor, self.account_changes.clone());

        let (account_branch_node_tx, mut account_branch_node_rx) = unbounded_channel();
        let mut hash_builder =
            HashBuilder::default().with_branch_node_sender(account_branch_node_tx);

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
                    StorageRoot::new_hashed(
                        self.tx,
                        hashed_address,
                        self.branch_node_update_sender.clone(),
                    )
                    .with_storage_changes(
                        self.storage_changes.get(&hashed_address).cloned().unwrap_or_default(),
                    )
                    .root()
                    .await?
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

        let root = hash_builder.root();
        drop(hash_builder);

        if let Some(sender) = &self.branch_node_update_sender {
            while let Some((nibbles, branch_node)) = account_branch_node_rx.recv().await {
                let _ = sender.send(BranchNodeUpdate::Account(nibbles, branch_node));
            }
        }

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
    pub storage_changes: PrefixSet,
    pub branch_node_update_sender: Option<BranchNodeUpdateSender>,
}

impl<'a, TX> StorageRoot<'a, TX> {
    /// Creates a new storage root calculator given an address
    pub fn new(tx: &'a TX, address: Address, sender: Option<BranchNodeUpdateSender>) -> Self {
        Self::new_hashed(tx, keccak256(&address), sender)
    }

    /// Creates a new storage root calculator given an address
    pub fn new_hashed(
        tx: &'a TX,
        hashed_address: H256,
        branch_node_update_sender: Option<BranchNodeUpdateSender>,
    ) -> Self {
        Self {
            tx,
            hashed_address,
            branch_node_update_sender,
            storage_changes: PrefixSet::default(),
        }
    }

    pub fn with_storage_changes(mut self, changes: PrefixSet) -> Self {
        self.storage_changes = changes;
        self
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
    pub async fn root(&self) -> Result<H256, StorageRootError> {
        tracing::debug!(target: "loader", hashed_address = ?self.hashed_address, "calculating storage root");

        let mut hashed_storage_cursor = self.tx.cursor_dup_read::<tables::HashedStorage>()?;
        let mut trie_cursor = StorageTrieCursor::new(
            self.tx.cursor_dup_write::<tables::StoragesTrie2>()?,
            self.hashed_address,
        );
        let mut walker = TrieWalker::new(&mut trie_cursor, PrefixSet::default() /* TODO: */);

        let (storage_branch_node_tx, mut storage_branch_node_rx) = unbounded_channel();
        let mut hash_builder =
            HashBuilder::default().with_branch_node_sender(storage_branch_node_tx);

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

            // TODO: check
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

        let root = hash_builder.root();
        drop(hash_builder);

        if let Some(sender) = &self.branch_node_update_sender {
            while let Some((nibbles, branch_node)) = storage_branch_node_rx.recv().await {
                let _ = sender.send(BranchNodeUpdate::Storage(
                    self.hashed_address,
                    nibbles,
                    branch_node,
                ));
            }
        }

        Ok(root)
    }
}

fn gather_changes<'a, TX>(
    tx: &TX,
    tid_range: Range<TransitionId>,
) -> Result<(PrefixSet, HashMap<H256, PrefixSet>), reth_interfaces::Error>
where
    TX: DbTx<'a>,
{
    let mut account_prefix_set = PrefixSet::default();
    let mut storage_prefix_set: HashMap<H256, PrefixSet> = HashMap::default();

    let mut account_cursor = tx.cursor_read::<tables::AccountChangeSet>()?;

    let mut walker = account_cursor.walk_range(tid_range.clone())?;

    while let Some((_, AccountBeforeTx { address, .. })) = walker.next().transpose()? {
        account_prefix_set.insert(Nibbles::unpack(keccak256(address)).get_data());
    }

    let mut storage_cursor = tx.cursor_dup_read::<tables::StorageChangeSet>()?;

    let start = TransitionIdAddress((tid_range.start, Address::zero()));
    let end = TransitionIdAddress((tid_range.end, Address::zero()));
    let mut walker = storage_cursor.walk_range(start..end)?;

    while let Some((TransitionIdAddress((_, address)), StorageEntry { key, .. })) =
        walker.next().transpose()?
    {
        storage_prefix_set
            .entry(keccak256(address))
            .or_default()
            .insert(Nibbles::unpack(keccak256(key)).get_data());
    }

    Ok((account_prefix_set, storage_prefix_set))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Transaction;
    use proptest::{prelude::ProptestConfig, proptest};
    use reth_db::{
        cursor::DbCursorRW, mdbx::test_utils::create_test_rw_db, tables, transaction::DbTxMut,
    };
    use reth_primitives::{keccak256, proofs::KeccakHasher, Account, Address, H256, U256};
    use reth_rlp::encode_fixed_size;
    use std::{
        collections::BTreeMap,
        ops::{Deref, DerefMut, Mul},
        str::FromStr,
    };
    use tokio::sync::mpsc;
    use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};

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
            tokio::runtime::Runtime::new().unwrap().block_on(async {
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

                let got = StorageRoot::new(tx.deref_mut(), address, None).root().await.unwrap();
                let expected = storage_root(storage.into_iter());
                assert_eq!(expected, got);
            });

        });
    }

    // TODO:
    // #[tokio::test]
    // This ensures we dont add empty accounts to the trie
    async fn test_empty_account() {
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
        test_state_root_with_state(state).await;
    }

    #[tokio::test]
    // This ensures we return an empty root when there are no storage entries
    async fn test_empty_storage_root() {
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

        let got = StorageRoot::new(tx.deref_mut(), address, None).root().await.unwrap();
        assert_eq!(got, EMPTY_ROOT);
    }

    #[tokio::test]
    // This ensures that the walker goes over all the storage slots
    async fn test_storage_root() {
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

        let got = StorageRoot::new(tx.deref_mut(), address, None).root().await.unwrap();

        assert_eq!(storage_root(storage.into_iter()), got);
    }

    type State = BTreeMap<Address, (Account, BTreeMap<H256, U256>)>;

    #[test]
    fn arbitrary_state_root() {
        proptest!(
            ProptestConfig::with_cases(10), | (state: State) | {
                tokio::runtime::Runtime::new().unwrap().block_on(async {
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
                    test_state_root_with_state(state).await
                })

            }
        );
    }

    async fn test_state_root_with_state(state: State) {
        let db = create_test_rw_db();
        let mut tx = Transaction::new(db.as_ref()).unwrap();

        for (address, (account, storage)) in &state {
            insert_account(&mut *tx, *address, *account, storage)
        }
        tx.commit().unwrap();
        let expected = state_root(state.into_iter());

        let got = StateRoot::new(tx.deref_mut()).root().await.unwrap();
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

    #[tokio::test]
    async fn account_and_storage_trie() {
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
        let account3_storage_root =
            StorageRoot::new(tx.deref_mut(), address3, None).root().await.unwrap();
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
        let (branch_node_tx, branch_node_rx) = mpsc::unbounded_channel();
        let loader = StateRoot::new(tx.deref()).with_branch_node_update_sender(branch_node_tx);
        assert_eq!(loader.root().await.unwrap(), computed_expected_root);

        // ----------------------------------------------------------------
        // Check account trie
        // ----------------------------------------------------------------

        drop(loader);
        let branch_node_stream = UnboundedReceiverStream::new(branch_node_rx);
        let updates = branch_node_stream.collect::<Vec<_>>().await;

        let account_updates = updates
            .iter()
            .filter_map(|u| {
                if let BranchNodeUpdate::Account(nibbles, node) = u {
                    Some((nibbles, node))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(account_updates.len(), 2);

        let (nibbles1a, node1a) = account_updates.last().unwrap();
        assert_eq!(nibbles1a.get_data(), [0xB]);
        assert_eq!(node1a.state_mask, 0b1011);
        assert_eq!(node1a.tree_mask, 0b0001);
        assert_eq!(node1a.hash_mask, 0b1001);
        assert_eq!(node1a.root_hash, None);
        assert_eq!(node1a.hashes.len(), 2);

        let (nibbles2a, node2a) = account_updates.first().unwrap();
        assert_eq!(nibbles2a.get_data(), [0xB, 0x0]);
        assert_eq!(node2a.state_mask, 0b10001);
        assert_eq!(node2a.tree_mask, 0b00000);
        assert_eq!(node2a.hash_mask, 0b10000);
        assert_eq!(node2a.root_hash, None);
        assert_eq!(node2a.hashes.len(), 1);

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

        let mut prefix_set = PrefixSet::default();
        prefix_set.insert(Nibbles::unpack(key4b).get_data());

        let expected_state_root =
            H256::from_str("8e263cd4eefb0c3cbbb14e5541a66a755cad25bcfab1e10dd9d706263e811b28")
                .unwrap();

        let (branch_node_tx, branch_node_rx) = mpsc::unbounded_channel();
        let loader = StateRoot::new(tx.deref())
            .with_branch_node_update_sender(branch_node_tx)
            .with_account_changes(prefix_set);
        assert_eq!(loader.root().await.unwrap(), expected_state_root);

        drop(loader);
        let branch_node_stream = UnboundedReceiverStream::new(branch_node_rx);
        let updates = branch_node_stream.collect::<Vec<_>>().await;

        let account_updates = updates
            .iter()
            .filter_map(|u| {
                if let BranchNodeUpdate::Account(nibbles, node) = u {
                    Some((nibbles, node))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(account_updates.len(), 2);

        let (nibbles1b, node1b) = account_updates.last().unwrap();
        assert_eq!(nibbles1b.get_data(), [0xB]);
        assert_eq!(node1b.state_mask, 0b1011);
        assert_eq!(node1b.tree_mask, 0b0001);
        assert_eq!(node1b.hash_mask, 0b1011);
        assert_eq!(node1b.root_hash, None);
        assert_eq!(node1b.hashes.len(), 3);
        assert_eq!(node1a.hashes[0], node1b.hashes[0]);
        assert_eq!(node1a.hashes[1], node1b.hashes[2]);

        // let node2b = &node_map[&vec![0xB, 0x0]];
        let (nibbles2b, node2b) = account_updates.first().unwrap();
        assert_eq!(nibbles2b.get_data(), [0xB, 0x0]);
        assert_eq!(node2a, node2b);

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
