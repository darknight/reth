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
    cursor::{DbCursorRO, DbCursorRW, DbDupCursorRO},
    models::{AccountBeforeTx, TransitionIdAddress},
    tables,
    transaction::{DbTx, DbTxMut},
    Error as DbError,
};
use reth_primitives::{
    keccak256, proofs::EMPTY_ROOT, Address, BlockNumber, StorageEntry, TransitionId, H256,
};
use reth_rlp::Encodable;
use thiserror::Error;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

pub type BranchNodeUpdateSender = UnboundedSender<BranchNodeUpdate>;

#[derive(Debug, Clone)]
pub enum BranchNodeUpdate {
    Account(Nibbles, BranchNodeCompact),
    Storage(H256, Nibbles, BranchNodeCompact),
}

#[derive(Debug)]
/// Collects all the updates to the trie and flushes them to the database in a sorted order.
struct TrieUpdates<'a, TX> {
    accounts: BTreeMap<Nibbles, BranchNodeCompact>,
    storages: BTreeMap<H256, BTreeMap<Nibbles, BranchNodeCompact>>,
    threshold: usize,
    count: usize,
    tx: &'a TX,
}

use std::collections::BTreeMap;

impl<'a, 'tx, TX> TrieUpdates<'a, TX>
where
    TX: DbTxMut<'tx> + DbTx<'tx>,
{
    fn new(tx: &'a TX) -> Self {
        Self { tx, accounts: BTreeMap::new(), storages: BTreeMap::new(), threshold: 100, count: 0 }
    }

    fn with_threshold(mut self, threshold: usize) -> Self {
        self.threshold = threshold;
        self
    }

    /// Adds an account update to the trie.
    fn add_account(&mut self, nibbles: Nibbles, branch_node: BranchNodeCompact) {
        self.accounts.insert(nibbles, branch_node);
        self.count += 1;

        if self.count == self.threshold {
            self.flush_to_db();
        }
    }

    fn add_storage(&mut self, address: H256, nibbles: Nibbles, branch_node: BranchNodeCompact) {
        self.storages.entry(address).or_insert_with(BTreeMap::new).insert(nibbles, branch_node);
        self.count += 1;

        if self.count == self.threshold {
            self.flush_to_db();
        }
    }

    #[tracing::instrument(skip_all, fields(count = self.count))]
    fn flush_to_db(&mut self) -> Result<(), StateRootError> {
        let mut account_cursor = self.tx.cursor_write::<tables::AccountsTrie2>()?;
        let mut storage_cursor = self.tx.cursor_dup_write::<tables::StoragesTrie2>()?;

        for (nibbles, branch_node) in std::mem::take(&mut self.accounts) {
            account_cursor.upsert(nibbles.hex_data.into(), branch_node.marshal())?;
        }

        for (hashed_address, storage) in std::mem::take(&mut self.storages) {
            for (nibbles, branch_node) in storage {
                storage_cursor.upsert(
                    hashed_address,
                    reth_primitives::StorageTrieEntry2 {
                        nibbles: nibbles.hex_data.into(),
                        node: branch_node.marshal(),
                    },
                )?;
            }
        }

        Ok(())
    }
}

pub struct StateRoot<'a, TX> {
    pub tx: &'a TX,
    pub account_changes: PrefixSet,
    pub storage_changes: HashMap<H256, PrefixSet>,
    pub flush_db_threshold: usize,
}

impl<'a, TX> StateRoot<'a, TX> {
    pub fn new(tx: &'a TX) -> Self {
        Self {
            tx,
            account_changes: PrefixSet::default(),
            storage_changes: HashMap::default(),
            // TODO: What should this be? How many trie updates do we want to keep in memory
            // before flushing to the database?
            flush_db_threshold: 1000,
        }
    }

    pub fn with_flush_db_threshold(mut self, threshold: usize) -> Self {
        self.flush_db_threshold = threshold;
        self
    }

    pub fn with_account_changes(mut self, changes: PrefixSet) -> Self {
        self.account_changes = changes;
        self
    }

    pub fn with_storage_changes(mut self, changes: HashMap<H256, PrefixSet>) -> Self {
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
    pub async fn incremental_root(
        tx: &'a TX,
        tid_range: Range<TransitionId>,
        branch_node_sender: Option<BranchNodeUpdateSender>,
    ) -> Result<H256, StateRootError> {
        tracing::debug!(target: "loader", "incremental state root");
        dbg!(&tid_range);
        let (account_prefixes, storage_prefixes) = gather_changes(tx, tid_range)?;
        dbg!(&account_prefixes, &storage_prefixes);

        let this = Self::new(tx)
            .with_account_changes(account_prefixes)
            .with_storage_changes(storage_prefixes);

        let root = this.root(branch_node_sender).await?;

        Ok(root)
    }

    /// Walks the entire hashed storage table entry for the given address and calculates the storage
    /// root
    pub async fn root(
        &self,
        branch_node_sender: Option<BranchNodeUpdateSender>,
    ) -> Result<H256, StateRootError> {
        tracing::debug!(target: "loader", "calculating state root");

        let (sender, maybe_receiver) = match branch_node_sender {
            Some(sender) => (sender, None),
            None => {
                let (sender, recv) = unbounded_channel();
                (sender, Some(recv))
            }
        };

        let mut hashed_account_cursor = self.tx.cursor_read::<tables::HashedAccount>()?;
        let mut trie_cursor = AccountTrieCursor(self.tx.cursor_write::<tables::AccountsTrie2>()?);
        let mut walker = TrieWalker::new(&mut trie_cursor, self.account_changes.clone());

        let (account_branch_node_tx, mut account_branch_node_rx) = unbounded_channel();
        let mut hash_builder =
            HashBuilder::default().with_branch_node_sender(account_branch_node_tx);

        while let Some(key) = walker.key() {
            if walker.can_skip_state {
                let key = Nibbles::unpack(key);
                tracing::info!(?key, "skipping state");
                hash_builder.add_branch_from_db(
                    key,
                    walker.hash().clone().unwrap(),
                    walker.children_are_in_trie(),
                );
            }

            let seek_key = match walker.first_uncovered_prefix() {
                Some(mut uncovered) => {
                    uncovered.resize(32, 0);
                    H256::from_slice(uncovered.as_slice())
                }
                None => {
                    tracing::info!("skipping, no prefix");
                    break
                }
            };

            walker.next().unwrap(); // TODO: handle

            let mut acc = hashed_account_cursor.seek(seek_key)?;
            let trie_key = walker.key();

            while let Some((hashed_address, account)) = acc {
                let packed_key = hashed_address.as_bytes();
                let unpacked_key = Nibbles::unpack(packed_key);

                if let Some(ref key) = trie_key {
                    if Nibbles::from(key.as_slice()) < unpacked_key {
                        tracing::info!("breaking, already detected");
                        break
                    }
                }

                // We assume we can always calculate a storage root without
                // OOMing. This opens us up to a potential DOS vector if
                // a contract had too many storage entries and they were
                // all buffered w/o us returning and committing our intermeditate
                // progress.
                // TODO: We can consider introducing the TrieProgress::Progress/Complete
                // abstraction inside StorageRoot, but let's give it a try as-is for now.
                let storage_root =
                    StorageRoot::new_hashed(self.tx, hashed_address, Some(sender.clone()))
                        .with_storage_changes(
                            self.storage_changes.get(&hashed_address).cloned().unwrap_or_default(),
                        )
                        .root()
                        .await?;

                let account = EthAccount::from(account).with_storage_root(storage_root);
                let mut account_rlp = Vec::with_capacity(account.length());
                account.encode(&mut account_rlp);

                hash_builder.add_leaf(unpacked_key, &account_rlp);

                acc = hashed_account_cursor.next()?;
            }
        }

        let root = hash_builder.root();
        drop(hash_builder);

        while let Some((nibbles, branch_node)) = account_branch_node_rx.recv().await {
            let _ = sender.send(BranchNodeUpdate::Account(nibbles, branch_node));
        }
        drop(sender);

        if let Some(mut receiver) = maybe_receiver {
            let mut updates = TrieUpdates::new(self.tx).with_threshold(self.flush_db_threshold);
            while let Some(update) = receiver.recv().await {
                match update {
                    BranchNodeUpdate::Account(nibbles, branch_node) => {
                        updates.add_account(nibbles, branch_node);
                    }
                    BranchNodeUpdate::Storage(hashed_address, nibbles, branch_node) => {
                        updates.add_storage(hashed_address, nibbles, branch_node);
                    }
                }
            }

            // flush once more to make sure that any leftover updates are also written
            updates.flush_to_db()?;
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
        let mut walker = TrieWalker::new(&mut trie_cursor, self.storage_changes.clone());

        let (storage_branch_node_tx, mut storage_branch_node_rx) = unbounded_channel();
        let mut hash_builder =
            HashBuilder::default().with_branch_node_sender(storage_branch_node_tx);

        while let Some(key) = walker.key() {
            if walker.can_skip_state {
                // do not add a branch node on empty storage
                if hashed_storage_cursor.seek_exact(self.hashed_address)?.is_none() {
                    return Ok(EMPTY_ROOT)
                }
                hash_builder.add_branch_from_db(
                    Nibbles::unpack(key),
                    walker.hash().unwrap(),
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

#[tracing::instrument(skip(tx))]
fn gather_changes<'a, TX>(
    tx: &TX,
    tid_range: Range<TransitionId>,
) -> Result<(PrefixSet, HashMap<H256, PrefixSet>), DbError>
where
    TX: DbTx<'a>,
{
    let mut account_prefix_set = PrefixSet::default();
    let mut storage_prefix_set: HashMap<H256, PrefixSet> = HashMap::default();

    let mut account_cursor = tx.cursor_read::<tables::AccountChangeSet>()?;

    let mut walker = account_cursor.walk_range(tid_range.clone())?;

    while let Some((key, AccountBeforeTx { address, info })) = walker.next().transpose()? {
        tracing::debug!(target: "loader", tid = ?key, address = ?address, prestate = ?info, "account change");
        account_prefix_set.insert(Nibbles::unpack(keccak256(address)));
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
            .insert(Nibbles::unpack(keccak256(key)));
        account_prefix_set.insert(Nibbles::unpack(keccak256(address)));
    }

    account_prefix_set.sort();
    for (_, storage_prefix_set) in storage_prefix_set.iter_mut() {
        storage_prefix_set.sort();
    }

    Ok((account_prefix_set, storage_prefix_set))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        trie::{DBTrieLoader, TrieProgress},
        trie_v2::account,
        Transaction,
    };
    use proptest::{prelude::ProptestConfig, proptest};
    use reth_db::{
        cursor::DbCursorRW, database::Database, mdbx::test_utils::create_test_rw_db, tables,
        transaction::DbTxMut,
    };
    use reth_primitives::{keccak256, proofs::KeccakHasher, Account, Address, H256, U256};
    use reth_rlp::encode_fixed_size;
    use std::{
        collections::BTreeMap,
        ops::{Deref, DerefMut, Mul},
        path::PathBuf,
        str::FromStr,
    };
    use tokio::sync::mpsc;
    use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};
    use tracing::Level;

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
            let storage_root = storage_root(storage.into_iter());
            let mut out = Vec::new();
            EthAccount::from(account).with_storage_root(storage_root).encode(&mut out);
            Some((address, out))
        });

        triehash::sec_trie_root::<KeccakHasher, _, _, _>(encoded_accounts)
    }

    fn storage_root_prehashed<I: Iterator<Item = (H256, U256)>>(storage: I) -> H256 {
        let encoded_storage = storage.map(|(k, v)| (k, encode_fixed_size(&v).to_vec()));
        H256(triehash::trie_root::<KeccakHasher, _, _, _>(encoded_storage).0)
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

    #[tokio::test]
    // This ensures we dont add empty accounts to the trie
    async fn test_empty_account() {
        let state: State = BTreeMap::from([
            (
                Address::random(),
                (
                    Account { nonce: 0, balance: U256::from(0), bytecode_hash: None },
                    BTreeMap::from([(H256::from_low_u64_be(0x4), U256::from(12))]),
                ),
            ),
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

        let got = StateRoot::new(tx.deref_mut()).root(None).await.unwrap();
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
    async fn sepolia_repro() {
        let _ = tracing_subscriber::fmt().with_env_filter("loader=trace").try_init();

        let path =
            PathBuf::from_str("/Users/georgios/Library/Application Support/reth/db/").unwrap();
        use reth_db::mdbx::{Env, WriteMap};
        let db = Env::<WriteMap>::open(&path, reth_db::mdbx::EnvKind::RW).unwrap();
        let mut tx = Transaction::new(&db).unwrap();

        let from = tx.get_block_transition(3208395).unwrap();
        let to = tx.get_block_transition(3208396).unwrap();
        dbg!(from, to);

        let incremental_root =
            StateRoot::incremental_root(tx.deref_mut(), from..to, None).await.unwrap();
        dbg!(&incremental_root);

        let loader = StateRoot::new(tx.deref_mut());
        let (sender, _) = tokio::sync::mpsc::unbounded_channel();
        let full_root = loader.root(Some(sender)).await.unwrap();
        drop(loader);

        dbg!(&full_root);

        //assert_eq!(incremental_root, full_root);

        // dbg!(&cita_root, &incremental_root);

        // assert_eq!(incremental_root, cita_root);
    }

    #[tokio::test]
    async fn storage_root_regression() {
        let db = create_test_rw_db();
        let mut tx = Transaction::new(db.as_ref()).unwrap();
        // Some address whose hash starts with 0xB041
        let address3 = Address::from_str("16b07afd1c635f77172e842a000ead9a2a222459").unwrap();
        let key3 = keccak256(address3);
        assert_eq!(key3[0], 0xB0);
        assert_eq!(key3[1], 0x41);

        let storage = BTreeMap::from(
            [
                ("1200000000000000000000000000000000000000000000000000000000000000", 0x42),
                ("1400000000000000000000000000000000000000000000000000000000000000", 0x01),
                ("3000000000000000000000000000000000000000000000000000000000E00000", 0x127a89),
                ("3000000000000000000000000000000000000000000000000000000000E00001", 0x05),
            ]
            .map(|(slot, val)| (H256::from_str(slot).unwrap(), U256::from(val))),
        );

        let mut hashed_storage_cursor = tx.cursor_dup_write::<tables::HashedStorage>().unwrap();
        for (hashed_slot, value) in storage.clone() {
            hashed_storage_cursor.upsert(key3, StorageEntry { key: hashed_slot, value }).unwrap();
        }
        tx.commit().unwrap();

        let account3_storage_root =
            StorageRoot::new(tx.deref_mut(), address3, None).root().await.unwrap();
        let expected_root = storage_root_prehashed(storage.into_iter());
        assert_eq!(expected_root, account3_storage_root);
    }

    #[tokio::test]
    async fn account_and_storage_trie() {
        let ether = U256::from(1e18);
        let storage = BTreeMap::from(
            [
                ("1200000000000000000000000000000000000000000000000000000000000000", 0x42),
                ("1400000000000000000000000000000000000000000000000000000000000000", 0x01),
                ("3000000000000000000000000000000000000000000000000000000000E00000", 0x127a89),
                ("3000000000000000000000000000000000000000000000000000000000E00001", 0x05),
            ]
            .map(|(slot, val)| (H256::from_str(slot).unwrap(), U256::from(val))),
        );

        let db = create_test_rw_db();
        let mut tx = Transaction::new(db.as_ref()).unwrap();

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

        // Populate account & storage trie DB tables
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
        let loader = StateRoot::new(tx.deref());
        assert_eq!(loader.root(Some(branch_node_tx)).await.unwrap(), computed_expected_root);

        // Check account trie
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

        // Check storage trie
        let storage_updates = updates
            .iter()
            .filter_map(|u| {
                if let BranchNodeUpdate::Storage(_, nibbles, node) = u {
                    Some((nibbles, node))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(storage_updates.len(), 1);

        let (nibbles3, node3) = storage_updates.first().unwrap();
        assert!(nibbles3.get_data().is_empty());
        assert_eq!(node3.state_mask, 0b1010);
        assert_eq!(node3.tree_mask, 0b0000);
        assert_eq!(node3.hash_mask, 0b0010);

        assert_eq!(node3.hashes.len(), 1);
        assert_eq!(node3.root_hash, Some(account3_storage_root));

        // Add an account
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
        let loader = StateRoot::new(tx.deref()).with_account_changes(prefix_set);
        assert_eq!(loader.root(Some(branch_node_tx)).await.unwrap(), expected_state_root);

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

        let (nibbles2b, node2b) = account_updates.first().unwrap();
        assert_eq!(nibbles2b.get_data(), [0xB, 0x0]);
        assert_eq!(node2a, node2b);
        tx.commit().unwrap();

        {
            let mut hashed_account_cursor = tx.cursor_write::<tables::HashedAccount>().unwrap();

            let account = hashed_account_cursor.seek_exact(key2).unwrap().unwrap();
            hashed_account_cursor.delete_current().unwrap();

            let mut account_prefix_set = PrefixSet::default();
            account_prefix_set.insert(Nibbles::unpack(account.0));

            let computed_expected_root: H256 = triehash::trie_root::<KeccakHasher, _, _, _>([
                (key1, encode_account(account1, None)),
                // DELETED: (key2, encode_account(account2, None)),
                (key3, encode_account(account3, Some(account3_storage_root))),
                (key4a, encode_account(account4a, None)),
                (key4b, encode_account(account4b, None)),
                (key5, encode_account(account5, None)),
                (key6, encode_account(account6, None)),
            ]);

            let (branch_node_tx, branch_node_rx) = mpsc::unbounded_channel();
            let loader = StateRoot::new(tx.deref_mut()).with_account_changes(account_prefix_set);
            assert_eq!(loader.root(Some(branch_node_tx)).await.unwrap(), computed_expected_root);
            drop(loader);

            let branch_node_stream = UnboundedReceiverStream::new(branch_node_rx);
            let updates = branch_node_stream.collect::<Vec<_>>().await;
            assert_eq!(updates.len(), 2);

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
            assert_eq!(account_updates.len(), 1);

            let (nibbles1c, node1c) = account_updates.first().unwrap();
            assert_eq!(nibbles1c.get_data(), [0xB]);

            assert_eq!(node1c.state_mask, 0b1011);
            assert_eq!(node1c.tree_mask, 0b0000);
            assert_eq!(node1c.hash_mask, 0b1011);

            assert_eq!(node1c.root_hash, None);

            assert_eq!(node1c.hashes.len(), 3);
            assert_ne!(node1c.hashes[0], node1b.hashes[0]);
            assert_eq!(node1c.hashes[1], node1b.hashes[1]);
            assert_eq!(node1c.hashes[2], node1b.hashes[2]);
            tx.drop().unwrap();
        }

        {
            let mut hashed_account_cursor = tx.cursor_write::<tables::HashedAccount>().unwrap();

            let account2 = hashed_account_cursor.seek_exact(key2).unwrap().unwrap();
            hashed_account_cursor.delete_current().unwrap();
            let account3 = hashed_account_cursor.seek_exact(key3).unwrap().unwrap();
            hashed_account_cursor.delete_current().unwrap();

            let mut account_prefix_set = PrefixSet::default();
            account_prefix_set.insert(Nibbles::unpack(account2.0));
            account_prefix_set.insert(Nibbles::unpack(account3.0));

            let computed_expected_root: H256 = triehash::trie_root::<KeccakHasher, _, _, _>([
                (key1, encode_account(account1, None)),
                // DELETED: (key2, encode_account(account2, None)),
                // DELETED: (key3, encode_account(account3, Some(account3_storage_root))),
                (key4a, encode_account(account4a, None)),
                (key4b, encode_account(account4b, None)),
                (key5, encode_account(account5, None)),
                (key6, encode_account(account6, None)),
            ]);

            let (branch_node_tx, branch_node_rx) = mpsc::unbounded_channel();
            let loader = StateRoot::new(tx.deref_mut()).with_account_changes(account_prefix_set);
            assert_eq!(loader.root(Some(branch_node_tx)).await.unwrap(), computed_expected_root);
            drop(loader);

            let branch_node_stream = UnboundedReceiverStream::new(branch_node_rx);
            let updates = branch_node_stream.collect::<Vec<_>>().await;
            assert_eq!(updates.len(), 1); // no storage root update

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
            assert_eq!(account_updates.len(), 1);

            let (nibbles1d, node1d) = account_updates.first().unwrap();
            assert_eq!(nibbles1d.get_data(), [0xB]);

            assert_eq!(node1d.state_mask, 0b1011);
            assert_eq!(node1d.tree_mask, 0b0000);
            assert_eq!(node1d.hash_mask, 0b1010);

            assert_eq!(node1d.root_hash, None);

            assert_eq!(node1d.hashes.len(), 2);
            assert_eq!(node1d.hashes[0], node1b.hashes[1]);
            assert_eq!(node1d.hashes[1], node1b.hashes[2]);
        }
    }

    #[tokio::test]
    async fn account_trie_around_extension_node() {
        let db = create_test_rw_db();
        let mut tx = Transaction::new(db.as_ref()).unwrap();

        let expected = extension_node_trie(&mut tx);

        let (sender, recv) = mpsc::unbounded_channel();
        let loader = StateRoot::new(tx.deref_mut());
        let got = loader.root(Some(sender)).await.unwrap();
        assert_eq!(expected, got);

        // Check account trie
        drop(loader);
        let branch_node_stream = UnboundedReceiverStream::new(recv);
        let updates = branch_node_stream.collect::<Vec<_>>().await;

        let account_updates = updates
            .into_iter()
            .filter_map(|u| {
                if let BranchNodeUpdate::Account(nibbles, node) = u {
                    Some((nibbles, node))
                } else {
                    None
                }
            })
            .collect::<BTreeMap<_, _>>();

        assert_account_updates(&account_updates);
    }

    #[tokio::test]

    async fn account_trie_around_extension_node_with_dbtrie() {
        let db = create_test_rw_db();
        let mut tx = Transaction::new(db.as_ref()).unwrap();

        let expected = extension_node_trie(&mut tx);

        let loader = StateRoot::new(tx.deref_mut());
        let got = loader.root(None).await.unwrap();
        assert_eq!(expected, got);

        drop(loader);

        // read the account updates from the db
        let mut accounts_trie = tx.cursor_read::<tables::AccountsTrie2>().unwrap();
        let mut walker = accounts_trie.walk(None).unwrap();
        let mut account_updates = BTreeMap::new();
        while let Some(item) = walker.next() {
            let item = item.unwrap();
            account_updates.insert(
                Nibbles::from(item.0.inner.0.as_ref()),
                BranchNodeCompact::unmarshal(&item.1).unwrap(),
            );
        }

        assert_account_updates(&account_updates);
    }

    use reth_db::mdbx::{Env, WriteMap};

    fn extension_node_trie(tx: &mut Transaction<'_, Env<WriteMap>>) -> H256 {
        let mut hashed_accounts = tx.cursor_write::<tables::HashedAccount>().unwrap();
        let mut hb = HashBuilder::new(None);

        let a = Account {
            nonce: 0,
            balance: U256::from(1u64),
            bytecode_hash: Some(H256::random()),
            ..Default::default()
        };
        let val = encode_account(a, None);
        use hex_literal::hex;

        let mut hashed_accounts = tx.cursor_write::<tables::HashedAccount>().unwrap();
        let mut hb = HashBuilder::new(None);

        for key in [
            hex!("30af561000000000000000000000000000000000000000000000000000000000"),
            hex!("30af569000000000000000000000000000000000000000000000000000000000"),
            hex!("30af650000000000000000000000000000000000000000000000000000000000"),
            hex!("30af6f0000000000000000000000000000000000000000000000000000000000"),
            hex!("30af8f0000000000000000000000000000000000000000000000000000000000"),
            hex!("3100000000000000000000000000000000000000000000000000000000000000"),
        ] {
            hashed_accounts.upsert(H256(key), a).unwrap();
            hb.add_leaf(Nibbles::unpack(&key), &val);
        }

        hb.root()
    }

    fn assert_account_updates(account_updates: &BTreeMap<Nibbles, BranchNodeCompact>) {
        assert_eq!(account_updates.len(), 2);

        let node = account_updates.get(&Nibbles::from(vec![0x3])).unwrap();
        let expected = BranchNodeCompact::new(0b0011, 0b0001, 0b0000, vec![], None);
        assert_eq!(node, &expected);

        let node = account_updates.get(&Nibbles::from(vec![0x3, 0x0, 0xA, 0xF])).unwrap();
        assert_eq!(node.state_mask, 0b101100000);
        assert_eq!(node.tree_mask, 0b000000000);
        assert_eq!(node.hash_mask, 0b001000000);

        assert_eq!(node.root_hash, None);
        assert_eq!(node.hashes.len(), 1);
    }
}
