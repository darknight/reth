use super::{H256, U256};
use crate::Bytes;
use derive_more::Deref;
use reth_codecs::{derive_arbitrary, main_codec, Compact};
use serde::{Deserialize, Serialize};

/// Account storage entry.
#[derive_arbitrary(compact)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct StorageEntry {
    /// Storage key.
    pub key: H256,
    /// Value on storage key.
    pub value: U256,
}

impl From<(H256, U256)> for StorageEntry {
    fn from((key, value): (H256, U256)) -> Self {
        StorageEntry { key, value }
    }
}

// NOTE: Removing main_codec and manually encode subkey
// and compress second part of the value. If we have compression
// over whole value (Even SubKey) that would mess up fetching of values with seek_by_key_subkey
impl Compact for StorageEntry {
    fn to_compact(self, buf: &mut impl bytes::BufMut) -> usize {
        // for now put full bytes and later compress it.
        buf.put_slice(&self.key.to_fixed_bytes()[..]);
        self.value.to_compact(buf) + 32
    }

    fn from_compact(buf: &[u8], len: usize) -> (Self, &[u8])
    where
        Self: Sized,
    {
        let key = H256::from_slice(&buf[..32]);
        let (value, out) = U256::from_compact(&buf[32..], len - 32);
        (Self { key, value }, out)
    }
}

#[main_codec]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
/// The nibbles are the keys for the AccountsTrie and the subkeys for the StorageTrie.
pub struct StoredNibbles {
    /// The inner nibble bytes
    pub inner: Bytes,
}

impl From<Vec<u8>> for StoredNibbles {
    fn from(inner: Vec<u8>) -> Self {
        Self { inner: inner.into() }
    }
}

/// TODO:
#[derive_arbitrary(compact)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord, Deref)]
pub struct StoredNibblesSubKey(StoredNibbles);

impl From<Vec<u8>> for StoredNibblesSubKey {
    fn from(inner: Vec<u8>) -> Self {
        Self(StoredNibbles { inner: inner.into() })
    }
}

impl Compact for StoredNibblesSubKey {
    fn to_compact(self, buf: &mut impl bytes::BufMut) -> usize {
        assert!(self.inner.len() <= 32);
        let mut padded = vec![0; 32];
        padded[..self.inner.len()].copy_from_slice(&self.inner[..]);
        buf.put_slice(&padded);
        buf.put_u8(self.inner.len() as u8);
        33 // 32 + 1
    }

    fn from_compact(buf: &[u8], _len: usize) -> (Self, &[u8])
    where
        Self: Sized,
    {
        let len = buf[32] as usize;
        let inner = Vec::from(&buf[..len]).into();
        (Self(StoredNibbles { inner }), &buf[33..])
    }
}

/// Account storage trie node.
#[derive_arbitrary(compact)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct StorageTrieEntry2 {
    /// The nibbles of the intermediate node
    pub nibbles: StoredNibblesSubKey,
    /// Encoded node.
    pub node: Vec<u8>,
}

// NOTE: Removing main_codec and manually encode subkey
// and compress second part of the value. If we have compression
// over whole value (Even SubKey) that would mess up fetching of values with seek_by_key_subkey
impl Compact for StorageTrieEntry2 {
    fn to_compact(self, buf: &mut impl bytes::BufMut) -> usize {
        let nibbles_len = self.nibbles.to_compact(buf);
        buf.put_slice(&self.node[..]);
        dbg!(nibbles_len);
        nibbles_len + self.node.len()
    }

    fn from_compact(buf: &[u8], len: usize) -> (Self, &[u8])
    where
        Self: Sized,
    {
        let (nibbles, _) = StoredNibblesSubKey::from_compact(buf, len);
        let node = Vec::from(&buf[33..len]);
        let this = Self { nibbles, node };
        (this, &buf[len..])
    }
}

/// Account storage trie node.
#[derive_arbitrary(compact)]
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct StorageTrieEntry {
    /// Hashed storage key.
    pub hash: H256,
    /// Encoded node.
    pub node: Vec<u8>,
}

// NOTE: Removing main_codec and manually encode subkey
// and compress second part of the value. If we have compression
// over whole value (Even SubKey) that would mess up fetching of values with seek_by_key_subkey
impl Compact for StorageTrieEntry {
    fn to_compact(self, buf: &mut impl bytes::BufMut) -> usize {
        // for now put full bytes and later compress it.
        buf.put_slice(&self.hash.to_fixed_bytes()[..]);
        buf.put_slice(&self.node[..]);
        self.node.len() + 32
    }

    fn from_compact(buf: &[u8], len: usize) -> (Self, &[u8])
    where
        Self: Sized,
    {
        let key = H256::from_slice(&buf[..32]);
        let node = Vec::from(&buf[32..len]);
        (Self { hash: key, node }, &buf[len..])
    }
}
