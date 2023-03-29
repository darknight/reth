use super::{H256, U256};
use reth_codecs::{derive_arbitrary, Compact};
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

use crate::Bytes;
use reth_codecs::main_codec;
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[main_codec]
/// The nibbles are the keys for the AccountsTrie and the subkeys for the StorageTrie.
pub struct Nibbles {
    /// The inner nibble bytes
    pub inner: Bytes,
}

/// Account storage trie node.
#[derive_arbitrary(compact)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct StorageTrieEntry2 {
    /// The nibbles of the intermediate node
    pub nibbles: Nibbles,
    /// Encoded node.
    pub node: Vec<u8>,
}

// NOTE: Removing main_codec and manually encode subkey
// and compress second part of the value. If we have compression
// over whole value (Even SubKey) that would mess up fetching of values with seek_by_key_subkey
impl Compact for StorageTrieEntry2 {
    fn to_compact(self, buf: &mut impl bytes::BufMut) -> usize {
        buf.put_slice(&self.nibbles.inner[..]);
        buf.put_slice(&self.node[..]);
        64 + self.node.len()
    }

    fn from_compact(buf: &[u8], len: usize) -> (Self, &[u8])
    where
        Self: Sized,
    {
        let node = Vec::from(&buf[64..len]);
        (Self { nibbles: Nibbles { inner: buf[..64].to_vec().into() }, node }, &buf[len..])
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
