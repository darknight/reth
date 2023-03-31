//! Implements [`Compress`] and [`Decompress`] for [`IntegerList`]

use crate::{
    table::{Compress, Decompress},
    Error,
};
use bytes::BufMut;
use reth_primitives::{bytes::Bytes, IntegerList};

impl Compress for IntegerList {
    type Compressed = Vec<u8>;

    fn compress(self) -> Self::Compressed {
        self.to_bytes()
    }

    fn compress_to_buf<W: BufMut>(self, buf: &mut W) {
        let mut writer = buf.writer();
        self.to_mut_bytes(writer);
    }
}

impl Decompress for IntegerList {
    fn decompress<B: Into<Bytes>>(value: B) -> Result<Self, Error> {
        IntegerList::from_bytes(&value.into()).map_err(|_| Error::DecodeError)
    }
}
