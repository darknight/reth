use std::fmt::Debug;

use revm_primitives::Bytes;

use super::nibbles::Nibbles;

#[derive(Debug, Clone)]
pub struct PrefixSet {
    keys: Vec<Nibbles>,
    sorted: bool,
    index: usize,
}

impl Default for PrefixSet {
    fn default() -> Self {
        PrefixSet::new()
    }
}

impl PrefixSet {
    pub fn new() -> Self {
        Self { keys: Vec::new(), sorted: true, index: 0 }
    }

    pub fn sort(&mut self) {
        self.keys.sort();
        self.keys.dedup();
        self.sorted = true;
    }

    pub(crate) fn contains<T: Into<Nibbles>>(&mut self, prefix: T) -> bool {
        let prefix = prefix.into();
        if self.keys.is_empty() {
            return false
        }

        if !self.sorted {
            self.sort();
        }

        while self.index > 0 && self.keys[self.index] > prefix {
            self.index -= 1;
        }

        loop {
            let current = &self.keys[self.index];

            if current.has_prefix(&prefix) {
                return true
            }

            if current > &prefix {
                return false
            }

            if self.index >= self.keys.len() - 1 {
                return false
            }

            self.index += 1;
        }
    }

    pub fn insert<T: Into<Nibbles>>(&mut self, nibbles: T) {
        self.keys.push(nibbles.into());
        self.sorted = false;
    }

    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_set() {
        let mut ps = PrefixSet::new();
        assert!(!ps.contains(b""));
        assert!(!ps.contains(b"a"));

        ps.insert(b"abc");
        ps.insert(b"fg");
        ps.insert(b"abc"); // duplicate
        ps.insert(b"ab");

        assert!(ps.contains(b""));
        assert!(ps.contains(b"a"));
        assert!(!ps.contains(b"aac"));
        assert!(ps.contains(b"ab"));
        assert!(ps.contains(b"abc"));
        assert!(!ps.contains(b"abcd"));
        assert!(!ps.contains(b"b"));
        assert!(ps.contains(b"f"));
        assert!(ps.contains(b"fg"));
        assert!(!ps.contains(b"fgk"));
        assert!(!ps.contains(b"fy"));
        assert!(!ps.contains(b"yyz"));
    }
}
