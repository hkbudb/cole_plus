use serde::{Serialize, Deserialize};
pub const DEFAULT_MAX_NODE_CAPACITY: usize = 92; // for internal node, (u32 ver, u64 offset, H256 hash) = 44 bytes, 4096 / 44 = 93, round down = 92
pub const DEFAULT_FANOUT: usize = 2;
pub const DEFAULT_HASH_LEVEL: usize = 0;
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum RollingResult {
    PatternFound,
    NoPatternFound,
    ReachCapacity,
}

/// Implements cyclic polynomial hash with poly x^23 + x^5 + 1
#[derive(Debug, Clone)]
pub struct CyclicPoly23Hasher {
    hash: u32, // only 23 bits used
}

impl Default for CyclicPoly23Hasher {
    fn default() -> Self {
        Self { hash: 0 }
    }
}

impl CyclicPoly23Hasher {
    pub fn set_hash(&mut self, h: u32) {
        self.hash = h & 0x7FFFFF; // mask to 23 bits
    }

    pub fn roll_byte(&mut self, b: u8) {
        self.hash <<= 8;
        self.hash |= b as u32;

        // Reduce hash to 23 bits using the polynomial
        for _ in 0..8 {
            if (self.hash & (1 << 23)) != 0 {
                self.hash ^= (1 << 5) | 1; // x^5 + 1
            }
            self.hash &= 0xFFFFFF; // keep 24 bits for shifting
        }
        self.hash &= 0x7FFFFF; // finally mask to 23 bits
    }

    pub fn next_match(&mut self, buf: &[u8], mask: u32) -> Option<usize> {
        self.set_hash(0); // reset for this buffer
        for (i, &b) in buf.iter().enumerate() {
            self.roll_byte(b);
            if (self.hash & mask) == 0 {
                return Some(i);
            }
        }
        None
    }
}

#[derive(Debug)]
pub struct RollingHash {
    pub mask: u32,
    pub hasher: CyclicPoly23Hasher,
    pub max_capacity: usize,
    pub counter: usize,
}

impl RollingHash {
    pub fn new(fanout: usize, level: usize, max_capacity: usize) -> Self {
        let mask = Self::generate_mask(fanout, level);
        Self {
            mask,
            hasher: CyclicPoly23Hasher::default(),
            max_capacity,
            counter: 0,
        }
    }

    pub fn generate_cut_point(&mut self, buf: &[u8]) -> RollingResult {
        self.counter += 1;
        self.hasher.set_hash(0);
        if self.counter >= self.max_capacity {
            self.reset_hasher();
            RollingResult::ReachCapacity
        } else {
            let r = self.hasher.next_match(buf, self.mask);
            if r.is_some() {
                self.reset_hasher();
                RollingResult::PatternFound
            } else {
                RollingResult::NoPatternFound
            }
        }
    }

    pub fn reset_hasher(&mut self) {
        self.hasher.set_hash(0);
        self.counter = 0;
    }

    fn generate_mask(fanout: usize, level: usize) -> u32 {
        let hash_size = 23;
        let total_len = hash_size * fanout;
        let log_len = (total_len as u32).ilog2() as usize + level;
        let s: String = vec!['1'; log_len].into_iter().collect();
        let mask = u32::from_str_radix(&s, 2).unwrap();
        mask
    }
}