use serde::{Serialize, Deserialize};
use gearhash::Hasher;
pub const DEFAULT_GEAR_HASH_LEVEL: usize = 0;
pub const DEFAULT_MAX_NODE_CAPACITY: usize = 92; // for internal node, (u32 ver, u64 offset, H256 hash) = 44 bytes, 4096 / 44 = 93, round down = 92
pub const DEFAULT_FANOUT: usize = 2;
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum CDCResult {
    PatternFound,
    NoPatternFound,
    ReachCapacity,    
}

#[derive(Debug)]
pub struct CDCHash<'a> {
    pub mask: u64, // a mask to determine gear hash pattern
    pub hasher: Hasher<'a>, // hasher object
    pub max_capacity: usize, // a pre-defined param to determine the maximum number of objects in a block
    pub counter: usize, // a counter to determine whether we reach the max_capacity
}

impl<'a> CDCHash<'a> {
    pub fn new(fanout: usize, level: usize, max_capacity: usize) -> Self {
        let mask = Self::generate_mask(fanout, level);
        Self {
            mask,
            hasher: Hasher::default(),
            max_capacity,
            counter: 0,
        }
    }

    // if the current buf includes a pattern, return PatternFound;
    // if no pattern is found, return NoPatternFound; 
    // if reach the capacity, return ReachCapacity
    pub fn generate_cut_point(&mut self, buf: &[u8]) -> CDCResult {
        self.counter += 1;
        // ensure the locality of this buf (remove the boundary bytes' effect)
        self.hasher.set_hash(0); // reset the hash value for this buf
        if self.counter >= self.max_capacity {
            // should early stop finding the pattern and generate a new block
            // reset the hasher
            self.reset_hasher();
            return CDCResult::ReachCapacity;
        } else {
            let r = self.hasher.next_match(buf, self.mask);
            if r.is_some() {
                // find a pattern
                // reset the hasher
                self.reset_hasher();
                return CDCResult::PatternFound;
            } else {
                return CDCResult::NoPatternFound;
            }
        }
    }

    // reset the hasher by setting the hash to 0 and counter to 0
    pub fn reset_hasher(&mut self) {
        self.hasher.set_hash(0);
        self.counter = 0;
    }

    // private function to help generate mask
    // mask includes log_2(32*fanout)+level number of '1'
    fn generate_mask(fanout: usize, level: usize) -> u64 {
        let hash_size = 32;
        let total_len = hash_size * fanout;
        let log_len = (total_len as u64).ilog2() as usize + level;
        // generate mask with log_len num of '1'
        let s: String = vec!['1'; log_len].into_iter().collect();
        let mask = u64::from_str_radix(&s, 2).unwrap();
        return mask;
    }
}