use primitive_types::{H160, H256};
use serde::{Serialize, Deserialize};
use serde::{ser::{Serializer, SerializeTuple}, de::{Visitor, self}};
use blake2b_simd::{Params, Hash};
use std::fmt::Debug;
use rug::Integer;

pub const ACC_ADDR_SIZE: usize = 20;
pub const STATE_ADDR_SIZE: usize = 32;
pub const VALUE_SIZE: usize = 32;
pub const VERSION_SIZE: usize = 4;
pub const VERPOS_SIZE: usize = 4;
pub const COMPOUND_KEY_SIZE: usize = ACC_ADDR_SIZE + STATE_ADDR_SIZE + VERSION_SIZE;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Address(pub H160);
/* Serialize and Deserialize the Address, note that the serialized length should be 20 bytes
 */
impl Serialize for Address {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer {
        let mut tuple = serializer.serialize_tuple(1)?;
        let v = self.0.to_fixed_bytes();
        tuple.serialize_element(&v)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for Address {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de> {

        struct AddrVisitor;
        impl<'de> Visitor<'de> for AddrVisitor {
            type Value = Address;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "", )
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                where
                    A: de::SeqAccess<'de>, {
                    let bytes: [u8; 20] = seq.next_element()?
                        .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                    let v = H160::from_slice(&bytes);
                    Ok(Address(v))
            }
        }
        deserializer.deserialize_tuple(1, AddrVisitor)
    }
}

impl Address {
    pub fn to_bytes(&self) -> Vec<u8> {
        let bytes = bincode::serialize(&self).unwrap();
        return bytes;
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let addr: Address = bincode::deserialize(&bytes).unwrap();
        return addr;
    }
}

impl From<H160> for Address {
    fn from(value: H160) -> Self {
        Address(value)
    }
}

pub const ADDRESS_SIZE: usize = 20;
/* Serialize and Deserialize the StateKey, note that the serialized length should be 32 bytes
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct StateKey(pub H256);

impl Serialize for StateKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer {
        let mut tuple = serializer.serialize_tuple(1)?;
        let v = self.0.to_fixed_bytes();
        tuple.serialize_element(&v)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for StateKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de> {

        struct StateKeyVisitor;
        impl<'de> Visitor<'de> for StateKeyVisitor {
            type Value = StateKey;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "", )
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                where
                    A: de::SeqAccess<'de>, {
                    let bytes: [u8; 32] = seq.next_element()?
                        .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                    let v = H256::from_slice(&bytes);
                    Ok(StateKey(v))
            }
        }
        deserializer.deserialize_tuple(1, StateKeyVisitor)
    }
}

impl StateKey {
    pub fn to_bytes(&self) -> Vec<u8> {
        let bytes = bincode::serialize(&self).unwrap();
        return bytes;
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let key: StateKey = bincode::deserialize(&bytes).unwrap();
        return key;
    }
}

impl From<H256> for StateKey {
    fn from(value: H256) -> Self {
        StateKey(value)
    }
}

pub const STATEKEY_SIZE: usize = 32;
/* Serialize and Deserialize the StateValue, note that the serialized length should be 32 bytes
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct StateValue(pub H256);

impl Serialize for StateValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer {
        let mut tuple = serializer.serialize_tuple(1)?;
        let v = self.0.to_fixed_bytes();
        tuple.serialize_element(&v)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for StateValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de> {

        struct StateValueVisitor;
        impl<'de> Visitor<'de> for StateValueVisitor {
            type Value = StateValue;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "", )
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                where
                    A: de::SeqAccess<'de>, {
                    let bytes: [u8; 32] = seq.next_element()?
                        .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                    let v = H256::from_slice(&bytes);
                    Ok(StateValue(v))
            }
        }
        deserializer.deserialize_tuple(1, StateValueVisitor)
    }
}

impl StateValue {
    pub fn to_bytes(&self) -> Vec<u8> {
        let bytes = bincode::serialize(&self).unwrap();
        return bytes;
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let value: StateValue = bincode::deserialize(&bytes).unwrap();
        return value;
    }
}

impl From<H256> for StateValue {
    fn from(value: H256) -> Self {
        StateValue(value)
    }
}

impl Into<H256> for StateValue {
    fn into(self) -> H256 {
        self.0
    }
}

pub const STATEVALUE_SIZE: usize = 32;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub struct AddrKey {
    pub addr: Address, // h160
    pub state_key: StateKey, // h256
}

impl AddrKey {
    pub fn to_bytes(&self) -> Vec<u8> {
        let bytes = bincode::serialize(&self).unwrap();
        return bytes;
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let addr: AddrKey = bincode::deserialize(&bytes).unwrap();
        return addr;
    }

    pub fn new(addr: Address, state_key: StateKey) -> Self {
        Self {
            addr,
             state_key,
        }
    }
}

pub const ADDRKEY_SIZE: usize = ADDRESS_SIZE + STATEKEY_SIZE;
pub const PAGE_N_SIZE: usize = 4;
pub const NUM_OF_VER_SIZE: usize = 4;
pub const MHT_INDEX_SIZE: usize = 4;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub struct CompoundKey {
    pub addr: AddrKey,
    pub version: u32,
}

impl CompoundKey {
    pub fn new(addr: AddrKey, version: u32) -> Self {
        Self {
            addr,
            version,
        }
    }
}

impl CompoundKey {
    pub fn to_bytes(&self) -> Vec<u8> {
        let bytes = bincode::serialize(&self).unwrap();
        return bytes;
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let key: CompoundKey = bincode::deserialize(&bytes).unwrap();
        return key;
    }
}

pub trait LoadAddr {
    fn addr(&self) -> Option<&AddrKey>;
}

impl LoadAddr for CompoundKey {
    fn addr(&self) -> Option<&AddrKey> {
        Some(&self.addr)
    }
}

/* A trait that is used to compute digests
 */
pub trait Digestible {
    fn to_digest(&self) -> H256;
}

impl Digestible for CompoundKey {
    fn to_digest(&self) -> H256 {
        let bytes = self.to_bytes();
        bytes_hash(&bytes)
    }
}

impl Digestible for StateValue {
    fn to_digest(&self) -> H256 {
        let bytes = self.to_bytes();
        bytes_hash(&bytes)
    }
}

// compute the hash of a byte array
pub fn bytes_hash(bytes: &[u8]) -> H256 {
    let mut hasher = Params::new().hash_length(32).to_state();
    hasher.update(bytes);
    let h = H256::from_slice(hasher.finalize().as_bytes());
    return h;
}

// given multiple H256 hash values, compute the hash of their concatenation (used in Merkle Hash Tree)
pub fn compute_concatenate_hash(v: &[H256]) -> H256 {
    let mut bytes = vec![];
    for elem in v {
        bytes.extend(elem.as_bytes());
    }
    bytes_hash(&bytes)
}

// the following codes are from Cheng's MPT
#[inline]
pub fn blake2b_hash_to_h160(input: Hash) -> H160 {
    H160::from_slice(input.as_bytes())
}

#[inline]
pub fn blake2b_hash_to_h256(input: Hash) -> H256 {
    H256::from_slice(input.as_bytes())
}

pub const DEFAULT_DIGEST_LEN: usize = 32;

#[inline]
pub fn blake2(size: usize) -> Params {
    let mut params = Params::new();
    params.hash_length(size);
    params
}

#[inline]
pub fn default_blake2() -> Params {
    blake2(DEFAULT_DIGEST_LEN)
}

impl Digestible for [u8] {
    fn to_digest(&self) -> H256 {
        let hash = default_blake2().hash(self);
        blake2b_hash_to_h256(hash)
    }
}

impl Digestible for std::vec::Vec<u8> {
    fn to_digest(&self) -> H256 {
        self.as_slice().to_digest()
    }
}

impl Digestible for str {
    fn to_digest(&self) -> H256 {
        self.as_bytes().to_digest()
    }
}

impl Digestible for std::string::String {
    fn to_digest(&self) -> H256 {
        self.as_bytes().to_digest()
    }
}

macro_rules! impl_digestible_for_numeric {
    ($x: ty) => {
        impl Digestible for $x {
            fn to_digest(&self) -> H256 {
                self.to_le_bytes().to_digest()
            }
        }
    };
    ($($x: ty),*) => {$(impl_digestible_for_numeric!($x);)*}
}

impl_digestible_for_numeric!(i8, i16, i32, i64);
impl_digestible_for_numeric!(u8, u16, u32, u64);
impl_digestible_for_numeric!(f32, f64);

// the following codes define the traits for in-memory tree
pub trait Num:
    PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Clone
    + Copy
    + Digestible
    + Default
    + Debug
    + Sized
    + Serialize
{

}

impl<T> Num for T where
    T: PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Clone
    + Copy
    + Digestible
    + Default
    + Debug
    + Serialize
{

}

pub trait Value:
    PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Digestible
    + Clone
    + Copy
    + Default
    + Debug
    + Serialize
{

}

impl<T> Value for T where
    T: PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Digestible
    + Clone
    + Copy
    + Default
    + Debug
    + Serialize
{

}

// the following code is for model learning
/* A trait to transform a type to the big integer type
 */
pub trait BigNum {
    fn to_big_integer(&self,) -> Integer;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ser_deser() {
        for i in 0..100 {
            let addr = Address(H160::from_low_u64_be(i));
            let v = addr.to_bytes();
            let de_addr: Address = Address::from_bytes(&v);
            assert_eq!(de_addr, addr);

            let state_key = StateKey(H256::from_low_u64_be(i));
            let v = state_key.to_bytes();
            let de_state_key: StateKey = StateKey::from_bytes(&v);
            assert_eq!(de_state_key, state_key);

            let addrkey = AddrKey {
                addr,
                state_key,
            };
            let compound_key = CompoundKey {
                addr: addrkey,
                version: i as u32,
            };
            let v = compound_key.to_bytes();
            let de_compound_key: CompoundKey = CompoundKey::from_bytes(&v);
            assert_eq!(de_compound_key, compound_key);

            let state_value = StateValue(H256::from_low_u64_be(i));
            let v = state_value.to_bytes();
            let de_state_value: StateValue = StateValue::from_bytes(&v);
            assert_eq!(de_state_value, state_value);
        }

    }
}
