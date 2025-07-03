#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("letus_new/include/Letus.h");
        type Letus;

        fn new_trie(path: String) -> UniquePtr<Letus>;

        fn LetusPut(self: &Letus, version: u64, key: String, value: String);

        fn LetusGet(self: &Letus, version: u64, key: String) -> String;

        fn LetusCommit(self: &Letus, version: u64);

        fn LetusFlush(self: &Letus, version: u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie() {
        let trie = ffi::new_trie(String::from("data"));
        trie.LetusPut(1, String::from("aaaa"), String::from("aaaa1"));
        trie.LetusPut(1, String::from("bbbb"), String::from("bbbb1"));
        trie.LetusCommit(1);

        let v1 = trie.LetusGet(1, String::from("aaaa"));
        println!("v1: {}", v1);
        let v2 = trie.LetusGet(1, String::from("bbbb"));
        println!("v2: {}", v2);
        let v3 = trie.LetusGet(1, String::from("cccc"));
        println!("v3: {}", v3);
    }
}
