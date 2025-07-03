#include "../include/Letus.h"

#include <iostream>
#include <string>

#include "../include/DMMTrie.hpp"
#include "../include/LSVPS.hpp"
#include "../include/VDLS.hpp"

std::unique_ptr<Letus> new_trie(rust::String path) {
    std::string cpath(path.c_str());
    LSVPS* page_store = new LSVPS(cpath + "/");
    VDLS* value_store = new VDLS(cpath + "/");
    DMMTrie* trie = new DMMTrie(0, page_store, value_store);
    page_store->RegisterTrie(trie);
    auto ptr = std::unique_ptr<Letus>(new Letus());
    ptr->trie = trie;
    return ptr;
}

void Letus::LetusPut(uint64_t version, rust::String key, rust::String value) const {
    std::string ckey(key.c_str());
    std::string cvalue(value.c_str());
    trie->Put(0, version, ckey, cvalue);
}

rust::String Letus::LetusGet(uint64_t version, rust::String key) const {
    std::string ckey(key.c_str());
    std::string value = trie->Get(0, version, ckey);
    return rust::String(value);
}

void Letus::LetusCommit(uint64_t version) const {
    trie->Commit(version);
}

void Letus::LetusFlush(uint64_t version) const {
    trie->Flush(0, version);
}