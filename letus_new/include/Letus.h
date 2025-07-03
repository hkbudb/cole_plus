#pragma once
#include "rust/cxx.h"
#include <memory>
#include "DMMTrie.hpp"

struct Letus {
  DMMTrie* trie;

  public:
  void LetusPut( uint64_t version, rust::String key, rust::String value) const;

  rust::String LetusGet(uint64_t version, rust::String key) const;

  void LetusCommit(uint64_t version) const;

  void LetusFlush(uint64_t version) const;
};

std::unique_ptr<Letus> new_trie(rust::String path);