
fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("src/LSVPS.cpp")
        .file("src/DMMTrie.cpp")
        .file("src/Letus.cpp")
        .compile("cxx-demo");
    println!("cargo:rustc-link-search=native=/usr/lib/");
    println!("cargo:rustc-link-lib=ssl");
    println!("cargo:rustc-link-lib=crypto");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=include/common.hpp");
    println!("cargo:rerun-if-changed=include/DMMTrie.hpp");
    println!("cargo:rerun-if-changed=include/Letus.hpp");
    println!("cargo:rerun-if-changed=include/LSVPS.hpp");
    println!("cargo:rerun-if-changed=include/VDLS.hpp");
    println!("cargo:rerun-if-changed=src/LSVPS.cc");
    println!("cargo:rerun-if-changed=src/DMMTrie.cc");
    println!("cargo:rerun-if-changed=src/Letus.cc");
}