pub mod types;
pub mod pager;
pub mod models;
pub mod cacher;
pub mod config;
pub mod merge_sort;
pub mod old_design_merge_sort;
use core::str;
use std::process::Command;
pub use std::{fs::{OpenOptions, File}, io::{Read, Write}};
pub use cdc_hash::DEFAULT_MAX_NODE_CAPACITY;

#[derive(Debug, Default)]
pub struct ColeStorageSize {
    pub tree_meta: usize,
    pub level_meta: usize,
    pub state_size: usize,
    pub mht_size: usize,
    pub model_size: usize,
    pub filter_size: usize,
    pub total_size: usize,
}

pub fn compute_cole_size_breakdown(path: &str) -> ColeStorageSize {
    let mut storage_size = ColeStorageSize::default();
    let path_files = Command::new("ls").arg(path).output().expect("read path files failure");
    let s = str::from_utf8(&path_files.stdout).unwrap();
    let lines: Vec<&str> = s.trim().split("\n").collect();
    let mut tree_meta = 0;
    let mut level_meta = 0;
    let mut state_size = 0;
    let mut mht_size = 0;
    let mut model_size = 0;
    let mut filter_size = 0;

    for line in lines {
        if line == "mht" {
            tree_meta = disk_usage_check_storage(path, line);
        }
        else if line.ends_with("lv") {
            level_meta += disk_usage_check_storage(path, line);
        }
        else if line.starts_with("s_") {
            state_size += disk_usage_check_storage(path, line);
        }
        else if line.starts_with("m_") {
            model_size += disk_usage_check_storage(path, line);
        }
        else if line.starts_with("h_") {
            mht_size += disk_usage_check_storage(path, line);
        }
        else if line.starts_with("f_") {
            filter_size += disk_usage_check_storage(path, line);
        }
        else if line.starts_with("lh_") {
            mht_size += disk_usage_check_storage(path, line);
        }
        else if line.starts_with("uh_") {
            mht_size += disk_usage_check_storage(path, line);
        }
        else if line.starts_with("uo_") {
            mht_size += disk_usage_check_storage(path, line);
        }
    }
    storage_size.tree_meta = tree_meta;
    storage_size.level_meta = level_meta;
    storage_size.state_size = state_size;
    storage_size.mht_size = mht_size;
    storage_size.model_size = model_size;
    storage_size.filter_size = filter_size;
    storage_size.total_size = disk_usage_check_storage(path, "");
    return storage_size;
}

fn disk_usage_check_storage(base: &str, file_name: &str) -> usize {
    let du = Command::new("du").arg("-b").arg(format!("{}/{}", base, file_name)).output().unwrap();
    let results: Vec<&str> = str::from_utf8(&du.stdout).unwrap().split("\t").collect();
    results[0].parse::<usize>().unwrap()
}

pub fn disk_usage_check_directory_storage(dir_path: &str) -> usize {
    let du = Command::new("du").arg("-b").arg(format!("{}", dir_path)).output().unwrap();
    let results: Vec<&str> = str::from_utf8(&du.stdout).unwrap().split("\t").collect();
    results[0].parse::<usize>().unwrap()
}

#[derive(Debug, Clone, Copy)]
pub struct MemCost {
    pub cache_size: usize,
    pub filter_size: usize,
    pub mht_size: usize,
}

impl MemCost {
    pub fn new(cache_size: usize, filter_size: usize, mht_size: usize) -> Self {
        Self {
            cache_size,
            filter_size,
            mht_size,
        }
    }

    pub fn size(&self) -> usize {
        return self.cache_size + self.filter_size + self.mht_size;
    }
}
#[cfg(test)]
mod tests {
}
