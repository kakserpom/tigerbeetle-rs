//! TigerBeetle LSM: forest of LSM trees (grooves, tables, manifests, compaction, scans).
//!
//! Port of `src/lsm/`. Depends bottom-up on the `tigerbeetle-core` crate only.
#![allow(clippy::doc_markdown)] // upstream terminology

pub mod binary_search;
pub mod direction;
pub mod free_set;
pub mod k_way_merge;
pub mod manifest_level;
pub mod node_pool;
pub mod scratch_memory;
pub mod segmented_array;
pub mod table_memory;
pub mod timestamp_range;
pub mod tree;
