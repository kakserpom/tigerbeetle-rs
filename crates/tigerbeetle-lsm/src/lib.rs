//! TigerBeetle LSM: forest of LSM trees (grooves, tables, manifests, compaction, scans).
//!
//! Port of `src/lsm/`. Depends bottom-up on the `tigerbeetle-core` crate only.
#![allow(clippy::doc_markdown)] // upstream terminology

pub mod free_set;
