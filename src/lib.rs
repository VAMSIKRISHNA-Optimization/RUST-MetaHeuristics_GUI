#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

pub mod algorithms;
pub mod algorithm_structure;
pub mod benchmarks;
pub mod core;
pub mod execution;
pub mod gui;

pub const DEFAULT_POPULATION_SIZE: usize = 50;
pub const DEFAULT_ITERATIONS     : usize = 500;
pub const DEFAULT_DIMENSIONS     : usize = 10;