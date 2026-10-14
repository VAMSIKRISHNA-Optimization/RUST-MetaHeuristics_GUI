use std::env;
use std::path::PathBuf;

fn main() 
{
    println!("cargo:rerun-if-changed=c_src/");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let c_file_path = manifest_dir
        .join("c_src")
        .join("CEC2019 Benchmark Suite")
        .join("cec19_func.cpp");

    assert!(
        c_file_path.exists(),
        "Could not find C++ benchmark file at: {:?}",
        c_file_path
    );

    cc::Build::new()
        .cpp(true)
        .define("WINDOWS", None) // Replicates -DWINDOWS from the original instructions
        .file(c_file_path)
        .warnings(false)
        .compile("cec19");
}