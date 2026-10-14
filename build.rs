fn main() 
{
    // Re-run this build script only if the C++ sources change
    println!("cargo:rerun-if-changed=c_src/");

    // Automatically compile C++ code into cec19.lib
    cc::Build::new()
        .cpp(true)
        // Ensure this path matches the exact location of cec19_func.cpp in your c_src directory
        .file("c_src/CEC2019 Benchmark/cec19_func.cpp") 
        .warnings(false)
        .compile("cec19"); // Generates and links cec19.lib automatically
}