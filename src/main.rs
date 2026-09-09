// Import your own library just like an external user would!
// Replace `metaheuristics_gui` with whatever your package name is in Cargo.toml
use MetaHeuristics_GUI::algorithms::pso_variants::PSO_Basic;
use MetaHeuristics_GUI::benchmarks::CEC2019;

mod gui; // The GUI stays strictly in the binary

fn main() 
{
    println!("cargo:rerun-if-changed=c_src/CEC2019 Benchmark Suite/cec19_func.cpp");

    cc::Build::new()
                .cpp(true) // Explicitly compile as C++
                .file("c_src/CEC2019 Benchmark Suite/cec19_func.cpp")
                .compile("cec19"); // This generates libcec19.a
}