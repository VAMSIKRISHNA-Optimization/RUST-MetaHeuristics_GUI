use crate::core::problem::Problem;

// Standard CEC C FFI signature
use std::os::raw::c_int;

#[link(name = "cec19", kind = "static")]
unsafe extern "C" 
{
    pub unsafe fn cec19_test_func
    (
        x: *mut f64,       // Array of decision variables
        f: *mut f64,       // Array to store fitness results
        nx: c_int,         // Number of dimensions (n)
        mx: c_int,         // Number of objective functions (m=1 for single objective)
        func_num: c_int,   // The benchmark function number (1-10)
    );
}

pub struct Cec2019 
{
    func_num: usize, // 1 to 10
    dim     : usize,
    bounds  : (f64, f64),
}

impl Cec2019 
{
    pub fn new(func_num: usize) -> Self 
    {
        assert!(func_num >= 1 && func_num <= 10, "CEC2019 only has functions F1 to F10.");
        
        // CEC2019 specific dimensions
        let dim = match func_num 
        {
            1 => 9,
            2 => 16,
            3 => 18,
            _ => 10, // F4 to F10 are 10D
        };

        Self 
        {
            func_num,
            dim,
            bounds: (-100.0, 100.0), // Standard search range
        }
    }
}

impl Problem for Cec2019 
{
    fn dimensions(&self) -> usize {
        self.dim
    }

    fn bounds(&self) -> (&[f64], &[f64]) 
    {
        // For simplicity, returning uniform static bounds slices
        // (In production, hold two Vec<f64> structs inside Cec2019)
        unimplemented!("Will return lower and upper bounds vectors")
    }

    fn name(&self) -> &str 
    {
        "CEC 2019 Instance"
    }

    fn suite_name(&self) -> &str 
    {
        "CEC 2019 Benchmark Suite"
    }

    fn optimal_value(&self) -> f64 
    {
        1.0 // CEC 2019 global optima are shifted to 1.0
    }

    fn evaluate(&self, solution: &[f64]) -> f64 
    {
        assert_eq!(solution.len(), self.dim, "Input vector dimension mismatch!");

        let mut fitness: f64 = 0.0;

        // SAFE FFI Bridge to C library
        unsafe 
        {
            cec19_test_func
            (
                solution.as_ptr(),
                &mut fitness as *mut f64,
                self.dim as i32,
                1, // single objective
                self.func_num as i32,
            );
        }

        fitness
    }
}