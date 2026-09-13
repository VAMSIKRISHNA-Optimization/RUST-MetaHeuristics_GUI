use crate::core::problem::Problem;

use core::panic;
// Standard CEC C FFI signature
use std::os::raw::c_int;

#[link(name = "cec19", kind = "static")]
unsafe extern "C" 
{
    pub unsafe fn cec19_test_func
    (
        x: *const f64,     // Array of decision variables
        f: *mut f64,       // Array to store fitness results
        nx: c_int,         // Number of dimensions (n)
        mx: c_int,         // Size of the population (m) sent for evaluation
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
    fn dimensions(&self) -> usize 
    {
        self.dim
    }

    fn function_number(&self) -> usize 
    {
        self.func_num
    }

    fn bounds(&self) -> (&[f64], &[f64]) 
    {
        match self.func_num
        {
            1 => (&[-8192.0; 9], &[8192.0; 9]), // F1: 9D
            2 => (&[-16384.0; 16], &[16384.0; 16]), // F2: 16D
            3 => (&[-4.0; 18], &[4.0; 18]), // F3: 18D
            _ => (&[-100.0; 10], &[100.0; 10]), // F4 to F10: 10D
        }
    }

    fn name(&self) -> &str 
    {
        match self.func_num 
        {
            1 => "F1: Storn's Chebyshev Polynomial Fitting Problem",
            2 => "F2: Inverse Hilbert Matrix Problem",
            3 => "F3: Lennard-Jones Minimum Energy Cluster Problem",
            4 => "F4: Shifted Rotated Rastrigin's Function",
            5 => "F5: Shifted Rotated Griewank's Function",
            6 => "F6: Shifted Rotated Weierstrass Function",
            7 => "F7: Shifted Rotated Schwefel's Function",
            8 => "F8: Shifted Rotated Expanded Schaffer's F6 Function",
            9 => "F9: Shifted Rotated HappyCat Function",
            10 => "F10: Shifted Rotated Ackley Function",
            _   =>  panic!("Invalid function number for CEC2019: {}", self.func_num),
        }
    }

    fn suite_name(&self) -> &str 
    {
        "CEC 2019 Benchmark Suite"
    }

    fn optimal_value(&self) -> f64 
    {
        return 1.000000000; // CEC 2019 global optima are shifted to 1.0 with 10 decimal places of precision
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
                1, // Single solution evaluation
                self.func_num as i32,
            );
        }

        fitness
    }

    fn evaluate_batch(&self, population: &ndarray::Array2<f64>, fitnesses: &mut ndarray::Array1<f64>) 
    {
        let (num_samples, _) = population.dim();
        assert_eq!(self.dimensions(), self.dim, "Dimension mismatch in batch evaluation!");
        assert_eq!(fitnesses.len(), num_samples, "Fitness score array size mismatch!");

        unsafe 
        {
            cec19_test_func
            (
                population.as_ptr(),
                fitnesses.as_mut_ptr(),
                self.dimensions() as i32,
                num_samples as i32, // Entire population evaluated in C++
                self.func_num as i32,
            );
        }
    }
}