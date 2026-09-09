#![allow(non_snake_case, non_camel_case_types)]

use crate::algorithm_structure::
{
    AlgorithmStatus, 
    BasicTuningParameters, 
    Bounding, BoundingStrategy, 
    CustomName, 
    FitnessEvaluation, 
    Initalize, Optimize,
};
use crate::core::problem::Problem;
use ndarray::{Array1, Array2};
use rand::Rng;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct PSO_HyperParameters 
{
    Inertia_Weight          : f64,
    Cognitive_Coefficient   : f64,
    Social_Coefficient      : f64,
    Kinetic_Energy          : f64,
}

impl PSO_HyperParameters 
{
    pub fn new(Inertia_Weight: f64, Cognitive_Coefficient: f64, Social_Coefficient: f64, Kinetic_Energy: f64) -> Self 
    {
        Self 
        {
            Inertia_Weight,
            Cognitive_Coefficient,
            Social_Coefficient,
            Kinetic_Energy,
        }
    }
}

impl Default for PSO_HyperParameters 
{
    fn default() -> Self 
    {
        // Clerc Constriction values (2002)
        Self 
        {
            Inertia_Weight          : 0.72980,
            Cognitive_Coefficient   : 1.49618,
            Social_Coefficient      : 1.49618,
            Kinetic_Energy          : 0.5,
        }
    }
}

#[derive(Debug)]
pub struct PSO<T: Debug> 
{
    PSO_Status: AlgorithmStatus,
    CustomName: CustomName<T>,

    Basic_Tuning_Parameters: BasicTuningParameters,

    HyperParameters     : PSO_HyperParameters,
    Bounding_Strategy   : BoundingStrategy,

    Population      : Array2<f64>,
    Fitness_Scores  : Array1<f64>,
    pBest_Solution  : Array2<f64>,
    pBest_Score     : Array1<f64>,
    gBest_Solution  : Array1<f64>,
    gBest_Score     : f64,
    Velocities      : Array2<f64>,

    lb: Array1<f64>,
    ub: Array1<f64>,

    Swarm_Size: usize,
    Dimensions: usize,

    Total_Iterations    : usize,
    Current_Iteration   : usize,

    Function_Evaluations        : usize,
    Total_Function_Evaluations  : usize,
}

impl<T: Debug> PSO<T> 
{
    pub fn new
    (
        Np: usize,
        Ite: usize,
        D: usize,
        NFEs: usize,
        CustomName: T,
        Inertia_Weight: f64,
        Cognitive_Coefficient: f64,
        Social_Coefficient: f64,
        Kinetic_Energy: f64,
        Bounding_Strategy: BoundingStrategy,
    ) -> Self 
    {
        Self 
        {
            PSO_Status              : AlgorithmStatus::Not_Initialized,
            Basic_Tuning_Parameters : BasicTuningParameters::new(Np, Ite, NFEs),
            CustomName              : CustomName::new(CustomName),
            HyperParameters         : PSO_HyperParameters::new
            (
                Inertia_Weight,
                Cognitive_Coefficient,
                Social_Coefficient,
                Kinetic_Energy,
            ),
            Bounding_Strategy       : Bounding_Strategy,
            
            Population      : Array2::from_elem((Np, D), f64::INFINITY),
            Fitness_Scores  : Array1::from_elem(Np, f64::INFINITY),
            pBest_Solution  : Array2::from_elem((Np, D), f64::INFINITY),
            pBest_Score     : Array1::from_elem(Np, f64::INFINITY),
            gBest_Solution  : Array1::from_elem(D, f64::INFINITY),
            gBest_Score     : f64::INFINITY,
            Velocities      : Array2::zeros((Np, D)),
            
            lb: Array1::from_elem(D, f64::NEG_INFINITY),
            ub: Array1::from_elem(D, f64::INFINITY),

            Swarm_Size: Np,
            Dimensions: D,
            Total_Iterations: Ite,
            Current_Iteration   : 0,
            Function_Evaluations: 0,
            Total_Function_Evaluations: NFEs,
        }
    }
}

impl<T: Debug> Initalize for PSO<T> 
{
    fn initialize(&mut self, Np: usize, Dim: usize, lb: &[f64], ub: &[f64]) 
    {
        assert_eq!(lb.len(), Dim, "Length of lb must match Dimensions");
        assert_eq!(ub.len(), Dim, "Length of ub must match Dimensions");

        self.Swarm_Size = Np;
        self.Dimensions = Dim;
        self.lb = Array1::from_vec(lb.to_vec());
        self.ub = Array1::from_vec(ub.to_vec());

        self.Population     = Array2::from_elem((Np, Dim), f64::INFINITY);
        self.Fitness_Scores = Array1::from_elem(Np, f64::INFINITY);
        self.pBest_Solution = Array2::from_elem((Np, Dim), f64::INFINITY);
        self.pBest_Score    = Array1::from_elem(Np, f64::INFINITY);
        self.gBest_Solution = Array1::from_elem(Dim, f64::INFINITY);
        self.gBest_Score    = f64::INFINITY;
        self.Velocities     = Array2::zeros((Np, Dim));

        let mut rng = rand::thread_rng();

        // Vectorized bounds initialization
        for p in 0..Np 
        {
            for d in 0..Dim 
            {
                let val = rng.gen_range(self.lb[d]..=self.ub[d]);
                self.Population[[p, d]]      = val;
                self.pBest_Solution[[p, d]]  = val;
            }
        }

        // Half-Difference Randomization for velocity initialization
        for p in 0..Np 
        {
            for d in 0..Dim 
            {
                self.Velocities[[p, d]] =
                    (rng.gen_range(0.0..=1.0) - 0.5) * (self.ub[d] - self.lb[d]);
            }
        }

        self.PSO_Status = AlgorithmStatus::Initialized 
        {
            Population: Np,
            Dimensions: Dim,
        };
    }

    fn is_initalized(&self) -> bool 
    {
        return matches!(self.PSO_Status, AlgorithmStatus::Initialized { .. });
    }
}

impl<T: Debug> Bounding for PSO<T> 
{
    fn bound_clamp(&mut self) 
    {
        for p in 0..self.Swarm_Size 
        {
            for d in 0..self.Dimensions 
            {
                let val = self.Population[[p, d]];
                if val < self.lb[d] 
                {
                    self.Population[[p, d]] = self.lb[d];
                } else if val > self.ub[d] 
                {
                    self.Population[[p, d]] = self.ub[d];
                }
            }
        }
    }

    fn bound_reflect(&mut self, damping_factor: f64) {
        for p in 0..self.Swarm_Size 
        {
            for d in 0..self.Dimensions 
            {
                let val = self.Population[[p, d]];
                if val < self.lb[d] 
                {
                    self.Population[[p, d]] =
                        self.lb[d] + (self.lb[d] - val) * damping_factor;
                } else if val > self.ub[d] 
                {
                    self.Population[[p, d]] =
                        self.ub[d] - (val - self.ub[d]) * damping_factor;
                }
            }
        }
    }

    fn bound_wrap(&mut self, overshoot_factor: f64) {
        for p in 0..self.Swarm_Size 
        {
            for d in 0..self.Dimensions 
            {
                let val = self.Population[[p, d]];
                if val < self.lb[d] 
                {
                    self.Population[[p, d]] =
                        self.ub[d] - (self.lb[d] - val) * overshoot_factor;
                } 
                else if val > self.ub[d] 
                {
                    self.Population[[p, d]] =
                        self.lb[d] + (val - self.ub[d]) * overshoot_factor;
                }
            }
        }
    }

    fn bound_reinitalize(&mut self) 
    {
        let mut rng = rand::thread_rng();
        for p in 0..self.Swarm_Size 
        {
            for d in 0..self.Dimensions 
            {
                let val = self.Population[[p, d]];
                if val < self.lb[d] || val > self.ub[d] 
                {
                    self.Population[[p, d]] = rng.gen_range(self.lb[d]..=self.ub[d]);
                }
            }
        }
    }
}

impl<T: Debug> FitnessEvaluation for PSO<T> {
    fn evaluate_fitness_single_population(&mut self, &dyn Problem) 
    {
        for i in 0..self.Swarm_Size 
        {
            // High-performance zero-copy row slice conversion
            let row_view = self.Population.row(i);
            let slice    = row_view.as_slice().unwrap();

            let fitness = problem.evaluate(slice);
            self.Fitness_Scores[i] = fitness;

            // Update pBest
            if fitness < self.pBest_Score[i] 
            {
                self.pBest_Score[i] = fitness;
                self.pBest_Solution
                    .row_mut(i)
                    .assign(&self.Population.row(i));
            }

            // Update gBest
            if fitness < self.gBest_Score 
            {
                self.gBest_Score = fitness;
                self.gBest_Solution
                    .assign(&self.Population.row(i));
            }
        }
    }

    fn evaluate_fitness_entire_population(&mut self, problem: &dyn Problem) 
    {
        unsafe 
        {
            // Extract raw pointers to the underlying contiguous data buffers
            let x_ptr = self.Population.as_mut_ptr();
            let f_ptr = self.Fitness_Scores.as_mut_ptr();

            // Cast dimensions to C-compatible integers
            let nx = self.Dimensions as c_int;
            let mx = self.Swarm_Size as c_int;
            let c_func_num = func_num as c_int;

            // Execute the batch evaluation directly on the C++ side
            cec19_test_func(x_ptr, f_ptr, nx, mx, c_func_num);
        }
        
    }

    fn get_best_score(&self) -> f64 
    {
        self.gBest_Score
    }

    fn get_best_solution(&self) -> &[f64] 
    {
        self.gBest_Solution.as_slice().unwrap()
    }
}

impl<T: Debug> Optimize for PSO<T> 
{
    fn step(&mut self, problem: &dyn Problem) 
    {
        let mut rng = rand::thread_rng();

        // Update velocities and positions
        for p in 0..self.Swarm_Size 
        {
            for d in 0..self.Dimensions 
            {
                let r1: f64 = rng.r#gen();
                let r2: f64 = rng.r#gen();

                let current_vel = self.Velocities[[p, d]];
                let pos         = self.Population[[p, d]];
                let pbest_pos   = self.pBest_Solution[[p, d]];
                let gbest_pos   = self.gBest_Solution[[p, d]];

                let mut new_vel = self.HyperParameters.Inertia_Weight * current_vel
                                + self.HyperParameters.Cognitive_Coefficient * r1 * (pbest_pos - pos)
                                + self.HyperParameters.Social_Coefficient * r2 * (gbest_pos - pos);

                // Velocity Clamping
                let max_vel = self.HyperParameters.Kinetic_Energy * (self.ub[d] - self.lb[d]);
                let min_vel = -max_vel;

                if new_vel > max_vel 
                {
                    new_vel = max_vel;
                } else if new_vel < min_vel 
                {
                    new_vel = min_vel;
                }

                self.Velocities[[p, d]] = new_vel;
                self.Population[[p, d]] += new_vel;
            }
        }

        // Apply selected Bounding Strategy
        match self.Bounding_Strategy 
        {
            BoundingStrategy::Clamping          => self.bound_clamp(),
            BoundingStrategy::Reflecting        => self.bound_reflect(0.5),
            BoundingStrategy::Wrapping          => self.bound_wrap(1.0),
            BoundingStrategy::ReInitialization  => self.bound_reinitalize(),
            _ => self.bound_clamp(),
        }

        // Evaluate updated fitness
        self.evaluate_fitness(problem);

        self.Current_Iteration += 1;
        self.Function_Evaluations = self.Swarm_Size * self.Current_Iteration;

        if self.Current_Iteration >= self.Total_Iterations 
        {
            self.PSO_Status = AlgorithmStatus::Completed;
        } 
        else 
        {
            self.PSO_Status = AlgorithmStatus::Running 
            {
                Iterations          : self.Current_Iteration,
                FunctionEvaluations : self.Function_Evaluations,
            };
        }
    }

    fn optimize(&mut self, problem: &dyn Problem) 
    {
        if !self.is_initalized() 
        {
            let dim      = problem.dimensions();
            let (lb, ub) = problem.bounds();
            self.initialize(self.Swarm_Size, dim, lb, ub);
        }

        // Evaluate initial generation
        self.evaluate_fitness(problem);

        while !self.is_done() 
        {
            self.step(problem);
        }
    }

    fn is_running(&self) -> bool 
    {
        return matches!(self.PSO_Status, AlgorithmStatus::Running { .. });
    }

    fn is_done(&self) -> bool {
        return matches!(self.PSO_Status, AlgorithmStatus::Completed);
    }
}