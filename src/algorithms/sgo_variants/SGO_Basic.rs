#![allow(non_snake_case, non_camel_case_types)]

use crate::algorithm_structure::
{
    AlgorithmStatus, 
    BasicTuningParameters, 
    Bounding, BoundingStrategy, 
    CustomName, 
    FitnessEvaluation, 
    Initalize, Optimize,
    GreedySelection,
};
use crate::core::problem::Problem;
use ndarray::{Array1, Array2};
use rand::Rng;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct SGO_HyperParameters 
{
    Self_Instrospection_Factor : f64,
}

impl SGO_HyperParameters 
{
    pub fn new(Self_Instrospection_Factor: f64) -> Self 
    {
        Self 
        {
            Self_Instrospection_Factor,
        }
    }
}

impl Default for SGO_HyperParameters 
{
    fn default() -> Self 
    {
        // Suresh Satapathy & Anima Naik, 2016
        Self 
        {
            Self_Instrospection_Factor  : 0.2,
        }
    }
}

#[derive(Debug)]
pub struct SGO<T: Debug> 
{
    SGO_Status: AlgorithmStatus,
    CustomName: CustomName<T>,

    Basic_Tuning_Parameters: BasicTuningParameters,

    HyperParameters     : SGO_HyperParameters,
    Bounding_Strategy   : BoundingStrategy,

    Population      : Array2<f64>,
    Fitness_Scores  : Array1<f64>,

    gBest_Solution  : Array1<f64>,
    gBest_Score     : f64,

    New_Solution : Array1<f64>,

    lb: Array1<f64>,
    ub: Array1<f64>,

    Population_Size  : usize,
    Population_Index : usize,
    Dimensions       : usize,

    Total_Iterations    : usize,
    Current_Iteration   : usize,

    Function_Evaluations        : usize,
    Total_Function_Evaluations  : usize,
}

impl<T: Debug> SGO<T> 
{
    pub fn new
    (
        Np: usize,
        Ite: usize,
        NFEs: usize,
        CustomName: T,
        Self_Instrospection_Factor: f64,
        Bounding_Strategy: BoundingStrategy,
        Optimization_Problem: &dyn Problem,
    ) -> Self 
    {
        Self 
        {
            SGO_Status              : AlgorithmStatus::Not_Initialized,
            Basic_Tuning_Parameters : BasicTuningParameters::new(Np, Ite, NFEs),
            CustomName              : CustomName::new(CustomName),
            HyperParameters         : SGO_HyperParameters::new
            (
                Self_Instrospection_Factor,
            ),
            Bounding_Strategy       : Bounding_Strategy,
            
            Population      : Array2::from_elem((Np, Optimization_Problem.dimensions()), f64::INFINITY),
            Fitness_Scores  : Array1::from_elem(Np, f64::INFINITY),

            gBest_Solution  : Array1::from_elem(Optimization_Problem.dimensions(), f64::INFINITY),
            gBest_Score     : f64::INFINITY,

            New_Solution : Array1::from_elem(Optimization_Problem.dimensions(), f64::INFINITY),

            lb: Array1::from_elem(Optimization_Problem.dimensions(), f64::NEG_INFINITY),
            ub: Array1::from_elem(Optimization_Problem.dimensions(), f64::INFINITY),

            Population_Size : Np,
            Population_Index: 0,
            Dimensions      : Optimization_Problem.dimensions(),

            Total_Iterations    : Ite,
            Current_Iteration   : 0,
            Function_Evaluations: 0,
            Total_Function_Evaluations: NFEs,
        }
    }
}


impl<T: Debug> Initalize for SGO<T> 
{
    fn initialize(&mut self, Np: usize, Dim: usize, lb: &[f64], ub: &[f64]) 
    {
        assert_eq!(lb.len(), Dim, "Length of lb must match Dimensions");
        assert_eq!(ub.len(), Dim, "Length of ub must match Dimensions");

        self.Population_Size = Np;
        self.Dimensions      = Dim;

        self.lb = Array1::from_vec(lb.to_vec());
        self.ub = Array1::from_vec(ub.to_vec());

        self.Population     = Array2::from_elem((Np, Dim), f64::INFINITY);
        self.Fitness_Scores = Array1::from_elem(Np, f64::INFINITY);

        self.gBest_Solution = Array1::from_elem(Dim, f64::INFINITY);
        self.gBest_Score    = f64::INFINITY;


        let mut rng = rand::thread_rng();

        // Vectorized bounds initialization
        for p in 0..Np 
        {
            for d in 0..Dim 
            {
                let val = rng.gen_range(self.lb[d]..=self.ub[d]);
                self.Population[[p, d]]      = val;
            }
        }

        self.SGO_Status = AlgorithmStatus::Initialized 
        {
            Population: Np,
            Dimensions: Dim,
        };
    }

    fn is_initalized(&self) -> bool 
    {
        return matches!(self.SGO_Status, AlgorithmStatus::Initialized { .. });
    }
}


impl<T: Debug> Bounding for SGO<T> 
{
    fn bound_clamp(&mut self) 
    {
        for d in 0..self.Dimensions 
        {
            let val = self.New_Solution[d];
            if val < self.lb[d] 
            {
                self.Population[[self.Population_Index, d]] = self.lb[d];
            } else if val > self.ub[d] 
            {
                self.Population[[self.Population_Index, d]] = self.ub[d];
            }
        }
    }

    fn bound_reflect(&mut self, damping_factor: f64) 
    {
        for d in 0..self.Dimensions 
        {
            let val = self.New_Solution[d];
            if val < self.lb[d] 
            {
                self.Population[[self.Population_Index, d]] =
                    self.lb[d] + (self.lb[d] - val) * damping_factor;
            } else if val > self.ub[d] 
            {
                self.Population[[self.Population_Index, d]] =
                    self.ub[d] - (val - self.ub[d]) * damping_factor;
            }
        }
        
    }

    fn bound_wrap(&mut self, overshoot_factor: f64) 
    {

        for d in 0..self.Dimensions 
        {
            let val = self.New_Solution[d];
            if val < self.lb[d] 
            {
                self.Population[[self.Population_Index, d]] =
                    self.ub[d] - (self.lb[d] - val) * overshoot_factor;
            } 
            else if val > self.ub[d] 
            {
                self.Population[[self.Population_Index, d]] =
                    self.lb[d] + (val - self.ub[d]) * overshoot_factor;
            }
        }
        
    }

    fn bound_reinitalize(&mut self) 
    {
        let mut rng = rand::thread_rng();
        for d in 0..self.Dimensions 
        {
            let val = self.New_Solution[d];
            if val < self.lb[d] || val > self.ub[d] 
            {
                self.Population[[self.Population_Index, d]] = rng.gen_range(self.lb[d]..=self.ub[d]);
            }
        }
}
    
}


impl<T: Debug> FitnessEvaluation for SGO<T> 
{
    fn evaluate_fitness_entire_population_one_by_one(&mut self, problem: &dyn Problem) 
    {
        for i in 0..self.Population_Size 
        {
            // High-performance zero-copy row slice conversion
            let row_view = self.Population.row(i).into_owned();
            let slice    = row_view.as_slice().unwrap();

            let fitness = problem.evaluate(slice);
            self.Fitness_Scores[i] = fitness;

            // // Greedy selection
            // if fitness < self.Fitness_Scores[i]
            // {
            //     self.Fitness_Scores[i] = fitness;
            //     self.Population
            //         .row_mut(i)
            //         .assign(&row_view);
            // }

            // Update gBest
            if fitness < self.gBest_Score 
            {
                self.gBest_Score = fitness;
                self.gBest_Solution
                    .assign(&row_view);
            }
        }
    }

    fn evaluate_fitness_entire_population_all_at_once(&mut self, problem: &dyn Problem) 
    {
        // Execute the batch evaluation directly on the C++ side
        problem.evaluate_batch(&self.Population, &mut self.Fitness_Scores);

    }

    fn get_population_member_by_index(&self, pop_index: usize) -> &[f64] 
    {
        let dim   = self.Dimensions as usize; 
        let start = pop_index * dim;
        
        // Slice the full contiguous buffer directly using the calculated offsets
        &self.Population.as_slice().unwrap()[start .. start + dim]
    }

    fn evaluate_single_population_member(&self, problem: &dyn Problem, pop_index: usize) -> f64
    {
        problem.evaluate(self.get_population_member_by_index(pop_index))
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


impl <T: Debug> GreedySelection for SGO<T> 
{
    fn greedy_selection(&mut self, new_fitness: f64, current_fitness: f64, pop_ind: usize)
    {
        // Update the population member if the new fitness is better
        if new_fitness < current_fitness 
        {
            self.Fitness_Scores[pop_ind] = new_fitness;
                self.Population
                    .row_mut(pop_ind)
                    .assign(&self.New_Solution);
        } 

        // Update gBest
        if new_fitness < self.gBest_Score 
        {
            self.gBest_Score = new_fitness;
            self.gBest_Solution
                .assign(&self.New_Solution);
        }

    }


}


impl<T: Debug> Optimize for SGO<T> 
{
    fn step(&mut self, problem: &dyn Problem) 
    {
        let mut rng = rand::thread_rng();

        // Get the gbest 
        if self.Current_Iteration == 0
        {
            self.evaluate_fitness_entire_population_one_by_one(problem);
            self.Function_Evaluations += self.Population_Size;
        }
        
        self.Population_Index = 0;

        // 1. IMPROVING PHASE
        for p in 0..self.Population_Size 
        {
            for d in 0..self.Dimensions 
            {
                let r1: f64 = rng.r#gen();

                let current_member = self.Population[[p, d]];
                let gbest_member   = self.gBest_Solution[d];

                self.New_Solution[d] = self.HyperParameters.Self_Instrospection_Factor * current_member
                            + r1 * (gbest_member - current_member);

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

            let new_fitness = problem.evaluate(self.New_Solution.as_slice().unwrap());

            self.greedy_selection(new_fitness, self.Fitness_Scores[p], p);
            self.Population_Index += 1;
        
        }



        // Evaluate updated fitness
        self.evaluate_fitness_entire_population_one_by_one(problem);
        self.Function_Evaluations += self.Population_Size;
        self.Population_Index = 0;

        // 2. ACQUIRING PHASE
        for p in 0..self.Population_Size 
        {
            
            let mut pr1 = rng.gen_range(0..self.Population_Size);
            while pr1 == p { pr1 = rng.gen_range(0..self.Population_Size); }

            let x_r1 = self.Population.row(pr1);

            for d in 0..self.Dimensions 
            {
                let r1: f64 = rng.r#gen();
                let r2: f64 = rng.r#gen();

                let current_member = self.Population[[p, d]];
                let gbest_member   = self.gBest_Solution[d];
                let rand_member    = x_r1[d];
                
                if self.Fitness_Scores[p] < self.Fitness_Scores[pr1]
                {
                    self.New_Solution[d] = current_member
                                + r1 * (current_member - rand_member)
                                + r2 * (gbest_member - current_member);
                }
                else 
                {
                    self.New_Solution[d] = current_member
                                + r1 * (rand_member - current_member)
                                + r2 * (gbest_member - current_member);
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

            let new_fitness = problem.evaluate(self.New_Solution.as_slice().unwrap());

            self.greedy_selection(new_fitness, self.Fitness_Scores[p], p);
            
            self.Population_Index += 1;
        }

        // Evaluate updated fitness
        self.Function_Evaluations += self.Population_Size;


        self.Current_Iteration    += 1;
        if self.Current_Iteration >= self.Total_Iterations 
        {
            self.SGO_Status = AlgorithmStatus::Completed;
        } 
        else 
        {
            self.SGO_Status = AlgorithmStatus::Running 
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
            self.Population_Index = 0;
            self.initialize(self.Population_Size, dim, lb, ub);
        }

        // Evaluate initial generation
        self.evaluate_fitness_entire_population_one_by_one(problem);

        while !self.is_done() 
        {
            self.step(problem);
        }
    }

    fn is_running(&self) -> bool 
    {
        return matches!(self.SGO_Status, AlgorithmStatus::Running { .. });
    }

    fn is_done(&self) -> bool {
        return matches!(self.SGO_Status, AlgorithmStatus::Completed);
    }
}