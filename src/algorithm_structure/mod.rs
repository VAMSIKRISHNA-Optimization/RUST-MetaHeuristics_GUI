use std::fmt::Debug;
use crate::core::problem::Problem;

#[derive(Debug)]
pub struct BasicTuningParameters
{
    Np  : usize, // Population Size
    Ite : usize, // Iterations
    NFEs: usize, // Number of Function Evaluations
}

impl BasicTuningParameters
{
    pub fn new(Np: usize, Ite: usize, NFEs: usize) -> Self
    {
        Self
        {
            Np,
            Ite,
            NFEs,
        }
    }

    pub fn get_total_iterations(&self) -> usize
    {
        self.Ite
    }
}

impl Default for BasicTuningParameters
{
    fn default() -> Self
    {
        Self
        {
            Np    : crate::DEFAULT_POPULATION_SIZE,
            Ite   : crate::DEFAULT_ITERATIONS,
            NFEs  : crate::DEFAULT_POPULATION_SIZE * crate::DEFAULT_ITERATIONS,
        }
    }
}


#[derive(Debug)]
pub struct CustomName<T>
{
    Algorithm_CustomName: T
}

impl<T: Debug> CustomName<T>
{
    pub fn new(Algorithm_CustomName: T) -> Self
    {
        Self{Algorithm_CustomName}
    }
}

impl Default for CustomName<String> 
{
    fn default() -> Self 
    {
        CustomName 
        {
            Algorithm_CustomName: String::from("No Name"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum AlgorithmStatus
{
    Not_Initialized,
    Initialized { Population: usize, Dimensions: usize },
    Running     { Iterations: usize, FunctionEvaluations: usize},
    Completed  
}


#[derive(Debug)]
pub enum BoundingStrategy
{
    Clamping,
    Reflecting,
    Wrapping,
    Penalization,
    ReInitialization,
}

pub trait Bounding
{
    fn bound_clamp(&mut self);
    fn bound_reflect(&mut self, damping_factor: f64);
    fn bound_wrap(&mut self, overshoot_factor: f64);
    fn bound_reinitalize(&mut self);

    // fn bound_penalize(&mut self);

}

pub trait Initalize
{
    fn initialize(&mut self, Np: usize, Dim: usize, lb: &[f64], ub: &[f64] );

    //Getters
    fn is_initalized(&self) -> bool;
}

pub trait FitnessEvaluation
{
    fn evaluate_fitness_entire_population_one_by_one(&mut self, problem: &dyn Problem);
    fn evaluate_fitness_entire_population_all_at_once(&mut self, problem: &dyn Problem);

    fn get_population_member_by_index(&self, pop_index: usize) -> &[f64];
    fn evaluate_single_population_member(&self, problem: &dyn Problem, pop_index: usize) -> f64
    {
        problem.evaluate(self.get_population_member_by_index(pop_index))
    }

   
    fn get_best_solution(&self) -> &[f64];
    fn get_best_score(&self) -> f64;
}

pub trait GreedySelection
{
    fn greedy_selection(&mut self, new_fitness: f64, current_fitness: f64, pop_ind: usize);
}

pub trait Optimize: Initalize + Bounding + FitnessEvaluation
{
    fn step(&mut self,     problem: &dyn Problem); // Intented to perform one iteration at a time, and update the GUI accordingly.
    fn optimize(&mut self, problem: &dyn Problem); // Intended to run the entire optimization process in one go, without GUI updates.);

    // Getters
    fn is_running(&self) -> bool;
    fn is_done(&self)    -> bool;

}