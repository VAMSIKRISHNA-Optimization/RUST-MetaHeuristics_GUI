pub trait Optimizer
{
    fn initialization(&mut self);
    fn step(&mut self);
    fn get_best_solution(&self) -> &[f64];
    fn get_best_solution_value(&self) -> f64;
}