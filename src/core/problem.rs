pub trait Problem 
{
    fn dimensions(&self) -> usize;
    fn bounds(&self) -> (&[f64], &[f64]);
    fn name(&self) -> &str;
    fn suite_name(&self) -> &str;
    fn optimal_value(&self) -> f64;

    fn evaluate(&self, solution: &[f64]) -> f64;
}