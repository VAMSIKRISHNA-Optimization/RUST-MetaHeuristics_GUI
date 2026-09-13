pub trait Problem 
{
    fn dimensions(&self) -> usize;
    fn function_number(&self) -> usize;
    fn bounds(&self) -> (&[f64], &[f64]);
    fn name(&self) -> &str;
    fn suite_name(&self) -> &str;
    fn optimal_value(&self) -> f64;

    fn evaluate(&self, solution: &[f64]) -> f64;

    // Batch evaluation: defaults to looping single evaluations, 
    // but can be overridden by CEC2019 for high-speed FFI evaluations.
    fn evaluate_batch(&self, population: &ndarray::Array2<f64>, fitnesses: &mut ndarray::Array1<f64>) 
    {
        for (row, fit) in population.rows().into_iter().zip(fitnesses.iter_mut()) 
        {
            *fit = self.evaluate(row.as_slice().unwrap());
        }
    }
}