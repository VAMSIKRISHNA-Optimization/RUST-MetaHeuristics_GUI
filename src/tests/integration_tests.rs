
#[cfg(test)]
mod tests 
{
    use crate::algorithm_structure::FitnessEvaluation;
// Adjust these paths if your modules are structured differently in crate root
    use crate::core::problem::Problem;
    use crate::algorithms::pso_variants::PSO_Basic::PSO;
    use crate::algorithms::sgo_variants::SGO_Basic::SGO;

    // Import the Optimize trait so .optimize() is accessible
    use crate::algorithm_structure::Optimize;
    
    // Import BoundingStrategy (adjust path if located elsewhere in your crate)
    use crate::algorithm_structure::BoundingStrategy;

    static  DIMENSIONS: usize = 3;

    struct SphereProblem 
    {
        dim: usize,
    }

    impl Problem for SphereProblem 
    {
        fn evaluate(&self, position: &[f64]) -> f64 
        {
            position.iter().map(|x| x * x).sum()
        }

        fn bounds(&self) -> (&[f64], &[f64]) 
        {
            ([-5.12; DIMENSIONS].as_slice(), [5.12; DIMENSIONS].as_slice())
        }

        fn dimensions(&self) -> usize 
        {
            self.dim
        }

        fn function_number(&self) -> usize 
        {
            1
        }

        fn name(&self) -> &str 
        {
            "Sphere"
        }

        fn suite_name(&self) -> &str 
        {
            "Benchmark"
        }

        fn optimal_value(&self) -> f64 
        {
            0.0
        }
    }

#[test]
    fn test_pso_sphere_convergence() {
        let problem = SphereProblem { dim: DIMENSIONS };
        
        let mut pso = PSO::new(
            30,                      // Swarm size
            200,                     // Max iterations
            problem.dimensions(),    // Dimension
            1.5,                     // c1
            1.5,                     // c2
            0.7,                     // w
            -5.12,                   // Min bound
            5.12,                    // Max bound
            BoundingStrategy::Clamping, // Bounding strategy (use your enum variant)
            &problem,                // Problem reference
        );

        pso.optimize(&problem);

        assert!(
            pso.get_best_score() < 1e-2,
            "PSO failed to converge. Best score: {}",
            pso.get_best_score()
        );
    }

    #[test]
    fn test_sgo_sphere_convergence() {
        let problem = SphereProblem { dim: DIMENSIONS };
        
        let mut sgo = SGO::new(
            30,                      // Population size
            200,                     // Max iterations
            problem.dimensions(),    // Dimension
            0.2,                     // Self-introspection factor
            1.0,                     // Additional parameter required by SGO
            BoundingStrategy::Clamping, // Bounding strategy
            &problem,                // Problem reference
        );

        sgo.optimize(&problem);

        assert!(
            sgo.get_best_score() < 1e-2,
            "SGO failed to converge. Best score: {}",
            sgo.get_best_score()
        );
    }

    #[test]
    fn test_pso_standalone_step_execution() {
        let problem = SphereProblem { dim: 2 };
        
        let mut pso = PSO::new(
            10,
            50,
            problem.dimensions(),
            1.0,
            1.0,
            0.5,
            -5.12,
            5.12,
            BoundingStrategy::Clamping,
            &problem,
        );

        pso.step(&problem);

        assert!(pso.get_best_score().is_finite());
        assert_eq!(pso.get_current_iteration(), 1); // Use getter method
    }

    #[test]
    fn test_sgo_minimum_population_boundary() 
    {
        let problem = SphereProblem { dim: 2 };
        
        let mut sgo = SGO::new(
            2,
            10,
            problem.dimensions(),
            0.1,
            1.0,
            BoundingStrategy::Clamping,
            &problem,
        );

        sgo.optimize(&problem);

        assert!(sgo.get_best_score().is_finite());
    }

    #[test]
    #[should_panic(expected = "Population size must be at least 2")]
    fn test_sgo_invalid_population_panic() 
    {
        let problem = SphereProblem { dim: 2 };
        
        let _sgo = SGO::new(
            1,
            10,
            problem.dimensions(),
            0.1,
            1.0,
            BoundingStrategy::Clamping,
            &problem,
        );
    }
}