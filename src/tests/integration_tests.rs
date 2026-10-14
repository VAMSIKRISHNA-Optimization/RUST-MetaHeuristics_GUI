#[cfg(test)]
mod tests {
    use crate::algorithm_structure::{ BoundingStrategy, FitnessEvaluation, Optimize,};
    use crate::algorithms::pso_variants::PSO_Basic::PSO;
    use crate::algorithms::sgo_variants::SGO_Basic::SGO;
    use crate::core::problem::Problem;

    struct SphereProblem {
        dim: usize,
        lb: Vec<f64>,
        ub: Vec<f64>,
    }

    impl SphereProblem {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                lb: vec![-5.12; dim],
                ub: vec![5.12; dim],
            }
        }
    }

    impl Problem for SphereProblem {
        fn evaluate(&self, position: &[f64]) -> f64 {
            position.iter().map(|x| x * x).sum()
        }

        fn bounds(&self) -> (&[f64], &[f64]) {
            (&self.lb, &self.ub)
        }

        fn dimensions(&self) -> usize {
            self.dim
        }

        fn function_number(&self) -> usize {
            1
        }

        fn name(&self) -> &str {
            "Sphere"
        }

        fn suite_name(&self) -> &str {
            "Benchmark"
        }

        fn optimal_value(&self) -> f64 {
            0.0
        }
    }

    #[test]
    fn test_pso_sphere_convergence() {
        let problem = SphereProblem::new(3);

        let mut pso = PSO::new(
            100,                         // Swarm size (Np)
            1000,                        // Max iterations
            100000,                       // NFEs
            "PSO_Test",                 // CustomName
            0.7298,                     // Inertia Weight (w)
            1.49618,                    // Cognitive Coefficient (c1)
            1.49618,                    // Social Coefficient (c2)
            0.5,                        // Kinetic Energy (V_max factor)
            BoundingStrategy::Clamping, // Bounding Strategy
            &problem,                   // Problem reference
        );

        pso.optimize(&problem);

        assert!(
            pso.get_best_score() < 1e-0,
            "PSO failed to converge. Best score: {}",
            pso.get_best_score()
        );
    }

    #[test]
    fn test_sgo_sphere_convergence() {
        let problem = SphereProblem::new(3);

        let mut sgo = SGO::new(
            30,                         // Population size (Np)
            200,                        // Max iterations
            6000,                       // NFEs
            "SGO_Test",                 // CustomName
            0.2,                        // Self-introspection factor
            BoundingStrategy::Clamping, // Bounding Strategy
            &problem,                   // Problem reference
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
        let problem = SphereProblem::new(2);

        let mut pso = PSO::new(
            10,
            50,
            500,
            "PSO_Step_Test",
            0.7298,
            1.49618,
            1.49618,
            0.5,
            BoundingStrategy::Clamping,
            &problem,
        );

        pso.step(&problem);

        assert!(pso.get_best_score().is_finite());
    }

    #[test]
    fn test_sgo_minimum_population_boundary() {
        let problem = SphereProblem::new(2);

        let mut sgo = SGO::new(
            2,
            10,
            20,
            "SGO_Min_Test",
            0.2,
            BoundingStrategy::Clamping,
            &problem,
        );

        sgo.optimize(&problem);

        assert!(sgo.get_best_score().is_finite());
    }

    #[test]
    #[should_panic(expected = "Population size must be at least 2")]
    fn test_sgo_invalid_population_panic() {
        let problem = SphereProblem::new(2);

        let _sgo = SGO::new(
            1,
            10,
            10,
            "SGO_Panic_Test",
            0.2,
            BoundingStrategy::Clamping,
            &problem,
        );
    }
}