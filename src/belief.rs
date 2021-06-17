use itertools::Itertools;
use rand::{
    distributions::WeightedIndex,
    prelude::{Distribution, StdRng},
};

#[derive(Clone)]
pub struct Belief {
    belief: Vec<Vec<f64>>,
}
impl Belief {
    pub fn uniform(n_cars: usize, n_policies: usize) -> Self {
        Self {
            belief: vec![vec![1.0 / n_policies as f64; n_policies]; n_cars],
        }
    }

    pub fn sample(&self, rng: &mut StdRng) -> Vec<usize> {
        self.belief
            .iter()
            .map(|weights| WeightedIndex::new(weights).unwrap().sample(rng))
            .collect_vec()
    }
}
