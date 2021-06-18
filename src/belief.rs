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

    pub fn get(&self, car_i: usize, policy_id: usize) -> f64 {
        assert_ne!(car_i, 0);
        self.belief[car_i][policy_id]
    }

    pub fn get_most_likely(&self, car_i: usize) -> usize {
        assert_ne!(car_i, 0);
        self.belief[car_i]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }

    pub fn is_uncertain(&self, car_i: usize, threshold: f64) -> bool {
        assert_ne!(car_i, 0);
        if self.belief[car_i].len() <= 1 {
            return false;
        }

        let mut values = self.belief[car_i].clone();

        // sort descending (switched a and b)
        values.sort_by(|a, b| b.partial_cmp(a).unwrap());

        (values[0] - values[1]) < threshold
    }
}
