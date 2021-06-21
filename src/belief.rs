use itertools::Itertools;
use rand::{
    distributions::WeightedIndex,
    prelude::{Distribution, StdRng},
};

use crate::{lane_change_policy::LongitudinalPolicy, road::Road};

fn predict_lane(road: &Road, car_i: usize) -> i32 {
    let car = &road.cars[car_i];
    let predicted_y = car.y + car.vel * car.theta.sin() * road.params.lane_change_time;
    Road::get_lane_i(predicted_y).min(1).max(0)
}

fn predict_long(road: &Road, car_i: usize) -> LongitudinalPolicy {
    let lane_i = road.cars[car_i].current_lane();
    let ahead_dist = road.dist_clear_ahead_in_lane(car_i, lane_i);
    let bparams = &road.params.belief;
    let car = &road.cars[car_i];
    if let Some((ahead_dist, ahead_car_i)) = ahead_dist {
        let ahead_car = &road.cars[ahead_car_i];
        if car.vel > ahead_car.vel + bparams.accelerate_delta_vel_thresh
            || ahead_dist < bparams.accelerate_ahead_dist_thresh
        {
            return LongitudinalPolicy::Accelerate;
        } else {
            return LongitudinalPolicy::Maintain;
        }
    }
    if car.vel < bparams.decelerate_vel_thresh {
        LongitudinalPolicy::Decelerate
    } else {
        LongitudinalPolicy::Accelerate
    }
}

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

    pub fn update(&mut self, road: &Road) {
        for (car_i, belief) in self.belief.iter_mut().enumerate().skip(1) {
            let pred_lane = predict_lane(road, car_i);
            let pred_long = predict_long(road, car_i);

            belief.clear();
            for &lane_i in &[0, 1] {
                for long_policy in [LongitudinalPolicy::Maintain, LongitudinalPolicy::Accelerate] {
                    let mut prob = 1.0;
                    if lane_i != pred_lane {
                        prob *= road.params.belief.different_lane_prob;
                    }
                    if long_policy != pred_long {
                        prob *= road.params.belief.different_longitudinal_prob;
                    }
                    belief.push(prob);
                }
            }
            if LongitudinalPolicy::Decelerate == pred_long {
                belief.push(1.0);
            } else {
                belief.push(road.params.belief.different_longitudinal_prob);
            }
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
