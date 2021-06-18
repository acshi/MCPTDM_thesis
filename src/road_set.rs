use rand::prelude::StdRng;

use crate::{cost::Cost, road::Road, side_policies::SidePolicy};

#[derive(Clone)]
pub struct RoadSet {
    roads: Vec<Road>,
}

impl RoadSet {
    pub fn new(roads: Vec<Road>) -> Self {
        Self { roads }
    }

    pub fn new_samples(road: &Road, rng: &mut StdRng, n: usize) -> Self {
        assert!(n > 0);

        if road.params.true_belief_sample_only {
            return Self {
                roads: vec![road.sim_estimate()],
            };
        }

        let mut roads = Vec::with_capacity(n);
        for _ in 0..n {
            roads.push(road.sample_belief(rng));
        }

        Self { roads }
    }

    pub fn ego_policy(&self) -> &SidePolicy {
        self.roads[0].ego_policy()
    }

    pub fn set_ego_policy(&mut self, policy: &SidePolicy) {
        for road in self.roads.iter_mut() {
            road.set_ego_policy(policy.clone());
        }
    }

    // pub fn is_debug(&self) -> bool {
    //     self.roads[0].debug
    // }

    pub fn timesteps(&self) -> usize {
        self.roads[0].timesteps
    }

    pub fn reset_car_traces(&mut self) {
        for road in self.roads.iter_mut() {
            road.reset_car_traces();
        }
    }

    #[allow(unused)]
    pub fn disable_car_traces(&mut self) {
        for road in self.roads.iter_mut() {
            road.car_traces = None;
        }
    }

    pub fn take_update_steps(&mut self, t: f64, dt: f64) {
        for road in self.roads.iter_mut() {
            road.take_update_steps(t, dt);
        }
    }

    pub fn make_traces(&self, depth_level: u32, include_obstacle_cars: bool) -> Vec<rvx::Shape> {
        let mut traces = Vec::new();
        for road in self.roads.iter() {
            traces.append(&mut road.make_traces(depth_level, include_obstacle_cars));
        }
        traces
    }

    pub fn cost(&self) -> Cost {
        self.roads.iter().map(|r| r.cost).sum::<Cost>() / self.roads.len() as f64
    }

    pub fn into_iter(self) -> impl Iterator<Item = Road> {
        self.roads.into_iter()
    }
}
