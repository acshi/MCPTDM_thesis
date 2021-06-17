use nalgebra::point;
use parry2d_f64::na::Point2;

use crate::{
    car::{PREFERRED_VEL_ESTIMATE_MIN, PRIUS_LENGTH},
    side_policies::{SidePolicy, SidePolicyTrait},
    Road,
};

const TRANSITION_DIST_MIN: f64 = 1.0 * PRIUS_LENGTH;
const TRANSITION_DIST_MAX: f64 = 100.0 * PRIUS_LENGTH;

#[derive(Clone, Copy, Debug)]
pub enum LongitudinalPolicy {
    Maintain,
    Accelerate,
    Decelerate,
}

#[derive(Clone)]
pub struct LaneChangePolicy {
    policy_id: u32,
    target_lane_i: i32,
    transition_time: f64,
    wait_for_clear: bool,
    long_policy: LongitudinalPolicy,
    start_vel: Option<f64>,
    start_lane_i: Option<i32>,
    waiting_done: bool,
    start_xy: Option<(f64, f64)>,
}

impl std::fmt::Debug for LaneChangePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        let policy_str = format_f!("{s.long_policy:?}");
        write_f!(f, "lane {s.target_lane_i}, {policy_str:10}")
    }
}

impl LaneChangePolicy {
    pub fn new(
        policy_id: u32,
        target_lane_i: i32,
        transition_time: f64,
        wait_for_clear: bool,
        long_policy: LongitudinalPolicy,
    ) -> Self {
        Self {
            policy_id,
            target_lane_i,
            transition_time,
            wait_for_clear,
            long_policy,
            start_vel: None,
            start_lane_i: None,
            waiting_done: false,
            start_xy: None,
        }
    }

    fn lane_change_trajectory(&mut self, road: &Road, car_i: usize) -> Vec<Point2<f64>> {
        let car = &road.cars[car_i];
        let (mut start_x, mut start_y) = *self.start_xy.get_or_insert((car.x, car.y));

        // reset start_xy after long enough
        // so that the trajectory doesn't run out
        if (start_x - car.x).abs() > 50.0 {
            start_x = car.x;
            start_y = car.y;
            self.start_xy = Some((car.x, car.y));
        }

        let transition_dist = (self.transition_time * car.vel)
            .max(TRANSITION_DIST_MIN)
            .min(TRANSITION_DIST_MAX);

        let target_y = Road::get_lane_y(self.target_lane_i);
        let target_x = start_x + transition_dist;
        // let progress = (road.t - start_time) / self.transition_time;

        vec![
            point!(start_x, start_y),
            point!(target_x, target_y),
            point!(target_x + 100.0, target_y), // then continue straight
        ]
    }

    fn lane_keep_trajectory(&mut self, road: &Road, car_i: usize) -> Vec<Point2<f64>> {
        let car = &road.cars[car_i];
        let start_lane_i = *self.start_lane_i.get_or_insert_with(|| car.current_lane());

        let transition_dist = (self.transition_time * car.vel)
            .max(TRANSITION_DIST_MIN)
            .min(TRANSITION_DIST_MAX);

        vec![
            point!(car.x, car.y),
            point!(car.x + transition_dist, Road::get_lane_y(start_lane_i)),
            point!(car.x + 100.0, Road::get_lane_y(start_lane_i)),
        ]
    }
}

impl SidePolicyTrait for LaneChangePolicy {
    fn choose_target_lane(&mut self, road: &Road, car_i: usize) -> i32 {
        if self.wait_for_clear && !self.waiting_done {
            return self
                .start_lane_i
                .unwrap_or_else(|| road.cars[car_i].current_lane());
        }
        self.target_lane_i
    }

    fn choose_follow_time(&mut self, _road: &Road, _car_i: usize) -> f64 {
        match self.long_policy {
            LongitudinalPolicy::Maintain => 0.6,
            LongitudinalPolicy::Accelerate => 0.2,
            LongitudinalPolicy::Decelerate => 1.0,
        }
    }

    fn choose_vel(&mut self, road: &Road, car_i: usize) -> f64 {
        let car = &road.cars[car_i];
        let target_vel = match self.long_policy {
            LongitudinalPolicy::Maintain => *self.start_vel.get_or_insert(car.vel),
            LongitudinalPolicy::Accelerate => (car.vel + 10.0).max(PREFERRED_VEL_ESTIMATE_MIN),
            LongitudinalPolicy::Decelerate => (car.vel - 10.0).max(0.0),
        };

        target_vel
    }

    fn choose_trajectory(&mut self, road: &Road, car_i: usize) -> Vec<Point2<f64>> {
        if self.wait_for_clear && !self.waiting_done {
            let car = &road.cars[car_i];
            self.waiting_done = road.lane_definitely_clear_between(
                car_i,
                self.target_lane_i,
                car.x - car.length * 2.0,
                car.x + car.length,
            );
        }
        if self.waiting_done || !self.wait_for_clear {
            self.lane_change_trajectory(road, car_i)
        } else {
            self.lane_keep_trajectory(road, car_i)
        }
    }

    fn policy_id(&self) -> u32 {
        self.policy_id
        // use std::hash::{Hash, Hasher};
        // let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // self.target_lane_i.hash(&mut hasher);
        // self.transition_time.to_bits().hash(&mut hasher);
        // if let Some(follow_time) = self.follow_time {
        //     follow_time.to_bits().hash(&mut hasher);
        // }
        // hasher.finish()
    }

    fn operating_policy(&self) -> SidePolicy {
        SidePolicy::LaneChangePolicy(self.clone())
    }
}
