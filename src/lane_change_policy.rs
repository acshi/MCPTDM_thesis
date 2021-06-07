use parry2d_f64::{math::Point, na::Point2};

use crate::{
    car::PRIUS_LENGTH,
    side_policies::{SidePolicy, SidePolicyTrait},
    LANE_WIDTH,
};

const TRANSITION_DIST_MIN: f64 = 1.0 * PRIUS_LENGTH;
const TRANSITION_DIST_MAX: f64 = 100.0 * PRIUS_LENGTH;

#[derive(Debug, Clone)]
pub struct LaneChangePolicy {
    target_lane_i: i32,
    transition_time: f64,
    follow_time: Option<f64>,
    start_xy: Option<(f64, f64)>,
}

impl LaneChangePolicy {
    pub fn new(target_lane_i: i32, transition_time: f64, follow_time: Option<f64>) -> Self {
        Self {
            target_lane_i,
            transition_time,
            follow_time,
            start_xy: None,
        }
    }
}

impl SidePolicyTrait for LaneChangePolicy {
    fn choose_follow_time(&mut self, road: &crate::Road, car_i: usize) -> f64 {
        self.follow_time
            .unwrap_or(road.cars[car_i].preferred_follow_time)
    }

    fn choose_trajectory(&mut self, road: &crate::Road, car_i: usize) -> Vec<Point2<f64>> {
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

        let target_y = (self.target_lane_i as f64 - 0.5) * LANE_WIDTH;
        let target_x = start_x + transition_dist;
        // let progress = (road.t - start_time) / self.transition_time;

        vec![
            Point::new(start_x, start_y),
            Point::new(target_x, target_y),
            Point::new(target_x + 100.0, target_y), // then continue straight
        ]
    }

    fn policy_id(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.target_lane_i.hash(&mut hasher);
        self.transition_time.to_bits().hash(&mut hasher);
        if let Some(follow_time) = self.follow_time {
            follow_time.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    fn operating_policy(&self) -> SidePolicy {
        SidePolicy::LaneChangePolicy(self.clone())
    }
}
