use std::f64::consts::PI;

use crate::{side_policies::SidePolicyTrait, Road, PRIUS_LENGTH};

const AHEAD_DIST_MIN: f64 = 0.2 * PRIUS_LENGTH;
const AHEAD_DIST_MAX: f64 = 10.0 * PRIUS_LENGTH;

#[derive(Debug, Clone)]
pub struct PurePursuitPolicy {
    target_y: f64, // lateral target position
    ahead_time: f64,
}

impl PurePursuitPolicy {
    pub fn new(target_y: f64, ahead_time: f64) -> Self {
        Self {
            target_y,
            ahead_time,
        }
    }
}

// https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/PurePursuit.html
impl SidePolicyTrait for PurePursuitPolicy {
    fn choose_steer(&mut self, road: &Road, car_i: usize) -> f64 {
        // if car_i == 0 {
        //     return PI / 4.0;
        // }

        let car = &road.cars[car_i];

        let target_ahead_dist = (self.ahead_time * car.vel)
            .min(AHEAD_DIST_MAX)
            .max(AHEAD_DIST_MIN);

        if car_i == 0 {
            eprintln_f!("{target_ahead_dist=:.2}");
        }

        let car_rear_y = car.y - car.length * car.theta.sin();

        let car_to_target_x = target_ahead_dist;
        let car_to_target_y = self.target_y - car.y;
        let ahead_dist = car_to_target_x.hypot(car_to_target_y);
        let angle_to_target = car_to_target_y.atan2(car_to_target_x);

        let target_steer = (2.0 * car.length * angle_to_target.sin() / ahead_dist).atan();
        target_steer
    }
}
