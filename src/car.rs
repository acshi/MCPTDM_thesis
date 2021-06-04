use std::{cell::RefCell, f64::consts::PI, rc::Rc};

use parry2d_f64::{
    na::{Isometry2, Vector2},
    shape::{Cuboid, Shape},
};
use rand::prelude::{Rng, StdRng};
use rvx::{Rvx, RvxColor};

use crate::{
    forward_control::ForwardControl,
    intelligent_driver::IntelligentDriverPolicy,
    lane_change_policy::LaneChangePolicy,
    pure_pursuit::PurePursuitPolicy,
    road::{LANE_WIDTH, ROAD_LENGTH},
    side_control::{SideControl, SideControlTrait},
    side_policies::SidePolicy,
    AHEAD_TIME_DEFAULT, LANE_CHANGE_TIME,
};

pub const PRIUS_WIDTH: f64 = 1.76;
pub const PRIUS_LENGTH: f64 = 4.57;
pub const PRIUS_MAX_STEER: f64 = 1.11; // from minimum turning radius of 4.34 meters and PRIUS_LENGTH
pub const MPH_TO_MPS: f64 = 0.44704;
pub const MPS_TO_MPH: f64 = 2.23694;
pub const SPEED_DEFAULT: f64 = 45.0 * MPH_TO_MPS;
pub const SPEED_LOW: f64 = 35.0 * MPH_TO_MPS;
pub const SPEED_HIGH: f64 = 55.0 * MPH_TO_MPS;
pub const FOLLOW_DIST_MIN: f64 = 0.5 * PRIUS_LENGTH;
pub const FOLLOW_TIME_LOW: f64 = 0.4;
pub const FOLLOW_TIME_HIGH: f64 = 2.0;
pub const FOLLOW_TIME_DEFAULT: f64 = (FOLLOW_TIME_LOW + FOLLOW_TIME_HIGH) / 2.0;

pub const PREFERRED_ACCEL_LOW: f64 = 0.2; // semi truck, 2min zero to sixty
pub const PREFERRED_ACCEL_HIGH: f64 = 11.2; // model s, 2.4s zero to sixty
pub const PREFERRED_ACCEL_DEFAULT: f64 = 2.0; // 16s zero to sixty, just under max accel for a prius (13s)
pub const BREAKING_ACCEL: f64 = 12.0;

// maybe needs to be a function of velocity too???
// also not really tuned/reasonably chosen, ahahah
pub const PREFERRED_STEER_ACCEL_LOW: f64 = 19.9;
pub const PREFERRED_STEER_ACCEL_HIGH: f64 = 20.1;
pub const PREFERRED_STEER_ACCEL_DEFAULT: f64 = 20.0;

#[derive(Clone, Debug)]
pub struct Car {
    pub car_i: usize,
    pub crashed: bool,

    // front-referenced kinematic bicycle model
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub vel: f64,
    pub steer: f64,

    pub width: f64,
    pub length: f64,

    // "attitude" properties/constants
    pub preferred_vel: f64,
    pub preferred_accel: f64,
    pub preferred_steer_accel: f64,
    pub follow_min_dist: f64,
    pub preferred_follow_time: f64,

    // current properties/goals
    pub target_follow_time: f64,

    pub forward_control: Option<ForwardControl>,
    pub side_control: Option<SideControl>,
    pub side_policy: Option<SidePolicy>,
}

impl Car {
    pub fn new(car_i: usize, lane_i: i32) -> Self {
        let lane_y = (lane_i as f64 - 0.5) * LANE_WIDTH;
        Self {
            car_i,
            crashed: false,

            x: 0.0,
            y: lane_y,
            theta: 0.0,
            vel: 0.0,
            steer: 0.0,

            width: PRIUS_WIDTH,
            length: PRIUS_LENGTH,

            preferred_vel: SPEED_DEFAULT,
            preferred_accel: PREFERRED_ACCEL_DEFAULT,
            preferred_steer_accel: PREFERRED_STEER_ACCEL_DEFAULT,
            follow_min_dist: FOLLOW_DIST_MIN,
            preferred_follow_time: FOLLOW_TIME_DEFAULT,

            target_follow_time: FOLLOW_TIME_DEFAULT,

            // policy: Some(Policy::AdapativeCruisePolicy(AdapativeCruisePolicy::new())),
            forward_control: Some(ForwardControl::IntelligentDriverPolicy(
                IntelligentDriverPolicy::new(),
            )),
            side_control: Some(SideControl::PurePursuitPolicy(PurePursuitPolicy::new(
                AHEAD_TIME_DEFAULT,
            ))),
            side_policy: Some(SidePolicy::LaneChangePolicy(LaneChangePolicy::new(
                lane_i,
                LANE_CHANGE_TIME,
                None,
            ))),
        }
    }

    pub fn random_new(car_i: usize, rng: &Rc<RefCell<StdRng>>) -> Self {
        let mut rng = rng.borrow_mut();

        let lane_i = rng.gen_range(0..=1);
        let mut car = Self::new(car_i, lane_i);
        car.preferred_vel = rng.gen_range(SPEED_LOW..SPEED_HIGH);
        car.x = rng.gen_range(0.0..ROAD_LENGTH) - ROAD_LENGTH / 2.0;
        car.preferred_accel = rng.gen_range(PREFERRED_ACCEL_LOW..PREFERRED_ACCEL_HIGH);
        car.preferred_steer_accel =
            rng.gen_range(PREFERRED_STEER_ACCEL_LOW..PREFERRED_STEER_ACCEL_HIGH);
        car.preferred_follow_time = rng.gen_range(FOLLOW_TIME_LOW..FOLLOW_TIME_HIGH);

        car
    }

    pub fn sim_estimate(&self) -> Self {
        let mut sim_car = self.clone();

        sim_car.preferred_vel = self.vel;
        sim_car.preferred_accel = PREFERRED_ACCEL_DEFAULT;
        sim_car.preferred_steer_accel = PREFERRED_STEER_ACCEL_DEFAULT;
        sim_car.follow_min_dist = FOLLOW_DIST_MIN;
        sim_car.preferred_follow_time = FOLLOW_TIME_DEFAULT;

        sim_car
    }

    pub fn follow_dist(&self) -> f64 {
        self.follow_min_dist + self.target_follow_time * self.vel
    }

    pub fn update(&mut self, dt: f64) {
        if !self.crashed {
            let theta = self.theta + self.steer;
            self.x += theta.cos() * self.vel * dt;
            self.y += theta.sin() * self.vel * dt;
            self.theta += self.vel * self.steer.sin() / self.length * dt;
        }
    }

    pub fn draw(&self, r: &mut Rvx, color: RvxColor) {
        // front dot
        r.draw(
            Rvx::circle()
                .scale(0.5)
                .translate(&[self.x, self.y])
                .color(RvxColor::WHITE.set_a(0.5)),
        );

        // back dot
        r.draw(
            Rvx::circle()
                .scale(0.5)
                .translate(&[
                    self.x - self.length * self.theta.cos(),
                    self.y - self.length * self.theta.sin(),
                ])
                .color(RvxColor::YELLOW.set_a(0.5)),
        );

        // front wheel
        r.draw(
            Rvx::square()
                .scale_xy(&[1.0, 0.5])
                .rot(self.theta + self.steer)
                .translate(&[self.x, self.y])
                .color(RvxColor::BLACK.set_a(0.9)),
        );

        // back wheel
        r.draw(
            Rvx::square()
                .scale_xy(&[1.0, 0.5])
                .rot(self.theta)
                .translate(&[
                    self.x - self.length * self.theta.cos(),
                    self.y - self.length * self.theta.sin(),
                ])
                .color(RvxColor::BLACK.set_a(0.9)),
        );

        let center_x = self.x - self.length / 2.0 * self.theta.cos();
        let center_y = self.y - self.length / 2.0 * self.theta.sin();

        r.draw(
            Rvx::square()
                .scale_xy(&[self.length, self.width])
                .rot(self.theta)
                .translate(&[center_x, center_y])
                .color(color),
        );

        r.draw(
            Rvx::text(&format!("{:.1}", self.car_i,), "Arial", 60.0)
                .rot(-PI / 2.0)
                .translate(&[self.x - self.length / 2.0, self.y + self.width / 2.0])
                .color(RvxColor::BLACK),
        );

        if false {
            r.draw(
            Rvx::text(
                &format!(
                    "MPH: {:.1}\nPref MPH: {:.1}\nLane: {}, y: {:.2}\nFollow time: {:.1}\nPref accel: {:.1}\nPref follow time: {:.1}\nPref follow: {:.1}",
                    self.vel * MPS_TO_MPH,
                    self.preferred_vel * MPS_TO_MPH,
                    self.current_lane(),
                    self.y,
                    self.target_follow_time,
                    self.preferred_accel,
                    self.preferred_follow_time,
                    self.follow_dist(),
                ),
                "Arial",
                40.0,
            )
            .rot(-PI / 2.0)
            .translate(&[self.x, self.y]),
        );
        } else {
            r.draw(
                Rvx::text(
                    &format!(
                        "MPH: {:.1}\nFollow time: {:.1}",
                        self.vel * MPS_TO_MPH,
                        self.target_follow_time,
                    ),
                    "Arial",
                    40.0,
                )
                .rot(-PI / 2.0)
                .translate(&[self.x, self.y]),
            );
        }

        if self.car_i == 0 {
            self.side_control.iter().for_each(|a| a.draw(r));
        }
    }

    pub fn current_lane(&self) -> i32 {
        // let lane_y = (lane_i as f64 - 0.5) * LANE_WIDTH;
        (self.y / LANE_WIDTH + 0.5).round() as i32
    }

    pub fn pose(&self) -> Isometry2<f64> {
        let center_x = self.x - self.length / 2.0 * self.theta.cos();
        let center_y = self.y - self.length / 2.0 * self.theta.sin();

        Isometry2::new(Vector2::new(center_x, center_y), self.theta)
    }

    pub fn shape(&self) -> impl Shape {
        Cuboid::new(Vector2::new(self.length / 2.0, self.width / 2.0))
    }
}
