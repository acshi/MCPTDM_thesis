use std::{f64::consts::PI, time::Duration};

use forward_policies::{ForwardPolicy, ForwardPolicyTrait};
use intelligent_driver::IntelligentDriverPolicy;
use parry2d_f64::{
    na::{Isometry2, Vector2},
    query::{self, ClosestPoints},
    shape::{Cuboid, Shape},
};
use pure_pursuit::PurePursuitPolicy;
use rand::{thread_rng, Rng};
use rate_timer::RateTimer;
use rvx::{Rvx, RvxColor};
use side_policies::{SidePolicy, SidePolicyTrait};

#[macro_use]
extern crate fstrings;

mod forward_policies;
mod intelligent_driver;
mod pure_pursuit;
mod rate_timer;
mod side_policies;

#[macro_use]
extern crate enum_dispatch;

const LANE_WIDTH: f64 = 3.7;
const ROAD_DASH_LENGTH: f64 = 3.0;
const ROAD_DASH_DIST: f64 = 9.0;
const ROAD_LENGTH: f64 = 500.0;

const N_CARS: usize = 1;

const PRIUS_WIDTH: f64 = 1.76;
const PRIUS_LENGTH: f64 = 4.57;
const PRIUS_MAX_STEER: f64 = 1.11; // from minimum turning radius of 4.34 meters and PRIUS_LENGTH
const MPH_TO_MPS: f64 = 0.44704;
const MPS_TO_MPH: f64 = 2.23694;
const SPEED_DEFAULT: f64 = 45.0 * MPH_TO_MPS;
const SPEED_LOW: f64 = 35.0 * MPH_TO_MPS;
const SPEED_HIGH: f64 = 55.0 * MPH_TO_MPS;
const FOLLOW_DIST_MIN: f64 = 1.0 * PRIUS_LENGTH;
const FOLLOW_TIME_LOW: f64 = 0.4;
const FOLLOW_TIME_HIGH: f64 = 2.0;
const FOLLOW_TIME_DEFAULT: f64 = (FOLLOW_TIME_LOW + FOLLOW_TIME_HIGH) / 2.0;

const PREFERRED_ACCEL_LOW: f64 = 0.2; // semi truck, 2min zero to sixty
const PREFERRED_ACCEL_HIGH: f64 = 11.2; // model s, 2.4s zero to sixty
const PREFERRED_ACCEL_DEFAULT: f64 = 2.0; // 16s zero to sixty, just under max accel for a prius (13s)
const BREAKING_ACCEL: f64 = 12.0;

const PREFERRED_STEER_ACCEL_LOW: f64 = 0.1;
const PREFERRED_STEER_ACCEL_HIGH: f64 = 0.3;
const PREFERRED_STEER_ACCEL_DEFAULT: f64 = 10.0;

const SIDE_MARGIN: f64 = 0.5;

#[derive(Clone, Debug)]
pub struct Car {
    // front-referenced kinematic bicycle model
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub vel: f64,
    pub steer: f64,

    pub width: f64,
    pub length: f64,

    pub preferred_vel: f64,
    pub preferred_accel: f64,
    pub preferred_steer_accel: f64,
    pub follow_min_dist: f64,
    pub follow_time: f64,
    pub crashed: bool,
    pub forward_policy: Option<ForwardPolicy>,
    pub side_policy: Option<SidePolicy>,
}

impl Car {
    fn new(left_side: bool) -> Self {
        let lane_y = LANE_WIDTH / 2.0 * if left_side { 1.0 } else { -1.0 };
        Self {
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
            follow_time: FOLLOW_TIME_DEFAULT,
            crashed: false,
            // policy: Some(Policy::AdapativeCruisePolicy(AdapativeCruisePolicy::new())),
            forward_policy: Some(ForwardPolicy::IntelligentDriverPolicy(
                IntelligentDriverPolicy::new(),
            )),
            side_policy: Some(SidePolicy::PurePursuitPolicy(PurePursuitPolicy::new(
                lane_y, 4.0,
            ))),
        }
    }

    fn random_new() -> Self {
        let left_side: bool = thread_rng().gen();
        let mut car = Self::new(left_side);
        car.preferred_vel = thread_rng().gen_range(SPEED_LOW..SPEED_HIGH);
        car.x = thread_rng().gen_range(0.0..ROAD_LENGTH) - ROAD_LENGTH / 2.0;
        car.preferred_accel = thread_rng().gen_range(PREFERRED_ACCEL_LOW..PREFERRED_ACCEL_HIGH);
        car.preferred_steer_accel =
            thread_rng().gen_range(PREFERRED_STEER_ACCEL_LOW..PREFERRED_STEER_ACCEL_HIGH);
        car.follow_time = thread_rng().gen_range(FOLLOW_TIME_LOW..FOLLOW_TIME_HIGH);

        car
    }

    fn follow_dist(&self) -> f64 {
        self.follow_min_dist + self.follow_time * self.vel
    }

    fn update(&mut self, dt: f64) {
        if !self.crashed {
            let theta = self.theta + self.steer;
            self.x += theta.cos() * self.vel * dt;
            self.y += theta.sin() * self.vel * dt;
            self.theta += self.vel * self.steer.sin() / self.length * dt;
        }
    }

    fn draw(&self, r: &mut Rvx, color: RvxColor) {
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
            Rvx::text(
                &format!(
                    "MPH: {:.1}\nPref MPH: {:.1}\nPref accel: {:.1}\nPref follow time: {:.1}\nPref follow: {:.1}\n{:?}",
                    self.vel * MPS_TO_MPH,
                    self.preferred_vel * MPS_TO_MPH,
                    self.preferred_accel,
                    self.follow_time,
                    self.follow_dist(),
                    self.forward_policy.as_ref().unwrap()
                ),
                "Arial",
                40.0,
            )
            .rot(-PI / 2.0)
            .translate(&[self.x, self.y]),
        );
    }

    fn pose(&self) -> Isometry2<f64> {
        let center_y = self.y + self.length / 2.0 * self.theta.sin();
        let center_x = self.x - self.length / 2.0 * self.theta.cos();

        Isometry2::new(Vector2::new(center_x, center_y), -self.theta)
    }

    fn shape(&self) -> impl Shape {
        Cuboid::new(Vector2::new(self.length / 2.0, self.width / 2.0))
    }
}

#[derive(Clone, Debug)]
pub struct Road {
    cars: Vec<Car>,
}

impl Road {
    fn new() -> Self {
        let mut road = Self {
            cars: vec![Car::new(false)],
        };

        road.cars[0].theta = PI / 16.0;
        road.cars[0].y = -LANE_WIDTH / 2.0 + 0.5;

        for _ in 0..100 {
            if road.cars.len() >= N_CARS {
                break;
            }

            let car = Car::random_new();
            if road.collides_any_car(&car) {
                continue;
            }
            road.cars.push(car);
        }

        road
    }

    fn double_borrow_mut(&mut self, i1: usize, i2: usize) -> (&mut Car, &mut Car) {
        assert_ne!(i1, i2);

        if i1 < i2 {
            let rem = &mut self.cars[i1..];
            let (first, second) = rem.split_at_mut(i2 - i1);
            (&mut first[0], &mut second[0])
        } else {
            let rem = &mut self.cars[i2..];
            let (second, first) = rem.split_at_mut(i1 - i2);
            (&mut first[0], &mut second[0])
        }
    }

    fn collides_any(&self, car_i: usize) -> bool {
        let car = &self.cars[car_i];
        let pose = car.pose();
        let shape = car.shape();
        for (i, c) in self.cars.iter().enumerate() {
            if i == car_i {
                continue;
            }

            if query::intersection_test(&pose, &shape, &c.pose(), &c.shape()).unwrap() {
                return true;
            }
        }
        false
    }

    fn collides_any_car(&self, car: &Car) -> bool {
        let pose = car.pose();
        let shape = car.shape();
        for c in self.cars.iter() {
            if query::intersection_test(&pose, &shape, &c.pose(), &c.shape()).unwrap() {
                return true;
            }
        }
        false
    }

    fn dist_clear_ahead(&self, car_i: usize) -> Option<(f64, usize)> {
        self.dist_clear(car_i, true)
    }

    fn dist_clear_behind(&self, car_i: usize) -> Option<(f64, usize)> {
        self.dist_clear(car_i, false)
    }

    fn dist_clear(&self, car_i: usize, ahead: bool) -> Option<(f64, usize)> {
        let mut min_dist = f64::MAX;
        let mut min_car_i = None;

        let car = &self.cars[car_i];
        let pose = self.cars[car_i].pose();
        let shape = self.cars[car_i].shape();
        for (i, c) in self.cars.iter().enumerate() {
            // skip cars behind this one
            if i == car_i || ahead && c.x < car.x || !ahead && c.x > car.x {
                continue;
            }

            match query::closest_points(
                &pose,
                &shape,
                &c.pose(),
                &c.shape(),
                4.0 * car.follow_dist(),
            ) {
                Ok(ClosestPoints::WithinMargin(a, b)) => {
                    let forward_dist = (a[0] - b[0]).abs();
                    let side_dist = (a[1] - b[1]).abs();
                    if side_dist > SIDE_MARGIN {
                        continue;
                    }

                    if forward_dist < min_dist {
                        min_dist = forward_dist;
                        min_car_i = Some(i);
                    }
                }
                _ => (),
            }

            // let d = query::distance(&pose, &shape, &c.pose(), &c.shape()).unwrap();
            // min_dist = min_dist.min(d);
        }

        Some((min_dist, min_car_i?))
    }

    fn update(&mut self, dt: f64) {
        for car_i in 0..self.cars.len() {
            if !self.cars[car_i].crashed {
                // forward policy
                let mut policy = self.cars[car_i].forward_policy.take().unwrap();
                let mut accel = policy.choose_accel(self, car_i);

                let car = &mut self.cars[car_i];
                accel = accel.max(-BREAKING_ACCEL).min(car.preferred_vel);
                car.vel = (car.vel + accel * dt).max(0.0).min(car.preferred_vel);
                self.cars[car_i].forward_policy = Some(policy);

                // side policy
                let mut policy = self.cars[car_i].side_policy.take().unwrap();
                let target_steer = policy.choose_steer(self, car_i);

                let car = &mut self.cars[car_i];
                let target_steer_accel = (target_steer - car.steer) / dt;
                let steer_accel = target_steer_accel
                    .max(-car.preferred_steer_accel)
                    .min(car.preferred_steer_accel);

                car.steer = (car.steer + steer_accel * dt)
                    .max(-PRIUS_MAX_STEER)
                    .min(PRIUS_MAX_STEER);
                self.cars[car_i].side_policy = Some(policy);
            }
        }

        for car in self.cars.iter_mut() {
            if !car.crashed {
                car.update(dt);
            }
        }

        for car_i in 0..self.cars.len() {
            if !self.cars[car_i].crashed {
                if self.collides_any(car_i) {
                    self.cars[car_i].crashed = true;
                }
            }
        }
    }

    fn draw(&self, r: &mut Rvx) {
        // draw a 'road'
        r.draw(
            Rvx::square()
                .scale_xy(&[ROAD_LENGTH, LANE_WIDTH * 2.0])
                .color(RvxColor::GRAY),
        );
        r.draw(
            Rvx::square()
                .scale_xy(&[ROAD_LENGTH, 0.2])
                .translate(&[0.0, -LANE_WIDTH])
                .color(RvxColor::WHITE),
        );
        r.draw(
            Rvx::square()
                .scale_xy(&[ROAD_LENGTH, 0.2])
                .translate(&[0.0, LANE_WIDTH])
                .color(RvxColor::WHITE),
        );

        // adjust for ego car
        r.set_translate_modifier(-self.cars[0].x, 0.0);

        // draw the dashes in the middle
        let dash_interval = ROAD_DASH_LENGTH + ROAD_DASH_DIST;
        let dash_offset = (self.cars[0].x / dash_interval).round() * dash_interval;
        for dash_i in -5..=5 {
            r.draw(
                Rvx::square()
                    .scale_xy(&[ROAD_DASH_LENGTH, 0.2])
                    .translate(&[dash_i as f64 * dash_interval + dash_offset, 0.0])
                    .color(RvxColor::WHITE),
            );
        }

        // draw the cars
        for (i, car) in self.cars.iter().enumerate() {
            if i == 0 {
                car.draw(r, RvxColor::GREEN.set_a(0.6));
            } else if car.crashed {
                car.draw(r, RvxColor::RED.set_a(0.6));
            } else if car.vel == 0.0 {
                car.draw(r, RvxColor::WHITE.set_a(0.6));
            } else {
                car.draw(r, RvxColor::BLUE.set_a(0.6));
            }
        }
    }
}

struct State {
    road: Road,
    r: Rvx,
}

impl State {
    fn update_graphics(&mut self) {
        let r = &mut self.r;
        r.clear();

        self.road.draw(r);

        r.set_global_rot(-PI / 2.0);
        r.commit_changes();
    }

    fn update(&mut self, dt: f64) {
        self.road.update(dt);
    }
}

fn main() {
    let mut state = State {
        road: Road::new(),
        r: Rvx::new("Self-Driving!", "0.0.0.0", 8000),
    };

    state.update_graphics();
    state.r.set_user_zoom(Some(0.4)); // 0.22
    std::thread::sleep(Duration::from_millis(500));
    state.r.set_user_zoom(None);

    let dt = 0.02;
    let mut rate = RateTimer::new(Duration::from_millis((dt * 1000.0) as u64));

    for i in 0..10000 {
        state.update(dt);
        state.update_graphics();
        rate.wait_until_ready();

        if i == 200 {
            for side_policy in state.road.cars[0].side_policy.iter_mut() {
                *side_policy =
                    SidePolicy::PurePursuitPolicy(PurePursuitPolicy::new(LANE_WIDTH / 2.0, 1.0));
            }
        }
    }

    std::thread::sleep(Duration::from_millis(1000));
}
