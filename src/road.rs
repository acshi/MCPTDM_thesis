use std::{cell::RefCell, f64::consts::PI, rc::Rc};

use itertools::Itertools;
use parry2d_f64::{
    math::Isometry,
    query::{self, ClosestPoints},
};
use rand::prelude::StdRng;
use rvx::{Rvx, RvxColor};

use crate::{car::PRIUS_WIDTH, side_control::SideControlTrait};
use crate::{
    car::{MPH_TO_MPS, PRIUS_MAX_STEER},
    forward_control::ForwardControlTrait,
};

use crate::side_policies::SidePolicyTrait;

use crate::car::{Car, BREAKING_ACCEL};

pub const LANE_WIDTH: f64 = 3.7;
pub const ROAD_DASH_LENGTH: f64 = 3.0;
pub const ROAD_DASH_DIST: f64 = 9.0;
pub const ROAD_LENGTH: f64 = 500.0;

pub const SIDE_MARGIN: f64 = 0.0;

const EFFICIENCY_WEIGHT: f64 = 1.0;
const SAFETY_WEIGHT: f64 = 40.0;
const SAFETY_MARGIN: f64 = LANE_WIDTH - PRIUS_WIDTH - 1.5;
const SMOOTHNESS_WEIGHT: f64 = 1.0;

#[derive(Clone, Debug)]
pub struct Road {
    pub t: f64, // current time in seconds
    pub cars: Vec<Car>,
    pub last_ego_policy_id: Option<u64>,
    pub reward: f64,
    pub debug: bool,
}

impl Road {
    pub fn new() -> Self {
        let mut road = Self {
            t: 0.0,
            cars: vec![Car::new(0, 0)],
            last_ego_policy_id: None,
            reward: 0.0,
            debug: true,
        };

        road.cars[0].preferred_vel = 90.0 * MPH_TO_MPS;
        road.cars[0].theta = PI / 16.0;
        road.cars[0].y = -LANE_WIDTH / 2.0 + 0.5;

        road
    }

    // fn double_borrow_mut(&mut self, i1: usize, i2: usize) -> (&mut Car, &mut Car) {
    //     assert_ne!(i1, i2);
    //     if i1 < i2 {
    //         let rem = &mut self.cars[i1..];
    //         let (first, second) = rem.split_at_mut(i2 - i1);
    //         (&mut first[0], &mut second[0])
    //     } else {
    //         let rem = &mut self.cars[i2..];
    //         let (second, first) = rem.split_at_mut(i1 - i2);
    //         (&mut first[0], &mut second[0])
    //     }
    // }

    pub fn add_random_car(&mut self, rng: &Rc<RefCell<StdRng>>) {
        for _ in 0..100 {
            let car = Car::random_new(self.cars.len(), rng);
            if self.collides_any_car(&car) {
                continue;
            }
            self.cars.push(car);
            return;
        }
        panic!();
    }

    pub fn add_obstacle(&mut self, x: f64, lane_i: i32) {
        let mut car = Car::new(self.cars.len(), lane_i);
        car.x = x;
        car.y += LANE_WIDTH / 2.0;
        car.theta = PI / 2.0;
        car.vel = 0.0;
        car.preferred_vel = 0.0;
        car.crashed = true;

        self.cars.push(car);
    }

    pub fn sim_estimate(&self) -> Self {
        let mut road = self.clone();
        road.cars = self.cars.iter().map(|c| c.sim_estimate()).collect();
        road.debug = false;
        road
    }

    pub fn collides_between(&self, car_i1: usize, car_i2: usize) -> bool {
        assert_ne!(car_i1, car_i2);

        let car_a = &self.cars[car_i1];
        let car_b = &self.cars[car_i2];

        if (car_a.x - car_b.x).abs() > (car_a.length + car_b.length) / 2.0 {
            return false;
        }

        parry2d_f64::query::intersection_test(
            &car_a.pose(),
            &car_a.shape(),
            &car_b.pose(),
            &car_b.shape(),
        )
        .unwrap()
    }

    #[allow(unused)]
    pub fn collides_any(&self, car_i: usize) -> bool {
        let car = &self.cars[car_i];
        let pose = car.pose();
        let shape = car.shape();
        for (i, c) in self.cars.iter().enumerate() {
            if i == car_i {
                continue;
            }

            if parry2d_f64::query::intersection_test(&pose, &shape, &c.pose(), &c.shape()).unwrap()
            {
                return true;
            }
        }
        false
    }

    pub fn collides_any_car(&self, car: &Car) -> bool {
        let pose = car.pose();
        let shape = car.shape();
        for c in self.cars.iter() {
            if parry2d_f64::query::intersection_test(&pose, &shape, &c.pose(), &c.shape()).unwrap()
            {
                return true;
            }
        }
        false
    }

    pub fn dist_clear_ahead(&self, car_i: usize) -> Option<(f64, usize)> {
        self.dist_clear(car_i, true)
    }

    // fn dist_clear_behind(&self, car_i: usize) -> Option<(f64, usize)> {
    //     self.dist_clear(car_i, false)
    // }

    pub fn dist_clear(&self, car_i: usize, ahead: bool) -> Option<(f64, usize)> {
        let car = &self.cars[car_i];

        let mut min_dist = f64::MAX;
        let mut min_car_i = None;

        let mut dist_thresh_sq = (car.follow_dist() * 4.0).powi(2);

        let pose = self.cars[car_i].pose();
        let shape = self.cars[car_i].shape();
        for (i, c) in self.cars.iter().enumerate() {
            if i == car_i {
                continue;
            }
            // skip cars behind this one
            if ahead && c.x < car.x || !ahead && c.x > car.x {
                // if i == 0 && car_i == 14 {
                //     eprintln_f!("Skipping {car_i} to ego: c.x {:.2} car.x {:.2}", c.x, car.x);
                // }
                continue;
            }

            let dist_sq = (c.x - car.x).powi(2) + (c.y - car.y).powi(2);
            if dist_sq >= dist_thresh_sq {
                continue;
            }

            match query::closest_points(&pose, &shape, &c.pose(), &c.shape(), f64::MAX) {
                Ok(ClosestPoints::WithinMargin(a, b)) => {
                    let forward_dist = (a[0] - b[0]).abs();

                    // to determine side-distance we have to be more clever, since the closest points
                    // do not necessarily give the closest side distance. So for that, we do another query.
                    let side_dist;
                    match query::closest_points(
                        &(Isometry::translation(b[0] - a[0] + car.length / 2.0, 0.0) * pose),
                        &shape,
                        &c.pose(),
                        &c.shape(),
                        SIDE_MARGIN,
                    ) {
                        Ok(ClosestPoints::WithinMargin(a, b)) => {
                            side_dist = (a[1] - b[1]).abs();
                            // if i == 0 && car_i == 14 {
                            //     eprintln_f!(
                            //         "From {car_i} to ego: {side_dist=:.2}, {a=:.2?}, {b=:.2?}",
                            //         a = a.coords.as_slice(),
                            //         b = b.coords.as_slice(),
                            //     );
                            // }
                        }
                        Ok(ClosestPoints::Intersecting) => side_dist = 0.0,
                        _ => side_dist = SIDE_MARGIN + 0.01,
                    }

                    // let side_dist = (a[1] - b[1]).abs();
                    // if i == 0 && car_i == 14 {
                    //     eprintln_f!("From {car_i} to ego: {forward_dist=:.2}, {side_dist=:.2}, {a=:.2?}, {b=:.2?}", a = a.coords.as_slice(), b = b.coords.as_slice());
                    // }
                    if side_dist > SIDE_MARGIN {
                        continue;
                    }

                    if forward_dist < min_dist {
                        min_dist = forward_dist;
                        min_car_i = Some(i);

                        dist_thresh_sq = min_dist.powi(2) + (SIDE_MARGIN + car.width / 2.0).powi(2);
                    }
                }
                Ok(ClosestPoints::Intersecting) => {
                    let dist = 0.0;
                    if dist < min_dist {
                        min_dist = dist;
                        min_car_i = Some(i);
                    }
                    // if i == 0 && car_i == 14 {
                    //     eprintln_f!("From {car_i} to ego: intersecting!");
                    // }
                }
                _ => (),
            }

            // let d = query::distance(&pose, &shape, &c.pose(), &c.shape()).unwrap();
            // min_dist = min_dist.min(d);
        }

        Some((min_dist, min_car_i?))
    }

    fn min_unsafe_dist(&self, car_i: usize) -> Option<f64> {
        let mut min_dist = None;

        let pose = self.cars[car_i].pose();
        let shape = self.cars[car_i].shape();
        for (i, c) in self.cars.iter().enumerate() {
            if i == car_i {
                continue;
            }

            match query::closest_points(&pose, &shape, &c.pose(), &c.shape(), SAFETY_MARGIN) {
                Ok(ClosestPoints::WithinMargin(a, b)) => {
                    let dist = (a - b).magnitude();
                    if dist < min_dist.unwrap_or(f64::MAX) {
                        min_dist = Some(dist);
                    }
                }
                Ok(ClosestPoints::Intersecting) => {
                    min_dist = Some(0.0);
                }
                _ => (),
            }

            // let d = query::distance(&pose, &shape, &c.pose(), &c.shape()).unwrap();
            // min_dist = min_dist.min(d);
        }

        min_dist
    }

    pub fn update(&mut self, dt: f64) {
        for car_i in 0..self.cars.len() {
            if true || !self.cars[car_i].crashed {
                // policy
                let trajectory;
                {
                    let mut policy = self.cars[car_i].side_policy.take().unwrap();
                    self.cars[car_i].target_follow_time = policy.choose_follow_time(self, car_i);
                    trajectory = policy.choose_trajectory(self, car_i);
                    self.cars[car_i].side_policy = Some(policy);
                }

                // forward control
                {
                    let mut control = self.cars[car_i].forward_control.take().unwrap();
                    let mut accel = control.choose_accel(self, car_i);

                    let car = &mut self.cars[car_i];
                    accel = accel.max(-BREAKING_ACCEL).min(car.preferred_vel);
                    car.vel = (car.vel + accel * dt).max(0.0).min(car.preferred_vel);
                    self.cars[car_i].forward_control = Some(control);
                }

                // side control
                {
                    let mut control = self.cars[car_i].side_control.take().unwrap();
                    let target_steer = control.choose_steer(self, car_i, &trajectory);

                    let car = &mut self.cars[car_i];
                    let target_steer_accel = (target_steer - car.steer) / dt;
                    let steer_accel = target_steer_accel
                        .max(-car.preferred_steer_accel)
                        .min(car.preferred_steer_accel);

                    car.steer = (car.steer + steer_accel * dt)
                        .max(-PRIUS_MAX_STEER)
                        .min(PRIUS_MAX_STEER);
                    self.cars[car_i].side_control = Some(control);
                }
            }
        }

        for car in self.cars.iter_mut() {
            if !car.crashed {
                car.update(dt);
            }
        }

        // for car_i in 0..self.cars.len() {
        //     if !self.cars[car_i].crashed {
        //         if self.collides_any(car_i) {
        //             self.cars[car_i].crashed = true;
        //         }
        //     }
        // }

        for (i1, i2) in (0..self.cars.len()).tuple_combinations() {
            if self.collides_between(i1, i2) {
                self.cars[i1].crashed = true;
                self.cars[i2].crashed = true;
            }
        }

        self.t += dt;

        self.update_reward(dt);
    }

    fn update_reward(&mut self, dt: f64) {
        let ecar = &self.cars[0];
        self.reward += EFFICIENCY_WEIGHT * (ecar.preferred_vel - ecar.vel).abs() * dt;

        let min_dist = self.min_unsafe_dist(0);
        if min_dist.is_some() {
            self.reward += SAFETY_WEIGHT * dt;
            // eprintln!("UNSAFE: {:.2}", min_dist.unwrap());
        }

        let policy_id = ecar.side_policy.as_ref().unwrap().policy_id();
        if let Some(last_policy_id) = self.last_ego_policy_id {
            if policy_id != last_policy_id {
                self.reward += SMOOTHNESS_WEIGHT;
                if self.debug {
                    eprintln_f!("policy change from {last_policy_id} to {policy_id}");
                }
            }
        }
        self.last_ego_policy_id = Some(policy_id);
    }

    pub fn final_reward(&self) -> f64 {
        self.reward / self.t
    }

    pub fn draw(&self, r: &mut Rvx) {
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
