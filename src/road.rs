use std::{cell::RefCell, f64::consts::PI, rc::Rc};

use itertools::Itertools;
use parry2d_f64::{
    math::Isometry,
    query::{self, ClosestPoints},
    shape::Shape,
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
const SAFETY_WEIGHT: f64 = 1000.0;
const SAFETY_MARGIN: f64 = LANE_WIDTH - PRIUS_WIDTH - 1.5;
const SMOOTHNESS_WEIGHT: f64 = 2.0;

#[derive(Clone, Debug)]
pub struct Road {
    pub t: f64,           // current time in seconds
    pub timesteps: usize, // current time in timesteps (related by DT)
    pub cars: Vec<Car>,
    pub last_ego_policy_id: Option<u64>,
    pub reward: f64,
    pub debug: bool,
}

fn range_dist(low_a: f64, high_a: f64, low_b: f64, high_b: f64) -> f64 {
    let sep1 = (low_a - high_b).max(0.0);
    let sep2 = (low_b - high_a).max(0.0);
    let sep = sep1.max(sep2);
    sep
}

impl Road {
    pub fn new() -> Self {
        let mut road = Self {
            t: 0.0,
            timesteps: 0,
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
        self.dist_clear::<true>(car_i)
    }

    // fn dist_clear_behind(&self, car_i: usize) -> Option<(f64, usize)> {
    //     self.dist_clear(car_i, false)
    // }

    pub fn dist_clear<const AHEAD: bool>(&self, car_i: usize) -> Option<(f64, usize)> {
        let car = &self.cars[car_i];

        let mut min_dist = f64::MAX;
        let mut min_car_i = None;

        let mut dist_thresh = car.follow_dist() * 4.0;

        let pose = self.cars[car_i].pose();
        let shape = self.cars[car_i].shape();
        let aabb = shape.compute_aabb(&pose);
        for (i, c) in self.cars.iter().enumerate() {
            if i == car_i {
                continue;
            }
            // skip cars behind this one
            if AHEAD && c.x < car.x || !AHEAD && c.x > car.x {
                // if i == 0 && car_i == 14 {
                //     eprintln_f!("Skipping {car_i} to ego: c.x {:.2} car.x {:.2}", c.x, car.x);
                // }
                continue;
            }

            if (c.x - car.x).abs() >= dist_thresh {
                continue;
            }

            if true {
                let other_aabb = c.shape().compute_aabb(&c.pose());

                let side_sep = range_dist(
                    aabb.mins[1],
                    aabb.maxs[1],
                    other_aabb.mins[1],
                    other_aabb.maxs[1],
                );

                // if car_i == 0 {
                //     eprintln_f!("ego from {i} {side_sep=:.2}");
                // }

                if side_sep <= SIDE_MARGIN {
                    let dist = other_aabb.mins[0] - aabb.maxs[0];
                    if dist < min_dist {
                        min_dist = dist;
                        min_car_i = Some(i);
                    }
                }
            } else {
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
                            _ => continue,
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

                            dist_thresh = min_dist.hypot(SIDE_MARGIN + car.width / 2.0);
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
            }

            // let d = query::distance(&pose, &shape, &c.pose(), &c.shape()).unwrap();
            // min_dist = min_dist.min(d);
        }

        Some((min_dist, min_car_i?))
    }

    fn min_unsafe_dist(&self, car_i: usize) -> Option<f64> {
        let car = &self.cars[car_i];

        let mut min_dist = None;
        let mut dist_thresh = car.length + SAFETY_WEIGHT;

        let pose = car.pose();
        let shape = car.shape();
        let aabb = shape.compute_aabb(&pose);
        for (i, c) in self.cars.iter().enumerate() {
            if i == car_i {
                continue;
            }

            if (c.x - car.x).abs() >= dist_thresh {
                continue;
            }

            if true {
                // let mut aabb_min_dist = None;
                let other_aabb = c.shape().compute_aabb(&c.pose());

                let side_sep = range_dist(
                    aabb.mins[1],
                    aabb.maxs[1],
                    other_aabb.mins[1],
                    other_aabb.maxs[1],
                );

                if side_sep <= SAFETY_MARGIN {
                    let longitidinal_sep = range_dist(
                        aabb.mins[0],
                        aabb.maxs[0],
                        other_aabb.mins[0],
                        other_aabb.maxs[0],
                    );
                    let dist = side_sep.max(longitidinal_sep);
                    if dist < min_dist.unwrap_or(SAFETY_MARGIN) {
                        min_dist = Some(dist);
                        dist_thresh = dist.hypot(SIDE_MARGIN + car.width / 2.0);
                    }
                }
            } else {
                // let mut ab = None;
                match query::closest_points(&pose, &shape, &c.pose(), &c.shape(), SAFETY_MARGIN) {
                    Ok(ClosestPoints::WithinMargin(a, b)) => {
                        let dist = (a - b).magnitude();
                        if dist < min_dist.unwrap_or(f64::MAX) {
                            min_dist = Some(dist);
                        }
                        // ab = Some((a, b));
                    }
                    Ok(ClosestPoints::Intersecting) => {
                        min_dist = Some(0.0);
                    }
                    _ => (),
                }
            }

            // if min_dist != aabb_min_dist {
            //     panic_f!("{i}:\n{aabb=:.2?}\n{other_aabb=:.2?}\n{side_sep=:.2}, {aabb_min_dist=:.2?}, {min_dist=:.2?}, {ab=:.2?}");
            // }

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
        self.timesteps += 1;

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
                    eprintln!("New policy:\n{:?}", ecar.side_policy.as_ref().unwrap());
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
            if i == 0 && car.crashed {
                car.draw(r, RvxColor::ORANGE.set_a(0.6));
            } else if i == 0 {
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
