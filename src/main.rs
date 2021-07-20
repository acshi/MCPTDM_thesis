use std::{
    f64::consts::PI,
    rc::Rc,
    time::{Duration, Instant},
};

use arg_parameters::Parameters;

use belief::Belief;
use car::{Car, MPH_TO_MPS};
use cfb::conditional_focused_branching;
use itertools::Itertools;
use mpdm::{
    make_obstacle_vehicle_policy_belief_states, make_obstacle_vehicle_policy_choices,
    make_policy_choices, mpdm_choose_policy,
};

use cost::Cost;
use rand::{
    distributions::WeightedIndex,
    prelude::{Distribution, StdRng},
    Rng, SeedableRng,
};
use rate_timer::RateTimer;
use reward::Reward;
use road::Road;
use road_set::RoadSet;
use rvx::Rvx;
#[allow(unused)]
use side_policies::SidePolicyTrait;

use crate::{eudm::dcp_tree_choose_policy, mcts::mcts_choose_policy, tree::tree_choose_policy};

#[allow(unused)]
#[macro_use]
extern crate fstrings;

mod arg_parameters;
mod belief;
mod car;
mod cfb;
mod cost;
mod delayed_policy;
mod eudm;
mod forward_control;
mod intelligent_driver;
mod lane_change_policy;
mod mcts;
mod mpdm;
mod open_loop_policy;
mod pure_pursuit;
mod rate_timer;
mod reward;
mod road;
mod road_set;
mod side_control;
mod side_policies;
mod tree;

#[macro_use]
extern crate enum_dispatch;

const AHEAD_TIME_DEFAULT: f64 = 0.6;

struct State {
    scenario_rng: StdRng,
    respawn_rng: StdRng,
    policy_rng: StdRng,
    params: Rc<Parameters>,
    road: Road,
    traces: Vec<rvx::Shape>,
    r: Option<Rvx>,
    timesteps: u32,
    reward: Reward,
}

fn check_for_duplicate_shapes(shapes: &[rvx::Shape]) {
    let mut ht = std::collections::HashMap::new();
    for shape in shapes.iter() {
        let entry = ht.entry(shape).or_insert(-1);
        *entry += 1;
    }
    let duplicates = ht.iter().map(|(_k, v)| *v).sum::<i32>();
    eprintln!(
        "Traces has {} shapes and {} duplicates",
        shapes.len(),
        duplicates
    );

    let (most_common_shape, most_common_count) = ht.iter().max_by_key(|a| a.1).unwrap();
    eprintln!(
        "The most common shape with {} copies is {:?}",
        *most_common_count + 1,
        most_common_shape
    );
}

impl State {
    fn update_graphics(&mut self) {
        if let Some(r) = self.r.as_mut() {
            r.clear();

            self.road.draw(r);
            r.draw_all(self.traces.iter().cloned());

            r.set_global_rot(-PI / 2.0);
            r.commit_changes();
        }
    }

    fn update(&mut self, dt: f64) {
        let replan_interval = (self.params.replan_dt / self.params.physics_dt).round() as u32;

        let real_time_start = Instant::now();

        // method chooses the ego policy
        let policy_rng = &mut self.policy_rng;
        if self.timesteps % replan_interval == 0 && !self.road.cars[0].crashed {
            let (policy, traces) = if false && self.road.super_debug() {
                mpdm_choose_policy(&self.params, &self.road, policy_rng)
            } else {
                match self.params.method.as_str() {
                    "fixed" => (None, Vec::new()),
                    "mpdm" => mpdm_choose_policy(&self.params, &self.road, policy_rng),
                    "eudm" => dcp_tree_choose_policy(&self.params, &self.road, policy_rng),
                    "tree" => tree_choose_policy(&self.params, &self.road, policy_rng),
                    "mcts" => mcts_choose_policy(&self.params, &self.road, policy_rng),
                    _ => panic!("invalid method '{}'", self.params.method),
                }
            };
            self.traces = traces;
            if false {
                check_for_duplicate_shapes(&self.traces);
            }

            if let Some(policy) = policy {
                self.road.set_ego_policy(policy);
            }
        }

        // random policy changes for the obstacle vehicles
        let policy_change_interval =
            (self.params.nonego_policy_change_dt / self.params.physics_dt).round() as u32;
        let timesteps = self.timesteps;
        if self.timesteps % policy_change_interval == 0 {
            let rng = &mut self.scenario_rng;
            let policy_choices = make_obstacle_vehicle_policy_choices(&self.params);

            for c in self.road.cars[1..].iter_mut() {
                if rng.gen_bool(
                    self.params.nonego_policy_change_prob * self.params.nonego_policy_change_dt,
                ) {
                    let new_policy_i = rng.gen_range(0..policy_choices.len());
                    let new_policy = policy_choices[new_policy_i].clone();

                    if self.road.debug && self.params.obstacle_car_debug {
                        eprintln_f!("{timesteps}: obstacle car {c.car_i} switching to policy {new_policy_i}: {new_policy:?}");
                    }

                    c.side_policy = Some(new_policy);
                }
            }
        }

        let last_ego_theta = self.road.cars[0].theta();
        let last_ego_vel = self.road.cars[0].vel;

        // actual simulation
        self.road.update_belief();
        self.road.update(dt);
        self.road.respawn_obstacle_cars(&mut self.respawn_rng);

        // final reporting reward (separate from cost function, though similar)
        self.reward.avg_vel += self.road.cars[0].vel * dt;
        if !self.road.ego_is_safe {
            self.reward.safety += dt;
        }
        let accel = (self.road.cars[0].vel - last_ego_vel) / dt;
        if accel <= -self.params.reward.uncomfortable_dec {
            self.reward.uncomfortable_dec += dt;
        }
        let curvature_change = (self.road.cars[0].theta() - last_ego_theta).abs() / dt;
        if curvature_change >= self.params.reward.large_curvature_change {
            self.reward.curvature_change += dt;
        }
        // eprintln_f!("{accel=:.2} {curvature_change=:.2}");

        let timestep_time = real_time_start.elapsed().as_secs_f64();
        self.reward.max_timestep_time = self.reward.max_timestep_time.max(timestep_time);

        self.timesteps += 1;
    }
}

fn debugging_scenarios(params: &Parameters, road: &mut Road) {
    match params.debugging_scenario {
        Some(1) => {
            road.cars.truncate(1);
            road.cars.push(Car::new(params, 1, 1));
            road.cars[1].side_policy =
                Some(make_obstacle_vehicle_policy_choices(params)[0].clone());
        }
        Some(2) => {
            road.cars.truncate(1);
            road.add_obstacle(5.0, 0);
            road.cars.push(Car::new(params, 2, 1));
            road.cars[2].side_policy =
                Some(make_obstacle_vehicle_policy_choices(params)[0].clone());
        }
        Some(3) => {
            road.cars.clear();
            let mut c = Car::new(params, 0, 0);
            c.vel = 15.0 * MPH_TO_MPS;
            c.side_policy = Some(make_policy_choices(params)[0].clone());
            road.last_ego = c.clone();
            road.cars.push(c);

            let mut c = Car::new(params, 1, 0);
            c.set_x(10.0);
            c.vel = 10.0 * MPH_TO_MPS;
            c.side_policy = Some(make_obstacle_vehicle_policy_choices(params)[0].clone());
            road.cars.push(c);
            let mut c = Car::new(params, 2, 1);
            c.set_x(10.0);
            c.vel = 10.0 * MPH_TO_MPS;
            c.side_policy = Some(make_obstacle_vehicle_policy_choices(params)[3].clone());
            road.cars.push(c);
        }
        _ => (),
    }
}

fn run_with_parameters(params: Parameters) -> (Cost, Reward) {
    let params = Rc::new(params);

    let mut full_seed = [0; 32];
    full_seed[0..8].copy_from_slice(&params.rng_seed.to_le_bytes());

    let mut scenario_rng = StdRng::from_seed(full_seed);

    let mut road = Road::new(params.clone());
    // road.add_obstacle(100.0, 0);
    while road.cars.len() < params.n_cars + 1 {
        road.add_random_car(&mut scenario_rng);
    }
    road.init_belief();
    debugging_scenarios(&params, &mut road);

    let mut state = State {
        scenario_rng,
        respawn_rng: StdRng::from_seed(full_seed),
        policy_rng: StdRng::from_seed(full_seed),
        road,
        r: None,
        timesteps: 0,
        params,
        traces: Vec::new(),
        reward: Default::default(),
    };

    let use_graphics = !state.params.run_fast;

    if use_graphics {
        let mut r = Rvx::new("Self-Driving!", [0, 0, 0, 0], 8000);
        // r.set_user_zoom(Some(0.4)); // 0.22
        std::thread::sleep(Duration::from_millis(500));
        r.set_user_zoom(None);
        state.r = Some(r);
    }

    let mut rate = RateTimer::new(Duration::from_millis(
        (state.params.physics_dt * 1000.0 / state.params.graphics_speedup) as u64,
    ));

    for _ in 0..state.params.max_steps {
        state.update(state.params.physics_dt);

        if use_graphics {
            state.update_graphics();
            rate.wait_until_ready();
        }

        // if i == 1000 {
        //     for side_policy in state.road.cars[0].side_policy.iter_mut() {
        //         *side_policy = side_policies::SidePolicy::LaneChangePolicy(
        //             lane_change_policy::LaneChangePolicy::new(1, LANE_CHANGE_TIME, None),
        //         );
        //     }
        // }
    }

    if use_graphics {
        std::thread::sleep(Duration::from_millis(1000));
    }

    let km_travelled = state.reward.avg_vel / 1000.0;

    state.reward.avg_vel /= state.road.t;
    state.reward.safety /= state.road.t;
    state.reward.uncomfortable_dec /= km_travelled;
    state.reward.curvature_change /= km_travelled;

    (state.road.cost, state.reward)
}

fn sample_from_road_set(base_set: RoadSet, rng: &mut StdRng, n: usize) -> RoadSet {
    let weights = base_set.inner().iter().map(|r| r.cost.weight).collect_vec();
    let weighted_distribution = WeightedIndex::new(weights).unwrap();

    let samples = (0..n)
        .map(|_| {
            let road_i = weighted_distribution.sample(rng);
            let mut road = base_set.inner()[road_i].clone();
            road.cost.weight = 1.0; // sampled, so weight is already taken into account
            road
        })
        .collect_vec();

    RoadSet::new(samples)
}

fn randomize_unimportant_vehicle_policies(
    params: &Parameters,
    roads: &mut RoadSet,
    belief: &Belief,
    selected_ids: &[usize],
    rng: &mut StdRng,
) {
    // eprintln!("Randomizing vehicles other than: {:?}", selected_ids);
    for road in roads.iter_mut() {
        let policies = make_obstacle_vehicle_policy_belief_states(params);
        let sampled_belief = belief.sample(rng);

        for car in road.cars[1..].iter_mut() {
            if !selected_ids.contains(&car.car_i) {
                car.side_policy = Some(policies[sampled_belief[car.car_i]].clone());
            }
        }
    }
}

fn road_set_for_scenario(
    params: &Parameters,
    true_road: &Road,
    rng: &mut StdRng,
    n: usize,
) -> RoadSet {
    if params.use_cfb {
        let (base_set, selected_ids) = conditional_focused_branching(params, true_road, n);
        if params.cfb.sample_from_base_set {
            let mut roads = sample_from_road_set(base_set, rng, n);
            if params.cfb.sample_unimportant_vehicle_policies {
                randomize_unimportant_vehicle_policies(
                    params,
                    &mut roads,
                    true_road.belief.as_ref().unwrap(),
                    &selected_ids,
                    rng,
                );
            }
            roads
        } else {
            base_set
        }
    } else {
        RoadSet::new_samples(true_road, rng, n)
    }
}

fn main() {
    arg_parameters::run_parallel_scenarios();
}
