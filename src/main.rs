use std::{
    cell::RefCell,
    f64::consts::PI,
    rc::Rc,
    sync::{Arc, Mutex},
    time::Duration,
};

use arg_parameters::Parameters;

use mpdm::mpdm_choose_policy;

use rand::{prelude::StdRng, SeedableRng};
use rate_timer::RateTimer;
use road::{Road, LANE_WIDTH};
use rvx::Rvx;
use side_policies::SidePolicyTrait;

use crate::eudm_dcp_tree::{dcp_tree_choose_policy, tree_choose_policy};

#[allow(unused)]
#[macro_use]
extern crate fstrings;

#[cfg(test)]
#[macro_use]
extern crate approx;

mod arg_parameters;
mod car;
mod delayed_policy;
mod eudm_dcp_tree;
mod forward_control;
mod intelligent_driver;
mod lane_change_policy;
mod mpdm;
mod pure_pursuit;
mod rate_timer;
mod reward;
mod road;
mod side_control;
mod side_policies;

#[macro_use]
extern crate enum_dispatch;

const AHEAD_TIME_DEFAULT: f64 = 0.6;
const LANE_CHANGE_TIME: f64 = 4.0;

struct State {
    // rng: Rc<RefCell<StdRng>>,
    params: Rc<Parameters>,
    road: Road,
    traces: Vec<rvx::Shape>,
    r: Option<Rvx>,
    timestep: u32,
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

        // let mut fake_sim_estimate = self.road.clone();
        // fake_sim_estimate.debug = false;

        if self.timestep % replan_interval == 0 {
            let (policy, traces) = match self.params.method.as_str() {
                "mpdm" => mpdm_choose_policy(&self.params, self.road.sim_estimate()),
                "eudm" => dcp_tree_choose_policy(&self.params, self.road.sim_estimate()),
                "tree" => tree_choose_policy(&self.params, self.road.sim_estimate()),
                _ => panic!("invalid method '{}'", self.params.method),
            };
            self.traces = traces;
            if false {
                check_for_duplicate_shapes(&self.traces);
            }

            let old_policy_id = self.road.cars[0]
                .side_policy
                .as_ref()
                .map(|p| p.policy_id());

            if Some(policy.policy_id()) != old_policy_id {
                self.road.cars[0].side_policy = Some(policy);
            }
        }

        self.road.update(dt);

        self.timestep += 1;
    }

    fn final_reward(&self) -> f64 {
        self.road.final_reward()
    }
}

fn run_with_parameters(
    params: Parameters,
    use_graphics: bool,
    n_scenarios: usize,
    n_scenarios_completed: Arc<Mutex<usize>>,
) {
    let params = Rc::new(params);

    let mut full_seed = [0; 32];
    full_seed[0..8].copy_from_slice(&params.rng_seed.to_le_bytes());

    let rng = Rc::new(RefCell::new(StdRng::from_seed(full_seed)));

    let mut road = Road::new(params.clone());
    road.add_obstacle(100.0, 0);

    while road.cars.len() < params.n_cars {
        road.add_random_car(&rng);
    }

    let mut state = State {
        road,
        r: None,
        timestep: 0,
        params,
        traces: Vec::new(),
    };

    if use_graphics {
        let mut r = Rvx::new("Self-Driving!", [0, 0, 0, 0], 8000);
        r.set_user_zoom(Some(0.4)); // 0.22
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

    *n_scenarios_completed.lock().unwrap() += 1;
    print!(
        "{}/{}: ",
        *n_scenarios_completed.lock().unwrap(),
        n_scenarios
    );

    println!("{:.4}", state.final_reward());

    if use_graphics {
        std::thread::sleep(Duration::from_millis(1000));
    }
}

fn main() {
    arg_parameters::run_parallel_scenarios();
}
