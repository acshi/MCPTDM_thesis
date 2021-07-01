use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
    sync::{
        atomic::{self, AtomicUsize},
        Mutex,
    },
    time::Instant,
};

use atomic::Ordering;
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::Deserialize;

use crate::{cost::Cost, run_with_parameters};

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct EudmParameters {
    pub dt: f64,
    pub layer_t: f64,
    pub search_depth: u32,
    pub samples_n: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct TreeParameters {
    pub dt: f64,
    pub layer_t: f64,
    pub search_depth: u32,
    pub samples_n: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct MctsParameters {
    pub dt: f64,
    pub layer_t: f64,
    pub search_depth: u32,
    pub samples_n: usize,
    pub prefer_same_policy: bool,
    pub choose_random_policy: bool,
    pub ucb_const: f64,
    pub bubble_up_max_weighted_leaf: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct MpdmParameters {
    pub dt: f64,
    pub forward_t: f64,
    pub samples_n: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct CostParameters {
    pub efficiency_low_speed_cost: f64,
    pub efficiency_high_speed_cost: f64,
    pub efficiency_high_speed_tolerance: f64,
    pub efficiency_weight: f64,
    pub safety_weight: f64,
    pub smoothness_weight: f64,
    pub uncomfortable_dec_weight: f64,
    pub curvature_change_weight: f64,
    pub safety_margin: f64,
    pub uncomfortable_dec: f64,
    pub large_curvature_change: f64,
    pub discount_factor: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct CfbParameters {
    pub key_vehicle_base_dist: f64,
    pub key_vehicle_dist_time: f64,
    pub uncertainty_threshold: f64,
    pub dangerous_delta_threshold: f64,
    pub max_n_for_cartesian_product: usize,
    pub dt: f64,
    pub horizon_t: f64,
    pub set_cost_weights: bool,
    pub sample_from_base_set: bool,
    pub sample_unimportant_vehicle_policies: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct BeliefParameters {
    pub different_lane_prob: f64,
    pub different_longitudinal_prob: f64,
    pub accelerate_delta_vel_thresh: f64,
    pub accelerate_ahead_dist_thresh: f64,
    pub decelerate_vel_thresh: f64,
    pub finished_waiting_dy: f64,
    pub skips_waiting_prob: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct SpawnParameters {
    pub remove_ahead_beyond: f64,
    pub remove_behind_beyond: f64,
    pub place_ahead_beyond: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Parameters {
    pub max_steps: u32,
    pub n_cars: usize,
    pub method: String,
    pub use_cfb: bool,
    pub extra_ego_accdec_policies: Vec<f64>,

    pub physics_dt: f64,
    pub replan_dt: f64,
    pub nonego_policy_change_prob: f64,
    pub nonego_policy_change_dt: f64,
    pub lane_change_time: f64,

    pub thread_limit: usize,
    pub rng_seed: u64,
    pub run_fast: bool,
    pub graphics_speedup: f64,
    pub debug_car_i: Option<usize>,
    pub debug_steps_before: usize,
    pub super_debug: bool,
    pub ego_policy_change_debug: bool,
    pub ego_state_debug: bool,
    pub separation_debug: bool,
    pub intelligent_driver_debug: bool,
    pub belief_debug: bool,
    pub cfb_debug: bool,
    pub obstacle_car_debug: bool,
    pub policy_report_debug: bool,
    pub ego_traces_debug: bool,

    pub only_ego_crashes_in_forward_sims: bool,
    pub only_crashes_with_ego: bool,
    pub obstacles_only_for_ego: bool,
    pub true_belief_sample_only: bool,
    pub debugging_scenario: Option<i32>,

    pub spawn: SpawnParameters,
    pub belief: BeliefParameters,
    pub cost: CostParameters,
    pub cfb: CfbParameters,
    pub eudm: EudmParameters,
    pub tree: TreeParameters,
    pub mpdm: MpdmParameters,
    pub mcts: MctsParameters,

    pub scenario_name: Option<String>,
}

impl Parameters {
    fn new() -> Result<Self, config::ConfigError> {
        let mut s = config::Config::new();
        s.merge(config::File::with_name("parameters"))?;
        s.try_into()
    }
}

fn create_scenarios(
    base_params: &Parameters,
    name_value_pairs: &[(String, Vec<String>)],
) -> Vec<Parameters> {
    if name_value_pairs.is_empty() {
        return vec![base_params.clone()];
    }

    let mut scenarios = Vec::new();
    let (name, values) = &name_value_pairs[0];

    if name.starts_with("tree.") && base_params.method != "tree"
        || name.starts_with("mpdm.") && base_params.method != "mpdm"
        || name.starts_with("eudm.") && base_params.method != "eudm"
        || name.starts_with("mcts.") && base_params.method != "mcts"
    {
        return create_scenarios(&base_params, &name_value_pairs[1..]);
    }

    for value in values.iter() {
        let mut value_set = vec![value.to_owned()];

        // Do we have a numeric range? special-case handle that!
        let range_parts = value.split("-").collect_vec();
        if range_parts.len() == 2 {
            let low: Option<usize> = range_parts[0].parse().ok();
            let high: Option<usize> = range_parts[1].parse().ok();
            if let (Some(low), Some(high)) = (low, high) {
                if low < high {
                    value_set.clear();
                    for v in low..=high {
                        value_set.push(v.to_string());
                    }
                }
            }
        }

        for val in value_set {
            let mut params = base_params.clone();
            match name.as_str() {
                "method" => params.method = val.parse().unwrap(),
                "use_cfb" => params.use_cfb = val.parse().unwrap(),
                "extra_ego_accdec_policies" => {
                    params.extra_ego_accdec_policies = val
                        .split(",")
                        .filter(|v| !v.is_empty())
                        .map(|v| v.parse::<f64>().unwrap())
                        .collect_vec()
                }
                "max_steps" => params.max_steps = val.parse().unwrap(),
                "n_cars" => params.n_cars = val.parse().unwrap(),
                "discount_factor" => params.cost.discount_factor = val.parse().unwrap(),
                "rng_seed" => params.rng_seed = val.parse().unwrap(),
                "run_fast" => params.run_fast = val.parse().unwrap(),
                "thread_limit" => params.thread_limit = val.parse().unwrap(),
                "tree.samples_n" => params.tree.samples_n = val.parse().unwrap(),
                "mpdm.samples_n" => params.mpdm.samples_n = val.parse().unwrap(),
                "eudm.samples_n" => params.eudm.samples_n = val.parse().unwrap(),
                "mcts.samples_n" => params.mcts.samples_n = val.parse().unwrap(),
                "mpdm.forward_t" => params.mpdm.forward_t = val.parse().unwrap(),
                "tree.search_depth" => params.tree.search_depth = val.parse().unwrap(),
                "eudm.search_depth" => params.eudm.search_depth = val.parse().unwrap(),
                "mcts.search_depth" => params.mcts.search_depth = val.parse().unwrap(),
                "tree.layer_t" => params.tree.layer_t = val.parse().unwrap(),
                "eudm.layer_t" => params.eudm.layer_t = val.parse().unwrap(),
                "mcts.layer_t" => params.mcts.layer_t = val.parse().unwrap(),
                "smoothness" => params.cost.smoothness_weight = val.parse().unwrap(),
                "safety" => params.cost.safety_weight = val.parse().unwrap(),
                "ud" => params.cost.uncomfortable_dec_weight = val.parse().unwrap(),
                "cc" => params.cost.curvature_change_weight = val.parse().unwrap(),
                "safety_margin" => params.cost.safety_margin = val.parse().unwrap(),
                _ => panic!("{} is not a valid parameter!", name),
            }
            if name_value_pairs.len() > 1 {
                scenarios.append(&mut create_scenarios(&params, &name_value_pairs[1..]));
            } else {
                scenarios.push(params);
            }
        }
    }

    // when there are multiple scenarios, always run them fast!
    if scenarios.len() > 1 {
        for scenario in scenarios.iter_mut() {
            scenario.run_fast = true;
        }
    }

    for s in scenarios.iter_mut() {
        let samples_n = match s.method.as_str() {
            "fixed" => "".to_owned(),
            "tree" => format_f!("samples_n_{s.tree.samples_n}"),
            "mpdm" => format_f!("samples_n_{s.mpdm.samples_n}"),
            "eudm" => format_f!("samples_n_{s.eudm.samples_n}"),
            "mcts" => format_f!("samples_n_{s.mcts.samples_n}"),
            _ => panic!("Unknown method {}", s.method),
        };

        let search_depth = match s.method.as_str() {
            "fixed" => "".to_owned(),
            "tree" => format_f!("search_depth_{s.tree.search_depth}"),
            "mpdm" => "".to_owned(),
            "eudm" => format_f!("search_depth_{s.eudm.search_depth}"),
            "mcts" => format_f!("search_depth_{s.mcts.search_depth}"),
            _ => panic!("Unknown method {}", s.method),
        };

        let layer_forward_t = match s.method.as_str() {
            "fixed" => "".to_owned(),
            "tree" => format_f!("layer_t_{s.tree.layer_t}"),
            "mpdm" => format_f!("forward_t_{s.mpdm.forward_t}"),
            "eudm" => format_f!("layer_t_{s.eudm.layer_t}"),
            "mcts" => format_f!("layer_t_{s.mcts.layer_t}"),
            _ => panic!("Unknown method {}", s.method),
        };

        let extra_ego_accdec = s
            .extra_ego_accdec_policies
            .iter()
            .map(|a| a.to_string())
            .join(",");

        s.scenario_name = Some(format_f!(
            "_method_{s.method}_use_cfb_{s.use_cfb}_extra_ego_accdec_policies_{extra_ego_accdec}\
             _{samples_n}_{search_depth}_{layer_forward_t}\
             _max_steps_{s.max_steps}_n_cars_{s.n_cars}\
             _discount_factor_{s.cost.discount_factor}_rng_seed_{s.rng_seed}_"
        ));
    }

    scenarios
}

pub fn run_parallel_scenarios() {
    let parameters_default = Parameters::new().unwrap();

    // let args = std::env::args().collect_vec();
    let mut name_value_pairs = Vec::<(String, Vec<String>)>::new();
    // let mut arg_i = 0;
    let mut name: Option<String> = None;
    let mut vals: Option<Vec<String>> = None;
    for arg in std::env::args()
        .skip(1)
        .chain(std::iter::once("::".to_owned()))
    {
        if arg == "--help" || arg == "help" {
            eprintln!("Usage: (<param name> [param value]* ::)*");
            eprintln!("For example: uwb_limit 8 12 16 24 32 :: forward_sim_steps 1000 :: rng_seed 0 1 2 3 4");
            eprintln!("Valid parameters and their default values:");
            let params_str = format!("{:?}", parameters_default)
                .replace(", file_name: None", "")
                .replace(", ", "\n\t")
                .replace("Parameters { ", "\t")
                .replace(" }", "");
            eprintln!("{}", params_str);
            std::process::exit(0);
        }
        if name.is_some() {
            if arg == "::" {
                let name = name.take().unwrap();
                if name_value_pairs.iter().any(|pair| pair.0 == name) {
                    panic!("Parameter {} has already been specified!", name);
                }
                name_value_pairs.push((name, vals.take().unwrap()));
            } else {
                vals.as_mut().unwrap().push(arg);
            }
        } else if arg != "::" {
            name = Some(arg);
            vals = Some(Vec::new());
        }
    }

    // for (name, vals) in name_value_pairs.iter() {
    //     eprintln!("{}: {:?}", name, vals);
    // }

    let mut base_scenario = parameters_default;
    base_scenario.scenario_name = Some("".to_owned());

    let scenarios = create_scenarios(&base_scenario, &name_value_pairs);
    // for (i, scenario) in scenarios.iter().enumerate() {
    //     eprintln!("{}: {:?}", i, scenario.file_name);
    // }

    let n_scenarios = scenarios.len();
    eprintln!("Starting to run {} scenarios", n_scenarios);
    if n_scenarios == 0 {
        return;
    }

    let thread_limit = scenarios[0].thread_limit;
    if thread_limit > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_limit as usize)
            .build_global()
            .unwrap();
    }

    let n_scenarios_completed = AtomicUsize::new(0);
    let cumulative_results = Mutex::new(BTreeMap::new());

    let cache_filename = "results.cache";
    // read the existing cache file
    {
        let mut cumulative_results = cumulative_results.lock().unwrap();
        if let Ok(file) = File::open(cache_filename) {
            let file = BufReader::new(file);
            for line in file.lines() {
                let line = line.unwrap();
                let parts = line.split_ascii_whitespace().collect_vec();
                let scenario_name = parts[0].to_owned();
                let compute_time: f64 = parts[1].parse().unwrap();
                let efficiency: f64 = parts[2].parse().unwrap();
                let safety: f64 = parts[3].parse().unwrap();
                let smoothness: f64 = parts[4].parse().unwrap();
                let cost = Cost {
                    efficiency,
                    safety,
                    smoothness,
                    ..Default::default()
                };
                cumulative_results.insert(scenario_name, (compute_time, cost));
            }
        }
    }

    let file = Mutex::new(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(cache_filename)
            .unwrap(),
    );

    if n_scenarios == 1 {
        let (cost, reward) = run_with_parameters(scenarios[0].clone());
        println_f!("{cost:?} {reward:?}");
    } else {
        scenarios.par_iter().for_each(|scenario| {
            let result = std::panic::catch_unwind(|| {
                let scenario_name = scenario.scenario_name.clone().unwrap();

                if cumulative_results
                    .lock()
                    .unwrap()
                    .contains_key(&scenario_name)
                {
                    n_scenarios_completed.fetch_add(1, Ordering::Relaxed);
                    return;
                }

                let start_time = Instant::now();
                let (cost, reward) = run_with_parameters(scenario.clone());
                let seconds = start_time.elapsed().as_secs_f64();

                n_scenarios_completed.fetch_add(1, Ordering::Relaxed);
                print!(
                    "{}/{}: ",
                    n_scenarios_completed.load(Ordering::Relaxed),
                    n_scenarios
                );
                println_f!("{cost} {reward} {seconds:6.2}");
                writeln_f!(
                    file.lock().unwrap(),
                    "{scenario_name} {cost} {reward} {seconds:6.2}"
                )
                .unwrap();

                cumulative_results
                    .lock()
                    .unwrap()
                    .insert(scenario_name, (seconds, cost));
            });
            if result.is_err() {
                eprintln!(
                    "PANIC for scenario: {:?}",
                    scenario.scenario_name.as_ref().unwrap()
                );
            }
        });
    }
}
