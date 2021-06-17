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

use atomic::Ordering::SeqCst;
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
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct MpdmParameters {
    pub dt: f64,
    pub forward_t: f64,
    pub samples_n: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct CostParameters {
    pub efficiency_weight: f64,
    pub safety_weight: f64,
    pub smoothness_weight: f64,
    pub uncomfortable_dec_weight: f64,
    pub curvature_change_weight: f64,
    pub safety_margin: f64,
    pub uncomfortable_dec: f64,
    pub large_curvature_change: f64,
    pub discount: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Parameters {
    pub max_steps: u32,
    pub n_cars: usize,
    pub method: String,

    pub physics_dt: f64,
    pub replan_dt: f64,
    pub nonego_policy_change_prob: f64,
    pub nonego_policy_change_dt: f64,

    pub thread_limit: usize,
    pub rng_seed: u64,
    pub run_fast: bool,
    pub graphics_speedup: f64,
    pub debug_car_i: Option<usize>,
    pub debug_steps_before: usize,
    pub super_debug: bool,
    pub only_crashes_with_ego: bool,
    pub obstacles_only_for_ego: bool,
    pub true_belief_sample_only: bool,
    pub debugging_scenario: Option<i32>,

    pub cost: CostParameters,
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
    for value in values.iter() {
        let mut params = base_params.clone();
        match name.as_str() {
            "method" => params.method = value.parse().unwrap(),
            "max_steps" => params.max_steps = value.parse().unwrap(),
            "n_cars" => params.n_cars = value.parse().unwrap(),
            "discount" => params.cost.discount = value.parse().unwrap(),
            "rng_seed" => params.rng_seed = value.parse().unwrap(),
            "run_fast" => params.run_fast = value.parse().unwrap(),
            "thread_limit" => params.thread_limit = value.parse().unwrap(),
            _ => panic!("{} is not a valid parameter!", name),
        }
        if name_value_pairs.len() > 1 {
            scenarios.append(&mut create_scenarios(&params, &name_value_pairs[1..]));
        } else {
            scenarios.push(params);
        }
    }

    // when there are multiple scenarios, always run them fast!
    if scenarios.len() > 1 {
        for scenario in scenarios.iter_mut() {
            scenario.run_fast = true;
        }
    }

    for s in scenarios.iter_mut() {
        s.scenario_name = Some(format_f!(
            "_method_{s.method}_max_steps_{s.max_steps}_n_cars_{s.n_cars}_discount_{s.cost.discount}_rng_seed_{s.rng_seed}_"
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
                    n_scenarios_completed.fetch_add(1, SeqCst);
                    return;
                }

                let start_time = Instant::now();
                let (cost, reward) = run_with_parameters(scenario.clone());
                let seconds = start_time.elapsed().as_secs_f64();

                n_scenarios_completed.fetch_add(1, SeqCst);
                print!("{}/{}: ", n_scenarios_completed.load(SeqCst), n_scenarios);
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
