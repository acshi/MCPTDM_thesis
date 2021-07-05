use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

use fstrings::{format_args_f, format_f, println_f, writeln_f};
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{run_with_parameters, CostBoundMode};

#[derive(Clone, Debug)]
pub(crate) struct Parameters {
    pub search_depth: u32,
    pub n_actions: u32,
    pub ucb_const: f64,
    pub rng_seed: u64,
    pub samples_n: usize,

    pub bound_mode: CostBoundMode,
    pub portion_bernoulli: f64,

    pub thread_limit: usize,
    pub scenario_name: Option<String>,
}

impl Parameters {
    fn new() -> Self {
        Self {
            search_depth: 4,
            n_actions: 4,
            ucb_const: -10000.0,
            rng_seed: 0,
            samples_n: 8,
            bound_mode: CostBoundMode::Normal,
            portion_bernoulli: 1.0,

            thread_limit: 1,
            scenario_name: None,
        }
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
                "thread_limit" => params.thread_limit = val.parse().unwrap(),
                "samples_n" => params.samples_n = val.parse().unwrap(),
                "bound_mode" => params.bound_mode = val.parse().unwrap(),
                "portion_bernoulli" => params.portion_bernoulli = val.parse().unwrap(),
                "ucb_const" => params.ucb_const = val.parse().unwrap(),
                "rng_seed" => params.rng_seed = val.parse().unwrap(),
                _ => panic!("{} is not a valid parameter!", name),
            }
            if name_value_pairs.len() > 1 {
                scenarios.append(&mut create_scenarios(&params, &name_value_pairs[1..]));
            } else {
                scenarios.push(params);
            }
        }
    }

    for s in scenarios.iter_mut() {
        s.scenario_name = Some(format_f!(
            "_samples_n_{s.samples_n}\
             _bound_mode_{s.bound_mode}\
             _portion_bernoulli_{s.portion_bernoulli}\
             _ucb_const_{s.ucb_const}\
             _rng_seed_{s.rng_seed}_"
        ));
    }

    scenarios
}

pub fn run_parallel_scenarios() {
    let parameters_default = Parameters::new();

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
            eprintln!("For example: limit 8 12 16 24 32 :: steps 1000 :: rng_seed 0 1 2 3 4");
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

    // only use the cache file to prevent recalculation when running a batch
    let cache_filename = "results.cache";
    if n_scenarios > 1 {
        // read the existing cache file
        {
            let mut cumulative_results = cumulative_results.lock().unwrap();
            if let Ok(file) = File::open(cache_filename) {
                let file = BufReader::new(file);
                for line in file.lines() {
                    let line = line.unwrap();
                    let parts = line.split_ascii_whitespace().collect_vec();
                    let scenario_name = parts[0].to_owned();
                    cumulative_results.insert(scenario_name, ());
                }
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
        let res = run_with_parameters(scenarios[0].clone());
        println_f!("{res}");
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

                let res = run_with_parameters(scenario.clone());

                n_scenarios_completed.fetch_add(1, Ordering::Relaxed);
                print!(
                    "{}/{}: ",
                    n_scenarios_completed.load(Ordering::Relaxed),
                    n_scenarios
                );
                println_f!("{res}");
                writeln_f!(file.lock().unwrap(), "{scenario_name} {res}").unwrap();

                cumulative_results.lock().unwrap().insert(scenario_name, ());
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
