use std::sync::{Arc, Mutex};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::run_with_parameters;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Parameters {
    pub rng_seed: u64,
    pub run_fast: bool,
    pub max_steps: u32,
    pub n_cars: usize,

    pub thread_limit: usize,
    pub file_name: Option<String>,
}

impl Default for Parameters {
    fn default() -> Self {
        Parameters {
            rng_seed: 0,
            run_fast: false,
            max_steps: 500,
            n_cars: 40,
            thread_limit: 1,
            file_name: None,
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
        let mut params = base_params.clone();
        match name.as_str() {
            "rng_seed" => params.rng_seed = value.parse().unwrap(),
            "run_fast" => params.run_fast = value.parse().unwrap(),
            "n_cars" => params.n_cars = value.parse().unwrap(),
            "max_steps" => params.max_steps = value.parse().unwrap(),
            "thread_limit" => params.thread_limit = value.parse().unwrap(),
            _ => panic!("{} is not a valid parameter!", name),
        }
        let not_for_file_name = ["run_fast"];
        if let Some(file_name) = params.file_name.as_mut() {
            if !not_for_file_name.contains(&name.as_str()) {
                if !file_name.is_empty() {
                    file_name.push_str("_");
                }
                file_name.push_str(name);
                file_name.push_str("_");
                file_name.push_str(value);
            }
        }

        if name_value_pairs.len() > 1 {
            scenarios.append(&mut create_scenarios(&params, &name_value_pairs[1..]));
        } else {
            scenarios.push(params);
        }
    }
    scenarios
}

pub fn run_parallel_scenarios() {
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
            let params_str = format!("{:?}", Parameters::default())
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

    let mut base_scenario = Parameters::default();
    base_scenario.file_name = Some("".to_owned());

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

    let n_scenarios_completed = Arc::new(Mutex::new(0));

    if n_scenarios == 1 {
        run_with_parameters(
            scenarios[0].clone(),
            !scenarios[0].run_fast,
            n_scenarios,
            n_scenarios_completed,
        );
    } else {
        scenarios.par_iter().for_each(|scenario| {
            let result = std::panic::catch_unwind(|| {
                run_with_parameters(
                    scenario.clone(),
                    false,
                    n_scenarios,
                    n_scenarios_completed.clone(),
                );
            });
            if result.is_err() {
                eprintln!(
                    "PANIC for scenario: {:?}",
                    scenario.file_name.as_ref().unwrap()
                );
            }
        });
    }
}
