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

use crate::{run_with_parameters, ChildSelectionMode, CostBoundMode};

#[derive(Clone, Debug)]
pub(crate) struct Parameters {
    pub search_depth: u32,
    pub n_actions: u32,
    pub ucb_const: f64,
    pub ucbv_const: f64,
    pub ucbd_const: f64,
    pub klucb_max_cost: f64,
    pub rng_seed: u64,
    pub samples_n: usize,

    pub bound_mode: CostBoundMode,
    pub final_choice_mode: CostBoundMode,
    pub selection_mode: ChildSelectionMode,
    pub prioritize_worst_particles_z: f64,
    pub consider_repeats_after_portion: f64,
    pub repeat_confidence_interval: f64,
    pub correct_future_std_dev_mean: bool,
    pub repeat_const: f64,
    pub repeat_particle_sign: i8,
    pub repeat_at_all_levels: bool,
    pub throwout_extreme_costs_z: f64,
    pub bootstrap_confidence_z: f64,
    pub use_final_selection_var: bool,
    pub gaussian_prior_std_dev: f64,
    pub thread_limit: usize,
    pub scenario_name: Option<String>,

    pub print_report: bool,
    pub stats_analysis: bool,
    pub is_single_run: bool,
}

impl Parameters {
    fn new() -> Self {
        Self {
            search_depth: 4,
            n_actions: 5,
            ucb_const: -2.2, // -3000 for UCB
            ucbv_const: 0.001,
            ucbd_const: 1.0,
            klucb_max_cost: 10000.0,
            rng_seed: 0,
            samples_n: 64,
            bound_mode: CostBoundMode::Marginal,
            final_choice_mode: CostBoundMode::Same,
            selection_mode: ChildSelectionMode::KLUCBP,
            prioritize_worst_particles_z: 1000.0,
            consider_repeats_after_portion: 0.0,
            repeat_confidence_interval: 1000.0,
            correct_future_std_dev_mean: false,
            repeat_const: -1.0,
            repeat_particle_sign: 1,
            repeat_at_all_levels: false,
            throwout_extreme_costs_z: 1000.0,
            bootstrap_confidence_z: 0.0,
            use_final_selection_var: false,
            gaussian_prior_std_dev: 1000.0,

            thread_limit: 1,
            scenario_name: None,

            print_report: false,
            stats_analysis: false,
            is_single_run: false,
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

    if name.starts_with("normal.") && base_params.bound_mode != CostBoundMode::Normal
        || name.starts_with("lower_bound.") && base_params.bound_mode != CostBoundMode::LowerBound
        || name.starts_with("marginal.") && base_params.bound_mode != CostBoundMode::Marginal
    {
        return create_scenarios(&base_params, &name_value_pairs[1..]);
    }

    if name.starts_with("ucb.") && base_params.selection_mode != ChildSelectionMode::UCB
        || name.starts_with("ucbv.") && base_params.selection_mode != ChildSelectionMode::UCBV
        || name.starts_with("ucbd.") && base_params.selection_mode != ChildSelectionMode::UCBd
        || name.starts_with("klucb.") && base_params.selection_mode != ChildSelectionMode::KLUCB
        || name.starts_with("klucb+.") && base_params.selection_mode != ChildSelectionMode::KLUCBP
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

            if name.ends_with(".ucb_const") {
                params.ucb_const = val.parse().unwrap();
            } else {
                match name.as_str() {
                    "search_depth" => params.search_depth = val.parse().unwrap(),
                    "n_actions" => params.n_actions = val.parse().unwrap(),
                    "thread_limit" => params.thread_limit = val.parse().unwrap(),
                    "samples_n" => params.samples_n = val.parse().unwrap(),
                    "bound_mode" => params.bound_mode = val.parse().unwrap(),
                    "final_choice_mode" => params.final_choice_mode = val.parse().unwrap(),
                    "selection_mode" => params.selection_mode = val.parse().unwrap(),
                    "prioritize_worst_particles_z" => {
                        params.prioritize_worst_particles_z = val.parse().unwrap()
                    }
                    "consider_repeats_after_portion" => {
                        params.consider_repeats_after_portion = val.parse().unwrap()
                    }
                    "repeat_confidence_interval" => {
                        params.repeat_confidence_interval = val.parse().unwrap()
                    }
                    "correct_future_std_dev_mean" => {
                        params.correct_future_std_dev_mean = val.parse().unwrap()
                    }
                    "repeat_const" => params.repeat_const = val.parse().unwrap(),
                    "repeat_particle_sign" => params.repeat_particle_sign = val.parse().unwrap(),
                    "repeat_at_all_levels" => params.repeat_at_all_levels = val.parse().unwrap(),
                    "throwout_extreme_costs_z" => {
                        params.throwout_extreme_costs_z = val.parse().unwrap()
                    }
                    "bootstrap_confidence_z" => {
                        params.bootstrap_confidence_z = val.parse().unwrap()
                    }
                    "use_final_selection_var" => {
                        params.use_final_selection_var = val.parse().unwrap()
                    }
                    "gaussian_prior_std_dev" => {
                        params.gaussian_prior_std_dev = val.parse().unwrap()
                    }
                    "ucb_const" => params.ucb_const = val.parse().unwrap(),
                    "ucbv.ucbv_const" => params.ucbv_const = val.parse().unwrap(),
                    "ucbd.ucbd_const" => {
                        params.ucbd_const = val.parse().unwrap();
                        assert!(params.ucbd_const <= 1.0);
                    }
                    "klucb.klucb_max_cost" => params.klucb_max_cost = val.parse().unwrap(),
                    "klucb+.klucb_max_cost" => params.klucb_max_cost = val.parse().unwrap(),
                    "rng_seed" => params.rng_seed = val.parse().unwrap(),
                    "print_report" => params.print_report = val.parse().unwrap(),
                    "stats_analysis" => params.stats_analysis = val.parse().unwrap(),
                    _ => panic!("{} is not a valid parameter!", name),
                }
            }
            if name_value_pairs.len() > 1 {
                scenarios.append(&mut create_scenarios(&params, &name_value_pairs[1..]));
            } else {
                scenarios.push(params);
            }
        }
    }

    for s in scenarios.iter_mut() {
        let ucbv_const = match s.selection_mode {
            ChildSelectionMode::UCBV => format!(",ucbv_const={}", s.ucbv_const),
            _ => "".to_string(),
        };

        let ucbd_const = match s.selection_mode {
            ChildSelectionMode::UCBd => format!(",ucbd_const={}", s.ucbd_const),
            _ => "".to_string(),
        };

        let klucb_max_cost = match s.selection_mode {
            ChildSelectionMode::KLUCB | ChildSelectionMode::KLUCBP => {
                format!(",klucb_max_cost={}", s.klucb_max_cost)
            }
            _ => "".to_string(),
        };

        s.scenario_name = Some(format_f!(
            ",search_depth={s.search_depth}\
             ,n_actions={s.n_actions}\
             ,samples_n={s.samples_n}\
             ,bound_mode={s.bound_mode}\
             ,final_choice_mode={s.final_choice_mode}\
             ,selection_mode={s.selection_mode}\
             ,prioritize_worst_particles_z={s.prioritize_worst_particles_z}\
             ,consider_repeats_after_portion={s.consider_repeats_after_portion}\
             ,repeat_confidence_interval={s.repeat_confidence_interval}\
             ,correct_future_std_dev_mean={s.correct_future_std_dev_mean}\
             ,repeat_const={s.repeat_const}\
             ,repeat_particle_sign={s.repeat_particle_sign}\
             ,repeat_at_all_levels={s.repeat_at_all_levels}\
             ,throwout_extreme_costs_z={s.throwout_extreme_costs_z}\
             ,bootstrap_confidence_z={s.bootstrap_confidence_z}\
             ,use_final_selection_var={s.use_final_selection_var}\
             ,gaussian_prior_std_dev={s.gaussian_prior_std_dev}\
             ,ucb_const={s.ucb_const}\
             {ucbv_const}\
             {ucbd_const}\
             {klucb_max_cost}\
             ,rng_seed={s.rng_seed}\
             ,"
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

    let many_scenarios = n_scenarios > 50000;

    if n_scenarios == 1 {
        let mut single_scenario = scenarios[0].clone();
        single_scenario.is_single_run = true;
        let res = run_with_parameters(single_scenario);
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
                if many_scenarios {
                    let completed = n_scenarios_completed.load(Ordering::Relaxed);
                    if completed % 500 == 0 {
                        println!(
                            "{}/{}: ",
                            n_scenarios_completed.load(Ordering::Relaxed),
                            n_scenarios
                        );
                    }
                } else {
                    print!(
                        "{}/{}: ",
                        n_scenarios_completed.load(Ordering::Relaxed),
                        n_scenarios
                    );
                    if scenario.stats_analysis {
                        println_f!("{res} {scenario.search_depth} {scenario.n_actions} {scenario.samples_n}");
                    } else {
                        println_f!("{res}");
                    }
                }
                if scenario.stats_analysis {
                    writeln_f!(
                        file.lock().unwrap(),
                        "{res} {scenario.search_depth} {scenario.n_actions} {scenario.samples_n}"
                    )
                    .unwrap();
                } else {
                    writeln_f!(file.lock().unwrap(), "{scenario_name} {res}").unwrap();
                }

                cumulative_results.lock().unwrap().insert(scenario_name, ());
            });
            if result.is_err() {
                eprintln!(
                    "PANIC for scenario: {:?}",
                    scenario.scenario_name.as_ref().unwrap()
                );
                panic!();
            }
        });
    }
}
