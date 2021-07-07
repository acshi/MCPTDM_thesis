mod arg_parameters;
mod klucb;

use arg_parameters::{run_parallel_scenarios, Parameters};
use fstrings::{eprintln_f, format_args_f, format_f, println_f, write_f};
use itertools::Itertools;
use klucb::klucb_bernoulli;
use rand::{
    prelude::{IteratorRandom, SliceRandom, StdRng},
    Rng, SeedableRng,
};
use rand_distr::{Bernoulli, Distribution, Normal};

#[derive(Clone, Copy, Debug)]
struct RunResults {
    chosen_cost: f64,
    chosen_true_cost: f64,
    true_best_cost: f64,
}

impl std::fmt::Display for RunResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        write_f!(
            f,
            "{s.chosen_cost:7.2} {s.chosen_true_cost:7.2} {s.true_best_cost:7.2}"
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CostBoundMode {
    Normal,
    LowerBound,
    Marginal,
}

impl std::fmt::Display for CostBoundMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "normal"),
            Self::LowerBound => write!(f, "lower_bound"),
            Self::Marginal => write!(f, "marginal"),
        }
    }
}

impl std::str::FromStr for CostBoundMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "normal" => Ok(Self::Normal),
            "lower_bound" => Ok(Self::LowerBound),
            "marginal" => Ok(Self::Marginal),
            _ => Err(format_f!("Invalid CostBoundMode '{s}'")),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ChildSelectionMode {
    UCB,
    UCBV,
    UCBd,
    KLUCB,
}

impl std::fmt::Display for ChildSelectionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UCB => write!(f, "ucb"),
            Self::UCBV => write!(f, "ucbv"),
            Self::UCBd => write!(f, "ucbd"),
            Self::KLUCB => write!(f, "klucb"),
        }
    }
}

impl std::str::FromStr for ChildSelectionMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "ucb" => Ok(Self::UCB),
            "ucbv" => Ok(Self::UCBV),
            "ucbd" => Ok(Self::UCBd),
            "klucb" => Ok(Self::KLUCB),
            _ => Err(format_f!("Invalid ChildSelectionMode '{s}'")),
        }
    }
}

#[derive(Clone)]
enum CostDistribution {
    Normal {
        d: Normal<f64>,
    },
    Bernoulli {
        d: Bernoulli,
        p: f64,
        magnitude: f64,
    },
}

impl CostDistribution {
    #[allow(unused)]
    fn normal(mean: f64, std_dev: f64) -> Self {
        Self::Normal {
            d: Normal::new(mean, std_dev).expect("valid mean and standard deviation"),
        }
    }

    fn bernoulli(p: f64, magnitude: f64) -> Self {
        Self::Bernoulli {
            d: Bernoulli::new(p).expect("probability from 0 to 1"),
            p,
            magnitude,
        }
    }

    fn mean(&self) -> f64 {
        match self {
            CostDistribution::Normal { d } => d.mean(),
            CostDistribution::Bernoulli { d: _, p, magnitude } => p * magnitude,
        }
    }

    fn sample(&self, rng: &mut StdRng) -> f64 {
        match self {
            CostDistribution::Normal { d } => d.sample(rng).max(0.0).min(2.0 * d.mean()),
            CostDistribution::Bernoulli { d, p: _, magnitude } => {
                if d.sample(rng) {
                    *magnitude
                } else {
                    0.0
                }
            }
        }
    }
}

#[derive(Clone)]
struct ProblemScenario {
    distribution: Option<CostDistribution>,
    children: Vec<ProblemScenario>,
    depth: u32,
}

impl ProblemScenario {
    fn inner_new(
        depth: u32,
        max_depth: u32,
        n_actions: u32,
        portion_bernoulli: f64,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            distribution: if depth == 0 {
                None
            } else {
                // let mean = rng.gen_range(0.0..100.0);
                // let std_dev = rng.gen_range(0.0..1000.0);

                // Some(CostDistribution::normal(mean, std_dev))

                // let p = rng.gen_range(0.0..=0.5);
                // let mag = rng.gen_range(0.0..=1000.0);
                // Some(CostDistribution::bernoulli(p, mag))
                let p = (0..=10).map(|i| i as f64 * 0.1).choose(rng).unwrap();
                let mag = 1000.0;

                if rng.gen_bool(portion_bernoulli) {
                    Some(CostDistribution::bernoulli(p, mag))
                } else {
                    let mean = p * mag;
                    let std_dev = (p * (1.0 - p)).sqrt() * mag;
                    Some(CostDistribution::normal(mean, std_dev))
                }
            },
            children: if depth < max_depth {
                (0..n_actions)
                    .map(|_| {
                        Self::inner_new(depth + 1, max_depth, n_actions, portion_bernoulli, rng)
                    })
                    .collect()
            } else {
                Vec::new()
            },
            depth,
        }
    }

    fn new(max_depth: u32, n_actions: u32, portion_bernoulli: f64, rng: &mut StdRng) -> Self {
        Self::inner_new(0, max_depth, n_actions, portion_bernoulli, rng)
    }
}

#[derive(Clone)]
struct Simulator<'a> {
    scenario: &'a ProblemScenario,
    cost: f64,
}

impl<'a> Simulator<'a> {
    fn new(scenario: &'a ProblemScenario) -> Self {
        Self {
            scenario,
            cost: 0.0,
        }
    }

    fn take_step(&mut self, policy: u32, rng: &mut StdRng) {
        let child = self
            .scenario
            .children
            .get(policy as usize)
            .expect("only take search_depth steps");
        // .expect("only take search_depth steps");
        let dist = child.distribution.as_ref().expect("not root-level node");
        let cost = dist.sample(rng); //.max(0.0).min(2.0 * dist.mean());
        self.cost += cost;

        self.scenario = child;
    }
}

#[derive(Clone)]
struct MctsNode<'a> {
    params: &'a Parameters,
    policy_choices: &'a [u32],

    policy: Option<u32>,
    depth: u32,
    n_trials: usize,
    expected_cost: Option<f64>,
    intermediate_costs: Vec<f64>,
    marginal_costs: Vec<f64>,

    sub_nodes: Option<Vec<MctsNode<'a>>>,
    costs: Vec<f64>,
}

impl<'a> MctsNode<'a> {
    fn variance(&self) -> f64 {
        let mean = self.mean_cost();
        self.costs.iter().map(|&c| (c - mean).powi(2)).sum::<f64>() / self.costs.len() as f64
    }

    fn min_child_expected_cost(&self) -> Option<f64> {
        self.sub_nodes.as_ref().and_then(|n| {
            n.iter()
                .filter_map(|n| n.expected_cost)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
        })
    }

    fn mean_cost(&self) -> f64 {
        self.costs.iter().copied().sum::<f64>() / self.costs.len() as f64
    }

    fn intermediate_cost(&self) -> f64 {
        if self.intermediate_costs.is_empty() {
            0.0
        } else {
            self.intermediate_costs.iter().sum::<f64>() / self.intermediate_costs.len() as f64
        }
    }

    fn marginal_cost(&self) -> f64 {
        if self.marginal_costs.is_empty() {
            0.0
        } else {
            self.marginal_costs.iter().sum::<f64>() / self.marginal_costs.len() as f64
        }
    }
}

fn find_and_run_trial(node: &mut MctsNode, sim: &mut Simulator, rng: &mut StdRng) -> f64 {
    let params = node.params;

    if let Some(policy) = node.policy.as_ref() {
        let prev_cost = sim.cost;
        sim.take_step(*policy, rng);
        node.intermediate_costs.push(sim.cost);
        node.marginal_costs.push(sim.cost - prev_cost);
    }

    let sub_depth = node.depth + 1;

    let mut trial_final_cost = None;
    if sub_depth > params.search_depth {
        trial_final_cost = Some(sim.cost);
    } else {
        // expand node?
        if node.sub_nodes.is_none() {
            let policy_choices = node.policy_choices;

            node.sub_nodes = Some(
                policy_choices
                    .iter()
                    .map(|p| MctsNode {
                        params,
                        policy_choices,
                        policy: Some(p.clone()),
                        depth: sub_depth,
                        n_trials: 0,
                        expected_cost: None,
                        intermediate_costs: Vec::new(),
                        marginal_costs: Vec::new(),
                        sub_nodes: None,
                        costs: Vec::new(),
                    })
                    .collect(),
            );
        }

        let sub_nodes = node.sub_nodes.as_mut().unwrap();

        // choose a node to recurse down into! First, try keeping the policy the same
        let mut has_run_trial = false;

        // then choose any unexplored branch
        if !has_run_trial {
            let unexplored = sub_nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.n_trials == 0)
                .map(|(i, _)| i)
                .collect_vec();
            if unexplored.len() > 0 {
                let sub_node_i = *unexplored.choose(rng).unwrap();
                trial_final_cost = Some(find_and_run_trial(&mut sub_nodes[sub_node_i], sim, rng));
                has_run_trial = true;
            }
        }

        // Everything has been explored at least once: UCB time!
        if !has_run_trial {
            let total_n = node.n_trials as f64;
            let ln_t = total_n.ln();
            let (_best_ucb, chosen_i) = sub_nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    let mean_cost = node.expected_cost.unwrap();
                    let ln_t_over_n = ln_t / node.n_trials as f64;

                    let upper_bound = match params.selection_mode {
                        ChildSelectionMode::UCB => {
                            let upper_margin = params.ucb_const * ln_t_over_n.sqrt();
                            mean_cost + upper_margin
                        }
                        ChildSelectionMode::UCBV => {
                            let variance = node.variance();
                            let upper_margin = params.ucb_const
                                * (params.ucbv_const * (variance * ln_t_over_n).sqrt()
                                    + ln_t_over_n);
                            mean_cost + upper_margin
                        }
                        ChildSelectionMode::UCBd => {
                            let n = node.n_trials as f64;
                            let a = (1.0 + n) / (n * n);
                            let b = (total_n * (1.0 + n).sqrt() / params.ucbd_const).ln();
                            let upper_margin = params.ucb_const * (a * (1.0 + 2.0 * b)).sqrt();
                            if !upper_margin.is_finite() {
                                eprintln_f!("{a=} {b=} {upper_margin=} {n=} {total_n=}");
                                panic!();
                            }
                            mean_cost + upper_margin
                        }
                        ChildSelectionMode::KLUCB => {
                            let scaled_mean =
                                (1.0 - mean_cost / params.klucb_max_cost).min(1.0).max(0.0);

                            // assert!(
                            //     scaled_mean >= 0.0,
                            //     "got mean cost: {:8.2} and costs: {:8.2?}",
                            //     mean_cost,
                            //     node.costs
                            // );
                            // assert!(
                            //     scaled_mean <= 1.0,
                            //     "got mean cost: {:8.2} and costs: {:8.2?}",
                            //     mean_cost,
                            //     node.costs
                            // );

                            let index =
                                -klucb_bernoulli(scaled_mean, params.ucb_const.abs() * ln_t_over_n);
                            // eprintln_f!(
                            //     "{total_n=:5}, {node.n_trials=:5}, {scaled_mean=:8.2}, {index=:8.6} would have been {:8.2}",
                            //     mean_cost + params.ucb_const * ln_t_over_n.sqrt()
                            // );
                            index
                        }
                    };
                    (upper_bound, i)
                })
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            trial_final_cost = Some(find_and_run_trial(&mut sub_nodes[chosen_i], sim, rng));
        }
    }

    let trial_final_cost = trial_final_cost.unwrap();

    node.costs.push(trial_final_cost);
    node.n_trials = node.costs.len();

    let expected_cost = match params.bound_mode {
        CostBoundMode::Normal => node.min_child_expected_cost().unwrap_or(node.mean_cost()),
        CostBoundMode::LowerBound => node
            .min_child_expected_cost()
            .unwrap_or(0.0)
            .max(node.intermediate_cost()),
        CostBoundMode::Marginal => {
            node.min_child_expected_cost().unwrap_or(0.0) + node.marginal_cost()
        }
    };

    node.expected_cost = Some(expected_cost);

    trial_final_cost
}

fn print_report(scenario: &ProblemScenario, node: &MctsNode, mut true_intermediate_cost: f64) {
    if node.n_trials > 0 {
        for _ in 0..node.depth {
            eprint!("    ");
        }
        let policy = node.policy.as_ref();
        let cost = node.expected_cost.unwrap();
        let mut additional_true_cost = 0.0;
        if let Some(dist_mean) = scenario.distribution.as_ref().map(|d| d.mean()) {
            additional_true_cost = dist_mean;
            true_intermediate_cost += additional_true_cost;
        }

        let intermediate_cost = node.intermediate_cost();
        let marginal_cost = node.marginal_cost();
        let _variance = node.variance();

        eprintln_f!(
            "n_trials: {node.n_trials}, {policy=:?}, {cost=:6.1}, \
             interm = {intermediate_cost:6.1?}, marginal = {marginal_cost:6.1?}, \
             true = {additional_true_cost:6.1} ({true_intermediate_cost:6.1}), \
             {node.marginal_costs=:.2?}, \
             {node.intermediate_costs=:.2?}, \
             {node.costs=:.2?}" //,
        );
    }
    if let Some(sub_nodes) = &node.sub_nodes {
        for (policy_i, sub_node) in sub_nodes.iter().enumerate() {
            print_report(
                &scenario.children[policy_i],
                sub_node,
                true_intermediate_cost,
            );
        }
    }
}

fn true_best_child_cost(scenario: &ProblemScenario) -> (f64, f64, usize) {
    let add_cost = scenario.distribution.as_ref().map_or(0.0, |d| d.mean());

    let best_child_cost = scenario
        .children
        .iter()
        .map(|c| {
            let a = true_best_child_cost(c);
            a.0 + a.1
        })
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap_or((0, 0.0));

    (add_cost, best_child_cost.1, best_child_cost.0)
}

fn true_best_cost(scenario: &ProblemScenario, debug: bool) -> (f64, usize) {
    let (add_cost, best_child_cost, best_child_i) = true_best_child_cost(scenario);
    let total_cost = best_child_cost + add_cost;

    if debug {
        for _ in 0..scenario.depth {
            eprint!("    ");
        }
        eprintln_f!("{add_cost=:6.1} + {best_child_cost=:6.1} = {total_cost=:6.1}");

        for child in scenario.children.iter() {
            true_best_cost(child, debug);
        }
    }

    (total_cost, best_child_i)
}

fn run_with_parameters(params: Parameters) -> RunResults {
    let policies = (0..params.n_actions).collect_vec();

    let mut node = MctsNode {
        params: &params,
        policy_choices: &policies,
        policy: None,

        depth: 0,
        n_trials: 0,
        expected_cost: None,
        intermediate_costs: Vec::new(),
        marginal_costs: Vec::new(),

        sub_nodes: None,
        costs: Vec::new(),
    };

    let mut full_seed = [0; 32];
    full_seed[0..8].copy_from_slice(&params.rng_seed.to_le_bytes());
    let mut rng = StdRng::from_seed(full_seed);

    let scenario = ProblemScenario::new(
        params.search_depth,
        params.n_actions,
        params.portion_bernoulli,
        &mut rng,
    );
    let sim = Simulator::new(&scenario);

    for _ in 0..params.samples_n {
        find_and_run_trial(&mut node, &mut sim.clone(), &mut rng);
    }

    if params.print_report {
        print_report(&scenario, &node, 0.0);
    }

    let chosen_policy = node
        .sub_nodes
        .as_ref()
        .unwrap()
        .iter()
        .min_by(|a, b| {
            a.expected_cost
                .unwrap()
                .partial_cmp(&b.expected_cost.unwrap())
                .unwrap()
        })
        .unwrap()
        .policy
        .unwrap();

    let chosen_true_cost = true_best_cost(&scenario.children[chosen_policy as usize], false).0;

    let (true_best_cost, true_best_policy) = true_best_cost(&scenario, false);
    println_f!(
        "{chosen_policy=}: {node.expected_cost=:.2?}, {chosen_true_cost=:.2}, {true_best_cost=:.2}: {true_best_policy=}"
    );

    RunResults {
        chosen_cost: node.expected_cost.unwrap(),
        chosen_true_cost,
        true_best_cost,
    }
}

fn main() {
    run_parallel_scenarios();
}
