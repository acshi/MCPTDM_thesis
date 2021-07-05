mod arg_parameters;

use arg_parameters::{run_parallel_scenarios, Parameters};
use fstrings::{eprintln_f, format_args_f, format_f, println_f, write_f};
use itertools::Itertools;
use rand::{
    prelude::{IteratorRandom, SliceRandom, StdRng},
    SeedableRng,
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
            CostBoundMode::Normal => write!(f, "normal"),
            CostBoundMode::LowerBound => write!(f, "lower_bound"),
            CostBoundMode::Marginal => write!(f, "marginal"),
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
            CostDistribution::Normal { d } => d.sample(rng),
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
    fn inner_new(depth: u32, max_depth: u32, n_actions: u32, rng: &mut StdRng) -> Self {
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
                Some(CostDistribution::bernoulli(p, mag))
            },
            children: if depth < max_depth {
                (0..n_actions)
                    .map(|_| Self::inner_new(depth + 1, max_depth, n_actions, rng))
                    .collect()
            } else {
                Vec::new()
            },
            depth,
        }
    }

    fn new(max_depth: u32, n_actions: u32, rng: &mut StdRng) -> Self {
        Self::inner_new(0, max_depth, n_actions, rng)
    }
}

#[derive(Clone)]
struct Simulator {
    scenario: ProblemScenario,
    cost: f64,
}

impl Simulator {
    fn new(scenario: &ProblemScenario) -> Self {
        Self {
            scenario: scenario.clone(),
            cost: 0.0,
        }
    }

    fn take_step(&mut self, policy: u32, rng: &mut StdRng) {
        let child = self
            .scenario
            .children
            .get(policy as usize)
            .expect("only take search_depth steps");
        let dist = child.distribution.as_ref().expect("not root-level node");
        let cost = dist.sample(rng); //.max(0.0).min(2.0 * dist.mean());
        self.cost += cost;

        self.scenario = child.clone();
    }
}

#[derive(Clone)]
enum MctsNodeInner<'a> {
    Branch {
        sub_nodes: Option<Vec<MctsNode<'a>>>,
    },
    Leaf {
        scores: Vec<f64>,
    },
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

    inner: MctsNodeInner<'a>,
}

fn find_and_run_trial(node: &mut MctsNode, sim: &mut Simulator, rng: &mut StdRng) {
    let params = node.params;

    if let Some(policy) = node.policy.as_ref() {
        let prev_cost = sim.cost;
        sim.take_step(*policy, rng);
        node.intermediate_costs.push(sim.cost);
        node.marginal_costs.push(sim.cost - prev_cost);
    }

    match &mut node.inner {
        MctsNodeInner::Branch { sub_nodes } => {
            let sub_depth = node.depth + 1;

            // expand node?
            if sub_nodes.is_none() {
                let sub_inner = if sub_depth >= params.search_depth {
                    MctsNodeInner::Leaf { scores: Vec::new() }
                } else {
                    MctsNodeInner::Branch { sub_nodes: None }
                };

                let policy_choices = node.policy_choices;

                *sub_nodes = Some(
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
                            inner: sub_inner.clone(),
                        })
                        .collect(),
                );
            }

            let sub_nodes = sub_nodes.as_mut().unwrap();

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
                    find_and_run_trial(&mut sub_nodes[sub_node_i], sim, rng);
                    has_run_trial = true;
                }
            }

            // Everything has been explored at least once: UCB time!
            if !has_run_trial {
                let ln_t = (node.n_trials as f64).ln();
                let (_best_ucb, chosen_i) = sub_nodes
                    .iter()
                    .enumerate()
                    .map(|(i, node)| {
                        let upper_margin = params.ucb_const * (ln_t / node.n_trials as f64).sqrt();
                        // eprintln!(
                        //     "n: {} expected: {:.2?} margin: {:.2}",
                        //     node.n_trials, node.expected_cost, upper_margin
                        // );
                        (node.expected_cost.unwrap() + upper_margin, i)
                    })
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();

                find_and_run_trial(&mut sub_nodes[chosen_i], sim, rng);
            }

            node.n_trials = sub_nodes.iter().map(|n| n.n_trials).sum::<usize>();

            let min_child_expected_cost = sub_nodes
                .iter()
                .filter_map(|n| n.expected_cost)
                .min_by(|a, b| a.partial_cmp(b).unwrap());

            let intermediate_cost = if node.intermediate_costs.is_empty() {
                0.0
            } else {
                node.intermediate_costs.iter().sum::<f64>() / node.intermediate_costs.len() as f64
            };

            let marginal_cost = if node.marginal_costs.is_empty() {
                0.0
            } else {
                node.marginal_costs.iter().sum::<f64>() / node.marginal_costs.len() as f64
            };

            if let Some(min_child_expected_cost) = min_child_expected_cost {
                let expected_cost = match params.bound_mode {
                    CostBoundMode::Normal => min_child_expected_cost,
                    CostBoundMode::LowerBound => {
                        // eprintln_f!("{node.expected_cost=:.2?} and {intermediate_cost=:.2}");
                        min_child_expected_cost.max(intermediate_cost)
                    }
                    CostBoundMode::Marginal => min_child_expected_cost + marginal_cost,
                };

                node.expected_cost = Some(expected_cost);
            }
        }
        MctsNodeInner::Leaf { scores } => {
            scores.push(sim.cost);
            node.n_trials = scores.len();
            node.expected_cost = Some(scores.iter().copied().sum::<f64>() / node.n_trials as f64);
        }
    }
}

fn print_report(scenario: &ProblemScenario, node: &MctsNode, mut true_intermediate_cost: f64) {
    if node.n_trials > 0 {
        for _ in 0..node.depth {
            eprint!("    ");
        }
        let policy = node.policy.as_ref();
        let cost = node.expected_cost.unwrap();
        let mean_intermediate_cost =
            node.intermediate_costs.iter().sum::<f64>() / node.intermediate_costs.len() as f64;

        let mut additional_true_cost = 0.0;
        if let Some(dist_mean) = scenario.distribution.as_ref().map(|d| d.mean()) {
            additional_true_cost = dist_mean;
            true_intermediate_cost += additional_true_cost;
        }

        eprintln_f!(
            "n_trials: {node.n_trials}, {policy=:?}, {cost=:6.1}, mean_interm = {mean_intermediate_cost:6.1?}, \
             true = {additional_true_cost:6.1} ({true_intermediate_cost:6.1}), \
             {node.intermediate_costs=:?}, {node.marginal_costs=:?}"
        );
    }

    match &node.inner {
        MctsNodeInner::Branch { sub_nodes } => {
            if let Some(sub_nodes) = sub_nodes {
                for (policy_i, sub_node) in sub_nodes.iter().enumerate() {
                    print_report(
                        &scenario.children[policy_i],
                        sub_node,
                        true_intermediate_cost,
                    );
                }
            }
        }
        MctsNodeInner::Leaf { .. } => (),
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

        inner: MctsNodeInner::Branch { sub_nodes: None },
    };

    let mut full_seed = [0; 32];
    full_seed[0..8].copy_from_slice(&params.rng_seed.to_le_bytes());
    let mut rng = StdRng::from_seed(full_seed);

    let scenario = ProblemScenario::new(params.search_depth, params.n_actions, &mut rng);
    let sim = Simulator::new(&scenario);

    for _ in 0..params.samples_n {
        find_and_run_trial(&mut node, &mut sim.clone(), &mut rng);
    }

    if false {
        print_report(&scenario, &node, 0.0);
    }

    let chosen_policy = match node.inner {
        MctsNodeInner::Branch { sub_nodes } => sub_nodes
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
            .unwrap(),
        MctsNodeInner::Leaf { scores: _ } => unreachable!(),
    };

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