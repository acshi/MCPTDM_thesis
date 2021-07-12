mod arg_parameters;
mod problem_scenario;

use arg_parameters::{run_parallel_scenarios, Parameters};
use fstrings::{eprintln_f, format_args_f, println_f, write_f};
use itertools::Itertools;
use problem_scenario::{ProblemScenario, Simulator, SituationParticle};
use progressive_mcts::klucb::klucb_bernoulli;
use progressive_mcts::{ChildSelectionMode, CostBoundMode};
use rand::{
    prelude::{SliceRandom, StdRng},
    SeedableRng,
};

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
    costs: Vec<(f64, SituationParticle)>,
}

impl<'a> MctsNode<'a> {
    fn variance(&self) -> f64 {
        let mean = self.mean_cost();
        self.costs
            .iter()
            .map(|(c, _)| (*c - mean).powi(2))
            .sum::<f64>()
            / self.costs.len() as f64
    }

    fn min_child_expected_cost(&self) -> Option<f64> {
        self.sub_nodes.as_ref().and_then(|nodes| {
            nodes
                .iter()
                .filter_map(|n| n.expected_cost)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
        })
    }

    fn mean_cost(&self) -> f64 {
        self.costs.iter().map(|(c, _)| *c).sum::<f64>() / self.costs.len() as f64
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

        // choose a node to recurse down into!
        let mut has_run_trial = false;

        // choose any unexplored branch
        if !has_run_trial {
            let unexplored = sub_nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.n_trials == 0)
                .map(|(i, _)| i)
                .collect_vec();
            if unexplored.len() > 0 {
                let sub_node_i = *unexplored.choose(rng).unwrap();
                possibly_modify_particle(&mut node.costs, &sub_nodes[sub_node_i], sim);
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
                    let n = node.n_trials as f64;
                    let ln_t_over_n = ln_t / n;

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
                            let index =
                                -klucb_bernoulli(scaled_mean, params.ucb_const.abs() * ln_t_over_n);
                            index
                        }
                        ChildSelectionMode::KLUCBP => {
                            let scaled_mean =
                                (1.0 - mean_cost / params.klucb_max_cost).min(1.0).max(0.0);
                            let index = -klucb_bernoulli(
                                scaled_mean,
                                params.ucb_const.abs() * (total_n / n).ln() / n,
                            );
                            index
                        }
                    };
                    (upper_bound, i)
                })
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            possibly_modify_particle(&mut node.costs, &sub_nodes[chosen_i], sim);
            trial_final_cost = Some(find_and_run_trial(&mut sub_nodes[chosen_i], sim, rng));
        }
    }

    let trial_final_cost = trial_final_cost.unwrap();

    node.costs.push((trial_final_cost, sim.particle));
    node.n_trials = node.costs.len();

    let expected_cost = match params.bound_mode {
        CostBoundMode::Normal => node.mean_cost(),
        CostBoundMode::BubbleBest => node.min_child_expected_cost().unwrap_or(node.mean_cost()),
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

fn possibly_modify_particle(
    costs: &[(f64, SituationParticle)],
    node: &MctsNode,
    sim: &mut Simulator,
) {
    if sim.depth != 0 {
        return;
    }

    // let orig_costs = costs.iter().map(|(c, _)| *c).collect_vec();

    let mean = costs.iter().map(|(c, _)| *c).sum::<f64>() / costs.len() as f64;
    let std_dev =
        (costs.iter().map(|(c, _)| (*c - mean).powi(2)).sum::<f64>() / costs.len() as f64).sqrt();

    let mut costs = costs.to_vec();
    // first remove duplicate particles (since we may have already replayed some)
    costs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap()
            .then_with(|| b.0.partial_cmp(&a.0).unwrap())
    });
    // will keep the first occuring of any duplicates, so we keep the highest-cost copy of each particle
    costs.dedup_by(|a, b| a.1 == b.1);

    // sort descending by cost, then particle
    costs.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let n = node.params.prioritize_worst_particles_n;
    for (_, particle) in costs.iter().take(n) {
        if node.costs.iter().all(|(_, p)| p != particle) {
            sim.particle = *particle;
            // eprintln!("(1) Replaying particle {:?}", sim.particle);
            return;
        }
    }

    let z = node.params.prioritize_worst_particles_z;
    if z >= 1000.0 {
        // take this high value to mean don't prioritize like this!
        return;
    }
    for (_c, particle) in costs.iter().take_while(|(c, _)| *c - mean >= std_dev * z) {
        if node.costs.iter().all(|(_, p)| p != particle) {
            sim.particle = *particle;
            // eprintln!(
            //     "(2) Replaying particle {:?} w/ c {}, mean {}, std_dev {}",
            //     sim.particle, _c, mean, std_dev
            // );
            return;
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
        let mut additional_true_cost = 0.0;
        if let Some(dist_mean) = scenario.distribution.as_ref().map(|d| d.mean()) {
            additional_true_cost = dist_mean;
            true_intermediate_cost += additional_true_cost;
        }

        let _intermediate_cost = node.intermediate_cost();
        let marginal_cost = node.marginal_cost();
        let _variance = node.variance();

        //  interm = {_intermediate_cost:6.1?}, \
        //  {node.intermediate_costs=:.2?}, \
        eprintln_f!(
            "n_trials: {node.n_trials}, {policy=:?}, {cost=:6.1}, \
             marginal = {marginal_cost:6.1?}, \
             true = {additional_true_cost:6.1} ({true_intermediate_cost:6.1}), \
             {node.marginal_costs=:.2?}, \
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
    let add_cost = scenario.expected_marginal_cost();

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
        params.bad_situation_p,
        params.bad_threshold_cost,
        &mut rng,
    );

    for i in 0..params.samples_n {
        find_and_run_trial(
            &mut node,
            &mut Simulator::sample(&scenario, i, params.bad_situation_p, &mut rng),
            &mut rng,
        );
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
