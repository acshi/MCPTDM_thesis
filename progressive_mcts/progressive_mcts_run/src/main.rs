mod arg_parameters;
mod problem_scenario;

use arg_parameters::{run_parallel_scenarios, Parameters};
#[allow(unused)]
use fstrings::{eprintln_f, format_args_f, println_f, write_f};
use itertools::Itertools;
use problem_scenario::{ProblemScenario, Simulator, SituationParticle};
use progressive_mcts::klucb::klucb_bernoulli;
use progressive_mcts::{ChildSelectionMode, CostBoundMode};
use rand::Rng;
use rand::{
    prelude::{SliceRandom, StdRng},
    SeedableRng,
};
use rolling_stats::Stats;

#[derive(Clone, Copy, Debug)]
struct RunResults {
    chosen_cost: f64,
    chosen_true_cost: f64,
    true_best_cost: f64,
    sum_repeated: usize,
    max_repeated: usize,
    repeated_cost_avg: f64,
}

impl std::fmt::Display for RunResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        write_f!(
            f,
            "{s.chosen_cost:7.2} {s.chosen_true_cost:7.2} {s.true_best_cost:7.2} {s.sum_repeated} {s.max_repeated} {s.repeated_cost_avg:7.3}"
        )
    }
}

#[derive(Clone, Debug)]
struct CostSet<T = ()> {
    throwout_extreme_z: f64,
    costs: Vec<(f64, T)>,
    raw_stats: Stats<f64>,
    stats: Stats<f64>,
}

impl<T: Clone> CostSet<T> {
    fn new(throwout_extreme_z: f64) -> Self {
        Self {
            throwout_extreme_z,
            costs: Vec::new(),
            raw_stats: Stats::new(),
            stats: Stats::new(),
        }
    }

    fn push(&mut self, cost: (f64, T)) {
        let cost_val = cost.0;

        self.costs.push(cost);

        self.raw_stats.update(cost_val);
        if self.throwout_extreme_z >= 1000.0 {
            self.stats = self.raw_stats.clone();
            return;
        }

        if self.costs.len() == 1 {
            self.stats.update(cost_val);
        } else {
            let z = (cost_val - self.raw_stats.mean) / self.raw_stats.std_dev;
            if z.abs() < self.throwout_extreme_z {
                self.stats.update(cost_val);
            }
        }
    }

    fn mean(&self) -> f64 {
        self.stats.mean
    }

    fn std_dev(&self) -> f64 {
        if self.stats.std_dev.is_finite() {
            self.stats.std_dev
        } else {
            1e12
        }
    }

    fn len(&self) -> usize {
        self.costs.len()
    }

    fn is_empty(&self) -> bool {
        self.costs.is_empty()
    }

    fn iter(&self) -> impl Iterator<Item = &(f64, T)> {
        self.costs.iter()
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
    expected_cost_std_dev: Option<f64>,
    intermediate_costs: CostSet,
    marginal_costs: CostSet,

    seen_particles: Vec<bool>,
    particles_repeated: usize,
    repeated_particle_costs: Vec<f64>,

    sub_nodes: Option<Vec<MctsNode<'a>>>,
    costs: CostSet<SituationParticle>,
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

    fn std_dev_of_mean(&self) -> f64 {
        self.costs.std_dev() / (self.costs.len() as f64).sqrt()
    }

    fn min_child_expected_cost_and_std_dev(&self) -> Option<(f64, f64)> {
        self.sub_nodes.as_ref().and_then(|nodes| {
            nodes
                .iter()
                .filter_map(|n| Some((n.expected_cost?, n.expected_cost_std_dev?)))
                .min_by(|a, b| a.partial_cmp(b).unwrap())
        })
    }

    fn mean_cost(&self) -> f64 {
        self.costs.mean()
    }

    fn intermediate_cost(&self) -> f64 {
        if self.intermediate_costs.is_empty() {
            0.0
        } else {
            self.intermediate_costs.mean()
        }
    }

    fn intermediate_cost_std_dev(&self) -> f64 {
        if self.intermediate_costs.is_empty() {
            0.0
        } else {
            self.intermediate_costs.std_dev() / (self.intermediate_costs.len() as f64).sqrt()
        }
    }

    fn marginal_cost(&self) -> f64 {
        if self.marginal_costs.is_empty() {
            0.0
        } else {
            self.marginal_costs.mean()
        }
    }

    fn marginal_cost_std_dev(&self) -> f64 {
        if self.marginal_costs.is_empty() {
            0.0
        } else {
            self.marginal_costs.std_dev() / (self.marginal_costs.len() as f64).sqrt()
        }
    }

    fn compute_selection_index(&self, total_n: f64, ln_t: f64) -> f64 {
        let params = self.params;
        let mean_cost = self.expected_cost.unwrap();
        let n = self.n_trials as f64;
        let ln_t_over_n = ln_t / n;
        let index = match params.selection_mode {
            ChildSelectionMode::UCB => {
                let upper_margin = params.ucb_const * ln_t_over_n.sqrt();
                mean_cost + upper_margin
            }
            ChildSelectionMode::UCBV => {
                let variance = self.variance();
                let upper_margin = params.ucb_const
                    * (params.ucbv_const * (variance * ln_t_over_n).sqrt() + ln_t_over_n);
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
                let scaled_mean = (1.0 - mean_cost / params.klucb_max_cost).min(1.0).max(0.0);
                let index = -klucb_bernoulli(scaled_mean, params.ucb_const.abs() * ln_t_over_n);
                index
            }
            ChildSelectionMode::KLUCBP => {
                let scaled_mean = (1.0 - mean_cost / params.klucb_max_cost).min(1.0).max(0.0);
                let index =
                    -klucb_bernoulli(scaled_mean, params.ucb_const.abs() * (total_n / n).ln() / n);
                index
            }
            ChildSelectionMode::Uniform => n,
            ChildSelectionMode::Random => unimplemented!("No index for Random ChildSelectionMode"),
        };
        index
    }

    fn choice_confidence_interval(&self, n_completed: usize) -> f64 {
        if self.sub_nodes.is_none() {
            // not expanded yet... no confidence!
            return 1000.0;
        }

        let mut best_i = None;
        let mut best_sub_node = None;
        let mut best_mean_cost = f64::MAX;
        for (i, sub_node) in self.sub_nodes.as_ref().unwrap().iter().enumerate() {
            if sub_node.mean_cost() < best_mean_cost {
                best_mean_cost = sub_node.mean_cost();
                best_i = Some(i);
                best_sub_node = Some(sub_node);
            }
        }
        let best_i = best_i.unwrap();
        let best_sub_node = best_sub_node.unwrap();

        let remaining_samples_n = self.params.samples_n - n_completed;

        let mut closest_z_gap = f64::MAX;
        for (i, sub_node) in self.sub_nodes.as_ref().unwrap().iter().enumerate() {
            if best_i == i {
                continue;
            }

            let mean_diff = sub_node.mean_cost() - best_sub_node.mean_cost();
            let expected_cost_std_dev = best_sub_node.expected_cost_std_dev.unwrap_or(10000.0);
            let z_gap = if self.params.correct_future_std_dev_mean {
                mean_diff / expected_cost_std_dev
                    * ((best_sub_node.costs.len() + remaining_samples_n) as f64).sqrt()
            } else {
                mean_diff / expected_cost_std_dev
            };
            // eprintln_f!("{best_i=}, {i=}, {mean_diff=:.0}, {z_gap=:.1}");
            closest_z_gap = closest_z_gap.min(z_gap);

            let expected_cost_std_dev = sub_node.expected_cost_std_dev.unwrap_or(10000.0);
            let z_gap = if self.params.correct_future_std_dev_mean {
                mean_diff / expected_cost_std_dev
                    * ((sub_node.costs.len() + remaining_samples_n) as f64).sqrt()
            } else {
                mean_diff / expected_cost_std_dev
            };
            // eprintln_f!("{best_i=}, {i=}, {mean_diff=:.0}, {z_gap=:.1}");
            closest_z_gap = closest_z_gap.min(z_gap);
        }

        closest_z_gap
    }

    fn update_expected_cost(&mut self, bound_mode: CostBoundMode) {
        let (expected_cost, std_dev) = match bound_mode {
            CostBoundMode::Normal => (self.mean_cost(), self.std_dev_of_mean()),
            CostBoundMode::BubbleBest => self
                .min_child_expected_cost_and_std_dev()
                .unwrap_or((self.mean_cost(), self.std_dev_of_mean())),
            CostBoundMode::LowerBound => {
                let (mut expected_cost, mut std_dev) = self
                    .min_child_expected_cost_and_std_dev()
                    .unwrap_or((0.0, 0.0));
                let intermediate_cost = self.intermediate_cost();
                if intermediate_cost > expected_cost {
                    expected_cost = intermediate_cost;
                    std_dev = self.intermediate_cost_std_dev();
                }
                (expected_cost, std_dev)
            }
            CostBoundMode::Marginal => {
                let (mut expected_cost, mut std_dev) = self
                    .min_child_expected_cost_and_std_dev()
                    .unwrap_or((0.0, 0.0));
                expected_cost += self.marginal_cost();
                std_dev += self.marginal_cost_std_dev();
                (expected_cost, std_dev)
            }
            CostBoundMode::Same => panic!("Bound mode cannot be 'Same'"),
        };
        self.expected_cost = Some(expected_cost);
        self.expected_cost_std_dev = Some(std_dev);
    }
}

fn find_and_run_trial(
    node: &mut MctsNode,
    sim: &mut Simulator,
    rng: &mut StdRng,
    n_completed: usize,
    choice_confidence_interval: Option<f64>,
) -> f64 {
    let params = node.params;

    if let Some(policy) = node.policy.as_ref() {
        let prev_cost = sim.cost;
        sim.take_step(*policy, rng);
        node.intermediate_costs.push((sim.cost, ()));
        node.marginal_costs.push((sim.cost - prev_cost, ()));
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
                        expected_cost_std_dev: None,
                        intermediate_costs: CostSet::new(params.throwout_extreme_costs_z),
                        marginal_costs: CostSet::new(params.throwout_extreme_costs_z),
                        seen_particles: vec![false; params.samples_n],
                        particles_repeated: 0,
                        repeated_particle_costs: Vec::new(),
                        sub_nodes: None,
                        costs: CostSet::new(params.throwout_extreme_costs_z),
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
                possibly_modify_particle(
                    &mut node.costs,
                    &mut sub_nodes[sub_node_i],
                    sim,
                    n_completed,
                    choice_confidence_interval,
                );
                trial_final_cost = Some(find_and_run_trial(
                    &mut sub_nodes[sub_node_i],
                    sim,
                    rng,
                    n_completed,
                    choice_confidence_interval,
                ));
                has_run_trial = true;
            }
        }

        // Everything has been explored at least once: UCB time!
        if !has_run_trial {
            let chosen_i = if params.selection_mode == ChildSelectionMode::Random {
                rng.gen_range(0..sub_nodes.len())
            } else {
                let total_n = node.n_trials as f64;
                let ln_t = total_n.ln();
                let (_best_ucb, chosen_i) = sub_nodes
                    .iter()
                    .enumerate()
                    .map(|(i, node)| {
                        let index = node.compute_selection_index(total_n, ln_t);
                        (index, i)
                    })
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                chosen_i
            };

            possibly_modify_particle(
                &mut node.costs,
                &mut sub_nodes[chosen_i],
                sim,
                n_completed,
                choice_confidence_interval,
            );
            trial_final_cost = Some(find_and_run_trial(
                &mut sub_nodes[chosen_i],
                sim,
                rng,
                n_completed,
                choice_confidence_interval,
            ));
        }
    }

    let trial_final_cost = trial_final_cost.unwrap();

    node.costs.push((trial_final_cost, sim.particle));
    node.seen_particles[sim.particle.id] = true;
    node.n_trials = node.costs.len();

    node.update_expected_cost(params.bound_mode);

    trial_final_cost
}

fn possibly_modify_particle(
    costs: &mut CostSet<SituationParticle>,
    node: &mut MctsNode,
    sim: &mut Simulator,
    n_completed: usize,
    choice_confidence_interval: Option<f64>,
) {
    if sim.depth != 0 && !node.params.repeat_at_all_levels {
        return;
    }

    let finished_portion = n_completed as f64 / node.params.samples_n as f64;
    if finished_portion < node.params.consider_repeats_after_portion {
        return;
    }

    if let Some(choice_confidence_interval) = choice_confidence_interval {
        if choice_confidence_interval >= node.params.repeat_confidence_interval {
            return;
        }
    }

    let mut z = node.params.prioritize_worst_particles_z;
    let repeat_const = node.params.repeat_const;
    if z >= 1000.0 && repeat_const >= 0.0 {
        // using repeat const and NOT z, so to disable z, we set to -1000
        z = -1000.0;
    }

    if z >= 1000.0 {
        // take this high z value to mean don't prioritize like this!
        return;
    }

    let mean = costs.mean();
    let std_dev = costs.std_dev();

    if repeat_const >= 0.0 {
        let repeat_n = (repeat_const
            * node.params.n_actions.pow(node.params.search_depth - 1) as f64
            / (node.params.samples_n as f64)) as usize;

        if node.particles_repeated >= repeat_n {
            return;
        }
    }

    if let Some((c, particle)) = costs
        .iter()
        .filter(|(c, particle)| !node.seen_particles[particle.id] && *c - mean >= std_dev * z)
        .max_by(|a, b| match node.params.repeat_particle_sign {
            -1 => b.0.partial_cmp(&a.0).unwrap(),
            1 => a.0.partial_cmp(&b.0).unwrap(),
            0 => {
                if node.particles_repeated % 2 == 0 {
                    a.0.partial_cmp(&b.0).unwrap()
                } else {
                    b.0.partial_cmp(&a.0).unwrap()
                }
            }
            _ => panic!("repeat_particle_sign must be -1, 1, or 0"),
        })
    {
        sim.particle = *particle;
        node.particles_repeated += 1;
        let z_score = (*c - costs.mean()) / costs.std_dev();
        if z_score.is_finite() {
            node.repeated_particle_costs.push(z_score);
        }
        eprintln_f!(
            "{n_completed}: Replaying particle {sim.particle.id:3} w/ z {z_score:.2}, {choice_confidence_interval:.2?}"
        );
    }
}

fn print_report(
    scenario: &ProblemScenario,
    node: &MctsNode,
    parent_n_trials: f64,
    mut true_intermediate_cost: f64,
) {
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

        let _costs_only = node.costs.iter().map(|(c, _)| *c).collect_vec();

        let index = node.compute_selection_index(parent_n_trials, parent_n_trials.ln());

        //  interm = {_intermediate_cost:6.1?}, \
        //  {node.intermediate_costs=:.2?}, \
        eprintln_f!(
            "n_trials: {node.n_trials}, {policy=:?}, {cost=:6.1}, \
             {index=:.8}, \
             marginal = {marginal_cost:6.1?}, \
             true = {additional_true_cost:6.1} ({true_intermediate_cost:6.1}), \
             {node.marginal_costs=:.2?}, \
             "
             //  {_costs_only=:.2?}, \
             //  {node.costs=:.2?}" //,
        );
    }
    if let Some(sub_nodes) = &node.sub_nodes {
        for (policy_i, sub_node) in sub_nodes.iter().enumerate() {
            print_report(
                &scenario.children[policy_i],
                sub_node,
                node.n_trials as f64,
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

fn set_final_choice_expected_values(params: &Parameters, node: &mut MctsNode) {
    if let Some(sub_nodes) = &mut node.sub_nodes {
        for sub_node in sub_nodes.iter_mut() {
            set_final_choice_expected_values(params, sub_node);
        }
    }

    if node.n_trials == 0 {
        return;
    }

    let final_choice_mode = if params.final_choice_mode == CostBoundMode::Same {
        params.bound_mode
    } else {
        params.final_choice_mode
    };

    node.update_expected_cost(final_choice_mode);
}

fn get_best_policy(node: &MctsNode) -> u32 {
    let chosen_policy = node
        .sub_nodes
        .as_ref()
        .unwrap()
        .iter()
        .min_by(|a, b| {
            let cost_a = a.expected_cost.unwrap_or(f64::MAX);
            let cost_b = b.expected_cost.unwrap_or(f64::MAX);
            cost_a.partial_cmp(&cost_b).unwrap()
        })
        .unwrap()
        .policy
        .unwrap();
    chosen_policy
}

fn print_variance_report(node: &MctsNode) {
    eprintln!(
        "Overall n: {:4}, expected cost: {:5.0} and std dev: {:5.0}",
        node.costs.len(),
        node.expected_cost.unwrap(),
        node.expected_cost_std_dev.unwrap(),
    );
    for (i, sub_node) in node.sub_nodes.as_ref().unwrap().iter().enumerate() {
        eprintln!(
            "{}:      n: {:4}, expected cost: {:5.0} and std dev: {:5.0}",
            i,
            sub_node.costs.len(),
            sub_node.expected_cost.unwrap(),
            sub_node.expected_cost_std_dev.unwrap(),
        );
    }

    eprintln!(
        "Choice confidence (z-score): {:.1}",
        node.choice_confidence_interval(node.params.samples_n)
    );
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
        expected_cost_std_dev: None,
        intermediate_costs: CostSet::new(params.throwout_extreme_costs_z),
        marginal_costs: CostSet::new(params.throwout_extreme_costs_z),
        seen_particles: vec![false; params.samples_n],
        particles_repeated: 0,
        repeated_particle_costs: Vec::new(),

        sub_nodes: None,
        costs: CostSet::new(params.throwout_extreme_costs_z),
    };

    let mut full_seed = [0; 32];
    full_seed[0..8].copy_from_slice(&params.rng_seed.to_le_bytes());
    let mut rng = StdRng::from_seed(full_seed);

    let scenario = ProblemScenario::new(params.search_depth, params.n_actions, &mut rng);

    for i in 0..params.samples_n {
        let mut choice_confidence_interval = None;
        if params.repeat_confidence_interval < 1000.0 {
            choice_confidence_interval = Some(node.choice_confidence_interval(i));
        }

        find_and_run_trial(
            &mut node,
            &mut Simulator::sample(&scenario, i, &mut rng),
            &mut rng,
            i,
            choice_confidence_interval,
        );
    }

    if params.print_report {
        print_report(&scenario, &node, node.n_trials as f64, 0.0);
    }

    set_final_choice_expected_values(&params, &mut node);
    let chosen_policy = get_best_policy(&node);

    // if params.print_report {
    //     print_report(&scenario, &node, 0.0);
    // }

    let chosen_true_cost = true_best_cost(&scenario.children[chosen_policy as usize], false).0;

    let (true_best_cost, _true_best_policy) = true_best_cost(&scenario, false);

    let mut sum_repeated = 0;
    let mut max_repeated = 0;
    let mut repeated_cost_avg = 0.0;
    let mut n_repeated_cost_avg = 0;
    if params.is_single_run {
        println_f!(
        "{chosen_policy=}: {node.expected_cost=:.2?}, {chosen_true_cost=:.2}, {true_best_cost=:.2}: {_true_best_policy=}");
    }

    for (i, sub_node) in node.sub_nodes.as_ref().unwrap().iter().enumerate() {
        if params.is_single_run {
            println_f!("{i}: {sub_node.particles_repeated=}");
        }
        sum_repeated += sub_node.particles_repeated;
        max_repeated = max_repeated.max(sub_node.particles_repeated);
        repeated_cost_avg += sub_node.repeated_particle_costs.iter().sum::<f64>();
        n_repeated_cost_avg += sub_node.repeated_particle_costs.len();
    }
    if sum_repeated > 0 {
        repeated_cost_avg /= n_repeated_cost_avg as f64;
    }
    if params.is_single_run {
        println_f!("total repeated: {sum_repeated}");
        println_f!("max repeated: {max_repeated}");
        println_f!("repeated avg: {repeated_cost_avg:.3}");
        print_variance_report(&node);
    }

    RunResults {
        chosen_cost: node.expected_cost.unwrap(),
        chosen_true_cost,
        true_best_cost,
        sum_repeated,
        max_repeated,
        repeated_cost_avg,
    }
}

fn main() {
    run_parallel_scenarios();
}
