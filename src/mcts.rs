use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
};

use itertools::Itertools;
use progressive_mcts::{
    cost_set::CostSet, klucb::klucb_bernoulli, ChildSelectionMode, CostBoundMode,
};
use rand::prelude::{SliceRandom, StdRng};

use crate::{
    arg_parameters::{MctsParameters, Parameters},
    cost::Cost,
    mpdm::make_policy_choices,
    road::{Particle, Road},
    road_set::RoadSet,
    road_set_for_scenario,
    side_policies::{SidePolicy, SidePolicyTrait},
};

fn compute_selection_index(
    mctsp: &MctsParameters,
    total_n: f64,
    ln_total_n: f64,
    n_trials: usize,
    cost: f64,
    mode: ChildSelectionMode,
) -> Option<f64> {
    if n_trials == 0 {
        return None;
    }

    let mean_cost = cost;
    let n = n_trials as f64;
    let ln_t_over_n = ln_total_n / n;
    let index = match mode {
        ChildSelectionMode::UCB => {
            let upper_margin = mctsp.ucb_const * ln_t_over_n.sqrt();
            assert!(upper_margin.is_finite(), "{}", n);
            mean_cost + upper_margin
        }
        ChildSelectionMode::KLUCB => {
            let scaled_mean = (1.0 - mean_cost / mctsp.klucb_max_cost).min(1.0).max(0.0);
            -klucb_bernoulli(scaled_mean, mctsp.ucb_const.abs() * ln_t_over_n)
        }
        ChildSelectionMode::KLUCBP => {
            let scaled_mean = (1.0 - mean_cost / mctsp.klucb_max_cost).min(1.0).max(0.0);
            -klucb_bernoulli(scaled_mean, mctsp.ucb_const.abs() * (total_n / n).ln() / n)
        }
        ChildSelectionMode::Uniform => n,
        ChildSelectionMode::Random => unimplemented!("No index for Random ChildSelectionMode"),
        _ => unimplemented!(),
    };
    Some(index)
}

#[derive(Clone)]
struct MctsNode<'a> {
    params: &'a Parameters,
    policy_choices: &'a [SidePolicy],
    policy: Option<SidePolicy>,
    traces: Vec<rvx::Shape>,

    depth: u32,
    n_trials: usize,
    expected_cost: Option<Cost>,

    costs: Vec<(Cost, Particle)>,
    intermediate_costs: Vec<Cost>,
    marginal_costs: CostSet<f64, Cost>,

    n_particles_repeated: usize,

    sub_nodes: Option<Vec<MctsNode<'a>>>,
}

impl<'a> std::fmt::Debug for MctsNode<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        write_f!(
            f,
            "MctsNode ({s.depth=}, {s.n_trials=}, {s.expected_cost=:?}, "
        )?;
        write_f!(
            f,
            "#intermediate costs = {}, #marginal costs = {}, ",
            s.intermediate_costs.len(),
            s.marginal_costs.len()
        )
    }
}

impl<'a> MctsNode<'a> {
    fn new(
        params: &'a Parameters,
        policy_choices: &'a [SidePolicy],
        policy: Option<SidePolicy>,
        depth: u32,
    ) -> Self {
        Self {
            params,
            policy_choices,
            policy,
            traces: Vec::new(),
            depth,
            n_trials: 0,
            expected_cost: None,
            costs: Vec::new(),
            intermediate_costs: Vec::new(),
            marginal_costs: CostSet::new(1000.0, params.mcts.preload_zeros),
            n_particles_repeated: 0,
            sub_nodes: None,
        }
    }

    fn min_child_expected_cost(&self) -> Option<Cost> {
        self.sub_nodes.as_ref().and_then(|sub_nodes| {
            sub_nodes
                .iter()
                .filter_map(|n| n.expected_cost)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
        })
    }

    fn min_child(&self) -> Option<&MctsNode<'a>> {
        self.sub_nodes
            .as_ref()
            .and_then(|sub_nodes| {
                sub_nodes
                    .iter()
                    .filter_map(|n| Some((n, n.expected_cost?)))
                    .min_by(|a, b| (a.1).partial_cmp(&b.1).unwrap())
            })
            .map(|(n, _)| n)
    }

    fn mean_cost(&self) -> Cost {
        self.costs.iter().map(|(c, _)| *c).sum::<Cost>() / self.costs.len() as f64
    }

    fn intermediate_cost(&self) -> Cost {
        if self.intermediate_costs.is_empty() {
            Cost::ZERO
        } else {
            self.intermediate_costs.iter().copied().sum::<Cost>()
                / self.intermediate_costs.len() as f64
        }
    }

    fn marginal_cost(&self) -> Cost {
        if self.marginal_costs.is_empty() {
            Cost::ZERO
        } else {
            self.marginal_costs.iter().map(|(_, c)| *c).sum::<Cost>()
                / self.marginal_costs.len() as f64
        }
    }

    fn marginal_cost_std_dev(&self) -> f64 {
        if self.marginal_costs.is_empty() {
            0.0
        } else if self.marginal_costs.len() == 1 {
            self.params.mcts.zero_mean_prior_std_dev * self.params.mcts.unknown_prior_std_dev_scalar
        } else {
            self.marginal_costs.std_dev() / (self.marginal_costs.len() as f64).sqrt()
        }
    }

    fn confidence_interval(mean1: f64, std_dev1: f64, mean2: f64, std_dev2: f64) -> f64 {
        let mean_diff = (mean1 - mean2).abs();
        let z_gap1 = mean_diff / std_dev1;
        let z_gap2 = mean_diff / std_dev2;
        z_gap1.min(z_gap2)
    }

    fn marginal_cost_confidence_interval(&self) -> f64 {
        let mut sorted_sub_nodes = self
            .sub_nodes
            .as_ref()
            .unwrap()
            .iter()
            .filter(|n| !n.marginal_costs.is_empty())
            .collect_vec();
        sorted_sub_nodes.sort_by(|a, b| a.marginal_cost().partial_cmp(&b.marginal_cost()).unwrap());

        if sorted_sub_nodes.len() < 2 {
            // not expanded enough yet for a comparison, so no confidence!
            return 0.0;
        }

        let best_sub_node = sorted_sub_nodes[0];
        let second_sub_node = sorted_sub_nodes[1];

        Self::confidence_interval(
            best_sub_node.marginal_cost().total(),
            best_sub_node.marginal_cost_std_dev(),
            second_sub_node.marginal_cost().total(),
            second_sub_node.marginal_cost_std_dev(),
        )
    }

    fn update_expected_cost(&mut self) {
        let mcts = &self.params.mcts;

        let expected_cost = match mcts.bound_mode {
            CostBoundMode::Normal => self.mean_cost(),
            CostBoundMode::BubbleBest => self
                .min_child_expected_cost()
                .unwrap_or_else(|| self.mean_cost()),
            CostBoundMode::LowerBound => self
                .min_child_expected_cost()
                .unwrap_or(Cost::ZERO)
                .max(&self.intermediate_cost()),
            CostBoundMode::Marginal => {
                let mut marginal_cost = self.marginal_cost();
                if self.marginal_costs.len() == 1 {
                    marginal_cost = marginal_cost * mcts.single_trial_discount_factor;
                }

                self.min_child_expected_cost().unwrap_or(Cost::ZERO) + marginal_cost
            }
            CostBoundMode::MarginalPrior => {
                let current_mean = self.marginal_cost();
                let current_std_dev = self.marginal_cost_std_dev();
                let (mean, _std_dev) = gaussian_update(
                    Cost::ZERO,
                    mcts.zero_mean_prior_std_dev.powi(2),
                    current_mean,
                    current_std_dev.powi(2),
                );

                // eprintln_f!("n: {}, {self.params.mcts.zero_mean_prior_std_dev=:.2}, {current_std_dev=:.2}, original: {:.2}, corrected: {:.2}",
                //             self.marginal_costs.len(), current_mean.total(), mean.total());

                self.min_child_expected_cost().unwrap_or(Cost::ZERO) + mean
            }
            CostBoundMode::Same => unimplemented!(),
        };

        self.expected_cost = Some(expected_cost);
    }

    fn get_or_expand_sub_nodes_mut(&mut self) -> &mut Vec<MctsNode<'a>> {
        let params = self.params;

        if self.sub_nodes.is_none() {
            let policy_choices = self.policy_choices;

            self.sub_nodes = Some(
                policy_choices
                    .iter()
                    .map(|p| Self::new(params, policy_choices, Some(p.clone()), self.depth + 1))
                    .collect(),
            );
        }

        self.sub_nodes.as_mut().unwrap()
    }

    fn get_or_expand_sub_nodes(&mut self) -> &Vec<MctsNode<'a>> {
        self.get_or_expand_sub_nodes_mut()
    }

    fn compute_expected_cost_index(&self, total_n: f64, ln_total_n: f64) -> Option<f64> {
        compute_selection_index(
            &self.params.mcts,
            total_n,
            ln_total_n,
            self.costs.len(),
            self.expected_cost.unwrap().total(),
            self.params.mcts.selection_mode,
        )
    }

    fn compute_marginal_cost_index(&self, total_n: f64, ln_total_n: f64) -> Option<f64> {
        compute_selection_index(
            &self.params.mcts,
            total_n,
            ln_total_n,
            self.marginal_costs.len(),
            self.marginal_cost().total(),
            self.params.mcts.selection_mode,
        )
    }

    fn min_child_marginal_index_mut(&mut self) -> Option<&mut MctsNode<'a>> {
        let total_n = self
            .get_or_expand_sub_nodes()
            .iter()
            .map(|a| a.marginal_costs.len())
            .sum::<usize>() as f64;
        let ln_total_n = total_n.ln();

        // if self.params.is_single_run {
        //     for (i, sub_node) in self.get_or_expand_sub_nodes().iter().enumerate() {
        //         eprintln_f!(
        //             "{i}: {:.2}",
        //             sub_node
        //                 .compute_marginal_cost_index(total_n, ln_total_n)
        //                 .unwrap_or(99999.9)
        //         )
        //     }
        // }

        self.sub_nodes.as_mut().and_then(|nodes| {
            nodes
                .iter_mut()
                .map(|a| {
                    (
                        a.compute_marginal_cost_index(total_n, ln_total_n)
                            .unwrap_or(-f64::MAX),
                        a,
                    )
                })
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .map(|a| a.1)
        })
    }

    fn get_best_policy_by_cost(&self) -> Option<&SidePolicy> {
        let chosen_policy = self
            .sub_nodes
            .as_ref()
            .unwrap()
            .iter()
            .min_by(|a, b| {
                let cost_a = a.expected_cost.map_or(f64::MAX, |c| c.total());
                let cost_b = b.expected_cost.map_or(f64::MAX, |c| c.total());
                cost_a.partial_cmp(&cost_b).unwrap()
            })?
            .policy
            .as_ref();
        chosen_policy
    }

    fn get_best_policy_by_visits(&self) -> Option<&SidePolicy> {
        let chosen_policy = self
            .sub_nodes
            .as_ref()
            .unwrap()
            .iter()
            .max_by(|a, b| a.costs.len().cmp(&b.costs.len()))?
            .policy
            .as_ref();
        chosen_policy
    }
}

fn gaussian_update<
    F: std::ops::Add<Output = F> + std::ops::Sub<Output = F> + std::ops::Mul<f64, Output = F> + Clone,
>(
    prior_mean: F,
    prior_variance: f64,
    mean: F,
    variance: f64,
) -> (F, f64) {
    let k = prior_variance / (prior_variance + variance);
    let new_mean = prior_mean.clone() + (mean - prior_mean) * k;
    let new_variance = prior_variance * (1.0 - k);
    (new_mean, new_variance)
}

fn possibly_modify_particle(costs: &mut [(Cost, Particle)], node: &mut MctsNode, road: &mut Road) {
    if node.depth > 1 {
        return;
    }

    let mctsp = &node.params.mcts;

    if mctsp.repeat_const >= 0.0 {
        let repeat_n = (mctsp.repeat_const / (mctsp.samples_n as f64)) as usize;
        if node.n_particles_repeated >= repeat_n {
            return;
        }
    }

    let z = mctsp.prioritize_worst_particles_z;
    if z >= 1000.0 {
        // take this high value to mean don't prioritize like this!
        return;
    }

    // let orig_costs = costs.iter().map(|(c, _)| *c).collect_vec();

    let mean = costs.iter().map(|(c, _)| *c).sum::<Cost>().total() / costs.len() as f64;
    let std_dev = (costs
        .iter()
        .map(|(c, _)| (c.total() - mean).powi(2))
        .sum::<f64>()
        / costs.len() as f64)
        .sqrt();

    // sort descending by cost, then particle
    costs.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // up to samples_n possible particles
    let mut node_seen_particles = vec![false; mctsp.samples_n];
    for (_, particle) in node.costs.iter() {
        if particle.id >= node_seen_particles.len() {
            node_seen_particles.resize(particle.id + 1, false);
        }
        node_seen_particles[particle.id] = true;
    }

    for (_c, particle) in costs
        .iter()
        .take_while(|(c, _)| c.total() - mean >= std_dev * z)
    {
        if particle.id >= node_seen_particles.len() || !node_seen_particles[particle.id] {
            for (car, policy) in road.cars.iter_mut().zip(&particle.policies).skip(1) {
                car.side_policy = Some(policy.clone());
            }
            road.sample_id = Some(particle.id);
            road.save_particle();
            node.n_particles_repeated += 1;
            // eprintln!(
            //     "{}: Replaying particle {} w/ c {:.2}, mean {:.2}, std_dev {:.2}: {:?}",
            //     node.depth,
            //     particle.id,
            //     _c.total(),
            //     mean,
            //     std_dev,
            //     node.policy,
            //     // node.costs.iter().map(|(c, p)| (c.total(), p)).collect_vec(),
            // );
            return;
        }
    }
}

fn run_step<'a>(node: &mut MctsNode<'a>, road: &mut Road) -> Option<Cost> {
    let mcts = &node.params.mcts;

    if let Some(ref policy) = node.policy {
        road.set_ego_policy(policy.clone());
        // road.reset_car_traces();
        if node.depth < 4 {
            road.reset_car_traces();
        } else {
            road.disable_car_traces();
        }
        let prev_cost = road.cost;
        road.take_update_steps(mcts.layer_t, mcts.dt);
        node.intermediate_costs.push(road.cost);
        let marginal_cost = road.cost - prev_cost;
        node.marginal_costs
            .push((marginal_cost.total(), marginal_cost));
        node.traces
            .append(&mut road.make_traces(node.depth - 1, false));

        return Some(road.cost);
    }
    None
}

fn find_and_run_trial(node: &mut MctsNode, road: &mut Road, rng: &mut StdRng) -> Cost {
    let params = node.params;
    let mcts = &params.mcts;

    run_step(node, road);

    let mut trial_final_cost = None;
    if node.depth + 1 > mcts.search_depth {
        trial_final_cost = Some(road.cost);
    } else {
        node.get_or_expand_sub_nodes();
        let sub_nodes = node.sub_nodes.as_mut().unwrap();

        // choose a node to recurse down into! First, try keeping the policy the same
        let mut has_run_trial = false;
        if mcts.prefer_same_policy {
            if let Some(ref policy) = node.policy {
                let policy_id = policy.policy_id();
                if sub_nodes[policy_id as usize].n_trials == 0 {
                    possibly_modify_particle(
                        &mut node.costs,
                        &mut sub_nodes[policy_id as usize],
                        road,
                    );
                    trial_final_cost = Some(find_and_run_trial(
                        &mut sub_nodes[policy_id as usize],
                        road,
                        rng,
                    ));
                    has_run_trial = true;
                }
            }
        }

        // then choose any unexplored branch
        if !has_run_trial {
            if mcts.choose_random_policy {
                let unexplored = sub_nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| n.n_trials == 0)
                    .map(|(i, _)| i)
                    .collect_vec();
                if !unexplored.is_empty() {
                    let sub_node_i = *unexplored.choose(rng).unwrap();
                    possibly_modify_particle(&mut node.costs, &mut sub_nodes[sub_node_i], road);
                    trial_final_cost =
                        Some(find_and_run_trial(&mut sub_nodes[sub_node_i], road, rng));
                    has_run_trial = true;
                }
            } else {
                for sub_node in sub_nodes.iter_mut() {
                    if sub_node.n_trials == 0 {
                        possibly_modify_particle(&mut node.costs, sub_node, road);
                        trial_final_cost = Some(find_and_run_trial(sub_node, road, rng));
                        has_run_trial = true;
                        break;
                    }
                }
            }
        }

        // Everything has been explored at least once: UCB time!
        if !has_run_trial {
            let total_n = node.n_trials as f64;
            let ln_total_n = (total_n).ln();
            let (_best_ucb, chosen_i) = sub_nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    let index = node.compute_expected_cost_index(total_n, ln_total_n);
                    (index, i)
                })
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            possibly_modify_particle(&mut node.costs, &mut sub_nodes[chosen_i], road);
            trial_final_cost = Some(find_and_run_trial(&mut sub_nodes[chosen_i], road, rng));
        }
    }

    let trial_final_cost = trial_final_cost.unwrap();

    node.costs
        .push((trial_final_cost, road.particle.clone().unwrap()));
    node.n_trials = node.costs.len();

    node.update_expected_cost();

    trial_final_cost
}

fn collect_traces(node: &mut MctsNode, traces: &mut Vec<rvx::Shape>) {
    traces.append(&mut node.traces);

    if let Some(sub_nodes) = node.sub_nodes.as_mut() {
        for sub_node in sub_nodes.iter_mut() {
            collect_traces(sub_node, traces);
        }
    }
}

fn print_report(node: &MctsNode) {
    if node.n_trials > 0 {
        for _ in 0..node.depth {
            eprint!("    ");
        }
        let policy_id = node.policy.as_ref().map(|p| p.policy_id());
        let expected_score = node.expected_cost.unwrap();
        let score = expected_score.total();
        eprintln_f!(
            "n_trials: {node.n_trials}, policy: {policy_id:?}, score: {score:.2}, cost: {expected_score=:.2?}"
        );
    }

    if let Some(sub_nodes) = &node.sub_nodes {
        for sub_node in sub_nodes.iter() {
            print_report(sub_node);
        }
    }
}

fn tree_exploration_report_best_path(
    f: &mut BufWriter<File>,
    node: &MctsNode,
) -> Result<(), std::io::Error> {
    let mut n = node;
    while let Some(min_child) = n.min_child() {
        write!(f, "{} ", min_child.policy.as_ref().unwrap().policy_id())?;
        n = min_child;
    }
    writeln!(f)?;
    Ok(())
}

fn tree_exploration_report(f: &mut BufWriter<File>, node: &MctsNode) -> Result<(), std::io::Error> {
    for child in node.sub_nodes.as_ref().unwrap() {
        write!(f, "{} ", child.n_trials)?;
    }
    writeln!(f)?;
    for child in node.sub_nodes.as_ref().unwrap() {
        if child.sub_nodes.is_some() {
            write!(f, "{} ", child.policy.as_ref().unwrap().policy_id())?;
            tree_exploration_report(f, child)?;
        }
    }
    Ok(())
}

fn all_mac_report(f: &mut BufWriter<File>, node: &MctsNode) -> Result<(), std::io::Error> {
    for mac in node.marginal_costs.iter() {
        writeln!(f, "{}", mac.0)?;
    }
    if let Some(ref sub_nodes) = node.sub_nodes {
        for child in sub_nodes.iter() {
            all_mac_report(f, child)?;
        }
    }
    Ok(())
}

fn bootstrap_run_trial<'a>(node: &mut MctsNode<'a>, roads: &mut RoadSet, n_completed: usize) {
    let is_single_run = !node.params.run_fast;

    // do search_depth single step trials so the total cost of a bootstrap run is the same
    // as a normal one
    for _ in 0..node.params.mcts.search_depth {
        let sub_node = node.min_child_marginal_index_mut().unwrap();
        let score = run_step(sub_node, &mut roads.pop()).unwrap();
        if is_single_run {
            eprintln!(
                "{}: Bootstrap trial: {:?} got {:.2}",
                n_completed,
                sub_node.policy.as_ref().unwrap(),
                score
            );
        }

        sub_node.update_expected_cost();
    }

    node.update_expected_cost();
}

pub fn mcts_choose_policy(
    params: &Parameters,
    true_road: &Road,
    rng: &mut StdRng,
) -> (Option<SidePolicy>, Vec<rvx::Shape>) {
    let mut params = params.clone();
    if let Some(total_forward_t) = params.mcts.total_forward_t {
        params.mcts.layer_t = total_forward_t / params.mcts.search_depth as f64;
    }
    let params = &params;

    let mut roads = road_set_for_scenario(
        params,
        true_road,
        rng,
        (params.mcts.samples_n as f64 * 1.2).ceil() as usize,
    );

    let policy_choices = make_policy_choices(params);
    let debug = true_road.debug
        && true_road.timesteps + params.debug_steps_before >= params.max_steps as usize;

    let mut node = MctsNode::new(params, &policy_choices, None, 0);
    node.get_or_expand_sub_nodes();

    let mut i = 0;
    loop {
        let marginal_confidence = node.marginal_cost_confidence_interval();
        // eprintln_f!("{i}: {marginal_confidence=:.2}");
        if params.mcts.bootstrap_confidence_z > marginal_confidence {
            bootstrap_run_trial(&mut node, &mut roads, i);
            continue;
        }

        let mut road = roads.pop();
        road.sample_id = Some(i);
        road.save_particle();
        find_and_run_trial(&mut node, &mut road, rng);

        i += 1;
        if i >= params.mcts.samples_n {
            if params.mcts.most_visited_best_cost_consistency
                && i <= params.mcts.samples_n * 12 / 10
            {
                // if we have this best policy inconsistency, do more trials to try to resolve it!
                let best_visits = node.get_best_policy_by_visits().map(|p| p.policy_id());
                let best_cost = node.get_best_policy_by_cost().map(|p| p.policy_id());
                if best_visits != best_cost {
                    // if params.is_single_run {
                    //     eprintln_f!("{best_visits:?} != {best_cost:?}");
                    // }
                    continue;
                }
            }
            break;
        }
    }

    let best_policy = node.get_best_policy_by_cost().cloned();

    let mut traces = Vec::new();
    collect_traces(&mut node, &mut traces);

    if debug && params.policy_report_debug {
        print_report(&node);
    }

    if debug && params.mcts.tree_exploration_report {
        let mut f = BufWriter::new(File::create("tree_exploration_report").unwrap());
        tree_exploration_report_best_path(&mut f, &node).unwrap();
        tree_exploration_report(&mut f, &node).unwrap();
    }

    if params.is_single_run && params.mcts.all_mac_report {
        if true_road.timesteps == 0 {
            let _ = std::fs::remove_file("all_mac_report");
        }

        let mut f = BufWriter::new(
            OpenOptions::new()
                .append(true)
                .create(true)
                .open("all_mac_report")
                .unwrap(),
        );
        all_mac_report(&mut f, &node).unwrap();
    }

    (best_policy, traces)
}
