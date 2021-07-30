use std::{
    fs::File,
    io::{BufWriter, Write},
};

use itertools::Itertools;
use progressive_mcts::{klucb::klucb_bernoulli, ChildSelectionMode, CostBoundMode};
use rand::prelude::{SliceRandom, StdRng};

use crate::{
    arg_parameters::Parameters,
    cost::Cost,
    mpdm::make_policy_choices,
    road::{Particle, Road},
    road_set_for_scenario,
    side_policies::{SidePolicy, SidePolicyTrait},
};

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
    marginal_costs: Vec<Cost>,

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
            self.marginal_costs.iter().copied().sum::<Cost>() / self.marginal_costs.len() as f64
        }
    }
}

fn possibly_modify_particle(costs: &mut [(Cost, Particle)], node: &MctsNode, road: &mut Road) {
    if node.depth > 1 {
        return;
    }

    let z = node.params.mcts.prioritize_worst_particles_z;
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
    let mut node_seen_particles = vec![false; node.params.mcts.samples_n];
    for (_, particle) in node.costs.iter() {
        node_seen_particles[particle.id] = true;
    }

    for (_c, particle) in costs
        .iter()
        .take_while(|(c, _)| c.total() - mean >= std_dev * z)
    {
        if !node_seen_particles[particle.id] {
            for (car, policy) in road.cars.iter_mut().zip(&particle.policies).skip(1) {
                car.side_policy = Some(policy.clone());
            }
            road.sample_id = Some(particle.id);
            road.save_particle();
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

fn find_and_run_trial(node: &mut MctsNode, road: &mut Road, rng: &mut StdRng) -> Cost {
    let params = node.params;
    let mcts = &params.mcts;

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
        node.marginal_costs.push(road.cost - prev_cost);
        node.traces
            .append(&mut road.make_traces(node.depth - 1, false));
    }

    let sub_depth = node.depth + 1;

    let mut trial_final_cost = None;
    if sub_depth > mcts.search_depth {
        trial_final_cost = Some(road.cost);
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
                        traces: Vec::new(),
                        depth: sub_depth,
                        n_trials: 0,
                        expected_cost: None,
                        costs: Vec::new(),
                        intermediate_costs: Vec::new(),
                        marginal_costs: Vec::new(),
                        sub_nodes: None,
                    })
                    .collect(),
            );
        }

        let sub_nodes = node.sub_nodes.as_mut().unwrap();

        // choose a node to recurse down into! First, try keeping the policy the same
        let mut has_run_trial = false;
        if mcts.prefer_same_policy {
            if let Some(ref policy) = node.policy {
                let policy_id = policy.policy_id();
                if sub_nodes[policy_id as usize].n_trials == 0 {
                    possibly_modify_particle(&mut node.costs, &sub_nodes[policy_id as usize], road);
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
                if unexplored.len() > 0 {
                    let sub_node_i = *unexplored.choose(rng).unwrap();
                    possibly_modify_particle(&mut node.costs, &sub_nodes[sub_node_i], road);
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
            let ln_t = (node.n_trials as f64).ln();
            let (_best_ucb, chosen_i) = sub_nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    let mean_cost = node.expected_cost.unwrap().total();
                    let n = node.n_trials as f64;
                    let ln_t_over_n = ln_t / n;

                    // the "index" by which the next child node is chosen
                    let index = match mcts.selection_mode {
                        ChildSelectionMode::UCB => {
                            let upper_margin = mcts.ucb_const * ln_t_over_n.sqrt();
                            mean_cost + upper_margin
                        }
                        ChildSelectionMode::KLUCB => {
                            if !road.params.run_fast && mean_cost >= mcts.klucb_max_cost {
                                eprintln_f!("High {mean_cost=:.2} > {mcts.klucb_max_cost}");
                            }
                            let scaled_mean =
                                (1.0 - mean_cost / mcts.klucb_max_cost).min(1.0).max(0.0);
                            -klucb_bernoulli(scaled_mean, mcts.ucb_const.abs() * ln_t_over_n)
                        }
                        _ => todo!(),
                    };

                    (index, i)
                })
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            possibly_modify_particle(&mut node.costs, &sub_nodes[chosen_i], road);
            trial_final_cost = Some(find_and_run_trial(&mut sub_nodes[chosen_i], road, rng));
        }
    }

    let trial_final_cost = trial_final_cost.unwrap();

    node.costs
        .push((trial_final_cost, road.particle.clone().unwrap()));
    node.n_trials = node.costs.len();

    let expected_cost = match mcts.bound_mode {
        CostBoundMode::Normal => node.mean_cost(),
        CostBoundMode::BubbleBest => node
            .min_child_expected_cost()
            .unwrap_or_else(|| node.mean_cost()),
        CostBoundMode::LowerBound => node
            .min_child_expected_cost()
            .unwrap_or(Cost::ZERO)
            .max(&node.intermediate_cost()),
        CostBoundMode::Marginal => {
            node.min_child_expected_cost().unwrap_or(Cost::ZERO) + node.marginal_cost()
        }
        CostBoundMode::Same => unimplemented!(),
    };

    node.expected_cost = Some(expected_cost);

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
    loop {
        if let Some(min_child) = n.min_child() {
            write!(f, "{} ", min_child.policy.as_ref().unwrap().policy_id())?;
            n = min_child;
        } else {
            break;
        }
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

    let roads = road_set_for_scenario(params, true_road, rng, params.mcts.samples_n);

    let policy_choices = make_policy_choices(params);
    let debug = true_road.debug
        && true_road.timesteps + params.debug_steps_before >= params.max_steps as usize;

    let mut node = MctsNode {
        params,
        policy_choices: &policy_choices,
        policy: None,
        traces: Vec::new(),
        depth: 0,
        n_trials: 0,
        expected_cost: None,
        costs: Vec::new(),
        intermediate_costs: Vec::new(),
        marginal_costs: Vec::new(),
        sub_nodes: None,
    };

    for mut road in roads.into_iter().cycle().take(params.mcts.samples_n) {
        road.save_particle();
        find_and_run_trial(&mut node, &mut road, rng);
    }

    let mut best_score = Cost::max_value();
    let mut best_policy = None;
    for sub_node in node.sub_nodes.as_ref().unwrap().iter() {
        if let Some(score) = sub_node.expected_cost {
            if score < best_score {
                best_score = score;
                best_policy = sub_node.policy.clone();
            }
        }
    }

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

    (best_policy, traces)
}
