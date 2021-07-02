use itertools::Itertools;
use rand::prelude::{SliceRandom, StdRng};

use crate::{
    arg_parameters::Parameters,
    cost::Cost,
    mpdm::{make_obstacle_vehicle_policy_belief_states, make_policy_choices},
    road::Road,
    road_set_for_scenario,
    side_policies::{SidePolicy, SidePolicyTrait},
};

#[derive(Clone)]
enum MctsNodeInner<'a> {
    Branch {
        sub_nodes: Option<Vec<MctsNode<'a>>>,
    },
    Leaf {
        scores: Vec<Cost>,
    },
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

    intermediate_costs: Vec<Cost>,

    inner: MctsNodeInner<'a>,
}

fn find_and_run_trial(node: &mut MctsNode, road: &mut Road, rng: &mut StdRng) {
    let params = node.params;
    let mcts = &params.mcts;

    if let Some(policy) = node.policy.as_ref() {
        road.set_ego_policy(policy.clone());
        // road.reset_car_traces();
        if node.depth < 4 {
            road.reset_car_traces();
        } else {
            road.disable_car_traces();
        }
        road.take_update_steps(mcts.layer_t, mcts.dt);
        node.intermediate_costs.push(road.cost);
        node.traces
            .append(&mut road.make_traces(node.depth - 1, false));
    }

    match &mut node.inner {
        MctsNodeInner::Branch { sub_nodes } => {
            let sub_depth = node.depth + 1;

            // expand node?
            if sub_nodes.is_none() {
                let sub_inner = if sub_depth >= mcts.search_depth {
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
                            traces: Vec::new(),
                            depth: sub_depth,
                            n_trials: 0,
                            expected_cost: None,
                            intermediate_costs: Vec::new(),
                            inner: sub_inner.clone(),
                        })
                        .collect(),
                );
            }

            let sub_nodes = sub_nodes.as_mut().unwrap();

            // choose a node to recurse down into! First, try keeping the policy the same
            let mut has_run_trial = false;
            if mcts.prefer_same_policy {
                if let Some(policy) = node.policy.as_ref() {
                    let policy_id = policy.policy_id();
                    if sub_nodes[policy_id as usize].n_trials == 0 {
                        find_and_run_trial(&mut sub_nodes[policy_id as usize], road, rng);
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
                        find_and_run_trial(&mut sub_nodes[sub_node_i], road, rng);
                        has_run_trial = true;
                    }
                } else {
                    for sub_node in sub_nodes.iter_mut() {
                        if sub_node.n_trials == 0 {
                            find_and_run_trial(sub_node, road, rng);
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
                        let upper_margin = mcts.ucb_const * (ln_t / node.n_trials as f64).sqrt();
                        // eprintln!("n: {} expected: {} margin: {}", a.scores.len(), a.mean_score, upper_margin);
                        (node.expected_cost.unwrap().total() + upper_margin, i)
                    })
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();

                find_and_run_trial(&mut sub_nodes[chosen_i], road, rng);
            }

            node.n_trials = sub_nodes.iter().map(|n| n.n_trials).sum::<usize>();
            node.expected_cost = sub_nodes
                .iter()
                .filter_map(|n| n.expected_cost)
                .min_by(|a, b| a.partial_cmp(b).unwrap());

            let intermediate_cost = node
                .intermediate_costs
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap());
            if let (Some(a), Some(b)) = (node.expected_cost, intermediate_cost) {
                node.expected_cost = Some(a.max(b));
            }
        }
        MctsNodeInner::Leaf { scores } => {
            scores.push(road.cost);
            node.n_trials = scores.len();
            if mcts.bubble_up_max_weighted_leaf {
                node.expected_cost = scores
                    .iter()
                    .max_by(|a, b| a.partial_cmp(&b).unwrap())
                    .copied();
            } else {
                node.expected_cost =
                    Some(scores.iter().copied().sum::<Cost>() / node.n_trials as f64);
            }
        }
    }
}

fn collect_traces(node: &mut MctsNode, traces: &mut Vec<rvx::Shape>) {
    traces.append(&mut node.traces);

    match &mut node.inner {
        MctsNodeInner::Branch { sub_nodes } => {
            if let Some(sub_nodes) = sub_nodes.as_mut() {
                for sub_node in sub_nodes.iter_mut() {
                    collect_traces(sub_node, traces);
                }
            }
        }
        MctsNodeInner::Leaf { .. } => (),
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

    match &node.inner {
        MctsNodeInner::Branch { sub_nodes } => {
            if let Some(sub_nodes) = sub_nodes.as_ref() {
                for sub_node in sub_nodes.iter() {
                    print_report(sub_node);
                }
            }
        }
        MctsNodeInner::Leaf { .. } => (),
    }
}

pub fn mcts_choose_policy(
    params: &Parameters,
    true_road: &Road,
    rng: &mut StdRng,
) -> (SidePolicy, Vec<rvx::Shape>) {
    let mut roads = road_set_for_scenario(params, true_road, rng, params.mcts.samples_n);
    // TODO REMOVE
    if true_road.super_debug() {
        if let Some(debug_car_i) = params.debug_car_i {
            let policy_belief_choices = make_obstacle_vehicle_policy_belief_states(params);
            for road in roads.iter_mut() {
                road.cars[debug_car_i].side_policy = Some(policy_belief_choices[4].clone());
            }
        }
    }

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
        intermediate_costs: Vec::new(),
        inner: MctsNodeInner::Branch { sub_nodes: None },
    };

    for mut road in roads.into_iter().cycle().take(params.mcts.samples_n) {
        find_and_run_trial(&mut node, &mut road, rng);
    }

    let mut best_score = Cost::max_value();
    let mut best_policy = None;
    if let MctsNodeInner::Branch { sub_nodes } = &node.inner {
        for sub_node in sub_nodes.as_ref().unwrap().iter() {
            if let Some(score) = sub_node.expected_cost {
                if score < best_score {
                    best_score = score;
                    best_policy = sub_node.policy.clone();
                }
            }
        }
    } else {
        unreachable!();
    }

    let mut traces = Vec::new();
    collect_traces(&mut node, &mut traces);

    if debug && params.policy_report_debug {
        print_report(&node);
    }

    (best_policy.unwrap(), traces)
}
