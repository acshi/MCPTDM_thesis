use rand::prelude::StdRng;

use crate::{
    arg_parameters::Parameters,
    cost::Cost,
    mpdm::make_policy_choices,
    road::Road,
    road_set::RoadSet,
    side_policies::{SidePolicy, SidePolicyTrait},
};

#[derive(Clone)]
struct ClassicTreeNode<'a> {
    params: &'a Parameters,
    policy_choices: &'a [SidePolicy],
    roads: RoadSet,
    depth: u32,
    traces: Vec<rvx::Shape>,
    debug: bool,
}

fn classic_tree_search(node: ClassicTreeNode) -> (ClassicTreeNode, SidePolicy) {
    let mut traces = Vec::new();

    let base_roads = &node.roads;
    let tree = &node.params.tree;
    let debug = node.debug;

    if debug {
        eprintln!(
            "{}: Classic tree search policies and costs and policy {}",
            base_roads.timesteps(),
            base_roads.ego_policy().policy_id(),
        );
        eprintln!(
            "Starting from base costs: {:7.2?} = {:7.2}",
            base_roads.cost(),
            base_roads.cost().total()
        );
    }

    let mut best_sub_policy = None;
    let mut best_node = None;
    let mut best_cost = Cost::max_value();
    let mut best_additional_traces = Vec::new();
    for (_i, sub_policy) in node.policy_choices.iter().enumerate() {
        let mut node = node.clone();
        // if debug && base_road.timesteps == 400 && _i == 2 {
        //     node.road.debug = true;
        // }

        node.roads.set_ego_policy(sub_policy);
        // if node.depth < 2 {
        node.roads.reset_car_traces();
        // } else {
        //     node.roads.disable_car_traces();
        // }
        node.roads.take_update_steps(tree.layer_t, tree.dt);
        traces.append(&mut node.roads.make_traces(node.depth, false));
        node.depth += 1;

        let _depth = node.depth;
        let mut node_traces = Vec::new();

        if node.depth < tree.search_depth {
            node = classic_tree_search(node).0;
            if _depth <= 1 {
                traces.append(&mut node.traces);
            } else {
                node_traces = std::mem::replace(&mut node.traces, Vec::new());
            }
        };

        if debug {
            eprintln_f!(
                "{_i}: {sub_policy:?}: {:7.2?} = {:7.2}",
                node.roads.cost(),
                node.roads.cost().total()
            );
        }

        if node.roads.cost() < best_cost {
            best_cost = node.roads.cost();
            best_sub_policy = Some(sub_policy);
            best_node = Some(node);
            best_additional_traces = node_traces;
        }
    }

    traces.append(&mut best_additional_traces);

    let mut best_node = best_node.unwrap();
    best_node.traces = traces;

    (best_node, best_sub_policy.unwrap().clone())
}

pub fn tree_choose_policy(
    params: &Parameters,
    true_road: &Road,
    rng: &mut StdRng,
) -> (SidePolicy, Vec<rvx::Shape>) {
    let roads = RoadSet::new_samples(true_road, rng, params.tree.samples_n);
    let policy_choices = make_policy_choices();
    let debug = true_road.debug && true_road.timesteps >= (params.max_steps - 100) as usize;

    let node = ClassicTreeNode {
        params,
        policy_choices: &policy_choices,
        roads,
        depth: 0,
        traces: Vec::new(),
        debug,
    };

    let (final_node, policy) = classic_tree_search(node);
    (policy, final_node.traces)
}
