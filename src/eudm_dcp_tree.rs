use crate::{
    arg_parameters::Parameters,
    delayed_policy::DelayedPolicy,
    lane_change_policy::LaneChangePolicy,
    reward::Reward,
    side_policies::{SidePolicy, SidePolicyTrait},
    Road,
};

#[derive(Clone)]
struct ClassicTreeNode<'a> {
    params: &'a Parameters,
    policy_choices: &'a [SidePolicy],
    road: Road,
    depth: u32,
    traces: Vec<rvx::Shape>,
}

fn take_update_steps(road: &mut Road, t: f64, dt: f64) {
    // For example, w/ t = 1.0, dt = 0.4 we get steps [0.2, 0.4, 0.4]
    let n_full_steps = (t / dt).floor() as i32;
    let remaining = t - dt * n_full_steps as f64;
    if remaining > 1e-6 {
        road.update(remaining);
    }
    for _ in 0..n_full_steps {
        road.update(dt);
    }
}

fn classic_tree_search(node: ClassicTreeNode) -> (ClassicTreeNode, SidePolicy) {
    let mut traces = Vec::new();

    let base_road = &node.road;
    let tree = &node.params.tree;
    let debug = node.depth == 0 && node.road.timesteps >= 4500;

    if debug {
        eprintln!(
            "{}: Classic tree search policies and rewards and policy {}",
            base_road.timesteps,
            base_road.cars[0].side_policy.as_ref().unwrap().policy_id(),
        );
        eprintln!(
            "Starting from base rewards: {:7.2?} = {:7.2}",
            base_road.reward,
            base_road.reward.total()
        );
    }

    let mut best_sub_policy = None;
    let mut best_node = None;
    let mut best_reward = Reward::max_value();
    for (_i, sub_policy) in node.policy_choices.iter().cloned().enumerate() {
        let mut node = node.clone();
        // if debug && base_road.timesteps == 400 && _i == 2 {
        //     node.road.debug = true;
        // }

        node.road.cars[0].side_policy = Some(sub_policy.clone());
        if node.depth < 2 {
            node.road.car_traces = Some(Vec::new());
        } else {
            node.road.car_traces = None;
        }
        take_update_steps(&mut node.road, tree.layer_t, tree.dt);
        node.depth += 1;
        node.road.reward.discount *= node.params.reward.discount.powf(tree.layer_t);
        traces.append(&mut node.road.make_traces(false));

        let _depth = node.depth;

        if node.depth < tree.search_depth {
            node = classic_tree_search(node).0;
            traces.append(&mut node.traces);
        };

        // if node.road.debug {
        //     eprintln!(
        //         "depth: {:.2}, smoothness reward: {:.2}",
        //         _depth, node.road.reward.smoothness
        //     );
        // }

        // if debug && base_road.timesteps >= 395 && _i < 3 {
        //     node.road.reward.safety += 1000.0;
        // }

        if debug {
            eprintln_f!(
                "{_i}: {sub_policy:?}: {:7.2?} = {:7.2}",
                node.road.reward,
                node.road.reward.total()
            );
        }

        if node.road.reward < best_reward {
            best_reward = node.road.reward;
            best_sub_policy = Some(sub_policy);
            best_node = Some(node);
        }
    }

    let mut best_node = best_node.unwrap();
    best_node.traces = traces;

    (best_node, best_sub_policy.unwrap())
}

fn dcp_tree_search(
    params: &Parameters,
    policy_choices: &[SidePolicy],
    road: Road,
) -> (SidePolicy, Vec<rvx::Shape>) {
    let mut traces = Vec::new();

    let unchanged_policy = road.cars[0].side_policy.as_ref().unwrap();
    let operating_policy = unchanged_policy.operating_policy();
    let eudm = &params.eudm;

    // let last_policy_id = operating_policy.policy_id();

    eprintln!("road.timesteps: {}", road.timesteps);
    let debug = road.timesteps >= params.max_steps as usize - 5;
    if debug {
        eprintln!(
            "{}: EUDM DCP-Tree search policies and rewards, starting with policy {}",
            road.timesteps,
            road.cars[0].side_policy.as_ref().unwrap().policy_id(),
        );
        eprintln!(
            "Starting from base rewards: {:7.2?} = {:7.2}",
            road.reward,
            road.reward.total()
        );
    }

    let mut best_sub_policy = None;
    let mut best_reward = Reward::max_value();

    // Let's first consider the ongoing policy, which may be mid-way through a transition
    // unlike everything else we will consider, which won't transition policies for at least some period
    {
        let mut ongoing_road = road.clone();
        ongoing_road.car_traces = Some(Vec::new());
        take_update_steps(
            &mut ongoing_road,
            eudm.layer_t * eudm.search_depth as f64,
            eudm.dt,
        );
        traces.append(&mut ongoing_road.make_traces(false));
        let reward = ongoing_road.reward;
        if reward < best_reward {
            best_reward = reward;
            best_sub_policy = None;
        }
    }

    let mut init_policy_road = road.clone();
    init_policy_road.cars[0].side_policy = Some(operating_policy.clone());

    for switch_depth in 1..=eudm.search_depth {
        init_policy_road.car_traces = Some(Vec::new());
        take_update_steps(&mut init_policy_road, eudm.layer_t, eudm.dt);
        traces.append(&mut init_policy_road.make_traces(false));

        if switch_depth == eudm.search_depth {
            if debug {
                eprintln_f!(
                    "{switch_depth=}: {operating_policy:?}: {:7.2?} = {:7.2}",
                    init_policy_road.reward,
                    init_policy_road.reward.total()
                );
            }

            let reward = init_policy_road.reward;
            if reward < best_reward {
                best_reward = reward;
                best_sub_policy = None;
            }
        } else {
            for (i, sub_policy) in policy_choices.iter().cloned().enumerate() {
                let mut road = init_policy_road.clone();
                if sub_policy.policy_id() == operating_policy.policy_id() {
                    continue;
                }
                road.cars[0].side_policy = Some(sub_policy.clone());

                road.car_traces = Some(Vec::new());
                for _ in switch_depth..eudm.search_depth {
                    take_update_steps(&mut road, eudm.layer_t, eudm.dt);
                }
                traces.append(&mut road.make_traces(false));

                if debug {
                    eprintln_f!(
                        "{switch_depth=} to {i}: {sub_policy:?}: {:7.2?} = {:7.2}",
                        road.reward,
                        road.reward.total()
                    );
                }

                let reward = road.reward;
                if reward < best_reward {
                    best_reward = reward;
                    if switch_depth == 1 {
                        best_sub_policy = Some(sub_policy);
                    } else {
                        best_sub_policy = None;
                    }
                }
            }
        }
    }

    // will be Some if we should switch policies after one layer, and None to stay the same
    if let Some(best_sub_policy) = best_sub_policy {
        (
            SidePolicy::DelayedPolicy(DelayedPolicy::new(
                operating_policy.clone(),
                best_sub_policy,
                eudm.layer_t,
            )),
            traces,
        )
    } else {
        (unchanged_policy.clone(), traces)
    }
}

fn make_policy_choices() -> Vec<SidePolicy> {
    let mut policy_choices = Vec::new();

    for &lane_i in &[0, 1] {
        for follow_time in vec![1.2, 0.6, 0.0] {
            policy_choices.push(SidePolicy::LaneChangePolicy(LaneChangePolicy::new(
                policy_choices.len() as u32,
                lane_i,
                2.0,
                follow_time,
            )));
        }
    }

    policy_choices
}

pub fn dcp_tree_choose_policy(params: &Parameters, road: Road) -> (SidePolicy, Vec<rvx::Shape>) {
    let policy_choices = make_policy_choices();
    dcp_tree_search(params, &policy_choices, road)
}

pub fn tree_choose_policy(params: &Parameters, road: Road) -> (SidePolicy, Vec<rvx::Shape>) {
    let policy_choices = make_policy_choices();

    let mut node = ClassicTreeNode {
        params,
        policy_choices: &policy_choices,
        road,
        depth: 0,
        traces: Vec::new(),
    };

    if let Some(traces) = node.road.car_traces.as_mut() {
        traces.clear();
    }

    let (final_node, policy) = classic_tree_search(node);
    (policy, final_node.traces)
}
