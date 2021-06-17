use rand::prelude::StdRng;

use crate::{
    arg_parameters::Parameters,
    cost::Cost,
    lane_change_policy::{LaneChangePolicy, LongitudinalPolicy},
    road::Road,
    road_set::RoadSet,
    side_policies::{SidePolicy, SidePolicyTrait},
};

pub fn make_obstacle_vehicle_policy_choices() -> Vec<SidePolicy> {
    let mut policy_choices = Vec::new();

    for &lane_i in &[0, 1] {
        for long_policy in vec![
            LongitudinalPolicy::Maintain,
            LongitudinalPolicy::Accelerate,
            LongitudinalPolicy::Decelerate,
        ] {
            policy_choices.push(SidePolicy::LaneChangePolicy(LaneChangePolicy::new(
                policy_choices.len() as u32,
                lane_i,
                4.0,
                true,
                long_policy,
            )));
        }
    }

    policy_choices
}

pub fn make_policy_choices() -> Vec<SidePolicy> {
    let mut policy_choices = Vec::new();

    for &lane_i in &[0, 1] {
        for long_policy in vec![
            LongitudinalPolicy::Maintain,
            LongitudinalPolicy::Accelerate,
            LongitudinalPolicy::Decelerate,
        ] {
            policy_choices.push(SidePolicy::LaneChangePolicy(LaneChangePolicy::new(
                policy_choices.len() as u32,
                lane_i,
                4.0,
                false,
                long_policy,
            )));
        }
    }

    policy_choices
}

fn evaluate_policy(
    params: &Parameters,
    roads: &RoadSet,
    policy: &SidePolicy,
) -> (Cost, Vec<rvx::Shape>) {
    let mut roads = roads.clone();
    roads.set_ego_policy(policy);

    let mpdm = &params.mpdm;
    roads.reset_car_traces();
    roads.take_update_steps(mpdm.forward_t, mpdm.dt);
    (roads.cost(), roads.make_traces(0, false))
}

pub fn mpdm_choose_policy(
    params: &Parameters,
    true_road: &Road,
    rng: &mut StdRng,
) -> (SidePolicy, Vec<rvx::Shape>) {
    let mut traces = Vec::new();
    let roads = RoadSet::new_samples(true_road, rng, params.mpdm.samples_n);
    let debug = true_road.debug
        && true_road.timesteps + params.debug_steps_before >= params.max_steps as usize;
    if debug {
        eprintln!(
            "{}: MPDM search policies and costs, starting with policy {}",
            roads.timesteps(),
            roads.ego_policy().policy_id(),
        );
        eprintln!(
            "Starting from base costs: {:7.2?} = {:7.2}",
            roads.cost(),
            roads.cost().total()
        );
    }

    let policy_choices = make_policy_choices();
    let mut best_cost = Cost::max_value();
    let mut best_policy = None;

    for (i, policy) in policy_choices.into_iter().enumerate() {
        // if roads.timesteps() >= 80 && i != 0 {
        //     continue;
        // }
        // if i == 0 || i == 3 {
        //     continue;
        // }

        let (cost, mut new_traces) = evaluate_policy(params, &roads, &policy);
        traces.append(&mut new_traces);
        // eprint!("{:.2} ", cost);
        // eprintln!("{:?}: {:.2} ", policy, cost);
        if debug {
            eprintln_f!("{i}: {policy:?}: {:7.2?} = {:7.2}", cost, cost.total());
        }

        if cost < best_cost {
            best_cost = cost;
            best_policy = Some(policy);
        }
    }
    // eprintln!();

    (best_policy.unwrap(), traces)
}
