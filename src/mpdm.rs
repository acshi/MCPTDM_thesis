use crate::{
    arg_parameters::Parameters,
    delayed_policy::DelayedPolicy,
    lane_change_policy::LaneChangePolicy,
    side_policies::{SidePolicy, SidePolicyTrait},
    Road, DT,
};

fn evaluate_policy(road: &Road, policy: SidePolicy) -> f64 {
    let mut road = road.clone();
    road.cars[0].side_policy = Some(policy);
    for _ in 0..400 {
        road.update(DT);
    }
    road.final_reward()
}

pub fn mpdm_choose_policy(_params: &Parameters, road: Road) -> SidePolicy {
    let mut policy_choices = Vec::new();

    for &lane_i in &[0, 1] {
        for follow_time in vec![Some(1.2), Some(0.6), Some(0.0)] {
            policy_choices.push(SidePolicy::LaneChangePolicy(LaneChangePolicy::new(
                lane_i,
                2.0,
                follow_time,
            )));
        }
    }

    let mut eudm_policy_choics = Vec::new();
    for &delay in &[1.0, 2.0] {
        for base_policy in policy_choices.iter() {
            eudm_policy_choics.push(SidePolicy::DelayedPolicy(DelayedPolicy::new(
                road.cars[0]
                    .side_policy
                    .as_ref()
                    .unwrap()
                    .operating_policy(),
                base_policy.clone(),
                delay,
            )))
        }
    }
    policy_choices = eudm_policy_choics;

    let mut best_reward = f64::MAX;
    let mut best_policy = None;

    for policy in policy_choices {
        let reward = evaluate_policy(&road, policy.clone());
        eprint!("{:.2} ", reward);
        // eprintln!("{:?}: {:.2} ", policy, reward);
        if reward < best_reward {
            best_reward = reward;
            best_policy = Some(policy);
        }
    }
    eprintln!();

    best_policy.unwrap()
}
