use crate::{
    arg_parameters::Parameters, lane_change_policy::LaneChangePolicy, side_policies::SidePolicy,
    Road,
};

fn evaluate_policy(params: &Parameters, road: &Road, policy: SidePolicy) -> f64 {
    let mut road = road.clone();
    road.cars[0].side_policy = Some(policy);

    let mpdm = &params.mpdm;
    let dt = mpdm.dt;
    let n_steps = (mpdm.forward_t / dt).round() as u32;
    for _ in 0..n_steps {
        road.update(dt);
    }
    road.final_reward()
}

pub fn mpdm_choose_policy(params: &Parameters, road: Road) -> (SidePolicy, Vec<rvx::Shape>) {
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

    let mut best_reward = f64::MAX;
    let mut best_policy = None;

    for policy in policy_choices {
        let reward = evaluate_policy(params, &road, policy.clone());
        eprint!("{:.2} ", reward);
        // eprintln!("{:?}: {:.2} ", policy, reward);
        if reward < best_reward {
            best_reward = reward;
            best_policy = Some(policy);
        }
    }
    eprintln!();

    best_policy.unwrap();
    panic!()
}
