use crate::{
    arg_parameters::Parameters, lane_change_policy::LaneChangePolicy, side_policies::SidePolicy,
    Road, DT,
};

fn evaluate_setup(road: &Road) -> f64 {
    let mut road = road.clone();
    for _ in 0..200 {
        road.update(DT);
    }
    road.final_reward()
}

pub fn mpdm_choose_policy(_params: &Parameters, road: Road, car_i: usize) -> SidePolicy {
    let mut policy_choices = Vec::new();

    for follow_time in vec![Some(1.2), Some(0.0)] {
        for &lane_i in &[0, 1] {
            policy_choices.push(SidePolicy::LaneChangePolicy(LaneChangePolicy::new(
                lane_i,
                2.0,
                follow_time,
            )));
        }
    }

    let mut best_reward = f64::MAX;
    let mut best_policy = None;

    for policy in policy_choices {
        let mut road = road.clone();
        road.cars[car_i].side_policy = Some(policy);
        let reward = evaluate_setup(&road);
        eprint!("{:.2} ", reward);
        if reward < best_reward {
            best_reward = reward;
            best_policy = road.cars.swap_remove(car_i).side_policy;
        }
    }
    eprintln!();

    best_policy.unwrap()
}
