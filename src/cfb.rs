use itertools::Itertools;

use crate::{
    arg_parameters::Parameters, mpdm::make_obstacle_vehicle_policy_choices, road::Road,
    road_set::RoadSet,
};

fn key_vehicles<'a>(params: &Parameters, road: &Road) -> Vec<usize> {
    let ego = &road.cars[0];
    let dx_thresh = params.cfb.key_vehicle_base_dist + ego.vel * params.cfb.key_vehicle_dist_time;

    let mut car_ids = Vec::new();
    for c in &road.cars[1..] {
        if c.crashed {
            continue;
        }

        let dx = (ego.x - c.x).abs();
        // eprintln_f!("ego to {c.car_i}: {dx=:.2}, {dx_thresh=:.2}");
        if dx <= dx_thresh {
            car_ids.push(c.car_i);
        }
    }

    car_ids
}

pub fn conditional_focused_branching(params: &Parameters, road: &Road, n: usize) -> RoadSet {
    let belief = road.belief.as_ref().unwrap();

    let key_car_ids = key_vehicles(params, road);
    // eprintln_f!("{key_car_ids=:?}");
    let uncertain_car_ids = key_car_ids
        .into_iter()
        .filter(|&car_i| belief.is_uncertain(car_i, params.cfb.uncertainty_threshold))
        .collect_vec();
    // eprintln_f!("{uncertain_car_ids=:?}");

    // For each car, perform an open-loop simulation with only that car, using each real policy.
    // I guess the ego-vehicle gets to keep using its real policy?
    // (And I guess tree search and mcts would both need to be able to produce policies that contain their full set of changes)
    // And then I guess all the other vehicles are just made to be constant-velocity?
    // Alternatively... maybe _no_ vehicles use the intelligent driver model? but then what keeps the ego vehicle in
    // a light acceleration from just speeding off into stuff? Or the simulation making any sense at all?
    // I imagine that the uncertain and the ego vehicle must follow their real closed-loop policies for this to do any good.

    let policies = make_obstacle_vehicle_policy_choices(params);

    let base_safety = road.cost.safety;
    let open_loop_sims = uncertain_car_ids
        .into_iter()
        .map(|car_i| {
            let road = road.open_loop_estimate(car_i);
            let costs = policies
                .iter()
                .map(|policy| {
                    let mut road = road.clone();
                    road.cars[car_i].side_policy = Some(policy.clone());
                    road.car_traces = None;
                    road.take_update_steps(params.cfb.horizon_t, params.cfb.dt);
                    // eprintln_f!("{car_i=} {road.cost:.2?} {policy:?}");
                    road.cost.safety - base_safety
                })
                .collect_vec();

            let worst_cost = *costs
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            if worst_cost > 0.0 {
                Some((car_i, worst_cost, costs))
            } else {
                None
            }
        })
        .collect_vec();

    // eprintln!("Open loop sim results:");
    // for open_loop_sim in open_loop_sims.iter() {
    //     eprintln_f!("{open_loop_sim:.2?}");
    // }

    let mut sim_road = road.sim_estimate();
    // Each car (besides ego) defaults to the policy that is most likely for it
    for c in sim_road.cars[1..].iter_mut() {
        let policy_i = belief.get_most_likely(c.car_i);
        c.side_policy = Some(policies[policy_i].clone());
    }

    // for each risky car, produces an iter of tuples, of (car_i, each policy_i)
    let risky_car_policies = open_loop_sims
        .iter()
        .filter_map(|a| a.as_ref())
        .map(|a| a.0)
        .map(|car_i| (0..policies.len()).map(move |p_i| (car_i, p_i)));

    let mut scenarios = Vec::new();

    for scenario in risky_car_policies.multi_cartesian_product() {
        let probability: f64 = scenario
            .iter()
            .map(|(car_i, policy_i)| belief.get(*car_i, *policy_i))
            .product();

        let mut sim_road = sim_road.clone();
        for (car_i, policy_i) in scenario.iter() {
            sim_road.cars[*car_i].side_policy = Some(policies[*policy_i].clone());
        }

        scenarios.push((probability, sim_road))
    }

    // sort descending and choose just the most probable
    scenarios.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scenarios.truncate(n);
    let mut roads = scenarios.into_iter().map(|a| a.1).collect_vec();

    if roads.is_empty() {
        roads.push(sim_road);
    }

    RoadSet::new(roads)
}
