use crate::{forward_policies::ForwardPolicyTrait, Road, BREAKING_ACCEL};

#[derive(Debug, Clone)]
pub struct IntelligentDriverPolicy;

impl IntelligentDriverPolicy {
    pub fn new() -> Self {
        Self
    }
}

// https://en.wikipedia.org/wiki/Intelligent_driver_model
impl ForwardPolicyTrait for IntelligentDriverPolicy {
    fn choose_accel(&mut self, road: &Road, car_i: usize) -> f64 {
        let car = &road.cars[car_i];

        let accel_free_road = car.preferred_accel * (1.0 - (car.vel / car.preferred_vel).powi(4));

        let accel;
        if let Some((forward_dist, c_i)) = road.dist_clear_ahead(car_i) {
            let approaching_rate = car.vel - road.cars[c_i].vel;

            let spacing_term = car.follow_min_dist
                + car.vel * car.follow_time
                + car.vel * approaching_rate / (2.0 * (car.preferred_accel * BREAKING_ACCEL));
            let accel_interaction = car.preferred_accel * (-(spacing_term / forward_dist).powi(2));

            accel = accel_free_road + accel_interaction;

            if car_i == 0 {
                eprintln_f!("{approaching_rate=:.2}, {spacing_term=:.2}, {accel_free_road=:.2}, {accel_interaction=:.2}");
            }
        } else {
            accel = accel_free_road;

            if car_i == 0 {
                eprintln_f!("{accel_free_road=:.2}");
            }
        }

        accel
    }
}
