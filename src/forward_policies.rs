use std::time::Duration;

use crate::intelligent_driver::IntelligentDriverPolicy;
use crate::{rate_timer::RateTimer, Road, BREAKING_ACCEL};

#[enum_dispatch]
#[derive(Debug, Clone)]
pub enum ForwardPolicy {
    AdapativeCruisePolicy,
    IntelligentDriverPolicy,
}

#[enum_dispatch(ForwardPolicy)]
pub trait ForwardPolicyTrait {
    fn choose_accel(&mut self, road: &Road, car_i: usize) -> f64;
}

#[derive(Clone)]
pub struct AdapativeCruisePolicy {
    update_rate: RateTimer,
    mode: AdapativeCruiseMode,
}

impl std::fmt::Debug for AdapativeCruisePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.mode)
    }
}

#[derive(Debug, Clone, Copy)]
enum AdapativeCruiseMode {
    Maintain,
    Break,
    Accelerate,
}

impl AdapativeCruisePolicy {
    pub fn new() -> Self {
        Self {
            update_rate: RateTimer::new(Duration::from_millis(100)),
            mode: AdapativeCruiseMode::Maintain,
        }
    }
}

impl ForwardPolicyTrait for AdapativeCruisePolicy {
    fn choose_accel(&mut self, road: &Road, car_i: usize) -> f64 {
        if self.update_rate.ready() {
            let backward_dist = road.dist_clear_behind(car_i).map_or(f64::MAX, |a| a.0);

            if let Some((forward_dist, _c_i)) = road.dist_clear_ahead(car_i) {
                let car = &road.cars[car_i];
                // let (car, c) = road.double_borrow_mut(car_i, c_i);
                if forward_dist < car.follow_dist() && forward_dist < backward_dist {
                    self.mode = AdapativeCruiseMode::Break;
                } else if forward_dist > car.follow_dist()
                    || (backward_dist < car.follow_dist() && backward_dist < forward_dist)
                {
                    self.mode = AdapativeCruiseMode::Accelerate;
                } else {
                    self.mode = AdapativeCruiseMode::Maintain;
                }

                if car_i == 0 {
                    eprintln!(
                        "ego forward_dist {:.2}, backward_dist {:.2}, follow_dist {:.2}, vel {:.2}, mode {:?}",
                        forward_dist, backward_dist.min(100.0), car.follow_dist(), car.vel, self.mode,
                    );
                }
            }
        }

        match self.mode {
            AdapativeCruiseMode::Maintain => 0.0,
            AdapativeCruiseMode::Break => -BREAKING_ACCEL,
            AdapativeCruiseMode::Accelerate => road.cars[car_i].preferred_accel,
        }
    }
}
