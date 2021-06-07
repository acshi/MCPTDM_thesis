use parry2d_f64::na::Point2;

use crate::delayed_policy::DelayedPolicy;
use crate::lane_change_policy::LaneChangePolicy;
use crate::Road;

#[enum_dispatch]
#[derive(Debug, Clone)]
pub enum SidePolicy {
    LaneChangePolicy,
    DelayedPolicy,
}

// impl SidePolicy {
//     fn operating_policy(&self) -> Self {

//     }
// }

#[enum_dispatch(SidePolicy)]
pub trait SidePolicyTrait {
    fn choose_follow_time(&mut self, road: &crate::Road, car_i: usize) -> f64 {
        road.cars[car_i].preferred_follow_time
    }

    fn choose_trajectory(&mut self, road: &Road, car_i: usize) -> Vec<Point2<f64>>;
    fn policy_id(&self) -> u64;
    fn operating_policy(&self) -> SidePolicy;
}
