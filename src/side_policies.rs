use crate::Road;

use crate::pure_pursuit::PurePursuitPolicy;

#[enum_dispatch]
#[derive(Debug, Clone)]
pub enum SidePolicy {
    PurePursuitPolicy,
}

#[enum_dispatch(SidePolicy)]
pub trait SidePolicyTrait {
    fn choose_steer(&mut self, road: &Road, car_i: usize) -> f64;
}
