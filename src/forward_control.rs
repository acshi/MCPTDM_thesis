use crate::intelligent_driver::IntelligentDriverPolicy;
use crate::Road;

#[enum_dispatch]
#[derive(Debug, Clone)]
pub enum ForwardControl {
    IntelligentDriverPolicy,
}

#[enum_dispatch(ForwardControl)]
pub trait ForwardControlTrait {
    fn choose_accel(&mut self, road: &Road, car_i: usize) -> f64;
}
