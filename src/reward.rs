#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reward {
    pub efficiency: f64,
    pub safety: f64,
    pub smoothness: f64,
    pub discount: f64,
}

impl Reward {
    pub fn new() -> Self {
        Self {
            efficiency: 0.0,
            safety: 0.0,
            smoothness: 0.0,
            discount: 1.0,
        }
    }

    pub fn max_value() -> Self {
        Self {
            efficiency: f64::MAX,
            safety: 0.0,
            smoothness: 0.0,
            discount: 1.0,
        }
    }

    pub fn total(&self) -> f64 {
        self.efficiency + self.safety + self.smoothness
    }
}

impl PartialOrd for Reward {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.total().partial_cmp(&other.total())
    }
}
