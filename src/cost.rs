#[derive(Clone, Copy, PartialEq)]
pub struct Cost {
    pub efficiency: f64,
    pub safety: f64,
    pub smoothness: f64,
    pub uncomfortable_dec: f64,
    pub curvature_change: f64,

    pub discount: f64,
    pub discount_factor: f64,
}

impl std::fmt::Display for Cost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        write_f!(f, "{s.efficiency:8.2} {s.safety:8.2} {s.smoothness:8.2} {s.uncomfortable_dec:8.2} {s.curvature_change:8.2}")
    }
}

impl std::fmt::Debug for Cost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        write_f!(
            f,
            "eff: {s.efficiency:8.2}, safe: {s.safety:8.2}, smooth: {s.smoothness:8.2}, ud: {s.uncomfortable_dec:8.2}, cc: {s.curvature_change:8.2}"
        )
    }
}

impl Cost {
    pub fn new(discount_factor: f64) -> Self {
        Self {
            efficiency: 0.0,
            safety: 0.0,
            smoothness: 0.0,
            uncomfortable_dec: 0.0,
            curvature_change: 0.0,
            discount: 1.0,
            discount_factor,
        }
    }

    pub fn max_value() -> Self {
        Self {
            efficiency: f64::MAX,
            safety: 0.0,
            smoothness: 0.0,
            uncomfortable_dec: 0.0,
            curvature_change: 0.0,
            discount: 1.0,
            discount_factor: 1.0,
        }
    }

    pub fn total(&self) -> f64 {
        self.efficiency
            + self.safety
            + self.smoothness
            + self.uncomfortable_dec
            + self.curvature_change
    }

    pub fn update_discount(&mut self, dt: f64) {
        self.discount *= self.discount_factor.powf(dt);
    }
}

impl Default for Cost {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.total().partial_cmp(&other.total())
    }
}

impl std::iter::Sum for Cost {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = Cost::new(1.0);
        for r in iter {
            sum += r;
        }
        sum
    }
}

impl std::ops::Mul<f64> for Cost {
    type Output = Cost;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            efficiency: self.efficiency * rhs,
            safety: self.safety * rhs,
            smoothness: self.smoothness * rhs,
            uncomfortable_dec: self.uncomfortable_dec * rhs,
            curvature_change: self.curvature_change * rhs,
            discount: self.discount,
            discount_factor: self.discount_factor,
        }
    }
}

impl std::ops::Div<f64> for Cost {
    type Output = Cost;

    fn div(self, rhs: f64) -> Self::Output {
        Self {
            efficiency: self.efficiency / rhs,
            safety: self.safety / rhs,
            smoothness: self.smoothness / rhs,
            uncomfortable_dec: self.uncomfortable_dec / rhs,
            curvature_change: self.curvature_change / rhs,
            discount: self.discount,
            discount_factor: self.discount_factor,
        }
    }
}

impl std::ops::DivAssign<f64> for Cost {
    fn div_assign(&mut self, rhs: f64) {
        self.efficiency /= rhs;
        self.safety /= rhs;
        self.smoothness /= rhs;
        self.uncomfortable_dec /= rhs;
        self.curvature_change /= rhs;
    }
}

impl std::ops::Add for Cost {
    type Output = Cost;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            efficiency: self.efficiency + rhs.efficiency,
            safety: self.safety + rhs.safety,
            smoothness: self.smoothness + rhs.smoothness,
            uncomfortable_dec: self.uncomfortable_dec + rhs.uncomfortable_dec,
            curvature_change: self.curvature_change + rhs.curvature_change,
            discount: self.discount,
            discount_factor: self.discount_factor,
        }
    }
}

impl std::ops::AddAssign for Cost {
    fn add_assign(&mut self, rhs: Self) {
        self.efficiency += rhs.efficiency;
        self.safety += rhs.safety;
        self.smoothness += rhs.smoothness;
        self.uncomfortable_dec += rhs.uncomfortable_dec;
        self.curvature_change += rhs.curvature_change;
    }
}
