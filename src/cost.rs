#[derive(Clone, Copy, PartialEq)]
pub struct Cost {
    pub efficiency: f64,
    pub safety: f64,
    pub smoothness: f64,
    pub uncomfortable_dec: f64,
    pub curvature_change: f64,

    pub discount: f64,
    pub discount_factor: f64,

    pub weight: f64,
}

impl std::fmt::Display for Cost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.normalize();
        write_f!(f, "{s.efficiency:8.2} {s.safety:8.2} {s.smoothness:8.2} {s.uncomfortable_dec:8.2} {s.curvature_change:8.2}")
    }
}

impl std::fmt::Debug for Cost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        write_f!(
            f,
            "eff: {s.efficiency:8.2}, safe: {s.safety:8.2}, smooth: {s.smoothness:8.2}, ud: {s.uncomfortable_dec:8.2}, cc: {s.curvature_change:8.2}, w: {s.weight}"
        )
    }
}

impl Cost {
    pub fn new(discount_factor: f64, weight: f64) -> Self {
        Self {
            efficiency: 0.0,
            safety: 0.0,
            smoothness: 0.0,
            uncomfortable_dec: 0.0,
            curvature_change: 0.0,
            discount: 1.0,
            discount_factor,
            weight,
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
            weight: 1.0,
        }
    }

    pub fn normalize(&self) -> Self {
        Self {
            efficiency: self.efficiency * self.weight,
            safety: self.safety * self.weight,
            smoothness: self.smoothness * self.weight,
            uncomfortable_dec: self.uncomfortable_dec * self.weight,
            curvature_change: self.curvature_change * self.weight,
            discount: 1.0,
            discount_factor: 1.0,
            weight: 1.0,
        }
    }

    fn unweighted_total(&self) -> f64 {
        self.efficiency
            + self.safety
            + self.smoothness
            + self.uncomfortable_dec
            + self.curvature_change
    }

    pub fn total(&self) -> f64 {
        self.weight * self.unweighted_total()
    }

    pub fn update_discount(&mut self, dt: f64) {
        self.discount *= self.discount_factor.powf(dt);
    }

    #[allow(unused)]
    pub fn max(&self, other: &Self) -> Self {
        if self > other {
            *self
        } else {
            *other
        }
    }

    #[allow(unused)]
    pub fn min(&self, other: &Self) -> Self {
        if self < other {
            *self
        } else {
            *other
        }
    }
}

impl Default for Cost {
    fn default() -> Self {
        Self::new(1.0, 1.0)
    }
}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.total().partial_cmp(&other.total())
    }
}

impl std::iter::Sum for Cost {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = Cost::new(1.0, 1.0);
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
            weight: self.weight,
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
            weight: self.weight,
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
        let a = self.normalize();
        let b = rhs.normalize();
        Self {
            efficiency: a.efficiency + b.efficiency,
            safety: a.safety + b.safety,
            smoothness: a.smoothness + b.smoothness,
            uncomfortable_dec: a.uncomfortable_dec + b.uncomfortable_dec,
            curvature_change: a.curvature_change + b.curvature_change,
            discount: self.discount,
            discount_factor: self.discount_factor,
            weight: 1.0,
        }
    }
}

impl std::ops::AddAssign for Cost {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
