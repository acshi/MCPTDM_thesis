#[derive(Default)]
pub struct Reward {
    pub avg_vel: f64,
    pub safety: f64,
    pub uncomfortable_dec: f64,
    pub curvature_change: f64,
}

impl std::fmt::Display for Reward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        write_f!(
            f,
            "{s.avg_vel:5.2} {s.safety:5.3} {s.uncomfortable_dec:5.3} {s.curvature_change:5.3}"
        )
    }
}

impl std::fmt::Debug for Reward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        write_f!(f, "avg_vel: {s.avg_vel:5.2}, safety: {s.safety:5.3}, ud: {s.uncomfortable_dec:5.3}, cc: {s.curvature_change:5.3}")
    }
}
