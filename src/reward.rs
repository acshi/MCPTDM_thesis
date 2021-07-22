#[derive(Default)]
pub struct Reward {
    pub avg_vel: f64,
    pub safety: f64,
    pub uncomfortable_dec: f64,
    pub curvature_change: f64,
    pub planning_times: Vec<f64>,
    pub mean_planning_time: Option<f64>,
    pub below95_planning_time: Option<f64>,
    pub below997_planning_time: Option<f64>,
    pub max_planning_time: Option<f64>,
    pub stddev_planning_time: Option<f64>,
}

impl Reward {
    pub fn calculate_timestep_metrics(&mut self) {
        self.planning_times
            .sort_by(|a, b| a.partial_cmp(b).unwrap());
        // 95% of the timesteps times will have a value <= the time of the 0.95 * len()
        let n = self.planning_times.len();
        self.below95_planning_time = Some(self.planning_times[n * 95 / 100]);
        self.below997_planning_time = Some(self.planning_times[n * 997 / 1000]);
        self.max_planning_time = Some(self.planning_times[n - 1]);

        let mean = self.planning_times.iter().sum::<f64>() / n as f64;
        self.mean_planning_time = Some(mean);

        // standard deviation of the mean or "standard error"
        let stddev = (self
            .planning_times
            .iter()
            .map(|t| (*t - mean).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt()
            / (n as f64).sqrt();
        self.stddev_planning_time = Some(stddev);
    }
}

impl std::fmt::Display for Reward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        write_f!(
            f,
            "{s.avg_vel:5.2} {s.safety:5.3} {s.uncomfortable_dec:5.3} {s.curvature_change:5.3} {:7.5} {:7.5} {:7.5} {:7.5} {:8.6}",
            s.mean_planning_time.unwrap(),
            s.below95_planning_time.unwrap(),
            s.below997_planning_time.unwrap(),
            s.max_planning_time.unwrap(),
            s.stddev_planning_time.unwrap()
        )
    }
}

impl std::fmt::Debug for Reward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        write_f!(f, "avg_vel: {s.avg_vel:.2}, safety: {s.safety:.3}, ud: {s.uncomfortable_dec:.3}, cc: {s.curvature_change:.3}")?;
        if let Some(t) = self.mean_planning_time {
            write_f!(f, ", mean ts: {:.2}", t * 1000.0)?;
        }
        if let Some(t) = self.below95_planning_time {
            write_f!(f, ", .95: {:.2}", t * 1000.0)?;
        }
        if let Some(t) = self.below997_planning_time {
            write_f!(f, ", .997: {:.2}", t * 1000.0)?;
        }
        if let Some(t) = self.max_planning_time {
            write_f!(f, ", max: {:.2}", t * 1000.0)?;
        }
        if let Some(t) = self.stddev_planning_time {
            write_f!(f, ", stddev: {:.3}", t * 1000.0)?;
        }
        Ok(())
    }
}
