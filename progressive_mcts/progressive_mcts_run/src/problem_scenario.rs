use rand::{
    prelude::{IteratorRandom, StdRng},
    Rng,
};
use rand_distr::{Bernoulli, Distribution, Normal};

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct SituationParticle {
    pub id: usize,
    pub bad_situation: Option<(u32, f64)>,
}

impl std::fmt::Debug for SituationParticle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {:?})", self.id, self.bad_situation)
    }
}

impl SituationParticle {
    pub fn sample(id: usize, p: f64, max_depth: u32, rng: &mut StdRng) -> Self {
        Self {
            id,
            bad_situation: if rng.gen_bool(p) {
                Some((rng.gen_range(0..max_depth), rng.gen_range(0.0..=1.0)))
            } else {
                None
            },
        }
    }
}

#[derive(Clone)]
pub enum CostDistribution {
    Normal {
        d: Normal<f64>,
        magnitude: f64,
    },
    Bernoulli {
        d: Bernoulli,
        p: f64,
        magnitude: f64,
    },
}

impl CostDistribution {
    #[allow(unused)]
    pub fn normal(magnitude: f64, mean: f64, std_dev: f64) -> Self {
        Self::Normal {
            d: Normal::new(mean, std_dev).expect("valid mean and standard deviation"),
            magnitude,
        }
    }

    pub fn bernoulli(p: f64, magnitude: f64) -> Self {
        Self::Bernoulli {
            d: Bernoulli::new(p).expect("probability from 0 to 1"),
            p,
            magnitude,
        }
    }

    pub fn mean(&self) -> f64 {
        match self {
            CostDistribution::Normal { d, magnitude: _ } => d.mean(),
            CostDistribution::Bernoulli { d: _, p, magnitude } => p * magnitude,
        }
    }

    #[allow(unused)]
    pub fn magnitude(&self) -> f64 {
        match self {
            // for normal, return magnitude of the corresponding bernoulli distribution
            CostDistribution::Normal { d: _, magnitude } => *magnitude, // d.std_dev().powi(2) / d.mean() + d.mean(),
            CostDistribution::Bernoulli {
                d: _,
                p: _,
                magnitude,
            } => *magnitude,
        }
    }

    pub fn sample(&self, rng: &mut StdRng) -> f64 {
        match self {
            CostDistribution::Normal { d, magnitude: _ } => {
                d.sample(rng).max(0.0).min(2.0 * d.mean())
            }
            CostDistribution::Bernoulli { d, p: _, magnitude } => {
                if d.sample(rng) {
                    *magnitude
                } else {
                    0.0
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct ProblemScenario {
    pub distribution: Option<CostDistribution>,
    pub bad_situation_p: f64,
    pub bad_threshold_cost: f64,
    pub bad_situation_threshold: f64,
    pub children: Vec<ProblemScenario>,
    pub depth: u32,
    pub max_depth: u32,
}

impl ProblemScenario {
    fn inner_new(
        depth: u32,
        max_depth: u32,
        n_actions: u32,
        portion_bernoulli: f64,
        bad_situation_p: f64,
        bad_threshold_cost: f64,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            distribution: if depth == 0 {
                None
            } else {
                // let mean = rng.gen_range(0.0..100.0);
                // let std_dev = rng.gen_range(0.0..1000.0);

                // Some(CostDistribution::normal(mean, std_dev))

                // let p = rng.gen_range(0.0..=0.5);
                // let mag = rng.gen_range(0.0..=1000.0);
                // Some(CostDistribution::bernoulli(p, mag))
                let p = (0..=10).map(|i| i as f64 * 0.1).choose(rng).unwrap();
                let mag = 1000.0;

                if rng.gen_bool(portion_bernoulli) {
                    Some(CostDistribution::bernoulli(p, mag))
                } else {
                    let mean = p * mag;
                    let std_dev = (p * (1.0 - p)).sqrt() * mag;
                    // std_dev^2 / mean^2 = (p - p^2) / p^2 = 1 / p - 1.0
                    // (std_dev^2 / mean^2 + 1.0)^-1 = p
                    // mag = (std_dev^2 / mean + mean)
                    Some(CostDistribution::normal(mag, mean, std_dev))
                }
            },
            children: if depth < max_depth {
                (0..n_actions)
                    .map(|_| {
                        Self::inner_new(
                            depth + 1,
                            max_depth,
                            n_actions,
                            portion_bernoulli,
                            bad_situation_p,
                            bad_threshold_cost,
                            rng,
                        )
                    })
                    .collect()
            } else {
                Vec::new()
            },
            bad_situation_p,
            bad_threshold_cost,
            bad_situation_threshold: rng.gen_range(0.0..=1.0),
            depth,
            max_depth,
        }
    }

    pub fn new(
        max_depth: u32,
        n_actions: u32,
        portion_bernoulli: f64,
        bad_situation_p: f64,
        bad_threshold_cost: f64,
        rng: &mut StdRng,
    ) -> Self {
        Self::inner_new(
            0,
            max_depth,
            n_actions,
            portion_bernoulli,
            bad_situation_p,
            bad_threshold_cost,
            rng,
        )
    }

    pub fn expected_marginal_cost(&self) -> f64 {
        if let Some(d) = &self.distribution {
            let bad_p =
                self.bad_situation_p / self.max_depth as f64 * (1.0 - self.bad_situation_threshold);

            bad_p * self.bad_threshold_cost + (1.0 - bad_p) * d.mean()
        } else {
            0.0
        }
    }
}

#[derive(Clone)]
pub struct Simulator<'a> {
    pub scenario: &'a ProblemScenario,
    pub particle: SituationParticle,
    pub depth: u32,
    pub cost: f64,
}

impl<'a> Simulator<'a> {
    pub fn sample(
        scenario: &'a ProblemScenario,
        id: usize,
        bad_situation_p: f64,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            scenario,
            particle: SituationParticle::sample(id, bad_situation_p, scenario.max_depth, rng),
            cost: 0.0,
            depth: 0,
        }
    }

    pub fn take_step(&mut self, policy: u32, rng: &mut StdRng) {
        let child = self
            .scenario
            .children
            .get(policy as usize)
            .expect("only take search_depth steps");
        // .expect("only take search_depth steps");
        let dist = child.distribution.as_ref().expect("not root-level node");

        // eprintln!(
        //     "depth: {} and {:?}",
        //     self.depth, self.particle.bad_situation_depth
        // );
        if self.particle.bad_situation.map_or(false, |(d, p)| {
            d == self.depth && p >= child.bad_situation_threshold
        }) {
            self.cost += child.bad_threshold_cost;
        } else {
            let cost = dist.sample(rng); //.max(0.0).min(2.0 * dist.mean());
            self.cost += cost;
        }

        self.scenario = child;
        self.depth += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use fstrings::{eprintln_f, format_args_f};
    use rand::SeedableRng;

    #[test]
    fn test_expected_marginal_cost() {
        let full_seed = [0; 32];
        let mut rng = StdRng::from_seed(full_seed);

        let bad_situation_p = 0.2;

        let scenario = ProblemScenario::new(4, 4, 0.5, bad_situation_p, 10000.0, &mut rng);

        let mut mean_cost = 0.0;
        let mut costs_n = 0;

        for i in 0..100000 {
            let mut sim = Simulator::sample(&scenario, i, bad_situation_p, &mut rng);
            sim.take_step(0, &mut rng);

            mean_cost += sim.cost;
            costs_n += 1;
        }

        mean_cost /= costs_n as f64;

        let true_mean_cost = scenario.children[0].expected_marginal_cost();

        let c0 = &scenario.children[0];
        let d_mean = c0.distribution.as_ref().unwrap().mean();
        eprintln_f!("{c0.bad_situation_threshold=:.4}, {d_mean=:.2}");

        assert_abs_diff_eq!(mean_cost, true_mean_cost, epsilon = 10.0);
    }
}
