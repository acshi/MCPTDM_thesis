use rand::{prelude::StdRng, Rng};
use rand_distr::{Bernoulli, Distribution, Normal, StandardNormal};

#[derive(Clone, Copy)]
pub struct SituationParticle {
    pub id: usize,
    pub gaussian_z: f64,
    pub bernoulli_p: f64,
}

impl Eq for SituationParticle {}

impl PartialOrd for SituationParticle {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl PartialEq for SituationParticle {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl std::hash::Hash for SituationParticle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.id);
    }
}

impl std::fmt::Debug for SituationParticle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", (self.id, self.gaussian_z, self.bernoulli_p))
    }
}

impl SituationParticle {
    pub fn sample(id: usize, rng: &mut StdRng) -> Self {
        Self {
            id,
            gaussian_z: StandardNormal.sample(rng),
            bernoulli_p: rng.gen_range(0.0..=1.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CostDistribution {
    normal_d: Normal<f64>,
    bernoulli_d: Bernoulli,
    bernoulli_p: f64,
    bernoulli_mag: f64,
}

impl CostDistribution {
    pub fn new(
        normal_mean: f64,
        normal_std_dev: f64,
        bernoulli_p: f64,
        bernoulli_mag: f64,
    ) -> Self {
        Self {
            normal_d: Normal::new(normal_mean, normal_std_dev)
                .expect("valid mean and standard deviation"),
            bernoulli_d: Bernoulli::new(bernoulli_p).expect("probability from 0 to 1"),
            bernoulli_p,
            bernoulli_mag,
        }
    }

    pub fn mean(&self) -> f64 {
        self.normal_d.mean() + self.bernoulli_p * self.bernoulli_mag
    }

    pub fn sample(&self, rng: &mut StdRng) -> f64 {
        // self.normal_d
        //     .sample(rng)
        //     .max(0.0)
        //     .min(2.0 * self.normal_d.mean())
        //     + if self.bernoulli_d.sample(rng) {
        //         self.bernoulli_mag
        //     } else {
        //         0.0
        //     }
        self.from_correlated(StandardNormal.sample(rng), rng.gen_range(0.0..=1.0))
    }

    pub fn from_correlated(&self, gaussian_z: f64, bernoulli_p: f64) -> f64 {
        self.normal_d
            .from_zscore(gaussian_z)
            .max(0.0)
            .min(2.0 * self.normal_d.mean())
            + if bernoulli_p <= self.bernoulli_p {
                self.bernoulli_mag
            } else {
                0.0
            }
    }
}

#[derive(Clone)]
pub struct ProblemScenario {
    pub distribution: Option<CostDistribution>,
    pub children: Vec<ProblemScenario>,
    pub depth: u32,
    pub max_depth: u32,
}

impl ProblemScenario {
    fn inner_new(depth: u32, max_depth: u32, n_actions: u32, rng: &mut StdRng) -> Self {
        Self {
            distribution: if depth == 0 {
                None
            } else {
                let normal_mean = rng.gen_range(0.0..100.0);
                let normal_std_dev = rng.gen_range(0.0..100.0);
                let bernoulli_p = rng.gen_range(0.0..=1.0);
                let bernoulli_mag = 1000.0;

                Some(CostDistribution::new(
                    normal_mean,
                    normal_std_dev,
                    bernoulli_p,
                    bernoulli_mag,
                ))
            },
            children: if depth < max_depth {
                (0..n_actions)
                    .map(|_| Self::inner_new(depth + 1, max_depth, n_actions, rng))
                    .collect()
            } else {
                Vec::new()
            },
            depth,
            max_depth,
        }
    }

    pub fn new(max_depth: u32, n_actions: u32, rng: &mut StdRng) -> Self {
        Self::inner_new(0, max_depth, n_actions, rng)
    }

    pub fn expected_marginal_cost(&self) -> f64 {
        self.distribution
            .as_ref()
            .map(|d| d.mean() * 2.0)
            .unwrap_or(0.0)
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
    pub fn sample(scenario: &'a ProblemScenario, id: usize, rng: &mut StdRng) -> Self {
        Self {
            scenario,
            particle: SituationParticle::sample(id, rng),
            depth: 0,
            cost: 0.0,
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
        self.cost += dist.sample(rng)
            + dist.from_correlated(self.particle.gaussian_z, self.particle.bernoulli_p);

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
        let full_seed = [1; 32];
        let mut rng = StdRng::from_seed(full_seed);

        let scenario = ProblemScenario::new(4, 4, &mut rng);

        let mut mean_cost = 0.0;
        let mut costs_n = 0;

        for i in 0..50000 {
            let mut sim = Simulator::sample(&scenario, i, &mut rng);
            sim.take_step(0, &mut rng);

            mean_cost += sim.cost;
            costs_n += 1;
        }

        mean_cost /= costs_n as f64;

        let true_mean_cost = scenario.children[0].expected_marginal_cost();

        let c0 = &scenario.children[0];
        let distribution = c0.distribution.as_ref().unwrap();
        eprintln_f!("{distribution=:.2?}");

        assert_abs_diff_eq!(mean_cost, true_mean_cost, epsilon = 10.0);
    }
}
