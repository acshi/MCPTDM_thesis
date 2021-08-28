use rand::{prelude::StdRng, Rng};
use rand_distr::{Distribution, Normal, StandardNormal};

#[derive(Clone, Copy)]
pub struct SituationParticle {
    pub id: usize,
    pub gaussian_z1: f64,
    pub gaussian_z2: f64,
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
        write!(f, "{:?}", (self.id, self.gaussian_z1, self.gaussian_z2))
    }
}

impl SituationParticle {
    pub fn sample(id: usize, rng: &mut StdRng) -> Self {
        Self {
            id,
            gaussian_z1: StandardNormal.sample(rng),
            gaussian_z2: StandardNormal.sample(rng),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CostDistribution {
    normal1: Normal<f64>,
    normal2: Normal<f64>,
}

impl CostDistribution {
    pub fn new(
        normal_mean1: f64,
        normal_std_dev1: f64,
        normal_mean2: f64,
        normal_std_dev2: f64,
    ) -> Self {
        Self {
            normal1: Normal::new(normal_mean1, normal_std_dev1)
                .expect("valid mean and standard deviation"),
            normal2: Normal::new(normal_mean2, normal_std_dev2)
                .expect("valid mean and standard deviation"),
        }
    }

    pub fn new_sampled(rng: &mut StdRng) -> Self {
        Self::new(
            rng.gen_range(0.0..100.0),
            rng.gen_range(0.0..100.0),
            rng.gen_range(0.0..100.0),
            rng.gen_range(0.0..100.0),
        )
    }

    pub fn mean(&self) -> f64 {
        self.normal1.mean() + self.normal2.mean()
    }

    pub fn sample(&self, rng: &mut StdRng) -> f64 {
        self.from_correlated(StandardNormal.sample(rng), StandardNormal.sample(rng))
    }

    pub fn from_correlated(&self, gaussian_z1: f64, gaussian_z2: f64) -> f64 {
        self.normal1
            .from_zscore(gaussian_z1)
            .max(0.0)
            .min(2.0 * self.normal1.mean())
            + self
                .normal2
                .from_zscore(gaussian_z2)
                .max(0.0)
                .min(2.0 * self.normal2.mean())
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
                Some(CostDistribution::new_sampled(rng))
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
            + dist.from_correlated(self.particle.gaussian_z1, self.particle.gaussian_z2);

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
    use rolling_stats::Stats;

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

    #[test]
    fn expected_std_dev() {
        let mut rng = StdRng::from_seed([0; 32]);

        let mut std_dev_stats = Stats::new();

        for _ in 0..1000 {
            let dist = CostDistribution::new_sampled(&mut rng);
            let mut stats = Stats::new();
            for _ in 0..2000 {
                let value = dist.sample(&mut rng);
                stats.update(value);
            }
            // eprintln!("{:.2}", stats.std_dev);
            std_dev_stats.update(stats.std_dev);
        }

        eprintln!("stats: {:.2?}", std_dev_stats);

        assert_eq!(std_dev_stats.mean, 0.0);
    }
}
