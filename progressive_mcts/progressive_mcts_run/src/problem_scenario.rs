use rand::{
    prelude::{IteratorRandom, StdRng},
    Rng,
};
use rand_distr::{Bernoulli, Distribution, Normal};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SituationParticle {
    pub id: usize,
    pub bad_situation_depth: Option<u32>,
}

impl std::fmt::Debug for SituationParticle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {:?})", self.id, self.bad_situation_depth)
    }
}

impl SituationParticle {
    pub fn sample(id: usize, p: f64, max_depth: u32, rng: &mut StdRng) -> Self {
        Self {
            id,
            bad_situation_depth: if rng.gen_bool(p) {
                Some(rng.gen_range(0..max_depth))
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
    },
    Bernoulli {
        d: Bernoulli,
        p: f64,
        magnitude: f64,
    },
}

impl CostDistribution {
    #[allow(unused)]
    pub fn normal(mean: f64, std_dev: f64) -> Self {
        Self::Normal {
            d: Normal::new(mean, std_dev).expect("valid mean and standard deviation"),
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
            CostDistribution::Normal { d } => d.mean(),
            CostDistribution::Bernoulli { d: _, p, magnitude } => p * magnitude,
        }
    }

    pub fn magnitude(&self) -> f64 {
        match self {
            // for normal, return magnitude of the corresponding bernoulli distribution
            CostDistribution::Normal { d } => d.std_dev().powi(2) / d.mean() + d.mean(),
            CostDistribution::Bernoulli {
                d: _,
                p: _,
                magnitude,
            } => *magnitude,
        }
    }

    pub fn sample(&self, rng: &mut StdRng) -> f64 {
        match self {
            CostDistribution::Normal { d } => d.sample(rng).max(0.0).min(2.0 * d.mean()),
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
                    Some(CostDistribution::normal(mean, std_dev))
                }
            },
            children: if depth < max_depth {
                (0..n_actions)
                    .map(|_| {
                        Self::inner_new(depth + 1, max_depth, n_actions, portion_bernoulli, rng)
                    })
                    .collect()
            } else {
                Vec::new()
            },
            depth,
            max_depth,
        }
    }

    pub fn new(max_depth: u32, n_actions: u32, portion_bernoulli: f64, rng: &mut StdRng) -> Self {
        Self::inner_new(0, max_depth, n_actions, portion_bernoulli, rng)
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

        eprintln!(
            "depth: {} and {:?}",
            self.depth, self.particle.bad_situation_depth
        );
        if self.particle.bad_situation_depth == Some(self.depth) {
            self.cost += dist.magnitude() * 2.5;
        } else {
            let cost = dist.sample(rng); //.max(0.0).min(2.0 * dist.mean());
            self.cost += cost;
        }

        self.scenario = child;
        self.depth += 1;
    }
}
