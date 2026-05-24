#[derive(Debug, Clone, Copy)]
pub struct Interval {
    pub min: f64,
    pub max: f64,
}

impl Interval {
    pub const fn new(min: f64, max: f64) -> Self { Self { min, max } }

    pub const EMPTY: Interval = Interval { min: f64::INFINITY, max: f64::NEG_INFINITY };
    pub const UNIVERSE: Interval = Interval { min: f64::NEG_INFINITY, max: f64::INFINITY };

    pub fn size(&self) -> f64 { self.max - self.min }

    pub fn contains(&self, x: f64) -> bool { self.min <= x && x <= self.max }

    pub fn surrounds(&self, x: f64) -> bool { self.min < x && x < self.max }

    pub fn clamp(&self, x: f64) -> f64 {
        if x < self.min { self.min } else if x > self.max { self.max } else { x }
    }

    pub fn expand(&self, delta: f64) -> Interval {
        let padding = delta / 2.0;
        Interval::new(self.min - padding, self.max + padding)
    }

    pub fn from_intervals(a: &Interval, b: &Interval) -> Interval {
        Interval::new(a.min.min(b.min), a.max.max(b.max))
    }
}

impl std::ops::Add<f64> for Interval {
    type Output = Interval;
    fn add(self, displacement: f64) -> Interval {
        Interval::new(self.min + displacement, self.max + displacement)
    }
}

impl std::ops::Add<Interval> for f64 {
    type Output = Interval;
    fn add(self, ival: Interval) -> Interval { ival + self }
}
