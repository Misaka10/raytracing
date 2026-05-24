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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains() {
        let i = Interval::new(0.0, 1.0);
        assert!(i.contains(0.5));
        assert!(i.contains(0.0));
        assert!(i.contains(1.0));
        assert!(!i.contains(-0.1));
        assert!(!i.contains(1.1));
    }

    #[test]
    fn test_surrounds() {
        let i = Interval::new(0.0, 1.0);
        assert!(i.surrounds(0.5));
        assert!(!i.surrounds(0.0));
        assert!(!i.surrounds(1.0));
    }

    #[test]
    fn test_clamp() {
        let i = Interval::new(0.0, 1.0);
        assert_eq!(i.clamp(0.5), 0.5);
        assert_eq!(i.clamp(-1.0), 0.0);
        assert_eq!(i.clamp(2.0), 1.0);
    }

    #[test]
    fn test_expand() {
        let i = Interval::new(0.0, 10.0);
        let e = i.expand(2.0);
        assert_eq!(e.min, -1.0);
        assert_eq!(e.max, 11.0);
    }

    #[test]
    fn test_size() {
        assert_eq!(Interval::new(3.0, 7.0).size(), 4.0);
    }

    #[test]
    fn test_empty() {
        assert!(Interval::EMPTY.size() < 0.0);
    }

    #[test]
    fn test_add_offset() {
        let i = Interval::new(1.0, 5.0) + 2.0;
        assert_eq!(i.min, 3.0);
        assert_eq!(i.max, 7.0);
    }
}
