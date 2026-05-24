use crate::vec3::{Point3, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub orig: Point3,
    pub dir: Vec3,
    pub tm: f64,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3, time: f64) -> Self {
        Self { orig: origin, dir: direction, tm: time }
    }

    pub fn at(&self, t: f64) -> Point3 {
        self.orig + self.dir * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_at() {
        let r = Ray::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 0.0, 0.0), 0.0);
        let p = r.at(5.0);
        assert_eq!(p.e, [6.0, 2.0, 3.0]);
    }

    #[test]
    fn test_at_zero() {
        let r = Ray::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 2.0, 3.0), 0.5);
        let p = r.at(0.0);
        assert_eq!(p.e, [1.0, 2.0, 3.0]);
    }
}
