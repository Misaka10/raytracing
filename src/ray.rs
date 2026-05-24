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
