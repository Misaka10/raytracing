use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub x: Interval,
    pub y: Interval,
    pub z: Interval,
}

impl Default for Aabb {
    fn default() -> Self {
        Self { x: Interval::EMPTY, y: Interval::EMPTY, z: Interval::EMPTY }
    }
}

impl Aabb {
    pub fn from_intervals(x: Interval, y: Interval, z: Interval) -> Self {
        let mut aabb = Self { x, y, z };
        aabb.pad_to_minimums();
        aabb
    }

    pub fn from_points(a: &Point3, b: &Point3) -> Self {
        let x = if a.e[0] <= b.e[0] { Interval::new(a.e[0], b.e[0]) } else { Interval::new(b.e[0], a.e[0]) };
        let y = if a.e[1] <= b.e[1] { Interval::new(a.e[1], b.e[1]) } else { Interval::new(b.e[1], a.e[1]) };
        let z = if a.e[2] <= b.e[2] { Interval::new(a.e[2], b.e[2]) } else { Interval::new(b.e[2], a.e[2]) };
        let mut aabb = Self { x, y, z };
        aabb.pad_to_minimums();
        aabb
    }

    pub fn from_boxes(box0: &Aabb, box1: &Aabb) -> Self {
        Self {
            x: Interval::from_intervals(&box0.x, &box1.x),
            y: Interval::from_intervals(&box0.y, &box1.y),
            z: Interval::from_intervals(&box0.z, &box1.z),
        }
    }

    pub fn axis_interval(&self, n: usize) -> &Interval {
        match n {
            1 => &self.y,
            2 => &self.z,
            _ => &self.x,
        }
    }

    pub fn hit(&self, r: &Ray, mut ray_t: Interval) -> bool {
        for axis in 0..3 {
            let ax = self.axis_interval(axis);
            let adinv = 1.0 / r.dir.e[axis];
            let t0 = (ax.min - r.orig.e[axis]) * adinv;
            let t1 = (ax.max - r.orig.e[axis]) * adinv;
            if t0 < t1 {
                if t0 > ray_t.min { ray_t.min = t0; }
                if t1 < ray_t.max { ray_t.max = t1; }
            } else {
                if t1 > ray_t.min { ray_t.min = t1; }
                if t0 < ray_t.max { ray_t.max = t0; }
            }
            if ray_t.max <= ray_t.min { return false; }
        }
        true
    }

    pub fn longest_axis(&self) -> usize {
        if self.x.size() > self.y.size() {
            if self.x.size() > self.z.size() { 0 } else { 2 }
        } else {
            if self.y.size() > self.z.size() { 1 } else { 2 }
        }
    }

    fn pad_to_minimums(&mut self) {
        let delta = 0.0001;
        if self.x.size() < delta { self.x = self.x.expand(delta); }
        if self.y.size() < delta { self.y = self.y.expand(delta); }
        if self.z.size() < delta { self.z = self.z.expand(delta); }
    }
}

impl std::ops::Add<Vec3> for Aabb {
    type Output = Aabb;
    fn add(self, offset: Vec3) -> Aabb {
        Aabb::from_intervals(
            self.x + offset.x(),
            self.y + offset.y(),
            self.z + offset.z(),
        )
    }
}

impl std::ops::Add<Aabb> for Vec3 {
    type Output = Aabb;
    fn add(self, bbox: Aabb) -> Aabb { bbox + self }
}
