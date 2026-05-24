use crate::hittable::HitRecord;
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::Vec3;
use rand::Rng;

#[derive(Clone)]
pub struct ConstantMedium {
    pub boundary: Box<super::Hittable>,
    pub neg_inv_density: f64,
    pub phase_function: Material,
}

impl ConstantMedium {
    pub fn new(boundary: super::Hittable, density: f64, mat: Material) -> Self {
        Self { boundary: Box::new(boundary), neg_inv_density: -1.0 / density, phase_function: mat }
    }

    pub fn hit(&self, r: &Ray, ray_t: &Interval, rec: &mut HitRecord) -> bool {
        let mut rec1 = HitRecord {
            p: Vec3::zero().into(), normal: Vec3::zero(), mat: self.phase_function.clone(),
            t: 0.0, u: 0.0, v: 0.0, front_face: false,
        };
        let mut rec2 = rec1.clone();

        if !self.boundary.hit(r, &Interval::UNIVERSE, &mut rec1) { return false; }
        if !self.boundary.hit(r, &Interval::new(rec1.t + 0.0001, f64::INFINITY), &mut rec2) { return false; }

        let t1 = rec1.t.max(ray_t.min);
        let t2 = rec2.t.min(ray_t.max);
        if t1 >= t2 { return false; }
        let t1 = if t1 < 0.0 { 0.0 } else { t1 };

        let ray_length = r.dir.length();
        let distance_inside = (t2 - t1) * ray_length;
        let mut rng = rand::thread_rng();
        let hit_distance = self.neg_inv_density * rng.gen::<f64>().ln();

        if hit_distance > distance_inside { return false; }

        rec.t = t1 + hit_distance / ray_length;
        rec.p = r.at(rec.t);
        rec.normal = Vec3::new(1.0, 0.0, 0.0);
        rec.front_face = true;
        rec.mat = self.phase_function.clone();
        true
    }
}
