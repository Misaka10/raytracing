use rand::Rng;

use crate::aabb::Aabb;
use crate::hittable::HitRecord;
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

#[derive(Clone)]
pub struct Quad {
    pub q: Point3,
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
    pub mat: Material,
    pub bbox: Aabb,
    pub normal: Vec3,
    d: f64,
    pub area: f64,
}

impl Quad {
    pub fn new(q: Point3, u: Vec3, v: Vec3, mat: Material) -> Self {
        let n = u.cross(&v);
        let normal = n.unit_vector();
        let d = normal.dot(&q);
        let w = n / n.dot(&n);
        let area = n.length();

        let bbox1 = Aabb::from_points(&q, &(q + u + v));
        let bbox2 = Aabb::from_points(&(q + u), &(q + v));
        let bbox = Aabb::from_boxes(&bbox1, &bbox2);

        Self { q, u, v, w, mat, bbox, normal, d, area }
    }

    pub fn hit(&self, r: &Ray, ray_t: &Interval, rec: &mut HitRecord) -> bool {
        let denom = self.normal.dot(&r.dir);
        if denom.abs() < 1e-8 {
            return false;
        }

        let t = (self.d - self.normal.dot(&r.orig)) / denom;
        if !ray_t.contains(t) {
            return false;
        }

        let intersection = r.at(t);
        let planar = intersection - self.q;
        let alpha = self.w.dot(&planar.cross(&self.v));
        let beta = self.w.dot(&self.u.cross(&planar));

        let unit = Interval::new(0.0, 1.0);
        if !unit.contains(alpha) || !unit.contains(beta) {
            return false;
        }

        rec.t = t;
        rec.p = intersection;
        rec.mat = self.mat.clone();
        rec.set_face_normal(r, &self.normal);
        rec.u = alpha;
        rec.v = beta;
        true
    }

    pub fn pdf_value(&self, origin: &Point3, direction: &Vec3) -> f64 {
        let test_ray = Ray::new(*origin, *direction, 0.0);
        let mut rec = HitRecord {
            p: Point3::zero(),
            normal: Vec3::zero(),
            mat: self.mat.clone(),
            t: 0.0,
            u: 0.0,
            v: 0.0,
            front_face: false,
        };
        if !self.hit(&test_ray, &Interval::new(0.001, f64::INFINITY), &mut rec) {
            return 0.0;
        }
        let dist_sq = rec.t * rec.t * direction.length_squared();
        let cosine = (direction.dot(&rec.normal) / direction.length()).abs();
        dist_sq / (cosine * self.area)
    }

    pub fn random<R: Rng>(&self, origin: &Point3, rng: &mut R) -> Vec3 {
        let p = self.q + rng.gen::<f64>() * self.u + rng.gen::<f64>() * self.v;
        p - *origin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quad_hit_center() {
        let q = Quad::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Material::lambertian_color(Vec3::zero()),
        );
        let r = Ray::new(Point3::new(1.0, 1.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 0.0);
        let mut rec = HitRecord {
            p: Point3::zero(),
            normal: Vec3::zero(),
            mat: Material::lambertian_color(Vec3::zero()),
            t: 0.0,
            u: 0.0,
            v: 0.0,
            front_face: false,
        };
        assert!(q.hit(&r, &Interval::new(0.001, f64::INFINITY), &mut rec));
        assert!((rec.t - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quad_miss_parallel() {
        let q = Quad::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Material::lambertian_color(Vec3::zero()),
        );
        let r = Ray::new(Point3::new(1.0, 1.0, -1.0), Vec3::new(1.0, 0.0, 0.0), 0.0);
        let mut rec = HitRecord {
            p: Point3::zero(),
            normal: Vec3::zero(),
            mat: Material::lambertian_color(Vec3::zero()),
            t: 0.0,
            u: 0.0,
            v: 0.0,
            front_face: false,
        };
        assert!(!q.hit(&r, &Interval::new(0.001, f64::INFINITY), &mut rec));
    }

    #[test]
    fn test_quad_outside_bounds() {
        let q = Quad::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Material::lambertian_color(Vec3::zero()),
        );
        let r = Ray::new(Point3::new(3.0, 3.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 0.0);
        let mut rec = HitRecord {
            p: Point3::zero(),
            normal: Vec3::zero(),
            mat: Material::lambertian_color(Vec3::zero()),
            t: 0.0,
            u: 0.0,
            v: 0.0,
            front_face: false,
        };
        assert!(!q.hit(&r, &Interval::new(0.001, f64::INFINITY), &mut rec));
    }

    #[test]
    fn test_quad_pdf_value() {
        let q = Quad::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Material::lambertian_color(Vec3::zero()),
        );
        let origin = Point3::new(1.0, 1.0, -1.0);
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let pdf = q.pdf_value(&origin, &dir);
        assert!(pdf > 0.0, "pdf should be > 0 for a ray that hits, got {}", pdf);
    }
}
