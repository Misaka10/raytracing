use crate::aabb::Aabb;
use crate::hittable::HitRecord;
use crate::interval::Interval;
use crate::material::Material;
use crate::onb::Onb;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};
use rand::Rng;

#[derive(Clone)]
pub struct Sphere {
    pub center: Ray,
    pub radius: f64,
    pub mat: Material,
    pub bbox: Aabb,
}

impl Sphere {
    pub fn stationary(center: Point3, radius: f64, mat: Material) -> Self {
        let rvec = Vec3::new(radius, radius, radius);
        let bbox = Aabb::from_points(&(center - rvec), &(center + rvec));
        Self { center: Ray::new(center, Vec3::zero(), 0.0), radius: radius.max(0.0), mat, bbox }
    }

    pub fn moving(center1: Point3, center2: Point3, radius: f64, mat: Material) -> Self {
        let rvec = Vec3::new(radius, radius, radius);
        let box1 = Aabb::from_points(&(center1 - rvec), &(center1 + rvec));
        let box2 = Aabb::from_points(&(center2 - rvec), &(center2 + rvec));
        let bbox = Aabb::from_boxes(&box1, &box2);
        Self { center: Ray::new(center1, center2 - center1, 0.0), radius: radius.max(0.0), mat, bbox }
    }

    pub fn hit(&self, r: &Ray, ray_t: &Interval, rec: &mut HitRecord) -> bool {
        let current_center = self.center.at(r.tm);
        let oc = current_center - r.orig;
        let a = r.dir.length_squared();
        let h = r.dir.dot(&oc);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = h * h - a * c;
        if discriminant < 0.0 { return false; }

        let sqrtd = discriminant.sqrt();
        let mut root = (h - sqrtd) / a;
        if !ray_t.surrounds(root) {
            root = (h + sqrtd) / a;
            if !ray_t.surrounds(root) { return false; }
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        let outward_normal = (rec.p - current_center) / self.radius;
        rec.set_face_normal(r, &outward_normal);
        get_sphere_uv(&outward_normal, &mut rec.u, &mut rec.v);
        rec.mat = self.mat.clone();
        true
    }

    pub fn pdf_value(&self, origin: &Point3, direction: &Vec3) -> f64 {
        let mut rec = HitRecord { p: Point3::zero(), normal: Vec3::zero(), mat: self.mat.clone(), t: 0.0, u: 0.0, v: 0.0, front_face: false };
        let test_ray = Ray::new(*origin, *direction, 0.0);
        if !self.hit(&test_ray, &Interval::new(0.001, f64::INFINITY), &mut rec) { return 0.0; }

        let dist_sq = (self.center.at(0.0) - *origin).length_squared();
        let cos_theta_max = (1.0 - self.radius * self.radius / dist_sq).sqrt();
        let solid_angle = 2.0 * std::f64::consts::PI * (1.0 - cos_theta_max);
        1.0 / solid_angle
    }

    pub fn random<R: Rng>(&self, origin: &Point3, rng: &mut R) -> Vec3 {
        let direction = self.center.at(0.0) - *origin;
        let distance_squared = direction.length_squared();
        let uvw = Onb::new(&direction);
        uvw.transform(&random_to_sphere(self.radius, distance_squared, rng))
    }
}

fn get_sphere_uv(p: &Vec3, u: &mut f64, v: &mut f64) {
    let theta = (-p.y()).acos();
    let phi = (-p.z()).atan2(p.x()) + std::f64::consts::PI;
    *u = phi / (2.0 * std::f64::consts::PI);
    *v = theta / std::f64::consts::PI;
}

fn random_to_sphere<R: Rng>(radius: f64, distance_squared: f64, rng: &mut R) -> Vec3 {
    let r1: f64 = rng.gen();
    let r2: f64 = rng.gen();
    let z = 1.0 + r2 * ((1.0 - radius * radius / distance_squared).sqrt() - 1.0);
    let phi = 2.0 * std::f64::consts::PI * r1;
    let x = phi.cos() * (1.0 - z * z).sqrt();
    let y = phi.sin() * (1.0 - z * z).sqrt();
    Vec3::new(x, y, z)
}
