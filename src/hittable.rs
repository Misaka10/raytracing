use crate::aabb::Aabb;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

#[derive(Clone)]
pub struct HitRecord {
    pub p: Point3,
    pub normal: Vec3,
    pub mat: Material,
    pub t: f64,
    pub u: f64,
    pub v: f64,
    pub front_face: bool,
}

impl HitRecord {
    pub fn set_face_normal(&mut self, r: &Ray, outward_normal: &Vec3) {
        self.front_face = r.dir.dot(outward_normal) < 0.0;
        self.normal = if self.front_face { *outward_normal } else { -*outward_normal };
    }
}

#[derive(Clone)]
pub enum Hittable {
    Sphere(crate::sphere::Sphere),
    Quad(crate::quad::Quad),
    HittableList(crate::hittable_list::HittableList),
    BvhNode(crate::bvh::BvhNode),
    Translate(Box<Hittable>, Vec3, Aabb),
    RotateY(Box<Hittable>, f64, f64, Aabb),
    ConstantMedium(crate::constant_medium::ConstantMedium),
}

impl Hittable {
    pub fn translate(object: Hittable, offset: Vec3) -> Self {
        let bbox = object.bounding_box() + offset;
        Hittable::Translate(Box::new(object), offset, bbox)
    }

    pub fn rotate_y(object: Hittable, angle: f64) -> Self {
        let radians = angle.to_radians();
        let sin_theta = radians.sin();
        let cos_theta = radians.cos();
        let obj_bbox = object.bounding_box();

        let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let x = i as f64 * obj_bbox.x.max + (1 - i) as f64 * obj_bbox.x.min;
                    let y = j as f64 * obj_bbox.y.max + (1 - j) as f64 * obj_bbox.y.min;
                    let z = k as f64 * obj_bbox.z.max + (1 - k) as f64 * obj_bbox.z.min;
                    let newx = cos_theta * x + sin_theta * z;
                    let newz = -sin_theta * x + cos_theta * z;
                    let tester = Vec3::new(newx, y, newz);
                    if tester.x() < min.e[0] { min.e[0] = tester.x(); }
                    if tester.y() < min.e[1] { min.e[1] = tester.y(); }
                    if tester.z() < min.e[2] { min.e[2] = tester.z(); }
                    if tester.x() > max.e[0] { max.e[0] = tester.x(); }
                    if tester.y() > max.e[1] { max.e[1] = tester.y(); }
                    if tester.z() > max.e[2] { max.e[2] = tester.z(); }
                }
            }
        }
        Hittable::RotateY(Box::new(object), sin_theta, cos_theta, Aabb::from_points(&min, &max))
    }

    pub fn hit(&self, r: &Ray, ray_t: &crate::interval::Interval, rec: &mut HitRecord) -> bool {
        match self {
            Hittable::Sphere(s) => s.hit(r, ray_t, rec),
            Hittable::Quad(q) => q.hit(r, ray_t, rec),
            Hittable::HittableList(l) => l.hit(r, ray_t, rec),
            Hittable::BvhNode(b) => b.hit(r, ray_t, rec),
            Hittable::Translate(object, offset, _) => {
                let offset_r = Ray::new(r.orig - *offset, r.dir, r.tm);
                if !object.hit(&offset_r, ray_t, rec) { return false; }
                rec.p = rec.p + *offset;
                true
            }
            Hittable::RotateY(object, sin_theta, cos_theta, _) => {
                let orig = Point3::new(
                    cos_theta * r.orig.x() - sin_theta * r.orig.z(),
                    r.orig.y(),
                    sin_theta * r.orig.x() + cos_theta * r.orig.z(),
                );
                let dir = Vec3::new(
                    cos_theta * r.dir.x() - sin_theta * r.dir.z(),
                    r.dir.y(),
                    sin_theta * r.dir.x() + cos_theta * r.dir.z(),
                );
                let rotated_r = Ray::new(orig, dir, r.tm);
                if !object.hit(&rotated_r, ray_t, rec) { return false; }
                rec.p = Point3::new(
                    cos_theta * rec.p.x() + sin_theta * rec.p.z(),
                    rec.p.y(),
                    -sin_theta * rec.p.x() + cos_theta * rec.p.z(),
                );
                rec.normal = Vec3::new(
                    cos_theta * rec.normal.x() + sin_theta * rec.normal.z(),
                    rec.normal.y(),
                    -sin_theta * rec.normal.x() + cos_theta * rec.normal.z(),
                );
                true
            }
            Hittable::ConstantMedium(cm) => cm.hit(r, ray_t, rec),
        }
    }

    pub fn bounding_box(&self) -> Aabb {
        match self {
            Hittable::Sphere(s) => s.bbox,
            Hittable::Quad(q) => q.bbox,
            Hittable::HittableList(l) => l.bbox,
            Hittable::BvhNode(b) => b.bbox(),
            Hittable::Translate(_, _, bbox) => *bbox,
            Hittable::RotateY(_, _, _, bbox) => *bbox,
            Hittable::ConstantMedium(cm) => cm.boundary.bounding_box(),
        }
    }

    pub fn pdf_value(&self, origin: &Point3, direction: &Vec3) -> f64 {
        match self {
            Hittable::Sphere(s) => s.pdf_value(origin, direction),
            Hittable::Quad(q) => q.pdf_value(origin, direction),
            Hittable::HittableList(l) => l.pdf_value(origin, direction),
            Hittable::BvhNode(b) => b.pdf_value(origin, direction),
            _ => 0.0,
        }
    }

    pub fn random(&self, origin: &Point3, rng: &mut impl rand::Rng) -> Vec3 {
        match self {
            Hittable::Sphere(s) => s.random(origin, rng),
            Hittable::Quad(q) => q.random(origin, rng),
            Hittable::HittableList(l) => l.random(origin, rng),
            Hittable::BvhNode(b) => b.random(origin, rng),
            _ => Vec3::new(1.0, 0.0, 0.0),
        }
    }
}
