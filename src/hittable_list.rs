use rand::Rng;

use crate::aabb::Aabb;
use crate::hittable::HitRecord;
use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

#[derive(Default, Clone)]
pub struct HittableList {
    pub objects: Vec<super::Hittable>,
    pub bbox: Aabb,
}

impl HittableList {
    pub fn new() -> Self {
        Self { objects: Vec::new(), bbox: Aabb::default() }
    }

    pub fn add(&mut self, object: super::Hittable) {
        self.bbox = Aabb::from_boxes(&self.bbox, &object.bounding_box());
        self.objects.push(object);
    }

    pub fn hit(&self, r: &Ray, ray_t: &Interval, rec: &mut HitRecord) -> bool {
        let mut hit_anything = false;
        let mut closest_so_far = ray_t.max;

        for object in &self.objects {
            let temp_interval = Interval::new(ray_t.min, closest_so_far);
            if object.hit(r, &temp_interval, rec) {
                hit_anything = true;
                closest_so_far = rec.t;
            }
        }
        hit_anything
    }

    pub fn pdf_value(&self, origin: &Point3, direction: &Vec3) -> f64 {
        if self.objects.is_empty() {
            return 0.0;
        }
        let weight = 1.0 / self.objects.len() as f64;
        let mut sum = 0.0;
        for obj in &self.objects {
            sum += weight * obj.pdf_value(origin, direction);
        }
        sum
    }

    pub fn random<R: Rng>(&self, origin: &Point3, rng: &mut R) -> Vec3 {
        if self.objects.is_empty() {
            return Vec3::new(1.0, 0.0, 0.0);
        }
        let idx = rng.gen_range(0..self.objects.len());
        self.objects[idx].random(origin, rng)
    }
}
