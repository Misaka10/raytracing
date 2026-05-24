use crate::aabb::Aabb;
use crate::hittable::HitRecord;
use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};
use rand::Rng;

#[derive(Clone)]
pub enum BvhNode {
    Leaf {
        object: Box<super::Hittable>,
        bbox: Aabb,
    },
    Split {
        left: Box<BvhNode>,
        right: Box<BvhNode>,
        bbox: Aabb,
    },
}

impl BvhNode {
    pub fn from_objects(objects: &mut [super::Hittable]) -> Self {
        let mut bbox = Aabb::default();
        for obj in objects.iter() {
            bbox = Aabb::from_boxes(&bbox, &obj.bounding_box());
        }

        let span = objects.len();

        if span == 1 {
            return BvhNode::Leaf { object: Box::new(objects[0].clone()), bbox };
        }

        if span == 2 {
            let left = Box::new(BvhNode::Leaf {
                object: Box::new(objects[0].clone()),
                bbox: objects[0].bounding_box(),
            });
            let right = Box::new(BvhNode::Leaf {
                object: Box::new(objects[1].clone()),
                bbox: objects[1].bounding_box(),
            });
            return BvhNode::Split { left, right, bbox };
        }

        let axis = bbox.longest_axis();
        objects.sort_by(|a, b| {
            let aa = a.bounding_box().axis_interval(axis).min;
            let bb = b.bounding_box().axis_interval(axis).min;
            aa.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = span / 2;
        let left = Box::new(BvhNode::from_objects(&mut objects[..mid]));
        let right = Box::new(BvhNode::from_objects(&mut objects[mid..]));

        BvhNode::Split { left, right, bbox }
    }

    pub fn bbox(&self) -> Aabb {
        match self {
            BvhNode::Leaf { bbox, .. } => *bbox,
            BvhNode::Split { bbox, .. } => *bbox,
        }
    }

    pub fn hit(&self, r: &Ray, ray_t: &Interval, rec: &mut HitRecord) -> bool {
        match self {
            BvhNode::Leaf { object, bbox } => {
                if !bbox.hit(r, *ray_t) { return false; }
                object.hit(r, ray_t, rec)
            }
            BvhNode::Split { left, right, bbox } => {
                if !bbox.hit(r, *ray_t) { return false; }
                let hit_left = left.hit(r, ray_t, rec);
                let right_interval = Interval::new(ray_t.min, if hit_left { rec.t } else { ray_t.max });
                let hit_right = right.hit(r, &right_interval, rec);
                hit_left || hit_right
            }
        }
    }

    pub fn pdf_value(&self, origin: &Point3, direction: &Vec3) -> f64 {
        match self {
            BvhNode::Leaf { object, .. } => object.pdf_value(origin, direction),
            BvhNode::Split { left, right, .. } => {
                0.5 * left.pdf_value(origin, direction) + 0.5 * right.pdf_value(origin, direction)
            }
        }
    }

    pub fn random<R: Rng>(&self, origin: &Point3, rng: &mut R) -> Vec3 {
        match self {
            BvhNode::Leaf { object, .. } => object.random(origin, rng),
            BvhNode::Split { left, right, .. } => {
                if rng.gen::<f64>() < 0.5 { left.random(origin, rng) } else { right.random(origin, rng) }
            }
        }
    }
}
