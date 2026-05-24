use crate::hittable_list::HittableList;
use crate::onb::Onb;
use crate::vec3::{self, Point3, Vec3};
use rand::Rng;

#[derive(Clone)]
pub enum Pdf {
    Sphere,
    Cosine(Onb),
    Hittable {
        objects: HittableList,
        origin: Point3,
    },
    Mixture(Box<Pdf>, Box<Pdf>),
}

impl Pdf {
    pub fn sphere() -> Self { Pdf::Sphere }

    pub fn cosine(normal: &Vec3) -> Self { Pdf::Cosine(Onb::new(normal)) }

    pub fn hittable(objects: &HittableList, origin: &Point3) -> Self {
        Pdf::Hittable { objects: objects.clone(), origin: *origin }
    }

    pub fn mixture(p0: Pdf, p1: Pdf) -> Self {
        Pdf::Mixture(Box::new(p0), Box::new(p1))
    }

    pub fn value(&self, direction: &Vec3) -> f64 {
        match self {
            Pdf::Sphere => 1.0 / (4.0 * std::f64::consts::PI),
            Pdf::Cosine(uvw) => {
                let cos_theta = direction.unit_vector().dot(uvw.w());
                cos_theta.max(0.0) / std::f64::consts::PI
            }
            Pdf::Hittable { objects, origin } => objects.pdf_value(origin, direction),
            Pdf::Mixture(p0, p1) => 0.5 * p0.value(direction) + 0.5 * p1.value(direction),
        }
    }

    pub fn generate<R: Rng>(&self, rng: &mut R) -> Vec3 {
        match self {
            Pdf::Sphere => vec3::random_unit_vector(rng),
            Pdf::Cosine(uvw) => uvw.transform(&vec3::random_cosine_direction(rng)),
            Pdf::Hittable { objects, origin } => objects.random(origin, rng),
            Pdf::Mixture(p0, p1) => {
                if rng.gen::<f64>() < 0.5 { p0.generate(rng) } else { p1.generate(rng) }
            }
        }
    }
}
