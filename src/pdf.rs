use rand::Rng;

use crate::onb::Onb;
use crate::vec3::{self, Vec3};

#[derive(Clone)]
pub enum Pdf {
    Sphere,
    Cosine(Onb),
    Mixture(Box<Pdf>, Box<Pdf>),
}

impl Pdf {
    pub fn sphere() -> Self {
        Pdf::Sphere
    }

    pub fn cosine(normal: &Vec3) -> Self {
        Pdf::Cosine(Onb::new(normal))
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
            Pdf::Mixture(p0, p1) => 0.5 * p0.value(direction) + 0.5 * p1.value(direction),
        }
    }

    pub fn generate<R: Rng>(&self, rng: &mut R) -> Vec3 {
        match self {
            Pdf::Sphere => vec3::random_unit_vector(rng),
            Pdf::Cosine(uvw) => uvw.transform(&vec3::random_cosine_direction(rng)),
            Pdf::Mixture(p0, p1) => {
                if rng.gen::<f64>() < 0.5 {
                    p0.generate(rng)
                } else {
                    p1.generate(rng)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn test_sphere_pdf_value() {
        let pdf = Pdf::sphere();
        let val = pdf.value(&Vec3::new(0.0, 0.0, 1.0));
        assert!((val - 1.0 / (4.0 * std::f64::consts::PI)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_pdf_value_normal() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let pdf = Pdf::cosine(&normal);
        let val = pdf.value(&Vec3::new(0.0, 0.0, 1.0));
        assert!(val > 0.0);
    }

    #[test]
    fn test_cosine_pdf_value_opposite() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let pdf = Pdf::cosine(&normal);
        let val = pdf.value(&Vec3::new(0.0, 0.0, -1.0));
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_mixture_pdf_value() {
        let p0 = Pdf::sphere();
        let p1 = Pdf::sphere();
        let mixture = Pdf::mixture(p0, p1);
        let val = mixture.value(&Vec3::new(0.0, 0.0, 1.0));
        assert!((val - 1.0 / (4.0 * std::f64::consts::PI)).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_generate() {
        let mut rng = SmallRng::seed_from_u64(42);
        let pdf = Pdf::sphere();
        for _ in 0..100 {
            let v = pdf.generate(&mut rng);
            assert!((v.length() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cosine_generate_hemisphere() {
        let mut rng = SmallRng::seed_from_u64(42);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let pdf = Pdf::cosine(&normal);
        for _ in 0..100 {
            let v = pdf.generate(&mut rng);
            assert!(v.dot(&normal) > 0.0, "direction should be in positive hemisphere");
        }
    }
}
