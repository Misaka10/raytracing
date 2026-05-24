use crate::hittable::HitRecord;
use crate::pdf::Pdf;
use crate::ray::Ray;
use crate::texture::Texture;
use crate::vec3::{self, Color, Vec3};
use rand::Rng;

#[derive(Clone)]
pub struct ScatterRecord {
    pub attenuation: Color,
    pub pdf_ptr: Option<Pdf>,
    pub skip_pdf: bool,
    pub skip_pdf_ray: Ray,
}

impl Default for ScatterRecord {
    fn default() -> Self {
        Self {
            attenuation: Color::zero(),
            pdf_ptr: None,
            skip_pdf: false,
            skip_pdf_ray: Ray::new(Vec3::zero().into(), Vec3::zero(), 0.0),
        }
    }
}

#[derive(Clone)]
pub enum Material {
    Lambertian { tex: Texture },
    Metal { albedo: Color, fuzz: f64 },
    Dielectric { refraction_index: f64 },
    DiffuseLight { tex: Texture },
    Isotropic { tex: Texture },
}

impl Material {
    pub fn lambertian_color(albedo: Color) -> Self {
        Material::Lambertian { tex: Texture::solid_color(albedo) }
    }
    pub fn lambertian_texture(tex: Texture) -> Self {
        Material::Lambertian { tex }
    }
    pub fn metal(albedo: Color, fuzz: f64) -> Self {
        Material::Metal { albedo, fuzz: fuzz.min(1.0) }
    }
    pub fn dielectric(ri: f64) -> Self { Material::Dielectric { refraction_index: ri } }
    pub fn diffuse_light_color(emit: Color) -> Self {
        Material::DiffuseLight { tex: Texture::solid_color(emit) }
    }
    pub fn diffuse_light_texture(tex: Texture) -> Self {
        Material::DiffuseLight { tex }
    }
    pub fn isotropic_color(albedo: Color) -> Self {
        Material::Isotropic { tex: Texture::solid_color(albedo) }
    }
    pub fn isotropic_texture(tex: Texture) -> Self {
        Material::Isotropic { tex }
    }

    pub fn emitted(&self, _r_in: &Ray, rec: &HitRecord, u: f64, v: f64, p: &Vec3) -> Color {
        match self {
            Material::DiffuseLight { tex } => {
                if rec.front_face { tex.value(u, v, p) } else { Color::zero() }
            }
            _ => Color::zero(),
        }
    }

    pub fn scatter<R: Rng>(&self, r_in: &Ray, rec: &HitRecord, srec: &mut ScatterRecord, rng: &mut R) -> bool {
        match self {
            Material::Lambertian { tex } => {
                srec.attenuation = tex.value(rec.u, rec.v, &rec.p);
                srec.pdf_ptr = Some(Pdf::cosine(&rec.normal));
                srec.skip_pdf = false;
                true
            }
            Material::Metal { albedo, fuzz } => {
                let mut reflected = vec3::reflect(&r_in.dir, &rec.normal);
                reflected = reflected.unit_vector() + *fuzz * vec3::random_unit_vector(rng);
                srec.attenuation = *albedo;
                srec.pdf_ptr = None;
                srec.skip_pdf = true;
                srec.skip_pdf_ray = Ray::new(rec.p, reflected, r_in.tm);
                true
            }
            Material::Dielectric { refraction_index } => {
                srec.attenuation = Color::new(1.0, 1.0, 1.0);
                srec.pdf_ptr = None;
                srec.skip_pdf = true;
                let ri = if rec.front_face { 1.0 / refraction_index } else { *refraction_index };
                let unit_dir = r_in.dir.unit_vector();
                let cos_theta = (-unit_dir).dot(&rec.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                let cannot_refract = ri * sin_theta > 1.0;
                let direction = if cannot_refract || reflectance(cos_theta, ri) > rng.gen::<f64>() {
                    vec3::reflect(&unit_dir, &rec.normal)
                } else {
                    vec3::refract(&unit_dir, &rec.normal, ri)
                };
                srec.skip_pdf_ray = Ray::new(rec.p, direction, r_in.tm);
                true
            }
            Material::Isotropic { tex } => {
                srec.attenuation = tex.value(rec.u, rec.v, &rec.p);
                srec.pdf_ptr = Some(Pdf::sphere());
                srec.skip_pdf = false;
                true
            }
            Material::DiffuseLight { .. } => false,
        }
    }

    pub fn scattering_pdf(&self, _r_in: &Ray, rec: &HitRecord, scattered: &Ray) -> f64 {
        match self {
            Material::Lambertian { .. } => {
                let cos_theta = rec.normal.dot(&scattered.dir.unit_vector());
                if cos_theta < 0.0 { 0.0 } else { cos_theta / std::f64::consts::PI }
            }
            Material::Isotropic { .. } => 1.0 / (4.0 * std::f64::consts::PI),
            _ => 0.0,
        }
    }
}

fn reflectance(cosine: f64, refraction_index: f64) -> f64 {
    let mut r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}
