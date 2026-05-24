use crate::color_io;
use crate::hittable::HitRecord;
use crate::hittable_list::HittableList;
use crate::interval::Interval;
use crate::material::Material;
use crate::pdf::Pdf;
use crate::ray::Ray;
use crate::vec3::{self, Point3, Vec3};
use crate::Hittable;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Camera {
    pub aspect_ratio: f64,
    pub image_width: u32,
    pub image_height: u32,
    pub samples_per_pixel: u32,
    pub max_depth: u32,
    pub background: Vec3,
    pub vfov: f64,
    pub lookfrom: Point3,
    pub lookat: Point3,
    pub vup: Vec3,
    pub defocus_angle: f64,
    pub focus_dist: f64,

    pixel_samples_scale: f64,
    sqrt_spp: u32,
    recip_sqrt_spp: f64,
    center: Point3,
    pixel00_loc: Point3,
    pixel_delta_u: Vec3,
    pixel_delta_v: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    defocus_disk_u: Vec3,
    defocus_disk_v: Vec3,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            aspect_ratio: 1.0, image_width: 100, samples_per_pixel: 10, max_depth: 10,
            background: Vec3::zero(), vfov: 90.0, lookfrom: Point3::new(0.0, 0.0, 0.0),
            lookat: Point3::new(0.0, 0.0, -1.0), vup: Vec3::new(0.0, 1.0, 0.0),
            defocus_angle: 0.0, focus_dist: 10.0,
            image_height: 0, pixel_samples_scale: 0.0, sqrt_spp: 0, recip_sqrt_spp: 0.0,
            center: Point3::zero(), pixel00_loc: Point3::zero(),
            pixel_delta_u: Vec3::zero(), pixel_delta_v: Vec3::zero(),
            u: Vec3::zero(), v: Vec3::zero(), w: Vec3::zero(),
            defocus_disk_u: Vec3::zero(), defocus_disk_v: Vec3::zero(),
        }
    }
}

impl Camera {
    pub fn new() -> Self { Self::default() }

    pub fn initialize(&mut self) {
        if self.image_height == 0 {
            self.image_height = (self.image_width as f64 / self.aspect_ratio) as u32;
        }
        if self.image_height < 1 { self.image_height = 1; }

        self.sqrt_spp = (self.samples_per_pixel as f64).sqrt() as u32;
        self.pixel_samples_scale = 1.0 / (self.sqrt_spp * self.sqrt_spp) as f64;
        self.recip_sqrt_spp = 1.0 / self.sqrt_spp as f64;

        self.center = self.lookfrom;

        let theta = self.vfov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h * self.focus_dist;
        let viewport_width = viewport_height * (self.image_width as f64 / self.image_height as f64);

        self.w = (self.lookfrom - self.lookat).unit_vector();
        self.u = self.vup.cross(&self.w).unit_vector();
        self.v = self.w.cross(&self.u);

        let viewport_u = viewport_width * self.u;
        let viewport_v = viewport_height * -self.v;

        self.pixel_delta_u = viewport_u / self.image_width as f64;
        self.pixel_delta_v = viewport_v / self.image_height as f64;

        let viewport_upper_left = self.center
            - self.focus_dist * self.w
            - viewport_u / 2.0
            - viewport_v / 2.0;
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);

        let defocus_radius = self.focus_dist * (self.defocus_angle / 2.0).to_radians().tan();
        self.defocus_disk_u = self.u * defocus_radius;
        self.defocus_disk_v = self.v * defocus_radius;
    }

    fn sample_square_stratified(&self, s_i: u32, s_j: u32, rng: &mut impl Rng) -> Vec3 {
        let px = ((s_i as f64 + rng.gen::<f64>()) * self.recip_sqrt_spp) - 0.5;
        let py = ((s_j as f64 + rng.gen::<f64>()) * self.recip_sqrt_spp) - 0.5;
        Vec3::new(px, py, 0.0)
    }

    fn get_ray(&self, i: u32, j: u32, s_i: u32, s_j: u32, rng: &mut impl Rng) -> Ray {
        let offset = self.sample_square_stratified(s_i, s_j, rng);
        let pixel_sample = self.pixel00_loc
            + (i as f64 + offset.x()) * self.pixel_delta_u
            + (j as f64 + offset.y()) * self.pixel_delta_v;

        let ray_origin = if self.defocus_angle <= 0.0 {
            self.center
        } else {
            let p = vec3::random_in_unit_disk(rng);
            self.center + p.x() * self.defocus_disk_u + p.y() * self.defocus_disk_v
        };
        let ray_direction = pixel_sample - ray_origin;
        let ray_time = rng.gen::<f64>();
        Ray::new(ray_origin, ray_direction, ray_time)
    }

    pub fn render(&self, world: &Hittable, lights: &Hittable, output_path: &str) -> anyhow::Result<()> {
        let lights_list = match lights {
            Hittable::HittableList(l) => l,
            _ => anyhow::bail!("lights must be HittableList"),
        };

        let w = self.image_width as usize;
        let h = self.image_height as usize;
        let total_pixels = w * h;

        println!("Rendering {}x{} with {} spp, {} bounces...", w, h, self.samples_per_pixel, self.max_depth);

        let pb = ProgressBar::new(total_pixels as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} pixels ({percent}%) [{eta}]")
                .unwrap()
                .progress_chars("##-"),
        );

        let counter = AtomicUsize::new(0);

        let pixel_data: Vec<[u16; 3]> = (0..h)
            .into_par_iter()
            .flat_map(|j| {
                let mut row_data: Vec<[u16; 3]> = Vec::with_capacity(w);
                let mut rng = SmallRng::from_entropy();

                for i in 0..w {
                    let mut pixel_color = Vec3::zero();
                    for s_j in 0..self.sqrt_spp {
                        for s_i in 0..self.sqrt_spp {
                            let r = self.get_ray(i as u32, j as u32, s_i, s_j, &mut rng);
                            pixel_color += ray_color(&r, self.max_depth, world, lights_list, &mut rng);
                        }
                    }
                    let rgb = color_io::pixel_to_10bit(&(self.pixel_samples_scale * pixel_color));
                    row_data.push(rgb);
                }

                counter.fetch_add(w, Ordering::Relaxed);
                pb.set_position(counter.load(Ordering::Relaxed) as u64);
                row_data
            })
            .collect();

        pb.finish_with_message("Done.");

        save_png(output_path, w as u32, h as u32, &pixel_data)?;
        println!("Wrote {}", output_path);
        Ok(())
    }
}

fn ray_color<R: Rng>(
    r: &Ray, depth: u32, world: &Hittable, lights: &HittableList, rng: &mut R,
) -> Vec3 {
    if depth == 0 { return Vec3::zero(); }

    let mut rec = HitRecord {
        p: Point3::zero(), normal: Vec3::zero(), mat: Material::lambertian_color(Vec3::zero()),
        t: 0.0, u: 0.0, v: 0.0, front_face: false,
    };

    if !world.hit(r, &Interval::new(0.001, f64::INFINITY), &mut rec) {
        return Vec3::zero();
    }

    let color_from_emission = rec.mat.emitted(r, &rec, rec.u, rec.v, &rec.p);

    let mut srec = crate::material::ScatterRecord::default();
    if !rec.mat.scatter(r, &rec, &mut srec, rng) {
        return color_from_emission;
    }

    if srec.skip_pdf {
        return srec.attenuation * ray_color(&srec.skip_pdf_ray, depth - 1, world, lights, rng);
    }

    let bsdf_pdf = srec.pdf_ptr.unwrap_or_else(|| Pdf::sphere());

    // 内联混合 PDF：直接使用 lights 引用，避免克隆
    let scattered_dir = if rng.gen::<f64>() < 0.5 {
        lights.random(&rec.p, rng)
    } else {
        bsdf_pdf.generate(rng)
    };
    let scattered_dir = scattered_dir.unit_vector(); // 提前归一化，省去下游重复 sqrt
    let scattered = Ray::new(rec.p, scattered_dir, r.tm);
    let pdf_val = 0.5 * lights.pdf_value(&rec.p, &scattered.dir)
        + 0.5 * bsdf_pdf.value(&scattered.dir);
    let scattering_pdf = rec.mat.scattering_pdf(r, &rec, &scattered);

    let sample_color = ray_color(&scattered, depth - 1, world, lights, rng);
    if pdf_val < 1e-160 {
        return color_from_emission;
    }
    let color_from_scatter = srec.attenuation * scattering_pdf * sample_color / pdf_val;

    color_from_emission + color_from_scatter
}

fn save_png(path: &str, width: u32, height: u32, data: &[[u16; 3]]) -> anyhow::Result<()> {
    use image::{ImageBuffer, Rgb};
    let mut buf: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(width, height);
    for (idx, pixel) in data.iter().enumerate() {
        let x = idx as u32 % width;
        let y = idx as u32 / width;
        buf.put_pixel(x, y, Rgb(*pixel));
    }
    buf.save(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_height_derived_from_aspect_ratio() {
        let mut cam = Camera::new();
        cam.image_width = 800;
        cam.image_height = 0;
        cam.aspect_ratio = 4.0 / 3.0;
        cam.initialize();
        assert_eq!(cam.image_height, 600);
    }

    #[test]
    fn test_explicit_height_overrides_aspect_ratio() {
        let mut cam = Camera::new();
        cam.image_width = 3840;
        cam.image_height = 2160;
        cam.aspect_ratio = 1.0;
        cam.initialize();
        assert_eq!(cam.image_height, 2160);
    }

    #[test]
    fn test_height_clamped_to_minimum() {
        let mut cam = Camera::new();
        cam.image_width = 100;
        cam.image_height = 0;
        cam.aspect_ratio = 1000.0;
        cam.initialize();
        assert!(cam.image_height >= 1);
    }

    #[test]
    fn test_4k_resolution() {
        let mut cam = Camera::new();
        cam.image_width = 3840;
        cam.image_height = 2160;
        cam.initialize();
        assert_eq!(cam.image_width, 3840);
        assert_eq!(cam.image_height, 2160);
    }
}
