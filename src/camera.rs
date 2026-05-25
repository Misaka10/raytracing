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
use serde_json::json;
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

    #[cfg(feature = "cuda")]
    /// Recursively search the hittable tree for the first area light quad.
    /// Uses BVH reference traversal (visit_leaves) to avoid subtree cloning.
    fn find_light_quad(&self, hittable: &Hittable) -> Option<([f32; 3], [f32; 3], [f32; 3], f32)> {
        self.find_light_quad_inner(hittable, true)
    }

    #[cfg(feature = "cuda")]
    fn find_light_quad_inner(&self, hittable: &Hittable, is_root: bool) -> Option<([f32; 3], [f32; 3], [f32; 3], f32)> {
        match hittable {
            Hittable::Quad(q) => {
                if matches!(&q.mat, crate::material::Material::DiffuseLight { .. }) {
                    let corner = [q.q.x() as f32, q.q.y() as f32, q.q.z() as f32];
                    let u = [q.u.x() as f32, q.u.y() as f32, q.u.z() as f32];
                    let v = [q.v.x() as f32, q.v.y() as f32, q.v.z() as f32];
                    let cross = [
                        u[1] * v[2] - u[2] * v[1],
                        u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0],
                    ];
                    let area = (cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]).sqrt();
                    let area_inv = if area > 0.0 { 1.0 / area } else { 0.0 };
                    return Some((corner, u, v, area_inv));
                }
                None
            }
            Hittable::HittableList(list) => {
                if is_root {
                    // BVH-wrapped at root: use visit_leaves for O(n) reference traversal
                    for obj in &list.objects {
                        if let Hittable::BvhNode(bvh) = obj {
                            let mut result = None;
                            bvh.visit_leaves(&mut |leaf| {
                                if result.is_none() {
                                    result = self.find_light_quad_inner(leaf, false);
                                }
                            });
                            if result.is_some() { return result; }
                        } else {
                            let result = self.find_light_quad_inner(obj, false);
                            if result.is_some() { return result; }
                        }
                    }
                    None
                } else {
                    for obj in &list.objects {
                        let result = self.find_light_quad_inner(obj, false);
                        if result.is_some() { return result; }
                    }
                    None
                }
            }
            Hittable::BvhNode(bvh) => {
                let mut result = None;
                bvh.visit_leaves(&mut |leaf| {
                    if result.is_none() {
                        result = self.find_light_quad_inner(leaf, false);
                    }
                });
                result
            }
            Hittable::Translate(inner, offset, _) => {
                self.find_light_quad_inner(inner, false).map(|(corner, u, v, area_inv)| {
                    let ox = offset.x() as f32;
                    let oy = offset.y() as f32;
                    let oz = offset.z() as f32;
                    ([corner[0] + ox, corner[1] + oy, corner[2] + oz], u, v, area_inv)
                })
            }
            Hittable::RotateY(inner, sin_theta, cos_theta, _) => {
                self.find_light_quad_inner(inner, false).map(|(corner, u, v, area_inv)| {
                    let st = *sin_theta as f32;
                    let ct = *cos_theta as f32;
                    let rotate = |p: [f32; 3]| -> [f32; 3] {
                        [ct * p[0] + st * p[2], p[1], -st * p[0] + ct * p[2]]
                    };
                    (rotate(corner), rotate(u), rotate(v), area_inv)
                })
            }
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    /// Recursively search the hittable tree for the glass sphere (dielectric material).
    /// Uses BVH reference traversal (visit_leaves) to avoid subtree cloning.
    fn find_glass_sphere(&self, hittable: &Hittable) -> Option<([f32; 3], f32)> {
        match hittable {
            Hittable::Sphere(s) => {
                if matches!(&s.mat, crate::material::Material::Dielectric { .. }) {
                    let center = [s.center.orig.x() as f32, s.center.orig.y() as f32, s.center.orig.z() as f32];
                    let radius = s.radius as f32;
                    return Some((center, radius));
                }
                None
            }
            Hittable::HittableList(list) => {
                for obj in &list.objects {
                    if let Hittable::BvhNode(bvh) = obj {
                        let mut result = None;
                        bvh.visit_leaves(&mut |leaf| {
                            if result.is_none() {
                                if let Hittable::Sphere(s) = leaf {
                                    if matches!(&s.mat, crate::material::Material::Dielectric { .. }) {
                                        let center = [s.center.orig.x() as f32, s.center.orig.y() as f32, s.center.orig.z() as f32];
                                        let radius = s.radius as f32;
                                        result = Some((center, radius));
                                    }
                                }
                            }
                        });
                        if result.is_some() { return result; }
                    } else {
                        let result = self.find_glass_sphere(obj);
                        if result.is_some() { return result; }
                    }
                }
                None
            }
            Hittable::BvhNode(bvh) => {
                let mut result = None;
                bvh.visit_leaves(&mut |leaf| {
                    if result.is_none() {
                        if let Hittable::Sphere(s) = leaf {
                            if matches!(&s.mat, crate::material::Material::Dielectric { .. }) {
                                let center = [s.center.orig.x() as f32, s.center.orig.y() as f32, s.center.orig.z() as f32];
                                let radius = s.radius as f32;
                                result = Some((center, radius));
                            }
                        }
                    }
                });
                result
            }
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn render_gpu(&self, world: &Hittable, output_path: &str, seed: Option<u64>, denoise: bool, calibrate: bool) -> anyhow::Result<()> {
        use crate::cuda::optix::{self, BridgeCameraParams, OptiXBridge};
        use crate::cuda::scene::GpuScene;

        let w = self.image_width as usize;
        let h = self.image_height as usize;
        let spp = self.samples_per_pixel;
        let sqrt_spp = self.sqrt_spp;

        // Find area light geometry for importance sampling (before world flattening)
        let light_info = self.find_light_quad(world);
        let sphere_info = self.find_glass_sphere(world);

        // Build GPU scene from world
        let gpu_scene = GpuScene::from_world(world);
        if gpu_scene.vertices.is_empty() || gpu_scene.indices.is_empty() {
            anyhow::bail!("GPU scene has no geometry");
        }

        eprintln!("GPU scene: {} triangles, {} vertices, {} materials",
            gpu_scene.tri_to_material.len(),
            gpu_scene.vertices.len() / 3,
            gpu_scene.materials.len());

        // Load PTX shaders
        let (ptx_raygen, ptx_ch, ptx_ms) = optix::load_ptx_shaders();

        // Init bridge
        let mut bridge = OptiXBridge::new(ptx_raygen, ptx_ch, ptx_ms)
            .ok_or_else(|| anyhow::anyhow!("Failed to initialize OptiX bridge"))?;

        // Build acceleration structure
        let tri_count = gpu_scene.tri_to_material.len() as i32;
        let vertex_count = (gpu_scene.vertices.len() / 3) as i32;
        if !bridge.build_accel(&gpu_scene.vertices, &gpu_scene.indices, &gpu_scene.normals, tri_count, vertex_count) {
            anyhow::bail!("Failed to build BVH: {}", bridge.get_error());
        }

        // Upload materials
        if !bridge.set_materials(&gpu_scene.materials) {
            anyhow::bail!("Failed to upload materials: {}", bridge.get_error());
        }

        // Upload per-triangle material indices
        if !bridge.set_tri_material(&gpu_scene.tri_to_material) {
            anyhow::bail!("Failed to upload tri_material: {}", bridge.get_error());
        }

        // Set render params
        bridge.set_render_params(sqrt_spp, self.max_depth, self.pixel_samples_scale as f32);

        // Set light params for importance sampling
        if let Some((corner, u, v, area_inv)) = light_info {
            bridge.set_light(&corner, &u, &v, area_inv);
        }
        if let Some((center, radius)) = sphere_info {
            bridge.set_sphere(&center, radius);
        }

        // Create pipeline
        if !bridge.create_pipeline(w as i32, h as i32) {
            anyhow::bail!("Failed to create pipeline: {}", bridge.get_error());
        }

        // Build camera params
        let cam = BridgeCameraParams {
            lookfrom: f64x3_to_f32x3(self.lookfrom),
            lookat:   f64x3_to_f32x3(self.lookat),
            vup:      f64x3_to_f32x3(self.vup),
            vfov: self.vfov as f32,
            aspect_ratio: self.aspect_ratio as f32,
            defocus_angle: self.defocus_angle as f32,
            focus_dist: self.focus_dist as f32,
            u: f64x3_to_f32x3(self.u),
            v: f64x3_to_f32x3(self.v),
            w: f64x3_to_f32x3(self.w),
            pixel00_loc: f64x3_to_f32x3(self.pixel00_loc),
            pixel_delta_u: f64x3_to_f32x3(self.pixel_delta_u),
            pixel_delta_v: f64x3_to_f32x3(self.pixel_delta_v),
            defocus_disk_u: f64x3_to_f32x3(self.defocus_disk_u),
            defocus_disk_v: f64x3_to_f32x3(self.defocus_disk_v),
        };

        let seed = seed.unwrap_or(0);
        let output_size = w * h * 3;
        let mut output = vec![0.0f32; output_size];

        if !calibrate {
            eprintln!("Rendering GPU {}x{} with {} spp...", w, h, spp);
        }
        let gpu_render_start = if calibrate { Some(std::time::Instant::now()) } else { None };
        if !bridge.render(&mut output, &cam, seed as u32) {
            anyhow::bail!("GPU render failed: {}", bridge.get_error());
        }

        // Compute elapsed time immediately after GPU render,
        // before any post-processing (denoise, PNG save).
        if let Some(start) = gpu_render_start {
            let elapsed = start.elapsed();
            let total_pixel_samples = (w * h * spp as usize) as f64;
            let px_per_ms = total_pixel_samples / (elapsed.as_secs_f64() * 1000.0);
            println!("{}", serde_json::to_string(&serde_json::json!({
                "pixel_samples_per_ms": px_per_ms,
            })).unwrap());
        }

        // Apply AI denoiser if requested
        if denoise {
            eprintln!("Applying AI denoiser (Tensor Core)...");
            if !bridge.denoise() {
                eprintln!("Warning: denoise failed: {}", bridge.get_error());
            }
        }

        // Convert float buffer to 16-bit PNG
        if !calibrate {
            save_png_gpu(output_path, w as u32, h as u32, &output)?;
            eprintln!("Wrote {}", output_path);
        }
        Ok(())
    }

    pub fn render(&self, world: &Hittable, lights: &Hittable, output_path: &str, seed: Option<u64>, json_progress: bool, calibrate: bool) -> anyhow::Result<()> {
        let lights_list = match lights {
            Hittable::HittableList(l) => l,
            _ => anyhow::bail!("lights must be HittableList"),
        };

        let w = self.image_width as usize;
        let h = self.image_height as usize;
        let total_pixels = w * h;

        if !calibrate {
            if json_progress {
                let msg = json!({
                    "type": "start",
                    "width": w,
                    "height": h,
                    "samples": self.samples_per_pixel,
                    "max_depth": self.max_depth,
                });
                println!("{}", serde_json::to_string(&msg).unwrap());
            } else {
                println!("Rendering {}x{} with {} spp, {} bounces...", w, h, self.samples_per_pixel, self.max_depth);
            }
        }

        let pb = if json_progress || calibrate {
            None
        } else {
            let bar = ProgressBar::new(total_pixels as u64);
            bar.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} pixels ({percent}%) [{eta}]")
                    .unwrap()
                    .progress_chars("##-"),
            );
            Some(bar)
        };

        let counter = AtomicUsize::new(0);
        let render_start = if calibrate { Some(std::time::Instant::now()) } else { None };

        // Golden ratio constant for deriving per-row seeds
        const GOLDEN_RATIO_U64: u64 = 0x9e3779b97f4a7c15;

        let pixel_data: Vec<[u16; 3]> = (0..h)
            .into_par_iter()
            .flat_map(|j| {
                let mut row_data: Vec<[u16; 3]> = Vec::with_capacity(w);
                let mut rng = match seed {
                    Some(s) => SmallRng::seed_from_u64(s.wrapping_add(GOLDEN_RATIO_U64.wrapping_mul(j as u64))),
                    None => SmallRng::from_entropy(),
                };

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

                let done = counter.fetch_add(w, Ordering::Relaxed) + w;
                if calibrate {
                    // no per-row output during calibration
                } else if json_progress {
                    let msg = json!({
                        "type": "progress",
                        "completed": done,
                        "total": total_pixels,
                    });
                    println!("{}", serde_json::to_string(&msg).unwrap());
                } else if let Some(ref bar) = pb {
                    bar.set_position(done as u64);
                }
                row_data
            })
            .collect();

        if let Some(ref bar) = pb {
            bar.finish_with_message("Done.");
        }

        // Compute elapsed time before any file I/O
        if let Some(start) = render_start {
            let elapsed = start.elapsed();
            let total_pixel_samples = (total_pixels * self.samples_per_pixel as usize) as f64;
            let px_per_ms = total_pixel_samples / (elapsed.as_secs_f64() * 1000.0);
            println!("{}", serde_json::to_string(&json!({
                "pixel_samples_per_ms": px_per_ms,
            })).unwrap());
        }

        if !calibrate {
            save_png(output_path, w as u32, h as u32, &pixel_data)?;

            if json_progress {
                let msg = json!({
                    "type": "done",
                    "output": output_path,
                });
                println!("{}", serde_json::to_string(&msg).unwrap());
            } else {
                println!("Wrote {}", output_path);
            }
        }
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

#[cfg(feature = "cuda")]
fn f64x3_to_f32x3(v: crate::vec3::Vec3) -> [f32; 3] {
    [v.x() as f32, v.y() as f32, v.z() as f32]
}

#[cfg(feature = "cuda")]
fn save_png_gpu(path: &str, width: u32, height: u32, data: &[f32]) -> anyhow::Result<()> {
    use crate::color_io::linear_to_gamma;
    use crate::interval::Interval;
    use image::{ImageBuffer, Rgb};

    let intensity = Interval::new(0.0, 0.9999);
    let mut buf: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(width, height);
    for (idx, chunk) in data.chunks(3).enumerate() {
        if chunk.len() < 3 { break; }
        let x = idx as u32 % width;
        let y = idx as u32 / width;
        // Match CPU pixel_to_10bit: sqrt gamma + 10-bit scaled to 16-bit
        let r = linear_to_gamma(chunk[0] as f64);
        let g = linear_to_gamma(chunk[1] as f64);
        let b = linear_to_gamma(chunk[2] as f64);
        let r10 = (1024.0 * intensity.clamp(r)) as u16;
        let g10 = (1024.0 * intensity.clamp(g)) as u16;
        let b10 = (1024.0 * intensity.clamp(b)) as u16;
        let scale = 65535.0 / 1023.0;
        let r16 = (r10 as f64 * scale) as u16;
        let g16 = (g10 as f64 * scale) as u16;
        let b16 = (b10 as f64 * scale) as u16;
        buf.put_pixel(x, y, Rgb([r16, g16, b16]));
    }
    buf.save(path)?;
    Ok(())
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
    use crate::bvh::BvhNode;
    use crate::material::Material;
    use crate::sphere::Sphere;
    use crate::vec3::Color;
    use std::fs;

    fn make_minimal_camera() -> Camera {
        let mut cam = Camera::new();
        cam.image_width = 10;
        cam.image_height = 6;
        cam.samples_per_pixel = 4;
        cam.max_depth = 5;
        cam.background = Vec3::zero();
        cam.vfov = 40.0;
        cam.lookfrom = Point3::new(278.0, 278.0, -800.0);
        cam.lookat = Point3::new(278.0, 278.0, 0.0);
        cam.vup = Vec3::new(0.0, 1.0, 0.0);
        cam.defocus_angle = 0.0;
        cam.initialize();
        cam
    }

    fn make_test_scene() -> (Hittable, Hittable) {
        let mat = Material::lambertian_color(Color::new(0.5, 0.5, 0.5));
        let light_mat = Material::diffuse_light_color(Color::new(4.0, 4.0, 4.0));
        let sphere = Hittable::Sphere(Sphere::stationary(
            Point3::new(278.0, 278.0, 0.0), 100.0, mat,
        ));
        let light_sphere = Hittable::Sphere(Sphere::stationary(
            Point3::new(278.0, 400.0, 0.0), 100.0, light_mat.clone(),
        ));

        let mut objects = vec![sphere, light_sphere.clone()];
        let bvh = BvhNode::from_objects(&mut objects);
        let world = Hittable::BvhNode(bvh);

        let mut lights = HittableList::new();
        lights.add(light_sphere);
        (world, Hittable::HittableList(lights))
    }

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

    #[test]
    fn test_render_with_seed_no_panic() {
        let cam = make_minimal_camera();
        let (world, lights) = make_test_scene();
        let out = std::env::temp_dir().join("rt_test_nopanic.png");
        let out_path = out.to_str().unwrap();

        let result = cam.render(&world, &lights, out_path, Some(42), false, false);
        assert!(result.is_ok());
        assert!(out.exists());
        let _ = fs::remove_file(&out);
    }

    #[test]
    fn test_seed_determinism() {
        let cam = make_minimal_camera();
        let (world, lights) = make_test_scene();
        let out_a = std::env::temp_dir().join("rt_test_seed_a.png");
        let out_b = std::env::temp_dir().join("rt_test_seed_b.png");
        let pa = out_a.to_str().unwrap();
        let pb = out_b.to_str().unwrap();

        cam.render(&world, &lights, pa, Some(42), false, false).unwrap();
        cam.render(&world, &lights, pb, Some(42), false, false).unwrap();

        let bytes_a = fs::read(&out_a).unwrap();
        let bytes_b = fs::read(&out_b).unwrap();
        assert_eq!(bytes_a, bytes_b, "same seed must produce byte-identical PNG");

        let _ = fs::remove_file(&out_a);
        let _ = fs::remove_file(&out_b);
    }

    #[test]
    fn test_json_progress_enabled_does_not_panic() {
        let cam = make_minimal_camera();
        let (world, lights) = make_test_scene();
        let out = std::env::temp_dir().join("rt_test_json.png");
        let out_path = out.to_str().unwrap();

        let result = cam.render(&world, &lights, out_path, Some(42), true, false);
        assert!(result.is_ok());
        assert!(out.exists());
        let _ = fs::remove_file(&out);
    }

    #[test]
    fn test_no_seed_produces_output() {
        let cam = make_minimal_camera();
        let (world, lights) = make_test_scene();
        let out = std::env::temp_dir().join("rt_test_noseed.png");
        let out_path = out.to_str().unwrap();

        let result = cam.render(&world, &lights, out_path, None, false, false);
        assert!(result.is_ok());
        assert!(out.exists());
        let _ = fs::remove_file(&out);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_png_gamma_matches_cpu_encoding() {
        use crate::color_io::pixel_to_10bit;
        use crate::vec3::Color;
        let tmp = std::env::temp_dir().join("rt_test_gpu_gamma.png");
        let p = tmp.to_str().unwrap();

        // Render a single white pixel and save via GPU path
        let data: Vec<f32> = vec![1.0, 1.0, 1.0];
        save_png_gpu(p, 1, 1, &data).unwrap();

        // Read back and verify the pixel is bright (not dark)
        let img = image::open(p).unwrap().into_rgb16();
        let px = img.get_pixel(0, 0);
        let gpu_r = px.0[0] as f64;

        // CPU encoding of the same value
        let cpu_px = pixel_to_10bit(&Color::new(1.0, 1.0, 1.0));
        let cpu_r = cpu_px[0] as f64;

        // GPU and CPU encodings should be very close
        let diff = (gpu_r - cpu_r).abs();
        assert!(diff < 100.0, "GPU gamma {} vs CPU {} diff too large", gpu_r, cpu_r);
        // White pixel should be bright (> 60000 in 16-bit)
        assert!(gpu_r > 60000.0, "GPU pixel {} should be bright white", gpu_r);

        let _ = std::fs::remove_file(&tmp);
    }
}
