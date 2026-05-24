//! Scene conversion: CPU HittableList → GPU triangle mesh.
//!
//! Spheres are tessellated into latitude-longitude triangle grids.
//! Quads are split into 2 triangles each.
//! BVH nodes are flattened by traversing object lists.
//! Materials are flattened into a uniform array for GPU access.

use crate::bvh::BvhNode;
use crate::material::Material;
use crate::quad::Quad;
use crate::sphere::Sphere;
use crate::texture::Texture;
use crate::Hittable;

const SPHERE_LAT: u32 = 32;
const SPHERE_LON: u32 = 32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuMaterial {
    pub mat_type: u32,
    pub albedo: [f32; 3],
    pub fuzz: f32,
    pub ir: f32,
    pub emission: [f32; 3],
}

pub struct GpuScene {
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    pub tri_to_material: Vec<u32>,
    pub materials: Vec<GpuMaterial>,
}

impl GpuScene {
    pub fn from_world(world: &Hittable) -> Self {
        let mut scene = GpuScene {
            vertices: Vec::new(),
            indices: Vec::new(),
            tri_to_material: Vec::new(),
            materials: Vec::new(),
        };

        // Phase 2: flatten the object tree, dedup materials, tessellate geometry
        let mut objects: Vec<Hittable> = Vec::new();
        flatten_hittable(world, &mut objects);

        // Deduplicate materials by Arc pointer identity
        let mut dedup_mats: Vec<&Material> = Vec::new();
        let mut mat_indices: Vec<u32> = Vec::with_capacity(objects.len());

        for obj in &objects {
            if let Some(mat) = get_material(obj) {
                let pos = dedup_mats.iter().position(|m| {
                    std::ptr::eq(*m as *const Material, mat as *const Material)
                });
                if let Some(idx) = pos {
                    mat_indices.push(idx as u32);
                } else {
                    let idx = dedup_mats.len() as u32;
                    dedup_mats.push(mat);
                    mat_indices.push(idx);
                }
            } else {
                mat_indices.push(0);
            }
        }

        for mat in &dedup_mats {
            scene.materials.push(material_to_gpu(mat));
        }

        // Ensure at least one material
        if scene.materials.is_empty() {
            scene.materials.push(GpuMaterial {
                mat_type: 0,
                albedo: [0.5, 0.5, 0.5],
                fuzz: 0.0,
                ir: 1.0,
                emission: [0.0, 0.0, 0.0],
            });
        }

        // Tessellate each object
        for (i, obj) in objects.iter().enumerate() {
            let mat_idx = mat_indices[i];
            match obj {
                Hittable::Sphere(s) => scene.tessellate_sphere(s, mat_idx),
                Hittable::Quad(q) => scene.tessellate_quad(q, mat_idx),
                Hittable::BvhNode(_)
                | Hittable::HittableList(_)
                | Hittable::Translate(..)
                | Hittable::RotateY(..)
                | Hittable::ConstantMedium(_) => {
                    // Already flattened or skipped for Phase 2
                }
            }
        }

        scene
    }

    fn tessellate_sphere(&mut self, sphere: &Sphere, mat_idx: u32) {
        let base_vert = (self.vertices.len() / 3) as u32;
        let cx = sphere.center.orig.x() as f32;
        let cy = sphere.center.orig.y() as f32;
        let cz = sphere.center.orig.z() as f32;
        let r = sphere.radius as f32;

        // Generate vertices (lat/lon grid with poles)
        for lat in 0..=SPHERE_LAT {
            let theta = std::f64::consts::PI * lat as f64 / SPHERE_LAT as f64;
            let sin_theta = theta.sin() as f32;
            let cos_theta = theta.cos() as f32;

            for lon in 0..=SPHERE_LON {
                let phi = 2.0 * std::f64::consts::PI * lon as f64 / SPHERE_LON as f64;
                let sin_phi = phi.sin() as f32;
                let cos_phi = phi.cos() as f32;

                self.vertices.push(cx + r * sin_theta * cos_phi);
                self.vertices.push(cy + r * cos_theta);
                self.vertices.push(cz + r * sin_theta * sin_phi);
            }
        }

        let verts_per_ring = SPHERE_LON + 1;
        for lat in 0..SPHERE_LAT {
            for lon in 0..SPHERE_LON {
                let a = base_vert + lat * verts_per_ring + lon;
                let b = a + verts_per_ring;
                let c = a + 1;
                let d = b + 1;

                self.indices.push(a);
                self.indices.push(b);
                self.indices.push(c);
                self.tri_to_material.push(mat_idx);

                self.indices.push(c);
                self.indices.push(b);
                self.indices.push(d);
                self.tri_to_material.push(mat_idx);
            }
        }
    }

    fn tessellate_quad(&mut self, quad: &Quad, mat_idx: u32) {
        let base_vert = (self.vertices.len() / 3) as u32;
        let q = quad.q;
        let u = quad.u;
        let v = quad.v;

        for corner in [q, q + u, q + v, q + u + v].iter() {
            self.vertices.push(corner.x() as f32);
            self.vertices.push(corner.y() as f32);
            self.vertices.push(corner.z() as f32);
        }

        // Triangle 0,1,2
        self.indices.push(base_vert);
        self.indices.push(base_vert + 1);
        self.indices.push(base_vert + 2);
        self.tri_to_material.push(mat_idx);

        // Triangle 1,3,2
        self.indices.push(base_vert + 1);
        self.indices.push(base_vert + 3);
        self.indices.push(base_vert + 2);
        self.tri_to_material.push(mat_idx);
    }
}

/// Extract material reference from a hittable object variant
fn get_material(h: &Hittable) -> Option<&Material> {
    match h {
        Hittable::Sphere(s) => Some(&s.mat),
        Hittable::Quad(q) => Some(&q.mat),
        Hittable::Translate(obj, _, _) => get_material(obj),
        Hittable::RotateY(obj, _, _, _) => get_material(obj),
        _ => None,
    }
}

/// Flatten a hittable tree into a list of leaf objects (no BVH, no lists)
fn flatten_hittable(obj: &Hittable, out: &mut Vec<Hittable>) {
    match obj {
        Hittable::HittableList(list) => {
            for child in &list.objects {
                flatten_hittable(child, out);
            }
        }
        Hittable::BvhNode(bvh) => {
            flatten_bvh(bvh, out);
        }
        Hittable::Translate(inner, _, _) | Hittable::RotateY(inner, _, _, _) => {
            flatten_hittable(inner, out);
        }
        Hittable::Sphere(_) | Hittable::Quad(_) | Hittable::ConstantMedium(_) => {
            out.push(obj.clone());
        }
    }
}

fn flatten_bvh(node: &BvhNode, out: &mut Vec<Hittable>) {
    match node {
        BvhNode::Leaf { object, .. } => {
            flatten_hittable(object, out);
        }
        BvhNode::Split { left, right, .. } => {
            flatten_bvh(left, out);
            flatten_bvh(right, out);
        }
    }
}

fn material_to_gpu(mat: &Material) -> GpuMaterial {
    match mat {
        Material::Lambertian { tex } => {
            let albedo = solid_color_albedo(tex);
            GpuMaterial { mat_type: 0, albedo, fuzz: 0.0, ir: 1.0, emission: [0.0, 0.0, 0.0] }
        }
        Material::Metal { albedo, fuzz } => GpuMaterial {
            mat_type: 1,
            albedo: [albedo.x() as f32, albedo.y() as f32, albedo.z() as f32],
            fuzz: *fuzz as f32,
            ir: 1.0,
            emission: [0.0, 0.0, 0.0],
        },
        Material::Dielectric { refraction_index } => GpuMaterial {
            mat_type: 2,
            albedo: [1.0, 1.0, 1.0],
            fuzz: 0.0,
            ir: *refraction_index as f32,
            emission: [0.0, 0.0, 0.0],
        },
        Material::DiffuseLight { tex } => {
            let emission = solid_color_albedo(tex);
            GpuMaterial { mat_type: 3, albedo: [0.0, 0.0, 0.0], fuzz: 0.0, ir: 1.0, emission }
        }
        Material::Isotropic { tex: _ } => GpuMaterial {
            mat_type: 4,
            albedo: [0.5, 0.5, 0.5],
            fuzz: 0.0,
            ir: 1.0,
            emission: [0.0, 0.0, 0.0],
        },
    }
}

/// Extract albedo from a solid color texture, or return a fallback gray
fn solid_color_albedo(tex: &Texture) -> [f32; 3] {
    match tex {
        Texture::SolidColor(c) => [c.x() as f32, c.y() as f32, c.z() as f32],
        _ => [0.5, 0.5, 0.5],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;
    use crate::vec3::{Color, Point3, Vec3};

    #[test]
    fn test_single_sphere_tessellation() {
        let s = Sphere::stationary(
            Point3::new(0.0, 0.0, 0.0),
            1.0,
            Material::lambertian_color(Color::new(0.5, 0.5, 0.5)),
        );
        let world = Hittable::Sphere(s);
        let scene = GpuScene::from_world(&world);

        let expected_tris = (SPHERE_LAT * SPHERE_LON * 2) as usize;
        assert_eq!(scene.tri_to_material.len(), expected_tris);
        assert_eq!(scene.indices.len(), expected_tris * 3);
        assert!(!scene.vertices.is_empty());
        assert_eq!(scene.materials.len(), 1);
    }

    #[test]
    fn test_sphere_vertices_in_bounds() {
        let s = Sphere::stationary(
            Point3::new(2.0, 3.0, 4.0),
            2.0,
            Material::lambertian_color(Color::new(0.5, 0.5, 0.5)),
        );
        let world = Hittable::Sphere(s);
        let scene = GpuScene::from_world(&world);

        for i in (0..scene.vertices.len()).step_by(3) {
            let dx = scene.vertices[i] - 2.0;
            let dy = scene.vertices[i + 1] - 3.0;
            let dz = scene.vertices[i + 2] - 4.0;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!((dist - 2.0).abs() < 0.2, "vertex dist {} not within ~2.0", dist);
        }
    }

    #[test]
    fn test_quad_tessellation() {
        let q = Quad::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Material::lambertian_color(Color::new(0.5, 0.5, 0.5)),
        );
        let world = Hittable::Quad(q);
        let scene = GpuScene::from_world(&world);

        assert_eq!(scene.tri_to_material.len(), 2);
        assert_eq!(scene.indices.len(), 6);
        assert_eq!(scene.vertices.len(), 12);
    }
}
