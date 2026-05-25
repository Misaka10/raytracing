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
    pub normals: Vec<f32>,
    pub indices: Vec<u32>,
    pub tri_to_material: Vec<u32>,
    pub materials: Vec<GpuMaterial>,
}

impl GpuScene {
    pub fn from_world(world: &Hittable) -> Self {
        let mut scene = GpuScene {
            vertices: Vec::new(),
            normals: Vec::new(),
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
                let pos = dedup_mats
                    .iter()
                    .position(|m| std::ptr::eq(*m as *const Material, mat as *const Material));
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

        // Tessellate each object, applying accumulated transforms
        for (i, obj) in objects.iter().enumerate() {
            let mat_idx = mat_indices[i];
            scene.tessellate_object(obj, mat_idx);
        }

        scene
    }

    fn tessellate_sphere(&mut self, sphere: &Sphere, mat_idx: u32) {
        let base_vert = (self.vertices.len() / 3) as u32;
        let cx = sphere.center.orig.x() as f32;
        let cy = sphere.center.orig.y() as f32;
        let cz = sphere.center.orig.z() as f32;
        let r = sphere.radius as f32;

        let total_verts = (SPHERE_LAT as usize + 1) * (SPHERE_LON as usize + 1) * 3;
        let total_indices = (SPHERE_LAT as usize) * (SPHERE_LON as usize) * 6;
        let total_tris = (SPHERE_LAT as usize) * (SPHERE_LON as usize) * 2;

        self.vertices.reserve(total_verts);
        self.normals.reserve(total_verts);
        self.indices.reserve(total_indices);
        self.tri_to_material.reserve(total_tris);

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

                // Analytic normal: (vertex - center) / radius = unit-sphere normal
                self.normals.push(sin_theta * cos_phi);
                self.normals.push(cos_theta);
                self.normals.push(sin_theta * sin_phi);
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

        // Reuse pre-computed face normal (already normalized in Quad::new)
        let nx = quad.normal.x() as f32;
        let ny = quad.normal.y() as f32;
        let nz = quad.normal.z() as f32;

        self.vertices.reserve(12); // 4 corners × 3 floats
        self.normals.reserve(12);
        self.indices.reserve(6); // 2 triangles × 3 indices
        self.tri_to_material.reserve(2);

        for corner in [q, q + u, q + v, q + u + v].iter() {
            self.vertices.push(corner.x() as f32);
            self.vertices.push(corner.y() as f32);
            self.vertices.push(corner.z() as f32);
            // All 4 corners share the same face normal
            self.normals.push(nx);
            self.normals.push(ny);
            self.normals.push(nz);
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
        Hittable::HittableList(list) => list.objects.first().and_then(|obj| get_material(obj)),
        Hittable::BvhNode(bvh) => get_bvh_material(bvh),
        _ => None,
    }
}

/// Helper: extract material from a BvhNode by recursing into leftmost leaf
fn get_bvh_material(bvh: &BvhNode) -> Option<&Material> {
    match bvh {
        BvhNode::Leaf { object, .. } => get_material(object),
        BvhNode::Split { left, .. } => get_bvh_material(left),
    }
}

/// Flatten a hittable tree into a list of leaf/transform objects (no BVH, no lists)
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
        // Keep transforms as-is — they'll be handled in tessellation
        Hittable::Translate(..)
        | Hittable::RotateY(..)
        | Hittable::Sphere(_)
        | Hittable::Quad(_)
        | Hittable::ConstantMedium(_) => {
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

impl GpuScene {
    /// Recursively tessellate an object, applying any transforms to vertex data
    fn tessellate_object(&mut self, obj: &Hittable, mat_idx: u32) {
        match obj {
            Hittable::Sphere(s) => self.tessellate_sphere(s, mat_idx),
            Hittable::Quad(q) => self.tessellate_quad(q, mat_idx),
            Hittable::Translate(inner, offset, _) => {
                let base_idx = self.vertices.len();
                self.tessellate_object(inner, mat_idx);
                // Apply offset to all newly added vertices; normals are direction vectors, unaffected by translation
                let ox = offset.x() as f32;
                let oy = offset.y() as f32;
                let oz = offset.z() as f32;
                for i in (base_idx..self.vertices.len()).step_by(3) {
                    self.vertices[i] += ox;
                    self.vertices[i + 1] += oy;
                    self.vertices[i + 2] += oz;
                }
            }
            Hittable::RotateY(inner, sin_theta, cos_theta, _) => {
                let base_idx = self.vertices.len();
                self.tessellate_object(inner, mat_idx);
                let st = *sin_theta as f32;
                let ct = *cos_theta as f32;
                // Rotate vertices
                for i in (base_idx..self.vertices.len()).step_by(3) {
                    let x = self.vertices[i];
                    let z = self.vertices[i + 2];
                    self.vertices[i] = ct * x + st * z;
                    self.vertices[i + 2] = -st * x + ct * z;
                }
                // Rotate normals (same rotation, direction vectors)
                for i in (base_idx..self.normals.len()).step_by(3) {
                    let nx = self.normals[i];
                    let nz = self.normals[i + 2];
                    self.normals[i] = ct * nx + st * nz;
                    self.normals[i + 2] = -st * nx + ct * nz;
                }
            }
            Hittable::ConstantMedium(_) => {
                // Skipped for Phase 3
            }
            Hittable::HittableList(list) => {
                for child in &list.objects {
                    self.tessellate_object(child, mat_idx);
                }
            }
            Hittable::BvhNode(bvh) => {
                // Defensive: recurse into BVH nodes via reference traversal (no cloning)
                bvh.visit_leaves(&mut |leaf| {
                    self.tessellate_object(leaf, mat_idx);
                });
            }
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

    #[test]
    fn test_gpu_material_size() {
        // Must match GPU-side GpuMaterialData (common.h) = 36 bytes (4-byte aligned)
        assert_eq!(std::mem::size_of::<GpuMaterial>(), 36);
    }

    #[test]
    fn test_gpu_material_field_offsets() {
        use std::mem::offset_of;
        // Verify fields are at expected offsets (no padding with 4-byte aligned GpuFloat3)
        assert_eq!(offset_of!(GpuMaterial, mat_type), 0);
        assert_eq!(offset_of!(GpuMaterial, albedo), 4);
        assert_eq!(offset_of!(GpuMaterial, fuzz), 16);
        assert_eq!(offset_of!(GpuMaterial, ir), 20);
        assert_eq!(offset_of!(GpuMaterial, emission), 24);
    }

    #[test]
    fn test_material_conversion_emission_preserved() {
        let tex = crate::texture::Texture::SolidColor(Color::new(1.0, 2.0, 3.0));
        let m = Material::DiffuseLight { tex };
        let g = material_to_gpu(&m);
        assert_eq!(g.mat_type, 3);
        assert_eq!(g.emission, [1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn test_sphere_vertex_normals_unit_length() {
        let s = Sphere::stationary(
            Point3::new(0.0, 0.0, 0.0),
            1.0,
            Material::lambertian_color(Color::new(0.5, 0.5, 0.5)),
        );
        let world = Hittable::Sphere(s);
        let scene = GpuScene::from_world(&world);

        assert_eq!(scene.vertices.len(), scene.normals.len());
        for i in (0..scene.normals.len()).step_by(3) {
            let nx = scene.normals[i];
            let ny = scene.normals[i + 1];
            let nz = scene.normals[i + 2];
            let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!(
                (n_len - 1.0).abs() < 0.001,
                "normal not unit length: ({},{},{}) len={}",
                nx,
                ny,
                nz,
                n_len
            );
        }
    }

    #[test]
    fn test_sphere_vertex_normals_direction() {
        // Unit sphere at origin: normal == position (both are (x,y,z)/r with r=1)
        let s = Sphere::stationary(
            Point3::new(0.0, 0.0, 0.0),
            1.0,
            Material::lambertian_color(Color::new(0.5, 0.5, 0.5)),
        );
        let world = Hittable::Sphere(s);
        let scene = GpuScene::from_world(&world);

        for i in (0..scene.vertices.len()).step_by(3) {
            let vx = scene.vertices[i];
            let vy = scene.vertices[i + 1];
            let vz = scene.vertices[i + 2];
            let nx = scene.normals[i];
            let ny = scene.normals[i + 1];
            let nz = scene.normals[i + 2];
            // For unit sphere at origin, normal == position
            assert!((vx - nx).abs() < 0.01, "vx={} nx={}", vx, nx);
            assert!((vy - ny).abs() < 0.01, "vy={} ny={}", vy, ny);
            assert!((vz - nz).abs() < 0.01, "vz={} nz={}", vz, nz);
        }
    }

    #[test]
    fn test_quad_vertex_normals_consistent() {
        let q = Quad::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
            Material::lambertian_color(Color::new(0.5, 0.5, 0.5)),
        );
        let world = Hittable::Quad(q);
        let scene = GpuScene::from_world(&world);

        assert_eq!(scene.normals.len(), 12); // 4 vertices * 3 floats
                                             // All 4 vertices should have the same normal
        let n0 = (&scene.normals[0..3]).to_vec();
        for vi in 0..4 {
            let base = vi * 3;
            for c in 0..3 {
                assert!(
                    (scene.normals[base + c] - n0[c]).abs() < 0.001,
                    "vertex {} component {} differs: {} vs {}",
                    vi,
                    c,
                    scene.normals[base + c],
                    n0[c]
                );
            }
        }
        // Quad in xy plane, normal should point in +z (or -z depending on winding)
        let n_len = (n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2]).sqrt();
        assert!((n_len - 1.0).abs() < 0.001, "face normal not unit length");
    }

    #[test]
    fn test_sphere_vertex_radius() {
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
            assert!(
                (dist - 2.0).abs() < 0.002,
                "vertex at distance {} from center, expected 2.0",
                dist
            );
        }
    }

    #[test]
    fn test_box_material_is_white_not_wall() {
        // Reproduces bug: box = Translate(RotateY(HittableList(6 white quads)))
        // get_material used to return None (hitting _ => None), causing mat_indices.push(0)
        // which assigned the first wall material (red/green) instead of white
        let white = Material::lambertian_color(Color::new(0.73, 0.73, 0.73));
        let red = Material::lambertian_color(Color::new(0.65, 0.05, 0.05));
        let box_geom = crate::quad_box::make_box(
            &Point3::new(0.0, 0.0, 0.0),
            &Point3::new(165.0, 330.0, 165.0),
            white,
        );
        let box_rotated = Hittable::rotate_y(box_geom, 15.0);
        let box_translated = Hittable::translate(box_rotated, Vec3::new(265.0, 0.0, 295.0));
        // Put a red wall BEFORE the box in the list — if get_material fails,
        // the box will fall back to index 0 (red) instead of white
        let red_quad = Hittable::Quad(Quad::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(0.0, 100.0, 0.0),
            red,
        ));
        let world = Hittable::HittableList(crate::hittable_list::HittableList {
            objects: vec![red_quad, box_translated],
            bbox: crate::aabb::Aabb::default(),
        });
        let scene = GpuScene::from_world(&world);
        // The red quad has 2 tris, the box has 12 tris
        assert_eq!(scene.tri_to_material.len(), 14);
        // Material 0 should be red, material 1+ should be white
        // The box tris start at index 2
        let box_mat_idx = scene.tri_to_material[2] as usize;
        let box_mat = &scene.materials[box_mat_idx];
        // Box material must be white (0.73, 0.73, 0.73), NOT red (0.65, 0.05, 0.05)
        assert!(
            (box_mat.albedo[0] - 0.73).abs() < 0.01,
            "box albedo[0] = {}, expected 0.73 (red wall albedo is 0.65). get_material returned \
             wrong material!",
            box_mat.albedo[0]
        );
        assert!(
            (box_mat.albedo[1] - 0.73).abs() < 0.01,
            "box albedo[1] = {}, expected 0.73",
            box_mat.albedo[1]
        );
        assert!(
            (box_mat.albedo[2] - 0.73).abs() < 0.01,
            "box albedo[2] = {}, expected 0.73",
            box_mat.albedo[2]
        );
    }

    #[test]
    fn test_get_material_penetrates_transform_chain() {
        let white = Material::lambertian_color(Color::new(0.73, 0.73, 0.73));
        let quad = Hittable::Quad(Quad::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            white,
        ));
        let list = Hittable::HittableList(crate::hittable_list::HittableList {
            objects: vec![quad],
            bbox: crate::aabb::Aabb::default(),
        });
        let rotated = Hittable::rotate_y(list, 15.0);
        let translated = Hittable::translate(rotated, Vec3::new(1.0, 2.0, 3.0));
        let mat = get_material(&translated);
        assert!(mat.is_some(), "get_material should penetrate Translate→RotateY→HittableList→Quad");
        let albedo = match mat.unwrap() {
            Material::Lambertian { tex } => solid_color_albedo(tex),
            _ => panic!("expected Lambertian material"),
        };
        assert!((albedo[0] - 0.73).abs() < 0.01);
        assert!((albedo[1] - 0.73).abs() < 0.01);
        assert!((albedo[2] - 0.73).abs() < 0.01);
    }

    #[test]
    fn test_box_with_transform_not_empty() {
        // This reproduces the bug where Translate(RotateY(HittableList(quads))) was silently dropped
        let white = Material::lambertian_color(Color::new(0.73, 0.73, 0.73));
        let box_geom = crate::quad_box::make_box(
            &Point3::new(0.0, 0.0, 0.0),
            &Point3::new(165.0, 330.0, 165.0),
            white,
        );
        let box_rotated = Hittable::rotate_y(box_geom, 15.0);
        let box_translated = Hittable::translate(box_rotated, Vec3::new(265.0, 0.0, 295.0));
        let world = Hittable::HittableList(crate::hittable_list::HittableList {
            objects: vec![box_translated],
            bbox: crate::aabb::Aabb::default(),
        });
        let scene = GpuScene::from_world(&world);
        // Box = 6 quads = 12 triangles = 24 vertices
        assert!(
            !scene.vertices.is_empty(),
            "Box should produce vertices (was silently dropped before fix)"
        );
        assert_eq!(
            scene.tri_to_material.len(),
            12,
            "6 quads * 2 tris = 12, got {}",
            scene.tri_to_material.len()
        );
        assert_eq!(
            scene.vertices.len(),
            24 * 3,
            "6 quads * 4 verts * 3 floats = 72, got {}",
            scene.vertices.len()
        );
        assert_eq!(
            scene.normals.len(),
            scene.vertices.len(),
            "normals count must match vertices count"
        );
    }
}
