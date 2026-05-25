use crate::hittable_list::HittableList;
use crate::vec3::{Point3, Vec3};

pub fn make_box(a: &Point3, b: &Point3, mat: crate::material::Material) -> super::Hittable {
    let min = Point3::new(a.x().min(b.x()), a.y().min(b.y()), a.z().min(b.z()));
    let max = Point3::new(a.x().max(b.x()), a.y().max(b.y()), a.z().max(b.z()));

    let dx = Vec3::new(max.x() - min.x(), 0.0, 0.0);
    let dy = Vec3::new(0.0, max.y() - min.y(), 0.0);
    let dz = Vec3::new(0.0, 0.0, max.z() - min.z());

    let mut sides = HittableList::new();
    sides.add(super::Hittable::Quad(crate::quad::Quad::new(
        Point3::new(min.x(), min.y(), max.z()),
        dx,
        dy,
        mat.clone(),
    )));
    sides.add(super::Hittable::Quad(crate::quad::Quad::new(
        Point3::new(max.x(), min.y(), max.z()),
        -dz,
        dy,
        mat.clone(),
    )));
    sides.add(super::Hittable::Quad(crate::quad::Quad::new(
        Point3::new(max.x(), min.y(), min.z()),
        -dx,
        dy,
        mat.clone(),
    )));
    sides.add(super::Hittable::Quad(crate::quad::Quad::new(
        Point3::new(min.x(), min.y(), min.z()),
        dz,
        dy,
        mat.clone(),
    )));
    sides.add(super::Hittable::Quad(crate::quad::Quad::new(
        Point3::new(min.x(), max.y(), max.z()),
        dx,
        -dz,
        mat.clone(),
    )));
    sides.add(super::Hittable::Quad(crate::quad::Quad::new(
        Point3::new(min.x(), min.y(), min.z()),
        dx,
        dz,
        mat,
    )));

    super::Hittable::HittableList(sides)
}
