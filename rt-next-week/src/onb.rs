use crate::vec3::Vec3;

#[derive(Clone)]
pub struct Onb {
    pub axis: [Vec3; 3],
}

impl Onb {
    pub fn new(n: &Vec3) -> Self {
        let w = n.unit_vector();
        let a = if w.x().abs() > 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
        let v = w.cross(&a).unit_vector();
        let u = w.cross(&v);
        Self { axis: [u, v, w] }
    }

    pub fn u(&self) -> &Vec3 { &self.axis[0] }
    pub fn v(&self) -> &Vec3 { &self.axis[1] }
    pub fn w(&self) -> &Vec3 { &self.axis[2] }

    pub fn transform(&self, v: &Vec3) -> Vec3 {
        v.e[0] * self.axis[0] + v.e[1] * self.axis[1] + v.e[2] * self.axis[2]
    }
}
