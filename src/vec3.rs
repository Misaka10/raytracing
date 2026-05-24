use rand::Rng;
use std::ops;

#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3 {
    pub e: [f64; 3],
}

pub type Point3 = Vec3;
pub type Color = Vec3;

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { e: [x, y, z] }
    }
    pub const fn zero() -> Self { Self::new(0.0, 0.0, 0.0) }

    pub fn x(&self) -> f64 { self.e[0] }
    pub fn y(&self) -> f64 { self.e[1] }
    pub fn z(&self) -> f64 { self.e[2] }

    pub fn length(&self) -> f64 { self.length_squared().sqrt() }
    pub fn length_squared(&self) -> f64 {
        self.e[0] * self.e[0] + self.e[1] * self.e[1] + self.e[2] * self.e[2]
    }

    pub fn near_zero(&self) -> bool {
        let s = 1e-8;
        self.e[0].abs() < s && self.e[1].abs() < s && self.e[2].abs() < s
    }

    pub fn dot(&self, other: &Vec3) -> f64 {
        self.e[0] * other.e[0] + self.e[1] * other.e[1] + self.e[2] * other.e[2]
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.e[1] * other.e[2] - self.e[2] * other.e[1],
            self.e[2] * other.e[0] - self.e[0] * other.e[2],
            self.e[0] * other.e[1] - self.e[1] * other.e[0],
        )
    }

    pub fn unit_vector(&self) -> Vec3 {
        *self / self.length()
    }

    pub fn random<R: Rng>(rng: &mut R) -> Vec3 {
        Vec3::new(rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>())
    }

    pub fn random_range<R: Rng>(rng: &mut R, min: f64, max: f64) -> Vec3 {
        Vec3::new(
            rng.gen_range(min..max),
            rng.gen_range(min..max),
            rng.gen_range(min..max),
        )
    }
}

impl ops::Index<usize> for Vec3 {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 { &self.e[i] }
}
impl ops::IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, i: usize) -> &mut f64 { &mut self.e[i] }
}

impl ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 { Vec3::new(-self.e[0], -self.e[1], -self.e[2]) }
}

impl ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.e[0] + rhs.e[0], self.e[1] + rhs.e[1], self.e[2] + rhs.e[2])
    }
}
impl ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.e[0] - rhs.e[0], self.e[1] - rhs.e[1], self.e[2] - rhs.e[2])
    }
}
impl ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.e[0] * rhs.e[0], self.e[1] * rhs.e[1], self.e[2] * rhs.e[2])
    }
}
impl ops::Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, t: f64) -> Vec3 { Vec3::new(self.e[0] * t, self.e[1] * t, self.e[2] * t) }
}
impl ops::Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 { v * self }
}
impl ops::Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, t: f64) -> Vec3 { self * (1.0 / t) }
}

impl ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        self.e[0] += rhs.e[0]; self.e[1] += rhs.e[1]; self.e[2] += rhs.e[2];
    }
}
impl ops::MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, t: f64) {
        self.e[0] *= t; self.e[1] *= t; self.e[2] *= t;
    }
}
impl ops::DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, t: f64) { *self *= 1.0 / t; }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.e[0], self.e[1], self.e[2])
    }
}

pub fn random_in_unit_disk<R: Rng>(rng: &mut R) -> Vec3 {
    loop {
        let p = Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0);
        if p.length_squared() < 1.0 { return p; }
    }
}

pub fn random_unit_vector<R: Rng>(rng: &mut R) -> Vec3 {
    loop {
        let p = Vec3::random_range(rng, -1.0, 1.0);
        let lensq = p.length_squared();
        if lensq > 1e-160 && lensq <= 1.0 { return p / lensq.sqrt(); }
    }
}

pub fn random_on_hemisphere<R: Rng>(normal: &Vec3, rng: &mut R) -> Vec3 {
    let on_unit_sphere = random_unit_vector(rng);
    if on_unit_sphere.dot(normal) > 0.0 { on_unit_sphere } else { -on_unit_sphere }
}

pub fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    *v - 2.0 * v.dot(n) * *n
}

pub fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = (-*uv).dot(n).min(1.0);
    let r_out_perp = etai_over_etat * (*uv + cos_theta * *n);
    let r_out_parallel = -(1.0 - r_out_perp.length_squared()).abs().sqrt() * *n;
    r_out_perp + r_out_parallel
}

pub fn random_cosine_direction<R: Rng>(rng: &mut R) -> Vec3 {
    let r1: f64 = rng.gen();
    let r2: f64 = rng.gen();
    let phi = 2.0 * std::f64::consts::PI * r1;
    let x = phi.cos() * r2.sqrt();
    let y = phi.sin() * r2.sqrt();
    let z = (1.0 - r2).sqrt();
    Vec3::new(x, y, z)
}
