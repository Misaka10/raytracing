use crate::vec3::Vec3;
use rand::Rng;

const POINT_COUNT: usize = 256;

#[derive(Clone)]
pub struct Perlin {
    randvec: [Vec3; POINT_COUNT],
    perm_x: [usize; POINT_COUNT],
    perm_y: [usize; POINT_COUNT],
    perm_z: [usize; POINT_COUNT],
}

impl Perlin {
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        let mut randvec = [Vec3::zero(); POINT_COUNT];
        for i in 0..POINT_COUNT {
            randvec[i] = Vec3::random_range(rng, -1.0, 1.0).unit_vector();
        }

        let mut p = Self {
            randvec,
            perm_x: [0; POINT_COUNT],
            perm_y: [0; POINT_COUNT],
            perm_z: [0; POINT_COUNT],
        };
        p.perm_x = perlin_generate_perm(rng);
        p.perm_y = perlin_generate_perm(rng);
        p.perm_z = perlin_generate_perm(rng);
        p
    }

    pub fn noise(&self, p: &Vec3) -> f64 {
        let u = p.x() - p.x().floor();
        let v = p.y() - p.y().floor();
        let w = p.z() - p.z().floor();

        let i = p.x().floor() as i32;
        let j = p.y().floor() as i32;
        let k = p.z().floor() as i32;

        let mut c: [[[Vec3; 2]; 2]; 2] = [[[Vec3::zero(); 2]; 2]; 2];
        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    let idx = self.perm_x[((i + di as i32) & 255) as usize]
                        ^ self.perm_y[((j + dj as i32) & 255) as usize]
                        ^ self.perm_z[((k + dk as i32) & 255) as usize];
                    c[di][dj][dk] = self.randvec[idx];
                }
            }
        }
        perlin_interp(&c, u, v, w)
    }

    pub fn turb(&self, p: &Vec3, depth: usize) -> f64 {
        let mut accum = 0.0;
        let mut temp_p = *p;
        let mut weight = 1.0;

        for _ in 0..depth {
            accum += weight * self.noise(&temp_p);
            weight *= 0.5;
            temp_p = temp_p * 2.0;
        }
        accum.abs()
    }
}

fn perlin_generate_perm<R: Rng>(rng: &mut R) -> [usize; POINT_COUNT] {
    let mut p = [0; POINT_COUNT];
    for i in 0..POINT_COUNT { p[i] = i; }
    permute(&mut p, rng);
    p
}

fn permute<R: Rng>(p: &mut [usize], rng: &mut R) {
    for i in (1..p.len()).rev() {
        let target = rng.gen_range(0..=i);
        p.swap(i, target);
    }
}

fn perlin_interp(c: &[[[Vec3; 2]; 2]; 2], u: f64, v: f64, w: f64) -> f64 {
    let uu = u * u * (3.0 - 2.0 * u);
    let vv = v * v * (3.0 - 2.0 * v);
    let ww = w * w * (3.0 - 2.0 * w);

    let mut accum = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let weight = Vec3::new(u - i as f64, v - j as f64, w - k as f64);
                let fi = i as f64;
                let fj = j as f64;
                let fk = k as f64;
                accum += (fi * uu + (1.0 - fi) * (1.0 - uu))
                    * (fj * vv + (1.0 - fj) * (1.0 - vv))
                    * (fk * ww + (1.0 - fk) * (1.0 - ww))
                    * c[i][j][k].dot(&weight);
            }
        }
    }
    accum
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_noise_range() {
        let mut rng = SmallRng::seed_from_u64(42);
        let perlin = Perlin::new(&mut rng);
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let v = perlin.noise(&Vec3::new(i as f64 * 0.1, j as f64 * 0.1, k as f64 * 0.1));
                    assert!(v >= -1.0 && v <= 1.0, "noise value {} out of range", v);
                }
            }
        }
    }

    #[test]
    fn test_turb_range() {
        let mut rng = SmallRng::seed_from_u64(99);
        let perlin = Perlin::new(&mut rng);
        let t = perlin.turb(&Vec3::new(0.5, 0.5, 0.5), 7);
        assert!(t >= 0.0, "turb should be non-negative, got {}", t);
    }

    #[test]
    fn test_deterministic() {
        let mut rng = SmallRng::seed_from_u64(42);
        let perlin = Perlin::new(&mut rng);
        let a = perlin.noise(&Vec3::new(0.5, 0.5, 0.5));
        let b = perlin.noise(&Vec3::new(0.5, 0.5, 0.5));
        assert_eq!(a, b);
    }
}
