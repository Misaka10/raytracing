use crate::interval::Interval;
use crate::perlin::Perlin;
use crate::vec3::{Color, Point3};
use image::GenericImageView;

#[derive(Clone)]
pub enum Texture {
    SolidColor(Color),
    Checker {
        inv_scale: f64,
        even: Box<Texture>,
        odd: Box<Texture>,
    },
    Image {
        data: Vec<u8>,
        width: u32,
        height: u32,
    },
    Noise {
        noise: Perlin,
        scale: f64,
    },
}

impl Texture {
    pub fn solid_color(albedo: Color) -> Self { Texture::SolidColor(albedo) }

    pub fn checker(scale: f64, even: Texture, odd: Texture) -> Self {
        Texture::Checker { inv_scale: 1.0 / scale, even: Box::new(even), odd: Box::new(odd) }
    }

    pub fn checker_colors(scale: f64, c1: Color, c2: Color) -> Self {
        Self::checker(scale, Texture::SolidColor(c1), Texture::SolidColor(c2))
    }

    pub fn image(filename: &str) -> anyhow::Result<Self> {
        let img = image::open(filename)?;
        let (width, height) = img.dimensions();
        let data = img.to_rgb8().into_raw();
        Ok(Texture::Image { data, width, height })
    }

    pub fn noise(scale: f64, perlin: Perlin) -> Self {
        Texture::Noise { noise: perlin, scale }
    }

    pub fn value(&self, u: f64, v: f64, p: &Point3) -> Color {
        match self {
            Texture::SolidColor(albedo) => *albedo,
            Texture::Checker { inv_scale, even, odd } => {
                let xi = (*inv_scale * p.x()).floor() as i32;
                let yi = (*inv_scale * p.y()).floor() as i32;
                let zi = (*inv_scale * p.z()).floor() as i32;
                if (xi + yi + zi) % 2 == 0 { even.value(u, v, p) } else { odd.value(u, v, p) }
            }
            Texture::Image { data, width, height } => {
                if *height == 0 {
                    return Color::new(0.0, 1.0, 1.0);
                }
                let u = Interval::new(0.0, 1.0).clamp(u);
                let v = 1.0 - Interval::new(0.0, 1.0).clamp(v);
                let i = ((u * *width as f64) as u32).min(*width - 1);
                let j = ((v * *height as f64) as u32).min(*height - 1);
                let idx = ((j * *width + i) * 3) as usize;
                let scale = 1.0 / 255.0;
                Color::new(
                    scale * data[idx] as f64,
                    scale * data[idx + 1] as f64,
                    scale * data[idx + 2] as f64,
                )
            }
            Texture::Noise { noise, scale } => {
                let s = *scale * p.z() + 10.0 * noise.turb(p, 7);
                Color::new(0.5, 0.5, 0.5) * (1.0 + s.sin())
            }
        }
    }
}
