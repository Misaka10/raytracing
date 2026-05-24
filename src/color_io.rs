use crate::interval::Interval;
use crate::vec3::Color;

pub fn linear_to_gamma(linear: f64) -> f64 {
    if linear > 0.0 { linear.sqrt() } else { 0.0 }
}

pub fn pixel_to_10bit(pixel_color: &Color) -> [u16; 3] {
    let mut r = pixel_color.x();
    let mut g = pixel_color.y();
    let mut b = pixel_color.z();

    if r.is_nan() { r = 0.0; }
    if g.is_nan() { g = 0.0; }
    if b.is_nan() { b = 0.0; }

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    let intensity = Interval::new(0.0, 0.9999);
    let r10 = (1024.0 * intensity.clamp(r)) as u16;
    let g10 = (1024.0 * intensity.clamp(g)) as u16;
    let b10 = (1024.0 * intensity.clamp(b)) as u16;

    [r10, g10, b10]
}

pub fn pixel_to_16bit(pixel_color: &Color) -> [u16; 3] {
    let mut r = pixel_color.x();
    let mut g = pixel_color.y();
    let mut b = pixel_color.z();

    if r.is_nan() { r = 0.0; }
    if g.is_nan() { g = 0.0; }
    if b.is_nan() { b = 0.0; }

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    let intensity = Interval::new(0.0, 0.9999);
    let r16 = (65536.0 * intensity.clamp(r)) as u16;
    let g16 = (65536.0 * intensity.clamp(g)) as u16;
    let b16 = (65536.0 * intensity.clamp(b)) as u16;

    [r16, g16, b16]
}

pub fn pixel_to_8bit(pixel_color: &Color) -> [u8; 3] {
    let mut r = pixel_color.x();
    let mut g = pixel_color.y();
    let mut b = pixel_color.z();

    if r.is_nan() { r = 0.0; }
    if g.is_nan() { g = 0.0; }
    if b.is_nan() { b = 0.0; }

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    let intensity = Interval::new(0.0, 0.999);
    let r8 = (256.0 * intensity.clamp(r)) as u8;
    let g8 = (256.0 * intensity.clamp(g)) as u8;
    let b8 = (256.0 * intensity.clamp(b)) as u8;

    [r8, g8, b8]
}
