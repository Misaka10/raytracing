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

    // 将 10-bit [0,1023] 缩放到 16-bit [0,65535] 以正确显示在 16-bit PNG 中
    let scale = 65535.0 / 1023.0;
    let r16 = (r10 as f64 * scale) as u16;
    let g16 = (g10 as f64 * scale) as u16;
    let b16 = (b10 as f64 * scale) as u16;

    [r16, g16, b16]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_to_gamma_zero() {
        assert_eq!(linear_to_gamma(0.0), 0.0);
    }

    #[test]
    fn test_linear_to_gamma_one() {
        assert!((linear_to_gamma(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_gamma_half() {
        let g = linear_to_gamma(0.25);
        assert!((g - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_gamma_negative() {
        assert_eq!(linear_to_gamma(-0.5), 0.0);
    }

    #[test]
    fn test_pixel_to_8bit_black() {
        let c = Color::zero();
        let p = pixel_to_8bit(&c);
        assert_eq!(p, [0, 0, 0]);
    }

    #[test]
    fn test_pixel_to_8bit_white() {
        let c = Color::new(1.0, 1.0, 1.0);
        let p = pixel_to_8bit(&c);
        assert!(p[0] >= 250);
    }

    #[test]
    fn test_pixel_to_10bit_not_zero() {
        let c = Color::new(1.0, 1.0, 1.0);
        let p = pixel_to_10bit(&c);
        // 10-bit 值已缩放到 16-bit 范围，白色应接近 65535
        assert!(p[0] > 60000);
        assert!(p[1] > 60000);
        assert!(p[2] > 60000);
    }

    #[test]
    fn test_pixel_to_10bit_black() {
        let c = Color::zero();
        let p = pixel_to_10bit(&c);
        assert_eq!(p, [0, 0, 0]);
    }

    #[test]
    fn test_pixel_nan_guard() {
        let c = Color::new(f64::NAN, f64::NAN, f64::NAN);
        let p = pixel_to_8bit(&c);
        assert_eq!(p, [0, 0, 0]);
    }

    #[test]
    fn test_pixel_to_16bit_white() {
        let c = Color::new(1.0, 1.0, 1.0);
        let p = pixel_to_16bit(&c);
        assert!(p[0] > 60000);
    }
}
