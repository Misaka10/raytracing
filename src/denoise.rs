#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    pub enabled: bool,
    pub radius: u32,
}

impl DenoiseConfig {
    pub fn disabled() -> Self {
        Self { enabled: false, radius: 0 }
    }

    pub fn median(radius: u32) -> Self {
        Self { enabled: true, radius }
    }
}

/// 对 RGB u16 像素数据应用中值滤波（逐通道独立排序取中值）
pub fn median_filter(data: &mut [[u16; 3]], width: u32, height: u32, radius: u32) {
    let r = radius as i32;
    let w = width as i32;
    let h = height as i32;
    let size = ((2 * r + 1) * (2 * r + 1)) as usize;

    let mut r_vals = vec![0u16; size];
    let mut g_vals = vec![0u16; size];
    let mut b_vals = vec![0u16; size];

    let mut result = vec![[0u16; 3]; data.len()];

    for y in 0..h {
        for x in 0..w {
            let mut n = 0;
            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = (x + dx).clamp(0, w - 1) as u32;
                    let ny = (y + dy).clamp(0, h - 1) as u32;
                    let p = data[(ny * width + nx) as usize];
                    r_vals[n] = p[0];
                    g_vals[n] = p[1];
                    b_vals[n] = p[2];
                    n += 1;
                }
            }
            r_vals[..n].sort_unstable();
            g_vals[..n].sort_unstable();
            b_vals[..n].sort_unstable();
            let mid = n / 2;
            result[(y as u32 * width + x as u32) as usize] = [r_vals[mid], g_vals[mid], b_vals[mid]];
        }
    }

    data.copy_from_slice(&result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_removes_spike() {
        // 3x3 图像，中心有一个亮噪点
        let mut data = vec![[100u16; 3]; 9];
        data[4] = [65535, 65535, 65535]; // 中心噪点
        median_filter(&mut data, 3, 3, 1);
        // 噪点应被周围暗像素的中值替代
        assert!(data[4][0] < 65535);
    }

    #[test]
    fn test_median_preserves_uniform() {
        let mut data = vec![[500u16, 200u16, 300u16]; 16];
        median_filter(&mut data, 4, 4, 1);
        // 均匀区域内部像素不变（clamp 边界可能有微小变化）
        assert_eq!(data[5][0], 500);
        assert_eq!(data[5][1], 200);
        assert_eq!(data[5][2], 300);
    }

    #[test]
    fn test_median_preserves_size() {
        let mut data = vec![[0u16; 3]; 25];
        let original_len = data.len();
        median_filter(&mut data, 5, 5, 2);
        assert_eq!(data.len(), original_len);
    }

    #[test]
    fn test_median_1x1_no_panic() {
        let mut data = vec![[500u16; 3]];
        median_filter(&mut data, 1, 1, 2);
        assert_eq!(data[0][0], 500);
    }

    #[test]
    fn test_denoise_config_disabled() {
        let cfg = DenoiseConfig::disabled();
        assert!(!cfg.enabled);
    }

    #[test]
    fn test_denoise_config_median() {
        let cfg = DenoiseConfig::median(2);
        assert!(cfg.enabled);
        assert_eq!(cfg.radius, 2);
    }
}
