//! Noise reduction and sharpening — the neighbourhood half of the pipeline.
//!
//! This is the pure-Rust mirror of the `detail()` function in
//! `src/shader.wgsl`; the two must stay in lockstep and are held there by
//! `tests/shader_parity.rs`. Everything here works on scene-referred linear
//! RGB, before the tone and colour chain in `pipeline::apply`.
//!
//! Both filters share a single pass over one 9×9 neighbourhood:
//!
//! * **Noise reduction** splits each pixel into luminance and a chroma
//!   difference. Luminance is filtered bilaterally — neighbours weighted by
//!   distance *and* by how close their luminance is — so flat areas smooth out
//!   while edges survive. Chroma just gets a plain Gaussian blur, because the
//!   blotchy colour noise of a high-ISO frame is low-frequency and chroma
//!   carries no detail worth protecting.
//! * **Sharpening** is an unsharp mask on luminance: subtract a small Gaussian
//!   blur to get the high-frequency detail, drop everything below a threshold
//!   so grain is not amplified, and add back a multiple of the rest.

use crate::pipeline::{AdjustParams, DETAIL_RADIUS, NR_RANGE, NR_SIGMA, luma};
use rayon::prelude::*;

/// True when no detail filtering is requested, so callers can skip the work
/// entirely — the common case.
pub fn is_neutral(p: &AdjustParams) -> bool {
    p.sharpen_amount <= 0.0 && p.nr_luma <= 0.0 && p.nr_chroma <= 0.0
}

/// One pixel of the detail stage, given a sampler into the source image.
///
/// `sample(dx, dy)` must return the pixel at the given offset with
/// clamp-to-edge behaviour, matching the GPU sampler.
pub fn pixel(p: &AdjustParams, center: [f32; 3], sample: impl Fn(i32, i32) -> [f32; 3]) -> [f32; 3] {
    let cl = luma(center);
    let s_nr = 2.0 * NR_SIGMA * NR_SIGMA;
    let s_sh = 2.0 * p.sharpen_radius.max(0.3).powi(2);
    let inv_range = 1.0 / NR_RANGE;

    let (mut bil_l, mut bil_w) = (0.0f32, 0.0f32);
    let (mut chroma_sum, mut chroma_w) = ([0.0f32; 3], 0.0f32);
    let (mut sharp_l, mut sharp_w) = (0.0f32, 0.0f32);

    for dy in -DETAIL_RADIUS..=DETAIL_RADIUS {
        for dx in -DETAIL_RADIUS..=DETAIL_RADIUS {
            let s = sample(dx, dy);
            let sl = luma(s);
            let d2 = (dx * dx + dy * dy) as f32;

            let w_spatial = (-d2 / s_nr).exp();
            let dl = (sl - cl) * inv_range;
            let w_range = (-0.5 * dl * dl).exp();
            let w = w_spatial * w_range;
            bil_l += sl * w;
            bil_w += w;

            for i in 0..3 {
                chroma_sum[i] += (s[i] - sl) * w_spatial;
            }
            chroma_w += w_spatial;

            let w_sharp = (-d2 / s_sh).exp();
            sharp_l += sl * w_sharp;
            sharp_w += w_sharp;
        }
    }

    // Noise reduction: blend luminance toward the bilateral average and chroma
    // toward the plain average.
    let nr_l = cl + (bil_l / bil_w.max(1e-6) - cl) * p.nr_luma;
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        let chroma = center[i] - cl;
        let blurred = chroma_sum[i] / chroma_w.max(1e-6);
        out[i] = chroma + (blurred - chroma) * p.nr_chroma;
    }

    // Sharpening: unsharp mask on the original luminance, thresholded, added
    // to the denoised luminance.
    let blur_l = sharp_l / sharp_w.max(1e-6);
    let d = cl - blur_l;
    let d = d.signum() * (d.abs() - p.sharpen_threshold).max(0.0);
    let final_l = nr_l + p.sharpen_amount * d;

    for v in &mut out {
        *v = (*v + final_l).max(0.0);
    }
    out
}

/// Run the detail stage over a whole tightly packed RGB image.
pub fn apply(width: usize, height: usize, data: &[f32], p: &AdjustParams) -> Vec<f32> {
    if is_neutral(p) {
        return data.to_vec();
    }
    let mut out = vec![0f32; width * height * 3];
    out.par_chunks_mut(width * 3).enumerate().for_each(|(y, row)| {
        for x in 0..width {
            let at = |dx: i32, dy: i32| -> [f32; 3] {
                let sx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                let sy = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                let o = (sy * width + sx) * 3;
                [data[o], data[o + 1], data[o + 2]]
            };
            let c = at(0, 0);
            let v = pixel(p, c, at);
            row[x * 3..x * 3 + 3].copy_from_slice(&v);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat field plus alternating per-pixel noise, which is what NR should
    /// remove and sharpening should leave alone (it is below the threshold).
    fn noisy(width: usize, height: usize, amp: f32) -> Vec<f32> {
        let mut v = vec![0f32; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let n = if (x + y) % 2 == 0 { amp } else { -amp };
                let o = (y * width + x) * 3;
                v[o] = 0.5 + n;
                v[o + 1] = 0.5 + n;
                v[o + 2] = 0.5 + n;
            }
        }
        v
    }

    fn variance(data: &[f32]) -> f32 {
        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;
        data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n
    }

    #[test]
    fn neutral_params_are_a_passthrough() {
        let src = noisy(32, 32, 0.02);
        let p = AdjustParams::default();
        assert!(is_neutral(&p));
        assert_eq!(apply(32, 32, &src, &p), src);
    }

    #[test]
    fn luma_nr_reduces_variance() {
        let src = noisy(32, 32, 0.02);
        let p = AdjustParams { nr_luma: 1.0, ..Default::default() };
        let out = apply(32, 32, &src, &p);
        assert!(
            variance(&out) < variance(&src) * 0.5,
            "NR did not smooth: {} vs {}",
            variance(&out),
            variance(&src)
        );
    }

    /// A step edge must survive luminance NR — that is the whole point of the
    /// bilateral range weight.
    #[test]
    fn luma_nr_preserves_an_edge() {
        let (w, h) = (32usize, 32usize);
        let mut src = vec![0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { 0.15 } else { 0.85 };
                let o = (y * w + x) * 3;
                src[o..o + 3].copy_from_slice(&[v, v, v]);
            }
        }
        let p = AdjustParams { nr_luma: 1.0, ..Default::default() };
        let out = apply(w, h, &src, &p);
        // Sample well away from the boundary on both sides.
        let left = out[(16 * w + 4) * 3];
        let right = out[(16 * w + 27) * 3];
        assert!(left < 0.2, "dark side bled: {left}");
        assert!(right > 0.8, "bright side bled: {right}");
    }

    #[test]
    fn sharpening_increases_edge_contrast() {
        let (w, h) = (32usize, 32usize);
        let mut src = vec![0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { 0.3 } else { 0.6 };
                let o = (y * w + x) * 3;
                src[o..o + 3].copy_from_slice(&[v, v, v]);
            }
        }
        let p = AdjustParams { sharpen_amount: 1.5, sharpen_radius: 1.5, ..Default::default() };
        let out = apply(w, h, &src, &p);
        let dark = out[(16 * w + 14) * 3];
        let bright = out[(16 * w + 17) * 3];
        assert!(bright - dark > 0.3, "no overshoot at the edge: {dark} .. {bright}");
    }

    /// Below-threshold detail is grain, and must be left alone.
    #[test]
    fn threshold_suppresses_small_detail() {
        let src = noisy(32, 32, 0.01);
        let p = AdjustParams {
            sharpen_amount: 2.0,
            sharpen_radius: 1.0,
            sharpen_threshold: 0.05,
            ..Default::default()
        };
        let out = apply(32, 32, &src, &p);
        for (a, b) in out.iter().zip(src.iter()) {
            assert!((a - b).abs() < 1e-5, "threshold let grain through: {a} vs {b}");
        }
    }

    #[test]
    fn chroma_nr_leaves_luminance_alone() {
        let (w, h) = (16usize, 16usize);
        let mut src = vec![0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 3;
                // Strong per-pixel colour noise, constant luminance.
                let s = if (x + y) % 2 == 0 { 0.1 } else { -0.1 };
                src[o..o + 3].copy_from_slice(&[0.5 + s, 0.5, 0.5 - s]);
            }
        }
        let p = AdjustParams { nr_chroma: 1.0, ..Default::default() };
        let out = apply(w, h, &src, &p);
        for i in 0..w * h {
            let a = luma([src[i * 3], src[i * 3 + 1], src[i * 3 + 2]]);
            let b = luma([out[i * 3], out[i * 3 + 1], out[i * 3 + 2]]);
            assert!((a - b).abs() < 1e-4, "chroma NR moved luminance: {a} vs {b}");
        }
    }
}
