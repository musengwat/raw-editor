//! Texture and clarity: local contrast at two scales.
//!
//! Both compare a pixel with a smoothed copy of the photo — a **base** — and
//! push the pixel further from it (or, with a negative slider, pull it
//! closer):
//!
//! - **texture** uses a base smoothed over a few pixels, so it acts on fine
//!   surface detail (bark, fabric, skin);
//! - **clarity** uses a base smoothed over a few percent of the frame, so it
//!   acts on local contrast (the modelling of a cloud, a face against its
//!   background).
//!
//! The 9×9 window of the detail stage is far too small for either, so the
//! two bases are worked out once, when the photo loads ([`LocalBase::build`]),
//! at a quarter of its resolution, and handed to the shader as one more
//! small texture. They are smoothed with a guided filter rather than a plain
//! blur: it leaves strong edges in the base, so an edge is not "detail" and
//! clarity does not ring it with a halo.
//!
//! The bases hold a fixed luminance of the **camera** RGB (as-shot white
//! balance, nothing else), and the pixel is measured the same way, so the
//! ratio pixel / base does not depend on any slider: moving exposure or
//! white balance never changes what counts as detail. The ratio then scales
//! the working pixel before the tone chain — a hue-preserving, scene-
//! referred local exposure change. Mirrored in the shader's `local_gain`.

use rayon::prelude::*;

use crate::develop::DevelopedImage;

/// The bases are built at 1/`SCALE` of the photo (when it is big enough to
/// spare the pixels).
const SCALE: usize = 4;
/// Photos with a long side below this are not scaled down.
const SCALE_ABOVE: usize = 1600;
/// The clarity base's filter runs this much smaller again…
const COARSE_SCALE: usize = 4;
/// …when the quarter-size plane's long side is at least this.
const COARSE_SCALE_ABOVE: usize = 400;
/// Radius of the texture base, as a fraction of the long side (~7 px on a
/// 24 MP frame at full size).
const FINE_RADIUS: f32 = 0.0012;
/// Radius of the clarity base, as a fraction of the long side.
const COARSE_RADIUS: f32 = 0.025;
/// How large a variation (variance, in stops²) counts as an edge the base
/// should keep rather than as detail to smooth away.
const FINE_EPS: f32 = 0.08;
const COARSE_EPS: f32 = 0.10;
/// At slider 1 the fine detail (in stops) is multiplied by 1 + this.
pub const TEXTURE_GAIN: f32 = 1.0;
/// Likewise for local contrast.
pub const CLARITY_GAIN: f32 = 0.8;
/// The pixel / base ratio is limited to this many times either way, so a
/// specular highlight or a dead pixel cannot be thrown to infinity.
pub const MAX_RATIO: f32 = 4.0;
const FLOOR: f32 = 1e-6;

fn f16_round(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

/// The smoothed copies of one photo that texture and clarity compare against.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalBase {
    pub width: usize,
    pub height: usize,
    /// Luminance weights on camera RGB — as-shot white balance times a
    /// green-heavy average. All positive, so the luminance of a real pixel
    /// is too.
    pub weights: [f32; 3],
    /// Row-major `[texture base, clarity base, 0]`, linear luminance,
    /// f16-exact (it is uploaded as an rgba16float texture, and the CPU
    /// mirror must read the very same numbers).
    pub base: Vec<[f32; 3]>,
}

impl LocalBase {
    pub fn build(img: &DevelopedImage) -> Self {
        let wb = img.colour.as_shot_wb;
        let weights = [0.25 * wb[0], 0.5 * wb[1], 0.25 * wb[2]];
        let (w, h) = (img.width, img.height);
        let k = if w.max(h) >= SCALE_ABOVE { SCALE } else { 1 };
        let (bw, bh) = (w.div_ceil(k), h.div_ceil(k));

        // Box-averaged luminance, then log2: detail is a ratio, and in stops
        // the filter's edge threshold means the same at every brightness.
        let mut log = vec![0f32; bw * bh];
        log.par_chunks_mut(bw).enumerate().for_each(|(by, row)| {
            for (bx, out) in row.iter_mut().enumerate() {
                let (mut sum, mut n) = (0f32, 0u32);
                for y in by * k..((by + 1) * k).min(h) {
                    for x in bx * k..((bx + 1) * k).min(w) {
                        let o = (y * w + x) * 3;
                        sum += img.data[o] * weights[0] + img.data[o + 1] * weights[1] + img.data[o + 2] * weights[2];
                        n += 1;
                    }
                }
                *out = (sum / n as f32).max(FLOOR).log2();
            }
        });

        let long = bw.max(bh) as f32;
        let fine: Vec<f32> = {
            let r = ((long * FINE_RADIUS).round() as usize).max(1);
            let (a, b) = crate::hdr::guided(&log, bw, bh, r, FINE_EPS);
            log.par_iter().zip(a.par_iter().zip(&b)).map(|(l, (a, b))| a * l + b).collect()
        };
        // The clarity base is smooth over dozens of pixels, so its filter
        // runs on a copy another COARSE_SCALE smaller; only the filter's
        // coefficients are scaled back up, and they are applied to the
        // full-size plane — which keeps its edges as sharp as they were
        // (the "fast guided filter").
        let ck = if long as usize >= COARSE_SCALE_ABOVE { COARSE_SCALE } else { 1 };
        let (cw, ch) = (bw.div_ceil(ck), bh.div_ceil(ck));
        let small = crate::hdr::box_downsample(&log, bw, bh, ck);
        let r = ((long * COARSE_RADIUS / ck as f32).round() as usize).max(2);
        let (a, b) = crate::hdr::guided(&small, cw, ch, r, COARSE_EPS);
        let base = (0..bw * bh)
            .into_par_iter()
            .map(|i| {
                let (x, y) = (i % bw, i / bw);
                let up = |g: &[f32]| crate::hdr::upsample(g, cw, ch, ck, x, y);
                let coarse = up(&a) * log[i] + up(&b);
                [f16_round(fine[i].exp2()), f16_round(coarse.exp2()), 0.0]
            })
            .collect();
        Self { width: bw, height: bh, weights, base }
    }

    /// The luminance the bases are in, for one camera-RGB pixel.
    pub fn luma(&self, cam: [f32; 3]) -> f32 {
        (cam[0] * self.weights[0] + cam[1] * self.weights[1] + cam[2] * self.weights[2]).max(FLOOR)
    }

    /// `[texture base, clarity base]` at a source uv, bilinear between texel
    /// centres and clamped at the edges — what the GPU's sampler returns.
    pub fn sample(&self, uv: [f32; 2]) -> [f32; 2] {
        let fx = (uv[0] * self.width as f32 - 0.5).clamp(0.0, (self.width - 1) as f32);
        let fy = (uv[1] * self.height as f32 - 0.5).clamp(0.0, (self.height - 1) as f32);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let at = |x: usize, y: usize| self.base[y * self.width + x];
        let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
        std::array::from_fn(|k| {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bot = c[k] + (d[k] - c[k]) * tx;
            top + (bot - top) * ty
        })
    }

    /// What to multiply the working pixel by. `cam` is the pixel's camera
    /// RGB as it sits in the texture (before optics and white balance).
    pub fn gain(&self, cam: [f32; 3], uv: [f32; 2], texture: f32, clarity: f32) -> f32 {
        gain(self.luma(cam), self.sample(uv), texture, clarity)
    }
}

/// The gain for a pixel of luminance `y` over bases `[fine, coarse]`.
/// Mirrors `local_gain` in the shader.
pub fn gain(y: f32, base: [f32; 2], texture: f32, clarity: f32) -> f32 {
    let ratio = |b: f32| (y / b.max(FLOOR)).clamp(1.0 / MAX_RATIO, MAX_RATIO);
    ratio(base[0]).powf(texture * TEXTURE_GAIN) * ratio(base[1]).powf(clarity * CLARITY_GAIN)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: usize, h: usize, px: impl Fn(usize, usize) -> f32) -> DevelopedImage {
        let data = (0..w * h).flat_map(|i| [px(i % w, i / w); 3]).collect();
        DevelopedImage::linear_srgb(w, h, data)
    }

    fn uv(x: usize, y: usize, w: usize, h: usize) -> [f32; 2] {
        [(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32]
    }

    #[test]
    fn a_flat_area_has_no_detail_to_change() {
        let img = image(64, 48, |_, _| 0.2);
        let base = LocalBase::build(&img);
        for (x, y) in [(0, 0), (31, 20), (63, 47)] {
            let g = base.gain([0.2; 3], uv(x, y, 64, 48), 1.0, 1.0);
            assert!((g - 1.0).abs() < 2e-3, "({x},{y}): {g}");
        }
    }

    #[test]
    fn sliders_at_zero_change_nothing_whatever_the_picture() {
        let img = image(64, 48, |x, y| 0.05 + 0.01 * ((x * 7 + y * 13) % 17) as f32);
        let base = LocalBase::build(&img);
        assert_eq!(base.gain([0.3, 0.1, 0.02], [0.3, 0.6], 0.0, 0.0), 1.0);
    }

    #[test]
    fn texture_amplifies_fine_detail_and_negative_texture_smooths_it() {
        // A ±10% checker on mid grey: pure fine detail.
        let (w, h) = (64, 48);
        let value = |x: usize, y: usize| if (x + y) % 2 == 0 { 0.22 } else { 0.18 };
        let base = LocalBase::build(&image(w, h, value));
        let contrast = |texture: f32, clarity: f32| {
            let at = |x, y| value(x, y) * base.gain([value(x, y); 3], uv(x, y, w, h), texture, clarity);
            at(30, 20) / at(31, 20)
        };
        let plain = contrast(0.0, 0.0);
        assert!((plain - 0.22 / 0.18).abs() < 1e-5);
        assert!(contrast(1.0, 0.0) > plain * 1.12, "+texture: {} vs {plain}", contrast(1.0, 0.0));
        assert!(contrast(-1.0, 0.0) < plain * 0.9, "−texture: {} vs {plain}", contrast(-1.0, 0.0));
        assert!(contrast(-1.0, 0.0) >= 0.99, "smoothed towards the base, not inverted: {}", contrast(-1.0, 0.0));
    }

    #[test]
    fn clarity_lifts_local_contrast_without_ringing_a_hard_edge() {
        // Two flat halves a stop and a half apart, with a soft bump on one.
        let (w, h) = (160, 96);
        let value = |x: usize, y: usize| {
            let side = if x < w / 2 { 0.1 } else { 0.3 };
            let (dx, dy) = (x as f32 - 120.0, y as f32 - 48.0);
            side * (1.0 + 0.25 * (-(dx * dx + dy * dy) / 18.0).exp())
        };
        let base = LocalBase::build(&image(w, h, value));
        let out = |x: usize, y: usize| value(x, y) * base.gain([value(x, y); 3], uv(x, y, w, h), 0.0, 1.0);
        // The bump stands further out of its surroundings.
        let (before, after) = (value(120, 48) / value(150, 10), out(120, 48) / out(150, 10));
        assert!(after > before * 1.05, "bump {before} -> {after}");
        // Along a row clear of the bump, each side of the edge stays within a
        // few percent of its level: no bright or dark fringe (a plain blur
        // base overshoots by tens of percent here).
        for x in 0..w {
            let level = if x < w / 2 { 0.1 } else { 0.3 };
            let v = out(x, 5);
            assert!((v / level - 1.0).abs() < 0.10, "x {x}: {v} vs {level}");
        }
    }

    #[test]
    fn the_ratio_is_limited() {
        // A single hot pixel a thousand times its surroundings.
        let g = gain(100.0, [0.1, 0.1], 1.0, 1.0);
        assert!((g - MAX_RATIO.powf(TEXTURE_GAIN) * MAX_RATIO.powf(CLARITY_GAIN)).abs() < 1e-3);
        let g = gain(0.0, [0.1, 0.1], 1.0, 0.0);
        assert!((g - (1.0 / MAX_RATIO).powf(TEXTURE_GAIN)).abs() < 1e-4);
        assert!(gain(0.2, [0.0, 0.0], 1.0, 1.0).is_finite());
    }

    #[test]
    fn large_photos_get_a_quarter_size_base_and_small_ones_a_full_one() {
        let small = LocalBase::build(&image(64, 48, |_, _| 0.2));
        assert_eq!((small.width, small.height), (64, 48));
        let big = LocalBase::build(&image(1602, 1001, |x, _| 0.1 + 0.2 * (x / 800) as f32));
        assert_eq!((big.width, big.height), (401, 251));
        // Every value is exactly representable in the f16 texture.
        for b in big.base.iter().step_by(97) {
            assert_eq!(b[0], f16_round(b[0]));
            assert_eq!(b[1], f16_round(b[1]));
        }
    }

    #[test]
    fn white_balance_goes_into_the_weights() {
        let mut img = image(16, 16, |_, _| 0.2);
        img.colour.as_shot_wb = [2.0, 1.0, 1.5];
        let base = LocalBase::build(&img);
        assert_eq!(base.weights, [0.5, 0.5, 0.375]);
        // The base is the same luminance the pixel is measured in.
        let y = base.luma([0.2; 3]);
        let b = base.sample([0.5, 0.5]);
        assert!((y / b[0] - 1.0).abs() < 2e-3 && (y / b[1] - 1.0).abs() < 2e-3);
    }

    #[test]
    fn sampling_is_bilinear_between_texel_centres() {
        let base = LocalBase { width: 2, height: 1, weights: [1.0; 3], base: vec![[1.0, 2.0, 0.0], [3.0, 6.0, 0.0]] };
        assert_eq!(base.sample([0.25, 0.5]), [1.0, 2.0]);
        assert_eq!(base.sample([0.5, 0.5]), [2.0, 4.0]);
        assert_eq!(base.sample([0.0, 0.0]), [1.0, 2.0], "clamped at the edge");
        assert_eq!(base.sample([2.0, 0.5]), [3.0, 6.0]);
    }
}
