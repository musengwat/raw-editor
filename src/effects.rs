//! Finishing effects: the vignette and film grain.
//!
//! - The **vignette** is an exposure change that grows from the middle of the
//!   *output* frame to its corners, so it follows the crop (Lightroom's
//!   "post-crop vignetting"). It is added to the exposure the tone chain
//!   runs with, not multiplied on afterwards: darkened corners go through
//!   the same tone curve as everything else, and brightened ones roll off
//!   into the highlights instead of clipping flat.
//! - **Grain** is luminance noise added at the very end, in display space
//!   (where its strength looks the same from shadows to highlights), tied
//!   to the *source* pixel grid so it stays put when the view pans or
//!   zooms and comes out the same in the export.
//!
//! Both are mirrored in `shader.wgsl` (`vignette_ev`, `grain_noise`,
//! `grain`).

use crate::pipeline::{AdjustParams, luma, smoothstep, srgb_encode};

/// A vignette at full amount changes the corners by this many stops.
pub const VIGNETTE_MAX_EV: f32 = 2.5;
/// Grain at full amount moves a mid-grey by up to this much of the
/// display range.
pub const GRAIN_MAX: f32 = 0.14;
/// Smallest grain size, in source pixels: one pixel per grain.
pub const GRAIN_MIN_SIZE: f32 = 1.0;

/// A vignette resolved for one render: the numbers the shader gets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vignette {
    /// Exposure change, in stops, where the vignette is at full strength.
    pub ev: f32,
    /// Distance from the centre (corner = 1) where it starts and where it
    /// reaches full strength.
    pub edges: [f32; 2],
    /// Exponent of the distance norm: 2 is an ellipse, larger is squarer.
    pub power: f32,
    /// Scale applied to the centred coordinates before the norm.
    pub k: [f32; 2],
    /// Brings the norm of a corner to 1.
    pub norm: f32,
}

impl Vignette {
    /// `None` when the amount is zero. `out_aspect` is the output frame's
    /// width / height (after crop and rotation).
    pub fn resolve(p: &AdjustParams, out_aspect: f32) -> Option<Self> {
        if p.vignette_amount == 0.0 {
            return None;
        }
        let mid = p.vignette_midpoint.clamp(0.0, 1.0);
        let feather = p.vignette_feather.clamp(0.0, 1.0);
        let e0 = mid * (1.0 - feather);
        let e1 = (mid + feather * (1.0 - mid)).max(e0 + 1e-3);
        let round = p.vignette_roundness.clamp(-1.0, 1.0);
        // Below zero the shape squares off towards the frame's own outline;
        // above zero it goes from an ellipse matching the frame to a circle.
        let power = 2.0 + 6.0 * (-round).max(0.0);
        let a = out_aspect.max(1e-3);
        let circle = [a * (2.0 / (a * a + 1.0)).sqrt(), (2.0 / (a * a + 1.0)).sqrt()];
        let t = round.max(0.0);
        let k = [1.0 + (circle[0] - 1.0) * t, 1.0 + (circle[1] - 1.0) * t];
        Some(Self { ev: p.vignette_amount * VIGNETTE_MAX_EV, edges: [e0, e1], power, k, norm: 2f32.powf(-1.0 / power) })
    }

    /// The exposure change at an output uv. Mirrors `vignette_ev` in the
    /// shader.
    pub fn ev_at(&self, out_uv: [f32; 2]) -> f32 {
        let q = [((out_uv[0] - 0.5) * 2.0 * self.k[0]).abs(), ((out_uv[1] - 0.5) * 2.0 * self.k[1]).abs()];
        let dist = (q[0].powf(self.power) + q[1].powf(self.power)).powf(1.0 / self.power) * self.norm;
        self.ev * smoothstep(self.edges[0], self.edges[1], dist)
    }

    /// As the two vec4s of the uniform block; all zero for no vignette.
    pub fn uniforms(v: Option<Self>) -> ([f32; 4], [f32; 4]) {
        match v {
            Some(v) => ([v.ev, v.edges[0], v.edges[1], v.power], [v.k[0], v.k[1], v.norm, 0.0]),
            None => ([0.0; 4], [0.0; 4]),
        }
    }
}

/// A repeatable pseudo-random value in −1..1 for an integer grid point.
/// Integer arithmetic only, so the shader's copy gives the same bits.
pub fn hash(x: i32, y: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9E37_79B1) ^ (y as u32).wrapping_mul(0x85EB_CA77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    (h >> 8) as f32 / 8_388_608.0 - 1.0
}

/// Grain at a position in source pixels (pixel centres at `x + 0.5`):
/// random values on a grid `size` pixels apart, blended between. At size 1
/// every source pixel has its own value. −1..1.
pub fn grain_noise(src_px: [f32; 2], size: f32) -> f32 {
    let size = size.max(GRAIN_MIN_SIZE);
    let p = [(src_px[0] - 0.5) / size, (src_px[1] - 0.5) / size];
    let i = [p[0].floor(), p[1].floor()];
    let f = [p[0] - i[0], p[1] - i[1]];
    let (x, y) = (i[0] as i32, i[1] as i32);
    let top = hash(x, y) + (hash(x + 1, y) - hash(x, y)) * f[0];
    let bottom = hash(x, y + 1) + (hash(x + 1, y + 1) - hash(x, y + 1)) * f[0];
    top + (bottom - top) * f[1]
}

/// Add grain `n` (−1..1) to a finished linear pixel. The noise is the same
/// in all three channels (it is luminance grain) and is added to the
/// display-encoded value, a little weaker towards black and white, as on
/// film. Values above white keep their excess. Mirrors `grain` in the
/// shader.
pub fn grain(c: [f32; 3], n: f32, amount: f32) -> [f32; 3] {
    let e = c.map(srgb_encode);
    let y = luma(e);
    let d = n * amount * GRAIN_MAX * (0.4 + 2.4 * y * (1.0 - y));
    std::array::from_fn(|i| srgb_decode((e[i] + d).clamp(0.0, 1.0)) + (c[i] - 1.0).max(0.0))
}

pub fn srgb_decode(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// Grain for the pixel at `src_px`, or the pixel unchanged when the amount
/// is zero.
pub fn apply_grain(p: &AdjustParams, c: [f32; 3], src_px: [f32; 2]) -> [f32; 3] {
    if p.grain_amount <= 0.0 { c } else { grain(c, grain_noise(src_px, p.grain_size), p.grain_amount) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(amount: f32, midpoint: f32, feather: f32, roundness: f32) -> AdjustParams {
        AdjustParams {
            vignette_amount: amount,
            vignette_midpoint: midpoint,
            vignette_feather: feather,
            vignette_roundness: roundness,
            ..Default::default()
        }
    }

    #[test]
    fn no_amount_no_vignette() {
        assert_eq!(Vignette::resolve(&AdjustParams::default(), 1.5), None);
        assert_eq!(Vignette::uniforms(None), ([0.0; 4], [0.0; 4]));
    }

    #[test]
    fn it_grows_from_the_centre_to_the_corners() {
        let v = Vignette::resolve(&params(-1.0, 0.5, 0.5, 0.0), 1.5).unwrap();
        assert_eq!(v.ev_at([0.5, 0.5]), 0.0, "the centre is untouched");
        let corner = v.ev_at([1.0, 1.0]);
        assert!((corner + VIGNETTE_MAX_EV).abs() < 1e-4, "full strength in the corner: {corner}");
        // Monotonic along the diagonal, and darker never brighter.
        let mut prev = 0.0;
        for i in 0..=50 {
            let t = 0.5 + 0.5 * i as f32 / 50.0;
            let ev = v.ev_at([t, t]);
            assert!(ev <= prev + 1e-6 && ev <= 0.0, "{t}: {ev} after {prev}");
            prev = ev;
        }
        // Symmetric in all four corners.
        for c in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]] {
            assert!((v.ev_at(c) - corner).abs() < 1e-5);
        }
        // A positive amount brightens instead.
        let up = Vignette::resolve(&params(0.4, 0.5, 0.5, 0.0), 1.5).unwrap();
        assert!((up.ev_at([0.0, 0.0]) - 0.4 * VIGNETTE_MAX_EV).abs() < 1e-4);
    }

    #[test]
    fn midpoint_and_feather_place_the_falloff() {
        // Start and end as distances from the centre (corner = 1).
        let v = |mid, feather| Vignette::resolve(&params(-1.0, mid, feather, 0.0), 1.0).unwrap().edges;
        assert_eq!(v(0.5, 0.5), [0.25, 0.75]);
        assert_eq!(v(0.5, 1.0), [0.0, 1.0]);
        let hard = v(0.6, 0.0);
        assert!(hard[0] == 0.6 && hard[1] > 0.6 && hard[1] < 0.61, "a hard edge, not a division by zero");
        // A small midpoint reaches further in.
        let (near, far) = (
            Vignette::resolve(&params(-1.0, 0.2, 0.5, 0.0), 1.0).unwrap(),
            Vignette::resolve(&params(-1.0, 0.8, 0.5, 0.0), 1.0).unwrap(),
        );
        assert!(near.ev_at([0.7, 0.5]) < far.ev_at([0.7, 0.5]));
    }

    #[test]
    fn roundness_goes_from_the_frames_shape_to_a_circle() {
        let aspect = 2.0;
        // At 0 the shape matches the frame: the middle of the long edge and
        // of the short edge are equally far out.
        let frame = Vignette::resolve(&params(-1.0, 0.5, 0.5, 0.0), aspect).unwrap();
        assert!((frame.ev_at([1.0, 0.5]) - frame.ev_at([0.5, 1.0])).abs() < 1e-5);
        // At +1 it is a circle in pixels: on a 2:1 frame the end of the long
        // axis is twice as far from the centre as the end of the short one.
        let circle = Vignette::resolve(&params(-1.0, 0.5, 0.5, 1.0), aspect).unwrap();
        assert!(circle.ev_at([1.0, 0.5]) < circle.ev_at([0.5, 1.0]) - 0.3);
        // …and a point a quarter of the width along x is as far as the end of
        // the short axis (both half a frame-height from the centre).
        assert!((circle.ev_at([0.75, 0.5]) - circle.ev_at([0.5, 1.0])).abs() < 1e-4);
        // The corner is still exactly full strength.
        assert!((circle.ev_at([1.0, 1.0]) + VIGNETTE_MAX_EV).abs() < 1e-4);
        // At −1 it squares off: the edge midpoints get much more of it.
        let square = Vignette::resolve(&params(-1.0, 0.5, 0.5, -1.0), aspect).unwrap();
        assert!(square.ev_at([0.8, 0.5]) < frame.ev_at([0.8, 0.5]) - 0.5);
        assert!((square.ev_at([1.0, 1.0]) + VIGNETTE_MAX_EV).abs() < 1e-4);
    }

    #[test]
    fn the_hash_is_repeatable_spread_out_and_in_range() {
        assert_eq!(hash(12, 34), hash(12, 34));
        assert_ne!(hash(12, 34), hash(34, 12));
        assert_ne!(hash(0, 0), hash(-1, 0));
        let n = 200;
        let (mut sum, mut sq) = (0f64, 0f64);
        for y in -n..n {
            for x in -n..n {
                let h = hash(x, y) as f64;
                assert!((-1.0..=1.0).contains(&h));
                sum += h;
                sq += h * h;
            }
        }
        let count = (4 * n * n) as f64;
        let (mean, var) = (sum / count, sq / count);
        assert!(mean.abs() < 0.01, "mean {mean}");
        // Uniform on −1..1 has variance 1/3.
        assert!((var - 1.0 / 3.0).abs() < 0.01, "variance {var}");
        // Neighbours are unrelated: no visible pattern along rows.
        let corr: f64 = (0..10_000).map(|i| hash(i, 7) as f64 * hash(i + 1, 7) as f64).sum::<f64>() / 10_000.0;
        assert!(corr.abs() < 0.02, "row correlation {corr}");
    }

    #[test]
    fn grain_noise_is_per_pixel_at_size_one_and_smooth_when_larger() {
        // At size 1 a pixel centre gets exactly its own hash.
        assert_eq!(grain_noise([10.5, 20.5], 1.0), hash(10, 20));
        assert_eq!(grain_noise([10.5, 20.5], 0.2), hash(10, 20), "sizes below one pixel are one pixel");
        // Continuous: a hair either side of a grid line gives nearly the same.
        let (a, b) = (grain_noise([8.4999, 3.2], 4.0), grain_noise([8.5001, 3.2], 4.0));
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        // Larger grain varies more slowly from pixel to pixel.
        let step = |size: f32| -> f32 {
            (0..2000).map(|i| (grain_noise([i as f32 + 0.5, 9.5], size) - grain_noise([i as f32 + 1.5, 9.5], size)).abs()).sum::<f32>() / 2000.0
        };
        assert!(step(4.0) < step(1.0) * 0.5, "{} vs {}", step(4.0), step(1.0));
    }

    #[test]
    fn grain_is_neutral_strongest_in_the_mid_tones_and_keeps_the_excess() {
        let grey = [0.2f32; 3];
        let up = grain(grey, 1.0, 0.5);
        assert!(up[0] > grey[0] && up[0] == up[1] && up[1] == up[2], "luminance only: {up:?}");
        let down = grain(grey, -1.0, 0.5);
        assert!(down[0] < grey[0]);
        assert_eq!(grain(grey, 0.0, 1.0), grey.map(|v| srgb_decode(srgb_encode(v))));
        // The same noise moves a mid-tone further (in display terms) than a
        // near-black or a near-white.
        let moved = |v: f32| srgb_encode(grain([v; 3], 1.0, 1.0)[0]) - srgb_encode(v);
        assert!(moved(0.2) > moved(0.001) * 1.5 && moved(0.2) > moved(0.95) * 1.5);
        // Never below black or (from below) above white.
        assert_eq!(grain([0.0; 3], -1.0, 1.0), [0.0; 3]);
        assert!(grain([0.98; 3], 1.0, 1.0)[0] <= 1.0);
        // Above white: the excess is carried through.
        let over = grain([1.5; 3], 0.0, 1.0);
        assert!((over[0] - 1.5).abs() < 1e-5);
        // Off means untouched, bit for bit.
        let c = [0.123, 0.456, 0.789];
        assert_eq!(apply_grain(&AdjustParams::default(), c, [3.5, 4.5]), c);
    }
}
