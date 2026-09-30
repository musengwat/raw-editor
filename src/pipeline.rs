//! Adjustment pipeline parameters and the pure-Rust mirror of the WGSL
//! shader math (`src/shader.wgsl`). Any change here MUST be mirrored in the
//! shader and covered by a parity test; export and tests use this path.

use serde::{Deserialize, Serialize};

pub const CONTRAST_PIVOT: f32 = 0.18;

// --- Tonal-range mask shape (display-domain luminance thresholds) ----------
// Masks are evaluated on `d = clamp(luma, 0, 1)^(1/2.2)`, an approximate
// display-referred position, so the ranges line up with what the eye sees
// rather than with scene-linear values (where 0.18 is mid grey).
pub const MASK_GAMMA: f32 = 2.2;
const HIGH_LO: f32 = 0.40;
const HIGH_HI: f32 = 0.95;
const SHAD_LO: f32 = 0.05;
const SHAD_HI: f32 = 0.60;
const WHITE_LO: f32 = 0.65;
const WHITE_HI: f32 = 1.00;
const BLACK_LO: f32 = 0.00;
const BLACK_HI: f32 = 0.30;
/// Whites are a gentler exposure push than highlights (EV at full mask).
const WHITES_EV: f32 = 0.75;
/// Blacks is an additive lift/crush of the floor, in scene-linear units.
const BLACKS_LIFT: f32 = 0.05;

/// Smallest gamma we will divide by, to keep `pow` finite.
pub const MIN_GAMMA: f32 = 0.01;

/// User-facing adjustment parameters. Serialized to the `.edit.json` sidecar.
///
/// Sign convention for the tonal-range controls follows Lightroom's basic
/// panel: **positive brightens that range, negative darkens it**. So negative
/// `highlights` is the "recovery"/compression direction and positive `shadows`
/// lifts the bottom end. (True highlight *reconstruction* of clipped channels
/// is deliberately out of scope — see README.)
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AdjustParams {
    /// Exposure in EV stops (gain = 2^ev).
    pub exposure_ev: f32,
    /// -1..1, pivot-based power curve in linear space.
    pub contrast: f32,
    /// -1..1, warm(+)/cool(-) as opposing R/B gains.
    pub temp: f32,
    /// -1..1, magenta(+)/green(-) as G gain.
    pub tint: f32,
    /// -1..1, luma-preserving saturation.
    pub saturation: f32,
    /// -1..1 EV applied through a broad upper-range mask.
    pub highlights: f32,
    /// -1..1 EV applied through a broad lower-range mask.
    pub shadows: f32,
    /// -1..1, a narrower push at the very top of the range.
    pub whites: f32,
    /// -1..1, additive lift/crush at the very bottom of the range.
    pub blacks: f32,
    /// Resolve-style lift, per channel. Neutral = 0.
    pub lift: [f32; 3],
    /// Resolve-style gamma, per channel. Neutral = 1.
    pub gamma: [f32; 3],
    /// Resolve-style gain, per channel. Neutral = 1.
    pub gain: [f32; 3],

    /// Unsharp-mask strength. 0 = off.
    pub sharpen_amount: f32,
    /// Gaussian sigma of the blur the unsharp mask subtracts, in source pixels.
    pub sharpen_radius: f32,
    /// Detail below this magnitude (linear luma) is left alone, so sharpening
    /// does not simply amplify grain.
    pub sharpen_threshold: f32,
    /// Luminance noise reduction, 0..1. Edge-preserving (bilateral).
    pub nr_luma: f32,
    /// Colour noise reduction, 0..1. Plain blur of the chroma difference,
    /// which is where high-ISO blotches live.
    pub nr_chroma: f32,
}

/// Half-width of the detail neighbourhood, in source pixels. Mirrored in
/// `shader.wgsl` as `DETAIL_R`.
pub const DETAIL_RADIUS: i32 = 4;
/// Spatial sigma of the noise-reduction kernel.
pub const NR_SIGMA: f32 = 2.0;
/// Range sigma of the bilateral weight: luminance differences much larger than
/// this read as an edge and are not averaged across.
pub const NR_RANGE: f32 = 0.08;

impl Default for AdjustParams {
    fn default() -> Self {
        Self {
            exposure_ev: 0.0,
            contrast: 0.0,
            temp: 0.0,
            tint: 0.0,
            saturation: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            whites: 0.0,
            blacks: 0.0,
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
            sharpen_amount: 0.0,
            sharpen_radius: 1.0,
            sharpen_threshold: 0.0,
            nr_luma: 0.0,
            nr_chroma: 0.0,
        }
    }
}

impl AdjustParams {
    /// True when every control sits at its neutral value.
    pub fn is_neutral(&self) -> bool {
        *self == Self::default()
    }
}

/// What the raw file says about its own colour: the camera's as-shot white
/// balance and the matrix from white-balanced camera RGB to linear sRGB.
///
/// The develop stage hands the GPU *camera* RGB, not yet white balanced or
/// matrixed, so that both can be changed live — an infrared conversion needs
/// white balance far outside anything the camera offers, and a camera matrix
/// that means nothing once the sensor sees infrared.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageColour {
    /// Per-channel gains, green = 1.
    pub as_shot_wb: [f32; 3],
    /// Row-major; row sums are 1, so white-balanced neutral stays neutral.
    pub cam_to_srgb: [[f32; 3]; 3],
}

pub const IDENTITY3: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl ImageColour {
    /// For data that is already linear sRGB: unit gains, identity matrix.
    pub const LINEAR_SRGB: Self = Self { as_shot_wb: [1.0; 3], cam_to_srgb: IDENTITY3 };
}

/// White balance and matrix as actually applied to one render.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Colour {
    pub wb: [f32; 3],
    pub matrix: [[f32; 3]; 3],
}

impl Colour {
    pub fn resolve(img: &ImageColour, _p: &AdjustParams) -> Self {
        Self { wb: img.as_shot_wb, matrix: img.cam_to_srgb }
    }
}

/// Camera RGB → working linear RGB for one pixel: white balance, then make a
/// clipped pixel neutral, then the matrix. Mirrors `to_working` in the shader.
///
/// `clip` is 0..1 rather than a flag because the GPU samples the clip mask
/// with bilinear filtering; blending keeps the edge of a blown area soft. A
/// saturated photosite only says "at least this much", so its channel ratios
/// are meaningless — white-balance gains applied to them turn blown highlights
/// magenta. The brightest channel (the one least likely to have clipped) is
/// the best luminance estimate, and neutral is the only honest colour. This
/// has to happen *after* white balance, which is why it lives here and not in
/// the develop stage.
pub fn to_working(col: &Colour, cam: [f32; 3], clip: f32) -> [f32; 3] {
    let mut c = [cam[0] * col.wb[0], cam[1] * col.wb[1], cam[2] * col.wb[2]];
    let m = c[0].max(c[1]).max(c[2]);
    for v in &mut c {
        *v += (m - *v) * clip;
    }
    let r = &col.matrix;
    std::array::from_fn(|i| r[i][0] * c[0] + r[i][1] * c[1] + r[i][2] * c[2])
}

/// GPU uniform block. Layout must match `Uniforms` in shader.wgsl.
/// The `vec4` members come first so they land on 16-byte offsets
/// (WGSL uniform address space requires it); the trailing scalars pack after.
/// Total size 208 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    /// White-balance gains `[r, g, b, _pad]`.
    pub wb: [f32; 4],
    /// Rows of the camera → working matrix, each `[_, _, _, _pad]`.
    pub m0: [f32; 4],
    pub m1: [f32; 4],
    pub m2: [f32; 4],
    /// `[r, g, b, _pad]`
    pub lift: [f32; 4],
    pub gamma: [f32; 4],
    pub gain: [f32; 4],
    /// NDC half-extent of the drawn quad.
    pub scale: [f32; 2],
    /// NDC translation of the quad, used for panning when zoomed in.
    pub offset: [f32; 2],
    /// 1 / source texture size, so the shader can step in source pixels.
    pub texel: [f32; 2],
    pub _pad: [f32; 2],
    pub exposure_ev: f32,
    pub contrast: f32,
    pub temp: f32,
    pub tint: f32,
    pub saturation: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    /// 1.0 when the render target is an sRGB format (hardware encodes);
    /// 0.0 when we must encode manually in the shader.
    pub target_is_srgb: f32,
    pub sharpen_amount: f32,
    pub sharpen_radius: f32,
    pub sharpen_threshold: f32,
    pub nr_luma: f32,
    pub nr_chroma: f32,
    /// WGSL rounds uniform struct sizes up to a multiple of the largest member
    /// alignment (16, for the vec4s), so the Rust side has to match or the
    /// buffer comes out too small.
    pub _tail: f32,
}

/// How the image sits in the viewport, plus the source texel size the detail
/// filters step by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// NDC half-extent of the quad.
    pub scale: [f32; 2],
    /// NDC translation, for panning when zoomed past the viewport.
    pub offset: [f32; 2],
    /// 1 / source texture size.
    pub texel: [f32; 2],
}

impl View {
    /// Aspect-fitted, unpanned.
    pub fn fit(scale: [f32; 2], texel: [f32; 2]) -> Self {
        Self { scale, offset: [0.0, 0.0], texel }
    }

    /// One output pixel per source pixel, filling the whole target.
    pub fn one_to_one(texel: [f32; 2]) -> Self {
        Self::fit([1.0, 1.0], texel)
    }
}

impl Uniforms {
    pub fn new(p: &AdjustParams, img: &ImageColour, view: View, target_is_srgb: bool) -> Self {
        let v4 = |a: [f32; 3]| [a[0], a[1], a[2], 0.0];
        let col = Colour::resolve(img, p);
        Self {
            wb: v4(col.wb),
            m0: v4(col.matrix[0]),
            m1: v4(col.matrix[1]),
            m2: v4(col.matrix[2]),
            lift: v4(p.lift),
            gamma: v4(p.gamma),
            gain: v4(p.gain),
            scale: view.scale,
            offset: view.offset,
            texel: view.texel,
            _pad: [0.0, 0.0],
            exposure_ev: p.exposure_ev,
            contrast: p.contrast,
            temp: p.temp,
            tint: p.tint,
            saturation: p.saturation,
            highlights: p.highlights,
            shadows: p.shadows,
            whites: p.whites,
            blacks: p.blacks,
            target_is_srgb: if target_is_srgb { 1.0 } else { 0.0 },
            sharpen_amount: p.sharpen_amount,
            sharpen_radius: p.sharpen_radius,
            sharpen_threshold: p.sharpen_threshold,
            nr_luma: p.nr_luma,
            nr_chroma: p.nr_chroma,
            _tail: 0.0,
        }
    }

    /// Uniforms that pass the image through untouched — used by the
    /// hold-to-compare "before" view and by neutral exports.
    pub fn passthrough(img: &ImageColour, view: View, target_is_srgb: bool) -> Self {
        Self::new(&AdjustParams::default(), img, view, target_is_srgb)
    }
}

/// Hermite interpolation between two edges, identical to WGSL `smoothstep`.
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Rec.709 luminance of a linear RGB triple.
pub fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// The four tonal-range masks at a given linear luminance, in the order
/// `[highlights, shadows, whites, blacks]`. Each is 0..1.
pub fn tone_masks(luma: f32) -> [f32; 4] {
    let d = luma.clamp(0.0, 1.0).powf(1.0 / MASK_GAMMA);
    [
        smoothstep(HIGH_LO, HIGH_HI, d),
        1.0 - smoothstep(SHAD_LO, SHAD_HI, d),
        smoothstep(WHITE_LO, WHITE_HI, d),
        1.0 - smoothstep(BLACK_LO, BLACK_HI, d),
    ]
}

/// CPU mirror of the fragment-shader adjustment chain, on linear sRGB input.
/// Returns *linear* output (no transfer function).
///
/// Order: exposure + white balance → contrast → highlights/shadows/whites/
/// blacks → lift/gamma/gain → saturation. Keep this in lockstep with
/// `fs_main` in `src/shader.wgsl`.
pub fn apply(p: &AdjustParams, rgb: [f32; 3]) -> [f32; 3] {
    // 1. Exposure and white balance as combined per-channel gains.
    let gain = (p.exposure_ev).exp2();
    let r_gain = 1.0 + 0.4 * p.temp;
    let b_gain = 1.0 - 0.4 * p.temp;
    let g_gain = 1.0 - 0.3 * p.tint;

    let mut c = [rgb[0] * gain * r_gain, rgb[1] * gain * g_gain, rgb[2] * gain * b_gain];

    // 2. Pivot-based contrast in linear space.
    let e = 1.0 + p.contrast;
    for v in &mut c {
        *v = CONTRAST_PIVOT * (v.max(0.0) / CONTRAST_PIVOT).powf(e);
    }

    // 3. Tonal ranges: one combined exposure multiplier from the highlight,
    //    shadow and white masks, plus an additive black-point lift. Masks are
    //    driven by luminance so hue stays put.
    let [mh, ms, mw, mb] = tone_masks(luma(c));
    let tone_gain = (p.highlights * mh + p.shadows * ms + p.whites * WHITES_EV * mw).exp2();
    let black_lift = p.blacks * BLACKS_LIFT * mb;
    for v in &mut c {
        *v = *v * tone_gain + black_lift;
    }

    // 4. Lift / gamma / gain, per channel:  out = gain * (in + lift*(1-in))^(1/gamma)
    for i in 0..3 {
        let x = c[i].max(0.0);
        let lifted = (x + p.lift[i] * (1.0 - x)).max(0.0);
        c[i] = p.gain[i] * lifted.powf(1.0 / p.gamma[i].max(MIN_GAMMA));
    }

    // 5. Luma-preserving saturation.
    let l = luma(c);
    let s = 1.0 + p.saturation;
    for v in &mut c {
        *v = l + (*v - l) * s;
    }
    c
}

/// sRGB OETF for one linear component.
pub fn srgb_encode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GREY: [f32; 3] = [0.18, 0.18, 0.18];

    #[test]
    fn identity_params_pass_through() {
        let p = AdjustParams::default();
        let out = apply(&p, [0.1, 0.5, 0.9]);
        for (o, i) in out.iter().zip([0.1, 0.5, 0.9]) {
            assert!((o - i).abs() < 1e-6);
        }
    }

    #[test]
    fn exposure_one_ev_doubles() {
        let p = AdjustParams { exposure_ev: 1.0, ..Default::default() };
        let out = apply(&p, GREY);
        assert!((out[1] - 0.36).abs() < 1e-6);
    }

    #[test]
    fn contrast_fixes_pivot() {
        let p = AdjustParams { contrast: 0.5, ..Default::default() };
        let out = apply(&p, GREY);
        // 0.18 is the pivot: unchanged by contrast.
        assert!((out[1] - 0.18).abs() < 1e-6);
        // Above pivot brightens, below darkens.
        let hi = apply(&p, [0.5; 3]);
        assert!(hi[0] > 0.5);
        let lo = apply(&p, [0.05; 3]);
        assert!(lo[0] < 0.05);
    }

    #[test]
    fn negative_saturation_desaturates_fully() {
        let p = AdjustParams { saturation: -1.0, ..Default::default() };
        let out = apply(&p, [0.6, 0.2, 0.1]);
        assert!((out[0] - out[1]).abs() < 1e-6 && (out[1] - out[2]).abs() < 1e-6);
    }

    #[test]
    fn warm_temp_raises_red_lowers_blue() {
        let p = AdjustParams { temp: 0.5, ..Default::default() };
        let out = apply(&p, GREY);
        assert!(out[0] > GREY[0] && out[2] < GREY[2]);
    }

    #[test]
    fn uniforms_layout_is_208_bytes() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 208);
        // Every vec4 member must sit on a 16-byte boundary for WGSL uniforms.
        let u = Uniforms::passthrough(&ImageColour::LINEAR_SRGB, View::fit([1.0, 1.0], [1.0, 1.0]), false);
        let base = &u as *const _ as usize;
        for field in [
            &u.wb as *const _ as usize,
            &u.m0 as *const _ as usize,
            &u.m1 as *const _ as usize,
            &u.m2 as *const _ as usize,
            &u.lift as *const _ as usize,
            &u.gamma as *const _ as usize,
            &u.gain as *const _ as usize,
        ] {
            assert_eq!((field - base) % 16, 0);
        }
    }

    #[test]
    fn sidecar_roundtrip() {
        let p = AdjustParams { exposure_ev: 0.7, contrast: -0.2, ..Default::default() };
        let json = serde_json::to_string(&p).unwrap();
        let back: AdjustParams = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    // --- Phase 3: tonal ranges -------------------------------------------

    #[test]
    fn masks_are_bounded_and_ordered() {
        for i in 0..=100 {
            let l = i as f32 / 100.0;
            for m in tone_masks(l) {
                assert!((0.0..=1.0).contains(&m), "mask out of range at luma {l}: {m}");
            }
        }
        // A bright pixel is mostly "highlights"/"whites"; a dark one mostly
        // "shadows"/"blacks".
        let [mh, ms, mw, mb] = tone_masks(0.95);
        assert!(mh > 0.9 && mw > 0.5 && ms < 0.01 && mb < 0.01);
        let [mh, ms, mw, mb] = tone_masks(0.0005);
        assert!(ms > 0.9 && mb > 0.9 && mh < 0.01 && mw < 0.01);
    }

    #[test]
    fn highlights_darken_the_top_without_touching_the_bottom() {
        let p = AdjustParams { highlights: -1.0, ..Default::default() };
        let bright = apply(&p, [0.9; 3]);
        let dark = apply(&p, [0.005; 3]);
        assert!(bright[0] < 0.9 * 0.75, "highlights should compress the top: {bright:?}");
        assert!((dark[0] - 0.005).abs() < 1e-4, "shadows must stay put: {dark:?}");
    }

    #[test]
    fn shadows_lift_the_bottom_without_touching_the_top() {
        let p = AdjustParams { shadows: 1.0, ..Default::default() };
        let dark = apply(&p, [0.01; 3]);
        let bright = apply(&p, [0.95; 3]);
        assert!(dark[0] > 0.01 * 1.5, "shadows should lift the bottom: {dark:?}");
        assert!((bright[0] - 0.95).abs() < 1e-3, "highlights must stay put: {bright:?}");
    }

    #[test]
    fn whites_and_blacks_move_the_end_points() {
        let w = apply(&AdjustParams { whites: 1.0, ..Default::default() }, [0.9; 3]);
        assert!(w[0] > 0.9);
        let b_up = apply(&AdjustParams { blacks: 1.0, ..Default::default() }, [0.002; 3]);
        let b_dn = apply(&AdjustParams { blacks: -1.0, ..Default::default() }, [0.002; 3]);
        assert!(b_up[0] > 0.002 && b_dn[0] < 0.002);
    }

    #[test]
    fn tonal_controls_are_neutral_at_zero() {
        let p = AdjustParams::default();
        for l in [0.0f32, 0.05, 0.18, 0.5, 0.9, 1.5] {
            let out = apply(&p, [l; 3]);
            assert!((out[0] - l).abs() < 1e-5, "luma {l} drifted to {}", out[0]);
        }
    }

    // --- Phase 3: lift / gamma / gain ------------------------------------

    #[test]
    fn gain_scales_and_lift_offsets() {
        let g = AdjustParams { gain: [2.0, 1.0, 1.0], ..Default::default() };
        let out = apply(&g, [0.25; 3]);
        assert!((out[0] - 0.5).abs() < 1e-5, "{out:?}");
        assert!((out[1] - 0.25).abs() < 1e-5);

        // lift pulls the floor up: 0 + lift*(1-0) = lift.
        let l = AdjustParams { lift: [0.1, 0.0, 0.0], ..Default::default() };
        let out = apply(&l, [0.0; 3]);
        assert!((out[0] - 0.1).abs() < 1e-5, "{out:?}");
        // ...and leaves 1.0 alone: 1 + lift*(1-1) = 1.
        let out = apply(&l, [1.0; 3]);
        assert!((out[0] - 1.0).abs() < 1e-5, "{out:?}");
    }

    #[test]
    fn gamma_above_one_brightens_midtones() {
        let p = AdjustParams { gamma: [1.5, 1.5, 1.5], ..Default::default() };
        let out = apply(&p, [0.25; 3]);
        assert!(out[0] > 0.25, "{out:?}");
        // Endpoints are fixed points of x^(1/gamma).
        assert!(apply(&p, [1.0; 3])[0] - 1.0 < 1e-5);
        assert!(apply(&p, [0.0; 3])[0].abs() < 1e-5);
    }

    #[test]
    fn negative_input_never_produces_nan() {
        let p = AdjustParams {
            lift: [-0.9, -0.9, -0.9],
            gamma: [0.2, 0.2, 0.2],
            gain: [1.4, 1.4, 1.4],
            blacks: -1.0,
            ..Default::default()
        };
        for v in [0.0f32, 1e-6, 0.01, 0.5, 3.0] {
            let out = apply(&p, [v; 3]);
            assert!(out.iter().all(|x| x.is_finite()), "non-finite at {v}: {out:?}");
        }
    }

    #[test]
    fn is_neutral_tracks_defaults() {
        assert!(AdjustParams::default().is_neutral());
        assert!(!AdjustParams { gamma: [1.0, 1.0, 1.1], ..Default::default() }.is_neutral());
    }

    #[test]
    fn smoothstep_matches_hermite() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        assert!((smoothstep(0.2, 0.6, 0.4) - 0.5).abs() < 1e-6);
    }

    // --- camera RGB → working RGB ---------------------------------------

    /// A plausible row-normalised camera matrix (rows sum to 1).
    const CAM: [[f32; 3]; 3] = [[1.6, -0.5, -0.1], [-0.2, 1.5, -0.3], [0.0, -0.4, 1.4]];

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn linear_srgb_colour_is_identity() {
        let col = Colour::resolve(&ImageColour::LINEAR_SRGB, &AdjustParams::default());
        assert!(close(to_working(&col, [0.1, 0.5, 1.7], 0.0), [0.1, 0.5, 1.7]));
    }

    #[test]
    fn white_balance_is_applied_before_the_matrix() {
        let col = Colour { wb: [2.0, 1.0, 1.5], matrix: CAM };
        let wbd = [0.2, 0.3, 0.4];
        let expect: [f32; 3] =
            std::array::from_fn(|i| CAM[i][0] * wbd[0] + CAM[i][1] * wbd[1] + CAM[i][2] * wbd[2]);
        assert!(close(to_working(&col, [0.1, 0.3, 0.4 / 1.5], 0.0), expect));
    }

    #[test]
    fn clipped_pixel_goes_neutral_whatever_the_white_balance() {
        // Green clipped low (the extended-ISO case): unbalanced ratios would
        // go magenta. Forced neutral after WB, and the matrix keeps it neutral.
        for wb in [[2.3, 1.0, 1.55], [0.4, 1.0, 5.0]] {
            let col = Colour { wb, matrix: CAM };
            let out = to_working(&col, [0.5, 0.7, 0.45], 1.0);
            assert!((out[0] - out[1]).abs() < 1e-5 && (out[1] - out[2]).abs() < 1e-5, "{out:?}");
            let m = (0.5 * wb[0]).max(0.7).max(0.45 * wb[2]);
            assert!((out[1] - m).abs() < 1e-5, "brightest channel is the level: {out:?}");
        }
    }

    #[test]
    fn partial_clip_blends_toward_neutral() {
        let col = Colour { wb: [1.0; 3], matrix: IDENTITY3 };
        let out = to_working(&col, [0.2, 0.6, 1.0], 0.5);
        assert!(close(out, [0.6, 0.8, 1.0]), "{out:?}");
    }
}
