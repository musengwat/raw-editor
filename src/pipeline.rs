//! Adjustment pipeline parameters and the pure-Rust mirror of the WGSL
//! shader math (`src/shader.wgsl`). Any change here MUST be mirrored in the
//! shader and covered by a parity test; export and tests use this path.

use serde::{Deserialize, Serialize};

pub const CONTRAST_PIVOT: f32 = 0.18;

/// User-facing adjustment parameters. Serialized to the `.edit.json` sidecar.
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
}

impl Default for AdjustParams {
    fn default() -> Self {
        Self { exposure_ev: 0.0, contrast: 0.0, temp: 0.0, tint: 0.0, saturation: 0.0 }
    }
}

/// GPU uniform block. Layout must match `Uniforms` in shader.wgsl
/// (vec2 + 6 x f32 = 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    /// NDC half-extent of the aspect-fitted quad.
    pub scale: [f32; 2],
    pub exposure_ev: f32,
    pub contrast: f32,
    pub temp: f32,
    pub tint: f32,
    pub saturation: f32,
    /// 1.0 when the render target is an sRGB format (hardware encodes);
    /// 0.0 when we must encode manually in the shader.
    pub target_is_srgb: f32,
}

impl Uniforms {
    pub fn new(p: &AdjustParams, scale: [f32; 2], target_is_srgb: bool) -> Self {
        Self {
            scale,
            exposure_ev: p.exposure_ev,
            contrast: p.contrast,
            temp: p.temp,
            tint: p.tint,
            saturation: p.saturation,
            target_is_srgb: if target_is_srgb { 1.0 } else { 0.0 },
        }
    }
}

/// CPU mirror of the fragment-shader adjustment chain, on linear sRGB input.
/// Returns *linear* output (no transfer function).
pub fn apply(p: &AdjustParams, rgb: [f32; 3]) -> [f32; 3] {
    let gain = (p.exposure_ev).exp2();
    let r_gain = 1.0 + 0.4 * p.temp;
    let b_gain = 1.0 - 0.4 * p.temp;
    let g_gain = 1.0 - 0.3 * p.tint;

    let mut c = [rgb[0] * gain * r_gain, rgb[1] * gain * g_gain, rgb[2] * gain * b_gain];

    let e = 1.0 + p.contrast;
    for v in &mut c {
        *v = CONTRAST_PIVOT * (v.max(0.0) / CONTRAST_PIVOT).powf(e);
    }

    let luma = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let s = 1.0 + p.saturation;
    for v in &mut c {
        *v = luma + (*v - luma) * s;
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
    fn uniforms_layout_is_32_bytes() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 32);
    }

    #[test]
    fn sidecar_roundtrip() {
        let p = AdjustParams { exposure_ev: 0.7, contrast: -0.2, ..Default::default() };
        let json = serde_json::to_string(&p).unwrap();
        let back: AdjustParams = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }
}
