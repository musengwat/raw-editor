//! Auto exposure: pick `exposure_ev` from the photo itself.
//!
//! The rendered image's median brightness (sRGB-encoded luminance, through
//! the whole pipeline — look, tone curve and all) is brought to
//! [`TARGET_MEDIAN`], about mid grey. Then a highlight guard: if that pushed
//! more than [`MAX_NEW_CLIP`] of the frame to white that the sensor had not
//! already clipped, exposure comes back down until it does not — by at most
//! [`MAX_PULLBACK_EV`], so a backlit subject is not left in the dark. Only
//! exposure changes; everything else in the params is kept.
//!
//! Brightening past [`FULL_BRIGHTEN_EV`] is only half applied. A median aimed
//! at mid grey cannot tell an underexposed frame from a low-key one (a dark
//! pond, a person in black against foliage): on the G85 test frames the two
//! low-key shots asked for +2.0 and +2.2 EV while the camera's own JPEGs kept
//! them dark; every ordinary frame landed within ±0.45 EV. Darkening is never
//! damped — an overexposed frame needs all of it.

use crate::develop::DevelopedImage;
use crate::pipeline::{AdjustParams, apply};

/// Median output brightness to aim for (sRGB-encoded; ~mid grey).
pub const TARGET_MEDIAN: f32 = 0.45;
/// Share of not-already-clipped pixels allowed to go white.
pub const MAX_NEW_CLIP: f32 = 0.01;
/// How far the highlight guard may pull exposure below the median's choice.
pub const MAX_PULLBACK_EV: f32 = 1.5;
pub const EV_RANGE: (f32, f32) = (-4.0, 4.0);
/// Brightening up to this much is applied in full; beyond it, half.
pub const FULL_BRIGHTEN_EV: f32 = 1.0;
/// At most this many pixels are looked at (evenly spaced). Plenty for a
/// median, and keeps a click on Auto well under a frame-drop's worth.
const SAMPLES: usize = 12_000;

fn encode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

struct Sample {
    rgb: [f32; 3],
    /// Clipped at the sensor already: going white is not Auto's doing.
    clipped: bool,
}

fn samples(preview: &DevelopedImage, p: &AdjustParams) -> Vec<Sample> {
    // Exposure-independent part of the pipeline: optics, WB, matrix.
    let work = preview.working(p);
    let n = preview.width * preview.height;
    let step = (n / SAMPLES).max(1);
    (0..n)
        .step_by(step)
        .map(|i| Sample { rgb: [work[i * 3], work[i * 3 + 1], work[i * 3 + 2]], clipped: preview.clip_at(i) > 0.0 })
        .collect()
}

fn median_and_new_clip(s: &[Sample], q: &AdjustParams) -> (f32, f32) {
    let mut lum = Vec::with_capacity(s.len());
    let (mut fresh, mut blown) = (0usize, 0usize);
    for px in s {
        let o = apply(q, px.rgb);
        lum.push(encode(0.2126 * o[0] + 0.7152 * o[1] + 0.0722 * o[2]));
        if !px.clipped {
            fresh += 1;
            if o.iter().any(|&v| v >= 1.0) {
                blown += 1;
            }
        }
    }
    if lum.is_empty() {
        return (TARGET_MEDIAN, 0.0);
    }
    let mid = lum.len() / 2;
    let (_, m, _) = lum.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    (*m, if fresh == 0 { 0.0 } else { blown as f32 / fresh as f32 })
}

/// Bisect `ev` in `lo..hi` for the first value where `too_far(ev)` holds
/// (`too_far` must be monotonic), to 0.01 EV.
fn bisect(mut lo: f32, mut hi: f32, too_far: impl Fn(f32) -> bool) -> f32 {
    for _ in 0..12 {
        let mid = 0.5 * (lo + hi);
        if too_far(mid) { hi = mid } else { lo = mid }
    }
    lo
}

fn soft_knee(ev: f32) -> f32 {
    if ev > FULL_BRIGHTEN_EV { FULL_BRIGHTEN_EV + 0.5 * (ev - FULL_BRIGHTEN_EV) } else { ev }
}

/// `p` with an automatic exposure for `preview` (a downscaled copy of the
/// photo is plenty; the GUI passes its histogram preview).
pub fn auto_exposure(preview: &DevelopedImage, p: &AdjustParams) -> AdjustParams {
    let s = samples(preview, p);
    let at = |ev: f32| median_and_new_clip(&s, &AdjustParams { exposure_ev: ev, ..*p });
    let (lo, hi) = EV_RANGE;
    let ev = soft_knee(bisect(lo, hi, |ev| at(ev).0 > TARGET_MEDIAN));
    let floor = (ev - MAX_PULLBACK_EV).max(lo);
    let ev = if at(ev).1 > MAX_NEW_CLIP { bisect(floor, ev, |e| at(e).1 > MAX_NEW_CLIP) } else { ev };
    // Slider-friendly value.
    AdjustParams { exposure_ev: (ev * 20.0).round() / 20.0, ..*p }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scene with a spread of tones (log-uniform over 4 stops, so its
    /// brightest part survives being brought to mid grey), scaled.
    fn scene(scale: f32) -> DevelopedImage {
        let (w, h) = (200, 100);
        let data = (0..w * h)
            .flat_map(|i| {
                let t = (i % 997) as f32 / 996.0;
                let v = scale * 2f32.powf(-4.0 + 4.0 * t);
                [v, v * 0.9, v * 0.8]
            })
            .collect();
        DevelopedImage::linear_srgb(w, h, data)
    }

    #[test]
    fn dark_photos_brighten_and_bright_ones_darken_to_mid_grey() {
        let p = AdjustParams::default();
        let dark = auto_exposure(&scene(0.1), &p).exposure_ev;
        let bright = auto_exposure(&scene(0.8), &p).exposure_ev;
        assert!(dark > 1.5, "dark scene got {dark} EV");
        assert!(bright < dark - 1.5, "bright scene got {bright} EV vs dark {dark}");
        // Corrections within the full-strength range land on the target;
        // so does any darkening.
        for img in [scene(0.4), scene(0.8), scene(2.0), scene(6.0)] {
            let q = auto_exposure(&img, &p);
            let (m, _) = median_and_new_clip(&samples(&img, &q), &q);
            assert!((m - TARGET_MEDIAN).abs() < 0.02, "median {m}");
        }
    }

    #[test]
    fn it_works_through_the_look_and_keeps_everything_else() {
        let p = AdjustParams { tone_curve: 1.0, saturation: 0.3, contrast: 0.2, crop: [0.1, 0.1, 0.9, 0.9], ..Default::default() };
        let img = scene(0.4);
        let q = auto_exposure(&img, &p);
        assert_eq!(AdjustParams { exposure_ev: p.exposure_ev, ..q }, p, "only exposure changes");
        let (m, _) = median_and_new_clip(&samples(&img, &q), &q);
        assert!((m - TARGET_MEDIAN).abs() < 0.02, "median {m} through the tone curve");
        // Running it again changes nothing.
        assert_eq!(auto_exposure(&img, &q).exposure_ev, q.exposure_ev);
    }

    #[test]
    fn highlights_are_guarded_but_not_at_any_cost() {
        // Dim foreground (60%) with a bright, unclipped sky (40%): the median
        // wants a big push that would blow the sky.
        let (w, h) = (100, 100);
        let data: Vec<f32> = (0..w * h).flat_map(|i| if i < 6000 { [0.01; 3] } else { [0.4; 3] }).collect();
        let img = DevelopedImage::linear_srgb(w, h, data);
        let p = AdjustParams::default();
        let q = auto_exposure(&img, &p);
        let (_, blown) = median_and_new_clip(&samples(&img, &q), &q);
        let unguarded = soft_knee(bisect(EV_RANGE.0, EV_RANGE.1, |ev| {
            median_and_new_clip(&samples(&img, &p), &AdjustParams { exposure_ev: ev, ..p }).0 > TARGET_MEDIAN
        }));
        assert!(q.exposure_ev < unguarded - 0.5, "guard pulled back: {} vs {unguarded}", q.exposure_ev);
        assert!(q.exposure_ev >= unguarded - MAX_PULLBACK_EV - 0.05, "but by at most {MAX_PULLBACK_EV} EV");
        assert!(blown <= 0.41, "{blown}");
    }

    #[test]
    fn sensor_clipped_pixels_do_not_hold_exposure_down() {
        let (w, h) = (100, 100);
        // A room needing ~+0.8 EV and a window the sensor already clipped.
        let data: Vec<f32> = (0..w * h).flat_map(|i| if i < 7000 { [0.1; 3] } else { [1.0; 3] }).collect();
        let mut img = DevelopedImage::linear_srgb(w, h, data);
        img.clip = (0..w * h).map(|i| if i < 7000 { 0.0 } else { 1.0 }).collect();
        let q = auto_exposure(&img, &AdjustParams::default());
        let (m, _) = median_and_new_clip(&samples(&img, &q), &q);
        assert!((m - TARGET_MEDIAN).abs() < 0.02, "a blown window must not keep the room dark: median {m}");
    }

    #[test]
    fn a_black_frame_does_not_blow_up() {
        let img = DevelopedImage::linear_srgb(10, 10, vec![0.0; 300]);
        let q = auto_exposure(&img, &AdjustParams::default());
        assert!(q.exposure_ev.is_finite() && q.exposure_ev <= EV_RANGE.1);
    }

    #[test]
    fn big_brightening_is_damped_but_darkening_is_not() {
        assert_eq!(soft_knee(0.6), 0.6);
        assert_eq!(soft_knee(3.0), 2.0);
        assert_eq!(soft_knee(-3.0), -3.0);
        // A frame that "needs" ~+2.8 EV by the median gets less than that.
        let q = auto_exposure(&scene(0.1), &AdjustParams::default());
        assert!((1.6..2.2).contains(&q.exposure_ev), "{}", q.exposure_ev);
    }

}
