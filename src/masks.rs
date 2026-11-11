//! Masks: local adjustments limited to part of the frame.
//!
//! A mask is a shape (a linear or radial gradient) plus a set of slider
//! offsets. For every pixel the shader — and [`locals`], its CPU twin — adds
//! each mask's offsets, scaled by the mask's weight there, to the global
//! slider values, and the ordinary tone chain then runs once with the result.
//! There is no second pipeline: a mask at weight 1 with +1 EV does exactly
//! what one more stop on the Exposure slider would.
//!
//! A mask can also be limited to a **range** of brightness, of colour, or
//! both ([`MaskRange`]) — "the blues inside this gradient" — and a mask of
//! kind `Range` has no shape at all: it is the range over the whole frame.
//! Ranges are judged on the pixel after the *global* exposure, white balance
//! and tone curve ([`RangeRef`]), so they follow what is on screen but never
//! shift while a mask's own sliders move.
//!
//! Geometry is stored in **source uv** (the developed image as it sits in the
//! texture, before the camera's orientation, rotation, straighten or crop),
//! so reframing a photo never moves its masks. Distances are measured in
//! "aspect space" — uv with x multiplied by width / height — so a circle is a
//! circle; the unit is the source height.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::pipeline::{AdjustParams, HSL_FULL_SAT, Locals, MASK_GAMMA, filmic, luma, smoothstep};

/// Mask slots per photo. Fixed so that `AdjustParams` stays `Copy` (undo,
/// copy / paste and the per-frame change checks rely on it) and the shader's
/// uniform block has a fixed size.
pub const MAX_MASKS: usize = 8;
/// Smallest radial feather: at 0 the edge would be a division by zero.
pub const MIN_FEATHER: f32 = 0.01;
/// Smallest radial semi-axis, in source heights.
pub const MIN_RADIUS: f32 = 0.005;
/// Smallest range feather (brightness 0..1, or a fraction of the hue
/// circle): the edges of a range are smoothsteps and need some width.
pub const MIN_RANGE_FEATHER: f32 = 1e-3;
/// How strongly the overlay tints a fully selected pixel.
pub const OVERLAY_OPACITY: f32 = 0.55;
/// The overlay tint (linear RGB): the traditional rubylith red.
pub const OVERLAY_COLOUR: [f32; 3] = [1.0, 0.08, 0.08];

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskKind {
    /// An unused slot.
    #[default]
    Off,
    /// Full effect at `a`, fading to none at `b`, constant across the line.
    Linear,
    /// An ellipse centred on `a` with semi-axes `b`, turned by `angle`; full
    /// effect inside, fading out over the outer `feather` of the radius.
    Radial,
    /// No shape: the whole frame, limited only by the mask's range.
    Range,
}

impl MaskKind {
    pub fn label(self) -> &'static str {
        match self {
            MaskKind::Off => "Off",
            MaskKind::Linear => "Linear gradient",
            MaskKind::Radial => "Radial gradient",
            MaskKind::Range => "Range",
        }
    }
}

/// What a mask adds to the global sliders where its weight is 1. Same units
/// and ranges as the sliders of the same names in `AdjustParams`.
#[derive(Clone, Copy, PartialEq, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MaskAdjust {
    pub exposure_ev: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub temp: f32,
    pub tint: f32,
    pub saturation: f32,
    /// Added to the sharpening amount.
    pub sharpen: f32,
    /// Added to luminance noise reduction.
    pub nr_luma: f32,
}

/// Limits on which pixels inside the shape a mask selects.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MaskRange {
    /// Select only brightnesses from `luma[0]` to `luma[1]`.
    pub luma_on: bool,
    /// Brightness as it shows on screen, 0 (black) to 1 (white).
    pub luma: [f32; 2],
    /// How far outside `luma` the selection fades out.
    pub luma_feather: f32,
    /// Select only colours around `hue`.
    pub colour_on: bool,
    /// Centre of the colour range, degrees (0 red, 120 green, 240 blue).
    pub hue: f32,
    /// Hues within this many degrees of the centre are fully selected.
    pub hue_width: f32,
    /// Degrees beyond `hue_width` over which the selection fades out.
    pub hue_feather: f32,
}

impl Default for MaskRange {
    fn default() -> Self {
        Self { luma_on: false, luma: [0.0, 1.0], luma_feather: 0.1, colour_on: false, hue: 220.0, hue_width: 25.0, hue_feather: 20.0 }
    }
}

impl MaskRange {
    fn luma_feather(&self) -> f32 {
        self.luma_feather.max(MIN_RANGE_FEATHER)
    }

    /// Centre, width and feather as fractions of the hue circle.
    fn hue_turns(&self) -> [f32; 3] {
        [(self.hue / 360.0).rem_euclid(1.0), self.hue_width.max(0.0) / 360.0, (self.hue_feather / 360.0).max(MIN_RANGE_FEATHER)]
    }

    /// Centre the brightness range on a picked brightness.
    pub fn pick_luma(&mut self, luma: f32) {
        self.luma_on = true;
        self.luma = [(luma - PICK_LUMA_HALF).max(0.0), (luma + PICK_LUMA_HALF).min(1.0)];
    }

    /// Centre the colour range on a picked hue (0..1 of the circle).
    pub fn pick_hue(&mut self, hue: f32) {
        self.colour_on = true;
        self.hue = (hue * 360.0).rem_euclid(360.0);
    }
}

/// Half-width of the brightness range a pick sets.
pub const PICK_LUMA_HALF: f32 = 0.15;

/// What a mask's range is judged on, for one pixel: its brightness and
/// colour after the global exposure, white balance and tone curve — close to
/// what the screen shows, and independent of every mask.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RangeRef {
    /// Display-domain brightness, 0..1.
    pub luma: f32,
    /// Hue, 0..1 of the circle (0 for a grey).
    pub hue: f32,
    /// HSV saturation; 0 for a grey.
    pub sat: f32,
}

impl RangeRef {
    /// For a working-RGB pixel (as [`crate::pipeline::to_working`] returns
    /// it, before the detail stage). Mirrors `range_ref` in the shader.
    pub fn of(p: &AdjustParams, rgb: [f32; 3]) -> Self {
        let gain = p.exposure_ev.exp2();
        let c = [rgb[0] * gain * (1.0 + 0.4 * p.temp), rgb[1] * gain * (1.0 - 0.3 * p.tint), rgb[2] * gain * (1.0 - 0.4 * p.temp)];
        let y = luma(c).clamp(0.0, 1.0);
        let y = y + p.tone_curve * (filmic(y) - y);
        let (hue, sat) = hue_sat(c);
        Self { luma: y.powf(1.0 / MASK_GAMMA), hue, sat }
    }
}

/// HSV hue (0..1) and saturation of a linear RGB colour; `(0, 0)` for a grey
/// or black. Mirrors `hue_sat` in the shader — keep the tie order of the
/// max-channel branches the same.
pub fn hue_sat(c: [f32; 3]) -> (f32, f32) {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    let d = mx - mn;
    if mx <= 1e-6 || d <= 1e-6 {
        return (0.0, 0.0);
    }
    let h6 = if mx == c[0] {
        (c[1] - c[2]) / d
    } else if mx == c[1] {
        (c[2] - c[0]) / d + 2.0
    } else {
        (c[0] - c[1]) / d + 4.0
    };
    let h = h6 / 6.0;
    (h - h.floor(), d / mx)
}

#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Mask {
    pub kind: MaskKind,
    /// The eye in the mask list: a disabled mask keeps its settings but does
    /// nothing.
    pub enabled: bool,
    /// Affect everything the shape leaves out instead.
    pub invert: bool,
    /// Overall strength, 0..1: scales every offset.
    pub amount: f32,
    /// Linear: where the effect is full. Radial: the centre. Source uv.
    pub a: [f32; 2],
    /// Linear: where the effect has faded to nothing (source uv). Radial: the
    /// semi-axes, in source heights.
    pub b: [f32; 2],
    /// Radial only: rotation of the ellipse in radians.
    pub angle: f32,
    /// Radial only: the fraction of the radius the edge fades over, 0..1.
    pub feather: f32,
    pub range: MaskRange,
    pub adjust: MaskAdjust,
}

impl Default for Mask {
    fn default() -> Self {
        Self {
            kind: MaskKind::Off,
            enabled: true,
            invert: false,
            amount: 1.0,
            a: [0.5, 0.5],
            b: [0.25, 0.25],
            angle: 0.0,
            feather: 0.5,
            range: MaskRange::default(),
            adjust: MaskAdjust::default(),
        }
    }
}

impl Mask {
    /// A linear gradient from `full` to `none` (source uv).
    pub fn linear(full: [f32; 2], none: [f32; 2]) -> Self {
        Self { kind: MaskKind::Linear, a: full, b: none, ..Default::default() }
    }

    /// A circle of `radius` source heights around `centre` (source uv).
    pub fn radial(centre: [f32; 2], radius: f32) -> Self {
        Self { kind: MaskKind::Radial, a: centre, b: [radius; 2], ..Default::default() }
    }

    /// The whole frame, to be limited by a range.
    pub fn whole() -> Self {
        Self { kind: MaskKind::Range, ..Default::default() }
    }

    /// What the mask list calls it.
    pub fn label(&self) -> &'static str {
        match (self.kind, self.range.luma_on, self.range.colour_on) {
            (MaskKind::Range, true, false) => "Brightness range",
            (MaskKind::Range, false, true) => "Colour range",
            (kind, ..) => kind.label(),
        }
    }

    /// In use and switched on.
    pub fn is_active(&self) -> bool {
        self.kind != MaskKind::Off && self.enabled
    }

    fn feather(&self) -> f32 {
        self.feather.clamp(MIN_FEATHER, 1.0)
    }

    fn radii(&self) -> [f32; 2] {
        self.b.map(|r| r.abs().max(MIN_RADIUS))
    }
}

/// The shape alone at a source uv, 0..1: no range, not inverted, not scaled
/// by the amount. Mirrors `mask_shape` in the shader.
pub fn shape(m: &Mask, uv: [f32; 2], aspect: f32) -> f32 {
    let p = [uv[0] * aspect, uv[1]];
    match m.kind {
        MaskKind::Off => 0.0,
        MaskKind::Range => 1.0,
        MaskKind::Linear => {
            let a = [m.a[0] * aspect, m.a[1]];
            let ab = [m.b[0] * aspect - a[0], m.b[1] - a[1]];
            let t = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-12);
            1.0 - smoothstep(0.0, 1.0, t)
        }
        MaskKind::Radial => {
            let d = [p[0] - m.a[0] * aspect, p[1] - m.a[1]];
            let (sin, cos) = m.angle.sin_cos();
            let r = m.radii();
            let (x, y) = ((cos * d[0] + sin * d[1]) / r[0], (-sin * d[0] + cos * d[1]) / r[1]);
            1.0 - smoothstep(1.0 - m.feather(), 1.0, (x * x + y * y).sqrt())
        }
    }
}

/// How much of a pixel's brightness / colour the mask's range lets through:
/// 1 with no range switched on. Mirrors `mask_range` in the shader.
pub fn range(r: &MaskRange, px: &RangeRef) -> f32 {
    let mut w = 1.0;
    if r.luma_on {
        let f = r.luma_feather();
        w *= smoothstep(r.luma[0] - f, r.luma[0], px.luma) * (1.0 - smoothstep(r.luma[1], r.luma[1] + f, px.luma));
    }
    if r.colour_on {
        let [hue, width, feather] = r.hue_turns();
        // Distance round the circle, 0..0.5.
        let d = px.hue - hue + 0.5;
        let dist = ((d - d.floor()) - 0.5).abs();
        // Greys have no hue to speak of: they fade out of any colour range.
        w *= (1.0 - smoothstep(width, width + feather, dist)) * (px.sat / HSL_FULL_SAT).clamp(0.0, 1.0);
    }
    w
}

/// What the mask selects at a pixel, 0..1: shape × range, inverted if the
/// mask is — but not scaled by its amount. This is what the overlay shows.
/// Mirrors `mask_select` in the shader.
pub fn selection(m: &Mask, uv: [f32; 2], aspect: f32, px: &RangeRef) -> f32 {
    if m.kind == MaskKind::Off {
        return 0.0;
    }
    let w = shape(m, uv, aspect) * range(&m.range, px);
    if m.invert { 1.0 - w } else { w }
}

/// How much of a mask's offsets a pixel gets: the selection times the
/// amount, or nothing for a slot that is off or disabled.
pub fn weight(m: &Mask, uv: [f32; 2], aspect: f32, px: &RangeRef) -> f32 {
    if m.is_active() { selection(m, uv, aspect, px) * m.amount } else { 0.0 }
}

/// Slots the shader has to look at: one past the last active mask.
pub fn count(masks: &[Mask; MAX_MASKS]) -> usize {
    masks.iter().rposition(Mask::is_active).map_or(0, |i| i + 1)
}

/// True when any mask touches the detail stage, which the CPU path otherwise
/// skips for photos with no global sharpening or noise reduction.
pub fn touch_detail(masks: &[Mask; MAX_MASKS]) -> bool {
    masks.iter().any(|m| m.is_active() && (m.adjust.sharpen != 0.0 || m.adjust.nr_luma != 0.0))
}

/// The slider values in force at one pixel: the global ones plus every
/// mask's offsets at its weight there. `uv` is the pixel's source uv,
/// `aspect` the source's width / height and `rgb` the pixel in working RGB
/// (before the detail stage), which the ranges are judged on. Mirrors
/// `locals` in the shader.
pub fn locals(p: &AdjustParams, uv: [f32; 2], aspect: f32, rgb: [f32; 3]) -> Locals {
    let mut l = Locals::of(p);
    let n = count(&p.masks);
    if n == 0 {
        return l;
    }
    let px = RangeRef::of(p, rgb);
    for m in &p.masks[..n] {
        let w = weight(m, uv, aspect, &px);
        let d = &m.adjust;
        l.exposure_ev += w * d.exposure_ev;
        l.contrast += w * d.contrast;
        l.temp += w * d.temp;
        l.tint += w * d.tint;
        l.saturation += w * d.saturation;
        l.highlights += w * d.highlights;
        l.shadows += w * d.shadows;
        l.whites += w * d.whites;
        l.blacks += w * d.blacks;
        l.sharpen_amount += w * d.sharpen;
        l.nr_luma += w * d.nr_luma;
    }
    // Stacked masks can push a sum past what one slider allows; keep the
    // values where the maths still means something.
    l.contrast = l.contrast.max(-1.0);
    l.saturation = l.saturation.max(-1.0);
    l.sharpen_amount = l.sharpen_amount.max(0.0);
    l.nr_luma = l.nr_luma.clamp(0.0, 1.0);
    l
}

/// The mask overlay: tint a finished pixel by how selected it is. A viewing
/// aid like the clipping overlay — never saved or exported. Mirrors the
/// overlay step in `fs_main`.
pub fn overlay(c: [f32; 3], shape: f32) -> [f32; 3] {
    let k = OVERLAY_OPACITY * shape;
    std::array::from_fn(|i| c[i] + (OVERLAY_COLOUR[i] - c[i]) * k)
}

/// One mask as the shader's `MaskU`: eight vec4s.
///
/// 0: a.xy, b.xy — 1: cos, sin, feather, kind — 2: amount, invert, sharpen,
/// nr_luma — 3: exposure, contrast, temp, tint — 4: saturation, highlights,
/// shadows, whites — 5: blacks — 6: brightness range (from, to, feather, on)
/// — 7: colour range (hue, width, feather as fractions of the circle, on).
pub type MaskUniform = [[f32; 4]; 8];

/// The uniform block's mask part: `[count, overlay slot + 1 or 0, 0, 0]` and
/// the slots. A slot that is off or disabled is sent as kind 0.
pub fn uniforms(masks: &[Mask; MAX_MASKS], overlay: Option<usize>) -> ([f32; 4], [MaskUniform; MAX_MASKS]) {
    let slots = masks.map(|m| {
        if m.kind == MaskKind::Off {
            return [[0.0; 4]; 8];
        }
        let (sin, cos) = m.angle.sin_cos();
        let b = if m.kind == MaskKind::Radial { m.radii() } else { m.b };
        let d = &m.adjust;
        let kind = match m.kind {
            MaskKind::Off => 0.0,
            MaskKind::Linear => 1.0,
            MaskKind::Radial => 2.0,
            MaskKind::Range => 3.0,
        };
        let r = &m.range;
        let [hue, hue_width, hue_feather] = r.hue_turns();
        let on = |b: bool| if b { 1.0 } else { 0.0 };
        [
            [m.a[0], m.a[1], b[0], b[1]],
            [cos, sin, m.feather(), kind],
            // A disabled mask keeps its shape (the overlay may still show
            // it) but has no amount, so it changes nothing.
            [if m.enabled { m.amount } else { 0.0 }, if m.invert { 1.0 } else { 0.0 }, d.sharpen, d.nr_luma],
            [d.exposure_ev, d.contrast, d.temp, d.tint],
            [d.saturation, d.highlights, d.shadows, d.whites],
            [d.blacks, 0.0, 0.0, 0.0],
            [r.luma[0], r.luma[1], r.luma_feather(), on(r.luma_on)],
            [hue, hue_width, hue_feather, on(r.colour_on)],
        ]
    });
    let overlay = overlay.filter(|&i| i < MAX_MASKS && masks[i].kind != MaskKind::Off);
    ([count(masks) as f32, overlay.map_or(0.0, |i| (i + 1) as f32), 0.0, 0.0], slots)
}

// --- The mask list -----------------------------------------------------------

/// Masks in use. They are kept packed at the front of the array.
pub fn used(masks: &[Mask; MAX_MASKS]) -> usize {
    masks.iter().filter(|m| m.kind != MaskKind::Off).count()
}

/// Add a mask in the first free slot; `None` when all are taken.
pub fn add(masks: &mut [Mask; MAX_MASKS], mask: Mask) -> Option<usize> {
    let i = masks.iter().position(|m| m.kind == MaskKind::Off)?;
    masks[i] = mask;
    Some(i)
}

/// Remove a mask, closing the gap so the rest stay packed and in order.
pub fn remove(masks: &mut [Mask; MAX_MASKS], i: usize) {
    if i < MAX_MASKS {
        masks.copy_within(i + 1.., i);
        masks[MAX_MASKS - 1] = Mask::default();
    }
}

/// Sidecar form of the slots: just the masks in use, as a list.
pub mod slots {
    use super::*;

    pub fn serialize<S: Serializer>(masks: &[Mask; MAX_MASKS], s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(masks.iter().filter(|m| m.kind != MaskKind::Off))
    }

    /// Unused entries and anything past the last slot are dropped rather
    /// than refused: a sidecar from a build with more slots still opens.
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[Mask; MAX_MASKS], D::Error> {
        let list = Vec::<Mask>::deserialize(d)?;
        let mut masks = [Mask::default(); MAX_MASKS];
        for (slot, m) in masks.iter_mut().zip(list.into_iter().filter(|m| m.kind != MaskKind::Off)) {
            *slot = m;
        }
        Ok(masks)
    }
}

// --- Handles on the image ----------------------------------------------------

/// What a pointer has hold of on a mask's on-image controls.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MaskGrab {
    /// The pin: moves the whole mask.
    Move,
    /// Linear: the full-effect end.
    Start,
    /// Linear: the no-effect end.
    End,
    /// Radial: an end of the first axis (`true` = the positive end).
    AxisX(bool),
    /// Radial: an end of the second axis.
    AxisY(bool),
    /// Radial: the knob that turns the ellipse.
    Rotate,
}

/// How far past the first axis's end the rotate knob sits, as a multiple of
/// that semi-axis.
const ROTATE_REACH: f32 = 1.3;

fn radial_axes(m: &Mask) -> ([f32; 2], [f32; 2]) {
    let (sin, cos) = m.angle.sin_cos();
    let r = m.radii();
    // Columns of the inverse of the rotation used in `shape`, scaled.
    ([cos * r[0], sin * r[0]], [-sin * r[1], cos * r[1]])
}

/// The mask's handles and where they are, in source uv.
pub fn handles(m: &Mask, aspect: f32) -> Vec<(MaskGrab, [f32; 2])> {
    match m.kind {
        MaskKind::Off | MaskKind::Range => Vec::new(),
        MaskKind::Linear => vec![
            (MaskGrab::Move, [(m.a[0] + m.b[0]) / 2.0, (m.a[1] + m.b[1]) / 2.0]),
            (MaskGrab::Start, m.a),
            (MaskGrab::End, m.b),
        ],
        MaskKind::Radial => {
            let (ax, ay) = radial_axes(m);
            let at = |v: [f32; 2], k: f32| [m.a[0] + v[0] * k / aspect, m.a[1] + v[1] * k];
            vec![
                (MaskGrab::Move, m.a),
                (MaskGrab::AxisX(true), at(ax, 1.0)),
                (MaskGrab::AxisX(false), at(ax, -1.0)),
                (MaskGrab::AxisY(true), at(ay, 1.0)),
                (MaskGrab::AxisY(false), at(ay, -1.0)),
                (MaskGrab::Rotate, at(ax, ROTATE_REACH)),
            ]
        }
    }
}

/// Points along the mask's outline in source uv, for drawing: for a linear
/// mask the three lines (full, middle, none) as pairs of end points, for a
/// radial one the ellipse as a closed polyline.
pub fn outline(m: &Mask, aspect: f32) -> Vec<Vec<[f32; 2]>> {
    match m.kind {
        MaskKind::Off | MaskKind::Range => Vec::new(),
        MaskKind::Linear => {
            // Perpendicular to a→b in aspect space, long enough to cross any frame.
            let ab = [(m.b[0] - m.a[0]) * aspect, m.b[1] - m.a[1]];
            let len = (ab[0] * ab[0] + ab[1] * ab[1]).sqrt().max(1e-6);
            let n = [-ab[1] / len * 4.0 / aspect, ab[0] / len * 4.0];
            [0.0, 0.5, 1.0]
                .into_iter()
                .map(|t| {
                    let c = [m.a[0] + (m.b[0] - m.a[0]) * t, m.a[1] + (m.b[1] - m.a[1]) * t];
                    vec![[c[0] - n[0], c[1] - n[1]], [c[0] + n[0], c[1] + n[1]]]
                })
                .collect()
        }
        MaskKind::Radial => {
            let (ax, ay) = radial_axes(m);
            let ring = |k: f32| -> Vec<[f32; 2]> {
                (0..=96)
                    .map(|i| {
                        let (s, c) = (i as f32 / 96.0 * std::f32::consts::TAU).sin_cos();
                        [m.a[0] + (ax[0] * c + ay[0] * s) * k / aspect, m.a[1] + (ax[1] * c + ay[1] * s) * k]
                    })
                    .collect()
            };
            // The edge, and inside it where the fade begins.
            vec![ring(1.0), ring(1.0 - m.feather())]
        }
    }
}

/// The mask after dragging `grab` to `to` (source uv). `start` is the mask
/// when the drag began and `from` where the pointer was then, so a drag is a
/// pure function of its two ends and never accumulates error.
pub fn drag(start: Mask, grab: MaskGrab, from: [f32; 2], to: [f32; 2], aspect: f32) -> Mask {
    let mut m = start;
    let d = [to[0] - from[0], to[1] - from[1]];
    match (start.kind, grab) {
        (MaskKind::Off | MaskKind::Range, _) => {}
        (MaskKind::Linear, MaskGrab::Move) => {
            m.a = [start.a[0] + d[0], start.a[1] + d[1]];
            m.b = [start.b[0] + d[0], start.b[1] + d[1]];
        }
        (MaskKind::Linear, MaskGrab::Start) => m.a = [start.a[0] + d[0], start.a[1] + d[1]],
        (MaskKind::Linear, MaskGrab::End) => m.b = [start.b[0] + d[0], start.b[1] + d[1]],
        (MaskKind::Radial, MaskGrab::Move) => m.a = [start.a[0] + d[0], start.a[1] + d[1]],
        (MaskKind::Radial, MaskGrab::AxisX(_) | MaskGrab::AxisY(_)) => {
            // The handle follows the pointer along its own axis: the new
            // semi-axis is the pointer's distance from the centre along it.
            let v = [(to[0] - start.a[0]) * aspect, to[1] - start.a[1]];
            let (sin, cos) = start.angle.sin_cos();
            if matches!(grab, MaskGrab::AxisX(_)) {
                m.b[0] = (cos * v[0] + sin * v[1]).abs().max(MIN_RADIUS);
            } else {
                m.b[1] = (-sin * v[0] + cos * v[1]).abs().max(MIN_RADIUS);
            }
        }
        (MaskKind::Radial, MaskGrab::Rotate) => {
            let v = [(to[0] - start.a[0]) * aspect, to[1] - start.a[1]];
            if v[0].abs() + v[1].abs() > 1e-6 {
                m.angle = v[1].atan2(v[0]);
            }
        }
        _ => {}
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASPECT: f32 = 1.5;

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    /// A mid-grey pixel: inside every brightness range that is off.
    const GREY: RangeRef = RangeRef { luma: 0.5, hue: 0.0, sat: 0.0 };

    fn selected(m: &Mask, uv: [f32; 2]) -> f32 {
        selection(m, uv, ASPECT, &GREY)
    }

    #[test]
    fn linear_fades_from_its_start_to_its_end() {
        let m = Mask::linear([0.5, 0.2], [0.5, 0.6]);
        assert_eq!(shape(&m, [0.1, 0.0], ASPECT), 1.0, "before the start: full");
        assert_eq!(shape(&m, [0.9, 0.2], ASPECT), 1.0, "on the start line, anywhere along it");
        assert!(near(shape(&m, [0.3, 0.4], ASPECT), 0.5), "half way");
        assert_eq!(shape(&m, [0.5, 0.6], ASPECT), 0.0);
        assert_eq!(shape(&m, [0.5, 0.95], ASPECT), 0.0, "past the end: nothing");
        let (a, b) = (shape(&m, [0.5, 0.3], ASPECT), shape(&m, [0.5, 0.5], ASPECT));
        assert!(a > 0.5 && b < 0.5 && near(a + b, 1.0), "smooth and symmetric: {a} {b}");
    }

    #[test]
    fn a_diagonal_linear_mask_is_constant_across_the_gradient_in_pixels() {
        // a→b runs at 45° on screen, so the lines of equal weight do too:
        // moving one step right and one step up (in pixels) changes nothing.
        // In uv that is +dx / aspect, −dx.
        let m = Mask::linear([0.2, 0.2], [0.2 + 0.4 / ASPECT, 0.6]);
        let p = [0.4, 0.4];
        let q = [p[0] + 0.1 / ASPECT, p[1] - 0.1];
        assert!(near(shape(&m, p, ASPECT), shape(&m, q, ASPECT)));
        assert!(shape(&m, p, ASPECT) > 0.0 && shape(&m, p, ASPECT) < 1.0);
    }

    #[test]
    fn radial_is_round_on_a_non_square_frame() {
        let m = Mask { feather: 0.5, ..Mask::radial([0.5, 0.5], 0.2) };
        assert_eq!(shape(&m, [0.5, 0.5], ASPECT), 1.0);
        // The edge is 0.2 source heights away in every direction — which is
        // 0.2 / aspect in u.
        for p in [[0.5 + 0.2 / ASPECT, 0.5], [0.5, 0.7], [0.5 - 0.2 / ASPECT, 0.5]] {
            assert!(shape(&m, p, ASPECT) < 1e-5, "{p:?}");
        }
        // Inside the unfeathered half: still full. Half way through the fade: half.
        assert_eq!(shape(&m, [0.5, 0.59], ASPECT), 1.0);
        assert!(near(shape(&m, [0.5, 0.65], ASPECT), 0.5));
        assert_eq!(shape(&m, [0.9, 0.9], ASPECT), 0.0);
    }

    #[test]
    fn radial_axes_turn_with_the_angle() {
        let m = Mask { b: [0.3, 0.1], angle: std::f32::consts::FRAC_PI_2, feather: 0.2, ..Mask::radial([0.5, 0.5], 0.1) };
        // The long axis now points down the frame.
        assert_eq!(shape(&m, [0.5, 0.7], ASPECT), 1.0);
        assert_eq!(shape(&m, [0.5 + 0.2 / ASPECT, 0.5], ASPECT), 0.0);
    }

    #[test]
    fn invert_amount_and_the_eye() {
        let mut m = Mask::radial([0.5, 0.5], 0.2);
        let inside = [0.5, 0.5];
        m.invert = true;
        assert_eq!(selected(&m, inside), 0.0);
        assert_eq!(selected(&m, [0.05, 0.05]), 1.0);
        m.invert = false;
        m.amount = 0.25;
        assert_eq!(weight(&m, inside, ASPECT, &GREY), 0.25);
        assert_eq!(selected(&m, inside), 1.0, "the overlay ignores the amount");
        m.enabled = false;
        assert_eq!(weight(&m, inside, ASPECT, &GREY), 0.0);
        assert_eq!(weight(&Mask::default(), inside, ASPECT, &GREY), 0.0, "an unused slot");
        assert_eq!(selected(&Mask { invert: true, ..Mask::default() }, inside), 0.0, "inverting nothing is still nothing");
    }

    #[test]
    fn a_zero_feather_or_radius_does_not_divide_by_zero() {
        let m = Mask { feather: 0.0, b: [0.0, 0.0], ..Mask::radial([0.5, 0.5], 0.0) };
        for p in [[0.5, 0.5], [0.5, 0.51], [0.0, 0.0]] {
            assert!(shape(&m, p, ASPECT).is_finite());
        }
        let flat = Mask::linear([0.3, 0.3], [0.3, 0.3]);
        assert!(shape(&flat, [0.3, 0.3], ASPECT).is_finite());
    }

    #[test]
    fn locals_add_each_masks_offsets_at_its_weight() {
        let mut p = AdjustParams { exposure_ev: 0.5, saturation: 0.1, ..Default::default() };
        assert_eq!(locals(&p, [0.5, 0.5], ASPECT, [0.18; 3]), Locals::of(&p), "no masks: the global values");
        p.masks[0] = Mask { adjust: MaskAdjust { exposure_ev: 1.0, ..Default::default() }, ..Mask::radial([0.5, 0.5], 0.2) };
        p.masks[1] = Mask {
            amount: 0.5,
            adjust: MaskAdjust { exposure_ev: -0.4, saturation: -3.0, ..Default::default() },
            ..Mask::linear([0.5, 0.0], [0.5, 1.0])
        };
        let l = locals(&p, [0.5, 0.5], ASPECT, [0.18; 3]);
        // Radial at full weight (+1), linear at half way × amount 0.5 (−0.1).
        assert!(near(l.exposure_ev, 0.5 + 1.0 - 0.1), "{}", l.exposure_ev);
        assert!(near(l.saturation, 0.1 - 0.75));
        let far = locals(&p, [0.02, 1.0], ASPECT, [0.18; 3]);
        assert!(near(far.exposure_ev, 0.5), "outside both: {}", far.exposure_ev);
        // Sums are kept where the maths still means something.
        let top = locals(&p, [0.02, 0.0], ASPECT, [0.18; 3]);
        assert_eq!(top.saturation, -1.0);
    }

    fn px(luma: f32, hue_deg: f32, sat: f32) -> RangeRef {
        RangeRef { luma, hue: hue_deg / 360.0, sat }
    }

    #[test]
    fn a_brightness_range_selects_between_its_ends() {
        let r = MaskRange { luma_on: true, luma: [0.4, 0.7], luma_feather: 0.1, ..Default::default() };
        for (luma, want) in [(0.2, 0.0), (0.3, 0.0), (0.35, 0.5), (0.4, 1.0), (0.55, 1.0), (0.7, 1.0), (0.75, 0.5), (0.8, 0.0), (1.0, 0.0)] {
            assert!(near(range(&r, &px(luma, 0.0, 0.0)), want), "luma {luma}");
        }
        // The full range lets everything through, black and white included.
        let all = MaskRange { luma_on: true, ..Default::default() };
        assert_eq!(range(&all, &px(0.0, 0.0, 0.0)), 1.0);
        assert_eq!(range(&all, &px(1.0, 0.0, 0.0)), 1.0);
        // Off: no limit at all.
        assert_eq!(range(&MaskRange::default(), &px(0.01, 0.0, 0.0)), 1.0);
        // A zero feather is a hard edge, not a division by zero.
        let hard = MaskRange { luma_feather: 0.0, ..r };
        assert_eq!(range(&hard, &px(0.39, 0.0, 0.0)), 0.0);
        assert_eq!(range(&hard, &px(0.5, 0.0, 0.0)), 1.0);
    }

    #[test]
    fn a_colour_range_wraps_round_red_and_leaves_greys_out() {
        let red = MaskRange { colour_on: true, hue: 350.0, hue_width: 20.0, hue_feather: 20.0, ..Default::default() };
        // 350° ± 20° fully in — across the 0° seam — fading out to ± 40°.
        for (hue, want) in [(350.0, 1.0), (335.0, 1.0), (5.0, 1.0), (20.0, 0.5), (320.0, 0.5), (35.0, 0.0), (180.0, 0.0)] {
            assert!(near(range(&red, &px(0.5, hue, 1.0)), want), "hue {hue}");
        }
        // Saturation: a grey is out, a muted colour partly in.
        assert_eq!(range(&red, &px(0.5, 350.0, 0.0)), 0.0);
        assert!(near(range(&red, &px(0.5, 350.0, HSL_FULL_SAT / 2.0)), 0.5));
        // A hue given outside 0..360 is the same colour.
        let same = MaskRange { hue: -10.0, ..red };
        assert!(near(range(&same, &px(0.5, 5.0, 1.0)), 1.0));
    }

    #[test]
    fn range_and_shape_multiply_and_invert_flips_the_product() {
        let mut m = Mask::radial([0.5, 0.5], 0.2);
        m.range = MaskRange { colour_on: true, hue: 240.0, ..Default::default() };
        let (blue, green) = (px(0.5, 240.0, 1.0), px(0.5, 120.0, 1.0));
        let (inside, outside) = ([0.5, 0.5], [0.05, 0.05]);
        assert_eq!(selection(&m, inside, ASPECT, &blue), 1.0);
        assert_eq!(selection(&m, inside, ASPECT, &green), 0.0, "wrong colour");
        assert_eq!(selection(&m, outside, ASPECT, &blue), 0.0, "outside the shape");
        m.invert = true;
        assert_eq!(selection(&m, inside, ASPECT, &blue), 0.0);
        assert_eq!(selection(&m, inside, ASPECT, &green), 1.0, "everything the mask left out");
        assert_eq!(selection(&m, outside, ASPECT, &blue), 1.0);
        // A range mask has no shape: the range alone, anywhere in the frame.
        let whole = Mask { range: m.range, ..Mask::whole() };
        assert_eq!(selection(&whole, outside, ASPECT, &blue), 1.0);
        assert_eq!(selection(&whole, outside, ASPECT, &green), 0.0);
        assert!(handles(&whole, ASPECT).is_empty() && outline(&whole, ASPECT).is_empty());
    }

    #[test]
    fn ranges_are_judged_after_the_global_exposure_white_balance_and_curve() {
        let grey = [0.18f32; 3];
        let base = RangeRef::of(&AdjustParams::default(), grey);
        assert!(near(base.luma, 0.18f32.powf(1.0 / MASK_GAMMA)) && base.sat == 0.0);
        let brighter = RangeRef::of(&AdjustParams { exposure_ev: 1.0, ..Default::default() }, grey);
        assert!(near(brighter.luma, 0.36f32.powf(1.0 / MASK_GAMMA)));
        let curved = RangeRef::of(&AdjustParams { tone_curve: 1.0, ..Default::default() }, grey);
        assert!(curved.luma > base.luma + 0.1, "the curve lifts mid-tones: {} vs {}", curved.luma, base.luma);
        // Warming a grey gives it an orange hue; cooling, a blue one.
        let warm = RangeRef::of(&AdjustParams { temp: 0.5, ..Default::default() }, grey);
        let cool = RangeRef::of(&AdjustParams { temp: -0.5, ..Default::default() }, grey);
        assert!(warm.sat > 0.0 && (warm.hue * 360.0 - 30.0).abs() < 31.0, "{warm:?}");
        assert!((cool.hue * 360.0 - 240.0).abs() < 31.0, "{cool:?}");
        // A mask's own offsets are not part of it.
        let mut p = AdjustParams::default();
        p.masks[0] = Mask { adjust: MaskAdjust { exposure_ev: 3.0, ..Default::default() }, ..Mask::whole() };
        assert_eq!(RangeRef::of(&p, grey), base);
        // Brighter than white is white; black has no hue.
        assert_eq!(RangeRef::of(&AdjustParams::default(), [4.0; 3]).luma, 1.0);
        assert_eq!(hue_sat([0.0; 3]), (0.0, 0.0));
    }

    #[test]
    fn hues_of_the_primaries() {
        for (c, deg) in [([1.0, 0.0, 0.0], 0.0), ([1.0, 1.0, 0.0], 60.0), ([0.0, 1.0, 0.0], 120.0), ([0.0, 0.0, 1.0], 240.0), ([1.0, 0.0, 1.0], 300.0)] {
            let (h, s) = hue_sat(c);
            assert!(near(h * 360.0, deg) && s == 1.0, "{c:?}: {h}");
        }
        // Just below red wraps to the top of the circle, not below zero.
        let (h, _) = hue_sat([1.0, 0.0, 0.1]);
        assert!(h > 0.9 && h < 1.0);
    }

    #[test]
    fn range_masks_are_named_for_what_they_limit() {
        let mut m = Mask::whole();
        assert_eq!(m.label(), "Range");
        m.range.luma_on = true;
        assert_eq!(m.label(), "Brightness range");
        m.range.colour_on = true;
        assert_eq!(m.label(), "Range");
        m.range.luma_on = false;
        assert_eq!(m.label(), "Colour range");
        assert_eq!(Mask { range: m.range, ..Mask::radial([0.5; 2], 0.1) }.label(), "Radial gradient");
    }

    #[test]
    fn picking_centres_the_range_on_the_pixel() {
        let mut r = MaskRange::default();
        r.pick_luma(0.1);
        assert!(r.luma_on && r.luma[0] == 0.0 && near(r.luma[1], 0.25));
        r.pick_hue(0.75);
        assert!(r.colour_on && near(r.hue, 270.0));
        assert_eq!(range(&r, &px(0.1, 270.0, 1.0)), 1.0);
    }

    #[test]
    fn locals_use_the_range_of_each_pixel() {
        let mut p = AdjustParams::default();
        p.masks[0] = Mask {
            range: MaskRange { luma_on: true, luma: [0.0, 0.3], luma_feather: 0.05, ..Default::default() },
            adjust: MaskAdjust { exposure_ev: 1.0, ..Default::default() },
            ..Mask::whole()
        };
        assert_eq!(locals(&p, [0.5, 0.5], ASPECT, [0.01; 3]).exposure_ev, 1.0, "a shadow gets it");
        assert_eq!(locals(&p, [0.5, 0.5], ASPECT, [0.5; 3]).exposure_ev, 0.0, "a mid-tone does not");
    }

    #[test]
    fn count_stops_after_the_last_active_mask() {
        let mut masks = [Mask::default(); MAX_MASKS];
        assert_eq!(count(&masks), 0);
        masks[0] = Mask::radial([0.5, 0.5], 0.2);
        masks[1] = Mask { enabled: false, ..Mask::radial([0.5, 0.5], 0.2) };
        assert_eq!(count(&masks), 1);
        assert_eq!(used(&masks), 2);
        masks[1].enabled = true;
        assert_eq!(count(&masks), 2);
    }

    #[test]
    fn the_list_stays_packed() {
        let mut masks = [Mask::default(); MAX_MASKS];
        for i in 0..MAX_MASKS {
            assert_eq!(add(&mut masks, Mask::radial([i as f32 * 0.1, 0.5], 0.2)), Some(i));
        }
        assert_eq!(add(&mut masks, Mask::linear([0.0; 2], [1.0; 2])), None, "full");
        remove(&mut masks, 2);
        assert_eq!(used(&masks), MAX_MASKS - 1);
        assert_eq!(masks[2].a[0], 0.3, "the next one moved up");
        assert_eq!(masks[MAX_MASKS - 1].kind, MaskKind::Off);
        assert_eq!(add(&mut masks, Mask::linear([0.0; 2], [1.0; 2])), Some(MAX_MASKS - 1));
    }

    #[test]
    fn sidecars_hold_only_the_masks_in_use() {
        let mut p = AdjustParams::default();
        let json = serde_json::to_value(p).unwrap();
        assert_eq!(json["masks"], serde_json::json!([]));
        p.masks[0] = Mask { invert: true, adjust: MaskAdjust { shadows: 0.4, ..Default::default() }, ..Mask::radial([0.3, 0.6], 0.2) };
        let json = serde_json::to_value(p).unwrap();
        assert_eq!(json["masks"].as_array().unwrap().len(), 1);
        let back: AdjustParams = serde_json::from_value(json).unwrap();
        assert_eq!(back, p);
        // A sidecar written before masks existed has no such field.
        let old: AdjustParams = serde_json::from_str(r#"{"exposure_ev": 0.5}"#).unwrap();
        assert_eq!(old.masks, [Mask::default(); MAX_MASKS]);
        // More masks than there are slots: the extra ones are dropped.
        let many = serde_json::json!({ "masks": vec![serde_json::to_value(p.masks[0]).unwrap(); MAX_MASKS + 3] });
        let back: AdjustParams = serde_json::from_value(many).unwrap();
        assert_eq!(used(&back.masks), MAX_MASKS);
    }

    #[test]
    fn handles_sit_on_the_shape() {
        let m = Mask { b: [0.3, 0.1], angle: 0.6, ..Mask::radial([0.5, 0.5], 0.1) };
        for (grab, at) in handles(&m, ASPECT) {
            let hard = Mask { feather: MIN_FEATHER, ..m };
            match grab {
                MaskGrab::Move => assert_eq!(at, m.a),
                // On the edge: just inside is selected, just outside is not.
                MaskGrab::AxisX(_) | MaskGrab::AxisY(_) => {
                    let toward = |k: f32| [m.a[0] + (at[0] - m.a[0]) * k, m.a[1] + (at[1] - m.a[1]) * k];
                    assert_eq!(shape(&hard, toward(0.97), ASPECT), 1.0, "{grab:?}");
                    assert_eq!(shape(&hard, toward(1.03), ASPECT), 0.0, "{grab:?}");
                }
                MaskGrab::Rotate => assert_eq!(shape(&hard, at, ASPECT), 0.0),
                _ => panic!("{grab:?} on a radial mask"),
            }
        }
        let l = Mask::linear([0.2, 0.2], [0.6, 0.8]);
        let h = handles(&l, ASPECT);
        assert!(h[0].0 == MaskGrab::Move && near(h[0].1[0], 0.4) && near(h[0].1[1], 0.5));
        assert_eq!((h[1].1, h[2].1), (l.a, l.b));
    }

    #[test]
    fn dragging_moves_resizes_and_turns() {
        let l = Mask::linear([0.2, 0.2], [0.6, 0.8]);
        let moved = drag(l, MaskGrab::Move, [0.4, 0.5], [0.5, 0.4], ASPECT);
        assert!(near(moved.a[0], 0.3) && near(moved.a[1], 0.1) && near(moved.b[0], 0.7) && near(moved.b[1], 0.7));
        let ended = drag(l, MaskGrab::End, l.b, [0.9, 0.9], ASPECT);
        assert_eq!((ended.a, ended.b), (l.a, [0.9, 0.9]));

        let r = Mask { b: [0.3, 0.1], ..Mask::radial([0.5, 0.5], 0.1) };
        // Pull the +x handle out to 0.4 source heights: 0.4 / aspect in u.
        let wide = drag(r, MaskGrab::AxisX(true), [0.7, 0.5], [0.5 + 0.4 / ASPECT, 0.5], ASPECT);
        assert!(near(wide.b[0], 0.4) && wide.b[1] == 0.1, "{:?}", wide.b);
        // The handle that was dragged is where the pointer is.
        let (_, at) = handles(&wide, ASPECT)[1];
        assert!(near(at[0], 0.5 + 0.4 / ASPECT) && near(at[1], 0.5));
        // Dragging through the centre cannot collapse or flip the shape.
        let tiny = drag(r, MaskGrab::AxisY(true), [0.5, 0.6], [0.5, 0.5], ASPECT);
        assert_eq!(tiny.b[1], MIN_RADIUS);
        // The rotate knob points the first axis at the pointer.
        let turned = drag(r, MaskGrab::Rotate, [0.8, 0.5], [0.5, 0.9], ASPECT);
        assert!(near(turned.angle, std::f32::consts::FRAC_PI_2));
        let (_, knob) = handles(&turned, ASPECT)[5];
        assert!(near(knob[0], 0.5) && knob[1] > 0.8);
    }

    #[test]
    fn outlines_follow_the_shape() {
        let r = Mask { feather: 0.4, ..Mask::radial([0.5, 0.5], 0.2) };
        let rings = outline(&r, ASPECT);
        assert_eq!(rings.len(), 2);
        let hard = Mask { feather: MIN_FEATHER, ..r };
        for p in &rings[1] {
            assert_eq!(shape(&r, *p, ASPECT), 1.0, "the inner ring is where the fade starts");
            assert_eq!(shape(&hard, *p, ASPECT), 1.0);
        }
        let l = Mask::linear([0.5, 0.2], [0.5, 0.6]);
        let lines = outline(&l, ASPECT);
        assert_eq!(lines.len(), 3);
        for (line, want) in lines.iter().zip([1.0, 0.5, 0.0]) {
            for p in line {
                assert!(near(shape(&l, *p, ASPECT), want), "{p:?}");
            }
        }
    }

    #[test]
    fn the_overlay_tints_by_the_shape() {
        let c = [0.2, 0.4, 0.6];
        assert_eq!(overlay(c, 0.0), c);
        let full = overlay(c, 1.0);
        assert!(full[0] > c[0] && full[1] < c[1] && full[2] < c[2]);
    }
}
