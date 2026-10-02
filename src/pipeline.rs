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

/// Smallest and largest white-balance gain the controls and the eyedropper
/// allow. Far wider than any visible-light camera needs: an infrared
/// conversion under warm light wants red well below 1 and blue several times
/// green.
pub const WB_GAIN_MIN: f32 = 0.1;
pub const WB_GAIN_MAX: f32 = 10.0;

/// How camera RGB becomes working RGB after white balance.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColourMode {
    /// The file's own camera→sRGB matrix — right for visible light.
    #[default]
    Camera,
    /// No matrix: white-balanced sensor RGB straight through. For infrared,
    /// where the camera matrix (calibrated for visible light) scrambles hues.
    SensorRgb,
}

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
    /// Custom white-balance gains `[r, g, b]`, green = 1. `None` = as shot.
    pub wb_gains: Option<[f32; 3]>,
    pub colour_mode: ColourMode,
    /// Channel mixer, row-major: output channel `i` is `mixer[i] · rgb`.
    /// Applied after the colour matrix (on white-balanced sensor RGB in
    /// Sensor RGB mode), so it is where infrared false-colour looks live.
    /// Rows that sum to 1 keep neutrals neutral. Identity = off.
    pub mixer: [[f32; 3]; 3],
    /// How much of the mixer applies: 0 = none (identity), 1 = all of it.
    /// This is the "look strength" control; values past 1 exaggerate.
    pub mixer_strength: f32,
    /// Per-colour adjustments, one `[hue, saturation, luminance]` per band in
    /// [`HSL_BANDS`] order, each -1..1, 0 = off. See [`hsl_adjust`].
    pub hsl: [[f32; 3]; 8],
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

    /// Radial hot-spot correction (IR-4): how much brighter the centre of the
    /// spot is than its surroundings, 0 = off, 1 = twice as bright.
    pub hotspot_amount: f32,
    /// Gaussian radius of the spot, as a fraction of the half-diagonal.
    pub hotspot_radius: f32,
    /// Spot centre in image coordinates (0..1, y down).
    pub hotspot_centre: [f32; 2],
    /// Per-channel weight of `hotspot_amount` — IR hot spots are usually
    /// coloured. Neutral = 1.
    pub hotspot_balance: [f32; 3],
    /// Apply the flat field saved for this photo's lens, if there is one.
    pub flat_field: bool,

    /// Extra clockwise quarter turns on top of the camera's orientation.
    pub rotation: u8,
    /// Fine rotation in degrees, clockwise, about the frame centre.
    pub straighten: f32,
    /// Crop `[x0, y0, x1, y1]`, normalised to the rotated (and straightened)
    /// frame. `[0, 0, 1, 1]` = no crop.
    pub crop: [f32; 4],
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
            wb_gains: None,
            colour_mode: ColourMode::Camera,
            mixer: IDENTITY3,
            mixer_strength: 1.0,
            hsl: [[0.0; 3]; 8],
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
            hotspot_amount: 0.0,
            hotspot_radius: 0.3,
            hotspot_centre: [0.5, 0.5],
            hotspot_balance: [1.0; 3],
            flat_field: false,
            rotation: 0,
            straighten: 0.0,
            crop: [0.0, 0.0, 1.0, 1.0],
        }
    }
}

impl AdjustParams {
    /// True when every control sits at its neutral value.
    pub fn is_neutral(&self) -> bool {
        *self == Self::default()
    }

    /// Put one panel section's controls back to neutral, leaving the rest.
    pub fn reset_section(&mut self, section: Section) {
        let d = Self::default();
        match section {
            Section::Colour => {
                self.wb_gains = d.wb_gains;
                self.colour_mode = d.colour_mode;
                self.mixer = d.mixer;
                self.mixer_strength = d.mixer_strength;
            }
            Section::Hsl => self.hsl = d.hsl,
            Section::Basic => {
                self.exposure_ev = d.exposure_ev;
                self.contrast = d.contrast;
                self.temp = d.temp;
                self.tint = d.tint;
                self.saturation = d.saturation;
            }
            Section::Tonal => {
                self.highlights = d.highlights;
                self.shadows = d.shadows;
                self.whites = d.whites;
                self.blacks = d.blacks;
            }
            Section::Wheels => {
                self.lift = d.lift;
                self.gamma = d.gamma;
                self.gain = d.gain;
            }
            Section::Detail => {
                self.nr_luma = d.nr_luma;
                self.nr_chroma = d.nr_chroma;
                self.sharpen_amount = d.sharpen_amount;
                self.sharpen_radius = d.sharpen_radius;
                self.sharpen_threshold = d.sharpen_threshold;
            }
            Section::Optics => {
                self.hotspot_amount = d.hotspot_amount;
                self.hotspot_radius = d.hotspot_radius;
                self.hotspot_centre = d.hotspot_centre;
                self.hotspot_balance = d.hotspot_balance;
                self.flat_field = d.flat_field;
            }
            Section::Geometry => {
                self.rotation = d.rotation;
                self.straighten = d.straighten;
                self.crop = d.crop;
            }
        }
    }

    /// True when every control in `section` sits at its neutral value.
    pub fn section_is_neutral(&self, section: Section) -> bool {
        let mut reset = *self;
        reset.reset_section(section);
        reset == *self
    }
}

/// The groups of controls in the side panel, each with its own Reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Colour,
    Basic,
    Hsl,
    Tonal,
    Wheels,
    Detail,
    Optics,
    Geometry,
}

impl Section {
    pub const ALL: [Section; 8] = [
        Section::Colour,
        Section::Basic,
        Section::Hsl,
        Section::Tonal,
        Section::Wheels,
        Section::Detail,
        Section::Optics,
        Section::Geometry,
    ];
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
    /// Clockwise quarter turns that display the frame upright, from the
    /// camera's orientation tag. Folded into the geometry so portrait shots
    /// open upright and exports carry upright pixels (Orientation = 1).
    pub turns: u8,
}

pub const IDENTITY3: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl ImageColour {
    /// For data that is already linear sRGB: unit gains, identity matrix.
    pub const LINEAR_SRGB: Self = Self { as_shot_wb: [1.0; 3], cam_to_srgb: IDENTITY3, turns: 0 };
}

/// Where each output pixel comes from: quarter turns (camera orientation plus
/// the user's), a fine straighten angle about the centre, then a crop. All of
/// it composes to one affine map from output uv to source uv, which the
/// shader applies before anything else — so lens corrections and the detail
/// stage still work on source pixels, where they belong.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    /// Output size in pixels.
    pub width: usize,
    pub height: usize,
    /// `src = [a[0][0]·u + a[0][1]·v + a[0][2], a[1][0]·u + a[1][1]·v + a[1][2]]`.
    pub a: [[f32; 3]; 2],
}

impl Geometry {
    /// Resolve for a `w`x`h` source. The crop is snapped to whole pixels of
    /// the rotated frame, so with no straightening every output pixel centre
    /// lands exactly on a source pixel centre and nothing is resampled.
    pub fn resolve(p: &AdjustParams, img: &ImageColour, w: usize, h: usize) -> Self {
        let turns = (img.turns + p.rotation) % 4;
        let (cw, ch) = if turns % 2 == 1 { (h as f64, w as f64) } else { (w as f64, h as f64) };
        let [x0, y0, x1, y1] = p.crop.map(|v| v.clamp(0.0, 1.0) as f64);
        let px0 = (x0 * cw).round().min(cw - 1.0);
        let py0 = (y0 * ch).round().min(ch - 1.0);
        let pw = ((x1 - x0) * cw).round().clamp(1.0, cw - px0);
        let ph = ((y1 - y0) * ch).round().clamp(1.0, ch - py0);
        let theta = (p.straighten as f64).to_radians();
        let (sin, cos) = theta.sin_cos();
        let map = |u: f64, v: f64| -> [f64; 2] {
            // Output uv → rotated-frame pixels → undo the straighten about the
            // centre → undo the quarter turns.
            let (x, y) = (px0 + u * pw - cw / 2.0, py0 + v * ph - ch / 2.0);
            let (x, y) = (cos * x + sin * y + cw / 2.0, -sin * x + cos * y + ch / 2.0);
            let (qx, qy) = (x / cw, y / ch);
            match turns {
                0 => [qx, qy],
                1 => [qy, 1.0 - qx],
                2 => [1.0 - qx, 1.0 - qy],
                _ => [1.0 - qy, qx],
            }
        };
        let o = map(0.0, 0.0);
        let du = map(1.0, 0.0);
        let dv = map(0.0, 1.0);
        let a = [
            [(du[0] - o[0]) as f32, (dv[0] - o[0]) as f32, o[0] as f32],
            [(du[1] - o[1]) as f32, (dv[1] - o[1]) as f32, o[1] as f32],
        ];
        Self { width: pw as usize, height: ph as usize, a }
    }

    pub fn source_uv(&self, uv: [f32; 2]) -> [f32; 2] {
        let a = &self.a;
        [a[0][0] * uv[0] + a[0][1] * uv[1] + a[0][2], a[1][0] * uv[0] + a[1][1] * uv[1] + a[1][2]]
    }

    /// The source pixel behind an output uv, or `None` where straightening
    /// left no source (the black corners).
    pub fn source_pixel(&self, uv: [f32; 2], w: usize, h: usize) -> Option<[usize; 2]> {
        let s = self.source_uv(uv);
        if !(0.0..=1.0).contains(&s[0]) || !(0.0..=1.0).contains(&s[1]) {
            return None;
        }
        Some([((s[0] * w as f32) as usize).min(w - 1), ((s[1] * h as f32) as usize).min(h - 1)])
    }

    /// True when the output is the source, pixel for pixel.
    pub fn is_identity(&self, w: usize, h: usize) -> bool {
        (self.width, self.height) == (w, h) && self.a == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    }
}

/// The largest crop with the frame's own aspect ratio, centred, that stays
/// inside a frame of `w`x`h` (after quarter turns) straightened by `degrees` —
/// so straightening never shows empty corners. Returns `[x0, y0, x1, y1]`.
pub fn straighten_crop(degrees: f32, w: f32, h: f32) -> [f32; 4] {
    let (s, c) = degrees.to_radians().sin_cos();
    let (s, c) = (s.abs(), c.abs());
    let k = (w / (w * c + h * s)).min(h / (w * s + h * c)).min(1.0);
    let m = (1.0 - k) / 2.0;
    [m, m, 1.0 - m, 1.0 - m]
}

/// White balance and matrix as actually applied to one render.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Colour {
    pub wb: [f32; 3],
    pub matrix: [[f32; 3]; 3],
}

impl Colour {
    /// The channel mixer is folded into the matrix here, so it costs nothing
    /// per pixel and the shader needs no code of its own for it.
    pub fn resolve(img: &ImageColour, p: &AdjustParams) -> Self {
        let base = match p.colour_mode {
            ColourMode::Camera => img.cam_to_srgb,
            ColourMode::SensorRgb => IDENTITY3,
        };
        // Look strength: blend the mixer towards identity.
        let s = p.mixer_strength;
        let mixer = std::array::from_fn(|i| std::array::from_fn(|j| IDENTITY3[i][j] + s * (p.mixer[i][j] - IDENTITY3[i][j])));
        Self { wb: p.wb_gains.unwrap_or(img.as_shot_wb), matrix: mat3_mul(&mixer, &base) }
    }
}

/// `a · b` for row-major 3x3 matrices.
pub fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

/// Camera RGB → working linear RGB for one pixel: white balance, then make a
/// clipped pixel neutral, then the matrix. Mirrors `to_working` in the shader.
/// `cam` must already carry the optics gain (`flatfield::Optics`), which the
/// shader applies at the top of its `to_working`.
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
/// Total size 400 bytes.
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
    /// Hot-spot strength per channel in `.rgb`, Gaussian radius in `.a`.
    pub hotspot: [f32; 4],
    /// Hot-spot centre (uv) in `.xy`, image aspect (w/h) in `.z`, flat field
    /// on (1.0) / off in `.w`.
    pub optics: [f32; 4],
    /// Per-colour bands `[hue, sat, lum, _]`; `hsl[0][3]` is 1.0 when any
    /// band is non-zero, so the shader can skip the stage.
    pub hsl: [[f32; 4]; 8],
    /// Output uv → source uv, the two rows of [`Geometry::a`] (`.w` unused).
    pub geom0: [f32; 4],
    pub geom1: [f32; 4],
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
    /// 1.0 = clipping overlay (see [`clip_view`]). A viewing aid, never
    /// exported. It also happens to be the slot that pads the struct to a
    /// multiple of 16 bytes, which WGSL requires — if it moves, add a pad.
    pub show_clip: f32,
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
    /// The source pixel under a viewport position given in NDC (x right, y
    /// up, both -1..1), or `None` off the image. Inverse of the quad placement
    /// in `vs_main`.
    pub fn image_pixel(&self, ndc: [f32; 2], size: [usize; 2]) -> Option<[usize; 2]> {
        let [u, v] = self.quad_uv(ndc)?;
        let px = ((u * size[0] as f32) as usize).min(size[0] - 1);
        let py = ((v * size[1] as f32) as usize).min(size[1] - 1);
        Some([px, py])
    }

    /// The quad's uv (0..1, v down) under a viewport position in NDC, or
    /// `None` off the quad. With a geometry this is the *output* uv; see
    /// [`Geometry::source_pixel`].
    pub fn quad_uv(&self, ndc: [f32; 2]) -> Option<[f32; 2]> {
        let cx = (ndc[0] - self.offset[0]) / self.scale[0];
        let cy = (ndc[1] - self.offset[1]) / self.scale[1];
        if !(-1.0..=1.0).contains(&cx) || !(-1.0..=1.0).contains(&cy) {
            return None;
        }
        Some([cx * 0.5 + 0.5, 0.5 - cy * 0.5])
    }

    /// Aspect-fitted, unpanned.
    pub fn fit(scale: [f32; 2], texel: [f32; 2]) -> Self {
        Self { scale, offset: [0.0, 0.0], texel }
    }

    /// One output pixel per source pixel, filling the whole target.
    pub fn one_to_one(texel: [f32; 2]) -> Self {
        Self::fit([1.0, 1.0], texel)
    }
}

/// The geometry for a render, with the source size recovered from the texel.
fn view_geometry(p: &AdjustParams, img: &ImageColour, view: View) -> Geometry {
    let (w, h) = ((1.0 / view.texel[0]).round() as usize, (1.0 / view.texel[1]).round() as usize);
    Geometry::resolve(p, img, w, h)
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
            hotspot: {
                let a = crate::flatfield::hotspot_amounts(p);
                [a[0], a[1], a[2], p.hotspot_radius.max(crate::flatfield::MIN_HOTSPOT_RADIUS)]
            },
            optics: [
                p.hotspot_centre[0],
                p.hotspot_centre[1],
                // texel = 1 / size, so texel.y / texel.x = width / height.
                view.texel[1] / view.texel[0],
                if p.flat_field { 1.0 } else { 0.0 },
            ],
            hsl: {
                let mut h = p.hsl.map(|b| [b[0], b[1], b[2], 0.0]);
                h[0][3] = if hsl_active(&p.hsl) { 1.0 } else { 0.0 };
                h
            },
            geom0: { let g = view_geometry(p, img, view); [g.a[0][0], g.a[0][1], g.a[0][2], 0.0] },
            geom1: { let g = view_geometry(p, img, view); [g.a[1][0], g.a[1][1], g.a[1][2], 0.0] },
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
            show_clip: 0.0,
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

    // 6. Per-colour hue / saturation / luminance.
    if hsl_active(&p.hsl) {
        c = hsl_adjust(&p.hsl, c);
    }
    c
}

/// The per-colour bands: name and centre hue in degrees. Spacing follows
/// Lightroom's — tighter around the warm hues, where skin and foliage live.
pub const HSL_BANDS: [(&str, f32); 8] = [
    ("Red", 0.0),
    ("Orange", 30.0),
    ("Yellow", 60.0),
    ("Green", 120.0),
    ("Aqua", 180.0),
    ("Blue", 240.0),
    ("Purple", 270.0),
    ("Magenta", 300.0),
];
/// Hue slider at ±1 rotates by this much (as a fraction of the circle: 30°).
pub const HSL_HUE_MAX: f32 = 30.0 / 360.0;
/// Luminance slider at ±1 is ±1 stop.
pub const HSL_LUM_EV: f32 = 1.0;
/// HSV saturation at which a colour gets the full adjustment; greyer colours
/// get proportionally less, so neutrals are never touched.
pub const HSL_FULL_SAT: f32 = 0.25;

pub fn hsl_active(hsl: &[[f32; 3]; 8]) -> bool {
    hsl.iter().flatten().any(|&v| v != 0.0)
}

/// The adjustment for a hue (0..1): linear interpolation between the two
/// nearest band centres, so neighbouring bands blend with no seams.
pub fn hsl_at(hsl: &[[f32; 3]; 8], h: f32) -> [f32; 3] {
    let deg = h * 360.0;
    let mut i = 7;
    for k in 0..8 {
        if deg >= HSL_BANDS[k].1 {
            i = k;
        }
    }
    let lo = HSL_BANDS[i].1;
    let hi = if i == 7 { 360.0 } else { HSL_BANDS[i + 1].1 };
    let t = (deg - lo) / (hi - lo);
    let (a, b) = (hsl[i], hsl[(i + 1) % 8]);
    std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t)
}

fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// Per-colour adjustment on linear RGB, via HSV: rotate hue, scale HSV
/// saturation, then scale the whole pixel by a luminance gain. Mirrors
/// `hsl_adjust` in the shader, branch for branch.
pub fn hsl_adjust(hsl: &[[f32; 3]; 8], c: [f32; 3]) -> [f32; 3] {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    let d = mx - mn;
    if mx <= 1e-6 || d <= 1e-6 {
        return c;
    }
    let h6 = if mx == c[0] {
        (c[1] - c[2]) / d
    } else if mx == c[1] {
        (c[2] - c[0]) / d + 2.0
    } else {
        (c[0] - c[1]) / d + 4.0
    };
    let h = fract(h6 / 6.0);
    let s = d / mx;
    let w = (s / HSL_FULL_SAT).clamp(0.0, 1.0);
    let [dh, ds, dl] = hsl_at(hsl, h).map(|v| v * w);
    let h = fract(h + dh * HSL_HUE_MAX);
    let s = s * (1.0 + ds);
    // HSV → RGB, the branch-free form the shader uses too.
    let ch = |n: f32| {
        let k = (n + h * 6.0) % 6.0;
        mx - mx * s * k.min(4.0 - k).clamp(0.0, 1.0)
    };
    let gain = (dl * HSL_LUM_EV).exp2();
    [ch(5.0) * gain, ch(3.0) * gain, ch(1.0) * gain]
}

/// Share of luminance the unclipped part of the clipping overlay keeps, so the
/// clipped pixels stand out.
pub const CLIP_VIEW_DIM: f32 = 0.35;

/// Clipping overlay, applied to the linear output of [`apply`]. A pixel with
/// any channel at or above white shows *which* channels clipped at full
/// strength (red, yellow, white...); everything else becomes dimmed grey. A
/// single warning colour would vanish into false-colour IR images, which are
/// full of saturated reds and magentas. Mirrored in the shader.
pub fn clip_view(c: [f32; 3]) -> [f32; 3] {
    let over = c.map(|v| if v >= 1.0 { 1.0 } else { 0.0 });
    if over.iter().any(|&o| o > 0.0) {
        over
    } else {
        [luma(c.map(|v| v.clamp(0.0, 1.0))) * CLIP_VIEW_DIM; 3]
    }
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

    /// Every control moved off neutral.
    fn all_moved() -> AdjustParams {
        AdjustParams {
            wb_gains: Some([2.0, 1.0, 1.5]),
            colour_mode: ColourMode::SensorRgb,
            mixer: [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            mixer_strength: 0.5,
            hsl: [[0.1, -0.2, 0.3]; 8],
            exposure_ev: 1.0,
            contrast: 0.2,
            temp: 0.1,
            tint: -0.1,
            saturation: 0.3,
            highlights: -0.4,
            shadows: 0.4,
            whites: 0.1,
            blacks: -0.1,
            lift: [0.1; 3],
            gamma: [1.1; 3],
            gain: [0.9; 3],
            sharpen_amount: 1.0,
            sharpen_radius: 2.0,
            sharpen_threshold: 0.01,
            nr_luma: 0.5,
            nr_chroma: 0.5,
            hotspot_amount: 0.5,
            hotspot_radius: 0.6,
            hotspot_centre: [0.4, 0.6],
            hotspot_balance: [1.2, 1.0, 0.8],
            flat_field: true,
            rotation: 1,
            straighten: 2.0,
            crop: [0.1, 0.1, 0.9, 0.8],
        }
    }

    fn hue_deg(c: [f32; 3]) -> f32 {
        let mx = c[0].max(c[1]).max(c[2]);
        let d = mx - c[0].min(c[1]).min(c[2]);
        let h = if mx == c[0] { (c[1] - c[2]) / d } else if mx == c[1] { (c[2] - c[0]) / d + 2.0 } else { (c[0] - c[1]) / d + 4.0 };
        (h * 60.0).rem_euclid(360.0)
    }

    fn band(name: &str) -> usize {
        HSL_BANDS.iter().position(|(n, _)| *n == name).unwrap()
    }

    #[test]
    fn hsl_bands_interpolate_between_centres() {
        let mut hsl = [[0.0; 3]; 8];
        hsl[band("Yellow")] = [1.0, 0.0, 0.0];
        assert_eq!(hsl_at(&hsl, 60.0 / 360.0), [1.0, 0.0, 0.0]);
        assert_eq!(hsl_at(&hsl, 90.0 / 360.0), [0.5, 0.0, 0.0], "halfway to green");
        assert_eq!(hsl_at(&hsl, 200.0 / 360.0), [0.0, 0.0, 0.0]);
        // Magenta wraps round to red.
        let mut hsl = [[0.0; 3]; 8];
        hsl[band("Red")] = [0.0, 1.0, 0.0];
        assert!((hsl_at(&hsl, 330.0 / 360.0)[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn hsl_leaves_greys_alone() {
        let hsl = [[1.0, 1.0, 1.0]; 8];
        assert_eq!(hsl_adjust(&hsl, GREY), GREY);
    }

    #[test]
    fn hsl_desaturating_blue_spares_red() {
        let mut hsl = [[0.0; 3]; 8];
        hsl[band("Blue")] = [0.0, -1.0, 0.0];
        let blue = hsl_adjust(&hsl, [0.1, 0.1, 0.6]);
        assert!((blue[0] - blue[2]).abs() < 1e-5, "fully desaturated: {blue:?}");
        let red = [0.6, 0.1, 0.1];
        let out = hsl_adjust(&hsl, red);
        assert!(out.iter().zip(red).all(|(o, i)| (o - i).abs() < 1e-6), "{out:?}");
    }

    #[test]
    fn hsl_hue_shift_rotates_by_up_to_thirty_degrees() {
        let mut hsl = [[0.0; 3]; 8];
        hsl[band("Red")] = [1.0, 0.0, 0.0];
        let out = hsl_adjust(&hsl, [0.8, 0.0, 0.0]);
        assert!((hue_deg(out) - 30.0).abs() < 0.01, "{out:?} at {}", hue_deg(out));
    }

    #[test]
    fn hsl_luminance_is_a_stop_per_unit() {
        let mut hsl = [[0.0; 3]; 8];
        hsl[band("Green")] = [0.0, 0.0, 1.0];
        let out = hsl_adjust(&hsl, [0.1, 0.5, 0.1]);
        for (o, i) in out.iter().zip([0.1, 0.5, 0.1]) {
            assert!((o - 2.0 * i).abs() < 1e-5, "{out:?}");
        }
    }

    #[test]
    fn mixer_strength_blends_towards_identity() {
        let swap = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let at = |s: f32| {
            let p = AdjustParams { colour_mode: ColourMode::SensorRgb, mixer: swap, mixer_strength: s, ..Default::default() };
            Colour::resolve(&ImageColour::LINEAR_SRGB, &p).matrix
        };
        assert_eq!(at(0.0), IDENTITY3);
        assert_eq!(at(1.0), swap);
        assert_eq!(at(0.5)[0], [0.5, 0.0, 0.5]);
    }

    fn geom(p: AdjustParams, turns: u8, w: usize, h: usize) -> Geometry {
        Geometry::resolve(&p, &ImageColour { turns, ..ImageColour::LINEAR_SRGB }, w, h)
    }

    fn close2(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5
    }

    #[test]
    fn default_geometry_is_identity() {
        assert!(geom(AdjustParams::default(), 0, 40, 30).is_identity(40, 30));
    }

    #[test]
    fn quarter_turns_map_corners_and_swap_size() {
        // Output top-left after a clockwise turn is the source's bottom-left.
        let cases = [(0, [0.0, 0.0]), (1, [0.0, 1.0]), (2, [1.0, 1.0]), (3, [1.0, 0.0])];
        for (turns, corner) in cases {
            let g = geom(AdjustParams::default(), turns, 40, 30);
            assert!(close2(g.source_uv([0.0, 0.0]), corner), "turns {turns}: {:?}", g.source_uv([0.0, 0.0]));
            let size = if turns % 2 == 1 { (30, 40) } else { (40, 30) };
            assert_eq!((g.width, g.height), size);
        }
    }

    #[test]
    fn source_pixel_follows_the_rotation() {
        // Turned clockwise, the output's top-left pixel is the source's
        // bottom-left one.
        let g = geom(AdjustParams { rotation: 1, ..Default::default() }, 0, 40, 30);
        assert_eq!(g.source_pixel([0.01, 0.01], 40, 30), Some([0, 29]));
        let s = geom(AdjustParams { straighten: 10.0, ..Default::default() }, 0, 40, 30);
        assert_eq!(s.source_pixel([0.0, 0.0], 40, 30), None, "uncropped straightened corner is empty");
    }

    #[test]
    fn camera_and_user_turns_add_up() {
        let p = AdjustParams { rotation: 3, ..Default::default() };
        // Camera says turn once, the user turns three more: back to upright.
        assert!(geom(p, 1, 40, 30).is_identity(40, 30));
    }

    #[test]
    fn crop_snaps_to_whole_pixels() {
        let p = AdjustParams { crop: [0.101, 0.2, 0.604, 0.7], ..Default::default() };
        let g = geom(p, 0, 100, 50);
        assert_eq!((g.width, g.height), (50, 25));
        // First output pixel centre lands on source pixel 10's centre.
        let s = g.source_uv([0.5 / 50.0, 0.5 / 25.0]);
        assert!(close2(s, [10.5 / 100.0, 10.5 / 50.0]), "{s:?}");
    }

    #[test]
    fn straighten_crop_leaves_no_empty_corners() {
        for deg in [-12.0f32, -3.0, 0.5, 5.0, 20.0] {
            let (w, h) = (4000.0, 3000.0);
            let p = AdjustParams { straighten: deg, crop: straighten_crop(deg, w, h), ..Default::default() };
            let g = geom(p, 0, w as usize, h as usize);
            for c in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] {
                let s = g.source_uv(c);
                assert!(s.iter().all(|v| (-1e-3..=1.0 + 1e-3).contains(v)), "{deg}°: corner {c:?} → {s:?}");
            }
            // And it is the largest such crop: the frame is not shrunk for nothing.
            let grown = straighten_crop(deg, w, h).map(|v| if v < 0.5 { v - 0.01 } else { v + 0.01 });
            let g = geom(AdjustParams { straighten: deg, crop: grown, ..Default::default() }, 0, w as usize, h as usize);
            assert!([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]].iter().any(|c| {
                let s = g.source_uv(*c);
                s.iter().any(|v| !(0.0..=1.0).contains(v))
            }) || deg.abs() < 1.0, "{deg}°: a 1% bigger crop should poke outside");
        }
    }

    #[test]
    fn resetting_every_section_resets_everything() {
        // Catches a new param that was never assigned to a section, and so
        // would survive every per-section Reset.
        let mut p = all_moved();
        for s in Section::ALL {
            assert!(!p.section_is_neutral(s), "{s:?} should start moved");
            p.reset_section(s);
            assert!(p.section_is_neutral(s));
        }
        assert!(p.is_neutral());
    }

    #[test]
    fn resetting_a_section_leaves_the_others_alone() {
        let moved = all_moved();
        for s in Section::ALL {
            let mut p = moved;
            p.reset_section(s);
            for other in Section::ALL.into_iter().filter(|&o| o != s) {
                assert!(!p.section_is_neutral(other), "resetting {s:?} touched {other:?}");
            }
        }
    }

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
    fn uniforms_layout_is_400_bytes() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 400);
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
            &u.hotspot as *const _ as usize,
            &u.optics as *const _ as usize,
            &u.hsl as *const _ as usize,
            &u.geom0 as *const _ as usize,
            &u.geom1 as *const _ as usize,
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

    #[test]
    fn sensor_rgb_mode_drops_the_matrix_and_custom_wb_overrides_as_shot() {
        let img = ImageColour { as_shot_wb: [2.0, 1.0, 1.5], cam_to_srgb: CAM, turns: 0 };
        let def = Colour::resolve(&img, &AdjustParams::default());
        assert_eq!(def, Colour { wb: [2.0, 1.0, 1.5], matrix: CAM });
        let p = AdjustParams {
            wb_gains: Some([0.7, 1.0, 3.2]),
            colour_mode: ColourMode::SensorRgb,
            ..Default::default()
        };
        assert_eq!(Colour::resolve(&img, &p), Colour { wb: [0.7, 1.0, 3.2], matrix: IDENTITY3 });
    }

    #[test]
    fn sidecars_from_before_ir1_load_as_as_shot_camera_mode() {
        let old = r#"{"exposure_ev":0.5,"contrast":0.1}"#;
        let p: AdjustParams = serde_json::from_str(old).unwrap();
        assert_eq!(p.wb_gains, None);
        assert_eq!(p.colour_mode, ColourMode::Camera);
        assert_eq!(p.exposure_ev, 0.5);
    }

    #[test]
    fn colour_settings_round_trip_through_json() {
        let p = AdjustParams {
            wb_gains: Some([0.62, 1.0, 2.4]),
            colour_mode: ColourMode::SensorRgb,
            ..Default::default()
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"sensor_rgb\""), "{json}");
        assert_eq!(serde_json::from_str::<AdjustParams>(&json).unwrap(), p);
    }

    #[test]
    fn image_pixel_inverts_the_quad_placement() {
        let size = [400, 300];
        // Fit view, quad fills the viewport: corners and centre.
        let v = View::fit([1.0, 1.0], [1.0 / 400.0, 1.0 / 300.0]);
        assert_eq!(v.image_pixel([-1.0, 1.0], size), Some([0, 0]));
        assert_eq!(v.image_pixel([1.0, -1.0], size), Some([399, 299]));
        assert_eq!(v.image_pixel([0.0, 0.0], size), Some([200, 150]));
        // Letterboxed: the band above the image is off it.
        let v = View::fit([1.0, 0.5], [1.0 / 400.0, 1.0 / 300.0]);
        assert_eq!(v.image_pixel([0.0, 0.75], size), None);
        assert_eq!(v.image_pixel([0.0, 0.5], size), Some([200, 0]));
        // Zoomed 2x and panned right by half the viewport: the viewport
        // centre now shows the image point a quarter of the way in.
        let v = View { scale: [2.0, 2.0], offset: [1.0, 0.0], texel: [0.0; 2] };
        assert_eq!(v.image_pixel([0.0, 0.0], size), Some([100, 150]));
    }

    #[test]
    fn clip_view_shows_which_channels_clipped() {
        assert_eq!(clip_view([1.3, 0.4, 0.2]), [1.0, 0.0, 0.0]);
        assert_eq!(clip_view([1.0, 1.2, 0.9]), [1.0, 1.0, 0.0]);
        assert_eq!(clip_view([2.0, 2.0, 2.0]), [1.0, 1.0, 1.0]);
        let g = clip_view([0.5, 0.5, 0.5]);
        assert!((g[0] - 0.5 * CLIP_VIEW_DIM).abs() < 1e-6 && g[0] == g[1] && g[1] == g[2]);
    }

    // --- channel mixer (IR-2) -------------------------------------------

    #[test]
    fn mixer_is_applied_after_the_matrix() {
        let img = ImageColour { as_shot_wb: [1.0; 3], cam_to_srgb: CAM, turns: 0 };
        let swap = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let p = AdjustParams { mixer: swap, ..Default::default() };
        let plain = to_working(&Colour::resolve(&img, &AdjustParams::default()), [0.2, 0.3, 0.5], 0.0);
        let mixed = to_working(&Colour::resolve(&img, &p), [0.2, 0.3, 0.5], 0.0);
        assert!(close(mixed, [plain[2], plain[1], plain[0]]), "{mixed:?} vs {plain:?}");
    }

    #[test]
    fn identity_mixer_changes_nothing() {
        let img = ImageColour { as_shot_wb: [2.0, 1.0, 1.5], cam_to_srgb: CAM, turns: 0 };
        assert_eq!(Colour::resolve(&img, &AdjustParams::default()).matrix, CAM);
    }

    #[test]
    fn sidecars_from_before_ir2_load_with_the_mixer_off() {
        let p: AdjustParams = serde_json::from_str(r#"{"colour_mode":"sensor_rgb"}"#).unwrap();
        assert_eq!(p.mixer, IDENTITY3);
        let m = AdjustParams { mixer: [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ..Default::default() };
        assert_eq!(serde_json::from_str::<AdjustParams>(&serde_json::to_string(&m).unwrap()).unwrap(), m);
    }
}

