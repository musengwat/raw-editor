//! Looks: a channel mixer plus, optionally, the colour and tone settings that
//! go with it.
//!
//! Built-ins come in groups ([`LookGroup`]). **Everyday**, **Film**,
//! **Black & white** and **Creative** looks are for any camera; they are
//! built on the camera-fitted tone curve and each one *owns* every look
//! field (anything it does not set is set to neutral), so switching from one
//! to another replaces the previous look completely. "Everyday — Natural" is
//! what photos open in by default ([`DEFAULT_LOOK`]). The film looks are
//! impressions of how those stocks are usually remembered, not measured
//! emulations.
//!
//! **Infrared** looks are false-colour starting points for the full-spectrum
//! GF1, grouped by the filter on the lens, because what each photosite
//! records depends entirely on the filter. They only set the fields they
//! name and leave the rest of the photo alone.
//!
//! **Full spectrum, no filter** and **Tiffen #16 orange** looks were tuned on
//! real daylight frames (P1320275–295, Oct 2 2026: water, grass, conifers,
//! a maple and asphalt) after the GF1's dust-shaker glass — which blocks IR
//! on this body — was removed. Each sets white balance so asphalt is neutral
//! and then mixes channels:
//!
//! - Behind the #16 (a ~550 nm long-pass), the **blue** photosites record IR
//!   plus a little orange light leaking through the blue dye. That leak is a
//!   steady ~17% of green on scenes with no IR, so `IR ≈ B − k·G` (written
//!   in white-balanced units as the row `[0, -k, 1 + k]`) isolates the IR:
//!   foliage comes out 3–6× brighter than asphalt.
//! - With no filter, every channel sees visible light and the extra IR lands
//!   mostly in **red** (the green and blue dyes only open up past ~800 nm),
//!   so `R − G` is the IR-ish signal: it puts trees bright and sky dark.
//!
//! **IR Chrome** looks are derived from the filter's physics, not tuned —
//! there are no shots through that filter yet. It blocks blue and passes
//! green, red and IR, so the blue photosites see almost pure IR.
//!
//! Kodak Aerochrome shifted its layers IR → red, red → green, green → blue;
//! the Aerochrome looks do the same with whatever the IR estimate is. Every
//! built-in row sums to 1, so anything white-balanced to neutral stays
//! neutral. Subtracting channels amplifies noise: at ISO 800 the #16 looks
//! and both IR mono looks are visibly grainy at 1:1 (the #16's blue gain is
//! ~4.6x before the subtraction), while the full-spectrum colour looks stay
//! clean. Shoot at base ISO and use the NR sliders. The #16 looks also render
//! about a stop darker — the filter eats light — so expect to raise exposure.

use serde::{Deserialize, Serialize};

use crate::pipeline::{AdjustParams, ColourMode, IDENTITY3};

/// What a look does to the white balance.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LookWb {
    /// Leave the photo's white balance alone (all built-in looks).
    Keep,
    /// Reset to the camera's as-shot white balance.
    AsShot,
    /// Set these gains, green = 1.
    Gains([f32; 3]),
}

/// Where a look sits in the picker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LookGroup {
    Everyday,
    Film,
    BlackWhite,
    Creative,
    Infrared,
    /// The user's own (every look loaded from the settings file).
    #[default]
    Saved,
}

impl LookGroup {
    pub const ALL: [LookGroup; 6] =
        [Self::Everyday, Self::Film, Self::BlackWhite, Self::Creative, Self::Infrared, Self::Saved];

    pub fn label(self) -> &'static str {
        match self {
            Self::Everyday => "Everyday",
            Self::Film => "Film",
            Self::BlackWhite => "Black & white",
            Self::Creative => "Creative",
            Self::Infrared => "Infrared",
            Self::Saved => "Saved",
        }
    }
}

/// The look photos open in when they have no sidecar and their camera has no
/// look of its own (`Config::default_look` can change it).
pub const DEFAULT_LOOK: &str = "Everyday — Natural";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Look {
    pub name: String,
    /// Picker group. Not stored for saved looks (they are always `Saved`).
    #[serde(default, skip)]
    pub group: LookGroup,
    /// `None` leaves the photo's colour mode as it is.
    pub colour_mode: Option<ColourMode>,
    pub wb: LookWb,
    pub mixer: [[f32; 3]; 3],
    /// `None` leaves the photo's saturation as it is.
    pub saturation: Option<f32>,
    /// Per-colour adjustments; `None` leaves the photo's alone.
    #[serde(default)]
    pub hsl: Option<[[f32; 3]; 8]>,
    /// Contrast and black level, for looks that emulate a film's tone.
    /// `None` leaves the photo's alone.
    #[serde(default)]
    pub contrast: Option<f32>,
    #[serde(default)]
    pub blacks: Option<f32>,
    /// Tone-curve strength, temperature/tint shift, the other tonal ranges and
    /// lift/gamma/gain (split toning). `None` leaves the photo's alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tone_curve: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temp: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tint: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highlights: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadows: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub whites: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lift: Option<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gamma: Option<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gain: Option<[f32; 3]>,
    /// What the look is for (filter, what it does), shown as a tooltip.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
}

impl Look {
    /// An infrared built-in: sets only what its builder calls name.
    fn builtin(name: &str, mixer: [[f32; 3]; 3], mode: Option<ColourMode>) -> Self {
        Self {
            name: name.into(),
            group: LookGroup::Infrared,
            colour_mode: mode,
            wb: LookWb::Keep,
            mixer,
            saturation: None,
            hsl: None,
            contrast: None,
            blacks: None,
            tone_curve: None,
            temp: None,
            tint: None,
            highlights: None,
            shadows: None,
            whites: None,
            lift: None,
            gamma: None,
            gain: None,
            note: String::new(),
        }
    }

    /// A look for any camera. It owns every look field — whatever the
    /// builder does not set is neutral — so it fully replaces the previous
    /// look. Starts from the camera-fitted tone curve and the camera's
    /// colour matrix; white balance is left alone.
    fn general(name: &str, group: LookGroup) -> Self {
        let d = AdjustParams::default();
        Self {
            name: name.into(),
            group,
            colour_mode: Some(ColourMode::Camera),
            wb: LookWb::Keep,
            mixer: IDENTITY3,
            saturation: Some(d.saturation),
            hsl: Some(d.hsl),
            contrast: Some(d.contrast),
            blacks: Some(d.blacks),
            tone_curve: Some(1.0),
            temp: Some(d.temp),
            tint: Some(d.tint),
            highlights: Some(d.highlights),
            shadows: Some(d.shadows),
            whites: Some(d.whites),
            lift: Some(d.lift),
            gamma: Some(d.gamma),
            gain: Some(d.gain),
            note: String::new(),
        }
    }

    fn sat(mut self, v: f32) -> Self {
        self.saturation = Some(v);
        self
    }

    fn contrast(mut self, v: f32) -> Self {
        self.contrast = Some(v);
        self
    }

    /// Highlights, shadows, whites, blacks.
    fn tones(mut self, highlights: f32, shadows: f32, whites: f32, blacks: f32) -> Self {
        self.highlights = Some(highlights);
        self.shadows = Some(shadows);
        self.whites = Some(whites);
        self.blacks = Some(blacks);
        self
    }

    /// Temperature and tint shift on top of the photo's white balance.
    fn warmth(mut self, temp: f32, tint: f32) -> Self {
        self.temp = Some(temp);
        self.tint = Some(tint);
        self
    }

    /// Lift (shadow colour), gamma and gain (highlight colour).
    fn grade(mut self, lift: [f32; 3], gamma: [f32; 3], gain: [f32; 3]) -> Self {
        self.lift = Some(lift);
        self.gamma = Some(gamma);
        self.gain = Some(gain);
        self
    }

    fn bands(mut self, hsl: &[(&str, [f32; 3])]) -> Self {
        self.hsl = Some(bands(hsl));
        self
    }

    fn mix(mut self, mixer: [[f32; 3]; 3]) -> Self {
        self.mixer = mixer;
        self
    }

    /// Black and white from these channel weights (they sum to 1).
    fn mono(self, w: [f32; 3]) -> Self {
        self.mix([w; 3])
    }

    fn wb(mut self, gains: [f32; 3]) -> Self {
        self.wb = LookWb::Gains(gains);
        self
    }

    fn note(mut self, note: &str) -> Self {
        self.note = note.into();
        self
    }

    /// A finished effect: tone and per-colour work on top of the mix.
    fn finish(mut self, saturation: f32, contrast: f32, blacks: f32, hsl: &[(&str, [f32; 3])]) -> Self {
        self.saturation = Some(saturation);
        self.contrast = Some(contrast);
        self.blacks = Some(blacks);
        self.hsl = Some(bands(hsl));
        self
    }

    /// Capture the current colour and tone settings of `p` as a look — white
    /// balance, colour mode, mixer, saturation, the per-colour panel, tone
    /// curve, contrast, temperature/tint, the tonal ranges and lift/gamma/
    /// gain. Exposure, detail, optics, geometry and look strength are per
    /// photo and deliberately left out (exposure depends on the aperture and
    /// light, not the look).
    pub fn from_params(name: &str, p: &AdjustParams) -> Self {
        Self {
            name: name.trim().into(),
            colour_mode: Some(p.colour_mode),
            wb: p.wb_gains.map_or(LookWb::AsShot, LookWb::Gains),
            mixer: p.mixer,
            saturation: Some(p.saturation),
            hsl: Some(p.hsl),
            contrast: Some(p.contrast),
            blacks: Some(p.blacks),
            tone_curve: Some(p.tone_curve),
            temp: Some(p.temp),
            tint: Some(p.tint),
            highlights: Some(p.highlights),
            shadows: Some(p.shadows),
            whites: Some(p.whites),
            lift: Some(p.lift),
            gamma: Some(p.gamma),
            gain: Some(p.gain),
            group: LookGroup::Saved,
            note: String::new(),
        }
    }

    pub fn apply(&self, p: &mut AdjustParams) {
        p.mixer = self.mixer;
        if let Some(m) = self.colour_mode {
            p.colour_mode = m;
        }
        match self.wb {
            LookWb::Keep => {}
            LookWb::AsShot => p.wb_gains = None,
            LookWb::Gains(g) => p.wb_gains = Some(g),
        }
        if let Some(s) = self.saturation {
            p.saturation = s;
        }
        if let Some(h) = self.hsl {
            p.hsl = h;
        }
        if let Some(c) = self.contrast {
            p.contrast = c;
        }
        if let Some(b) = self.blacks {
            p.blacks = b;
        }
        let set = |dst: &mut f32, v: Option<f32>| {
            if let Some(v) = v {
                *dst = v;
            }
        };
        set(&mut p.tone_curve, self.tone_curve);
        set(&mut p.temp, self.temp);
        set(&mut p.tint, self.tint);
        set(&mut p.highlights, self.highlights);
        set(&mut p.shadows, self.shadows);
        set(&mut p.whites, self.whites);
        if let Some(v) = self.lift {
            p.lift = v;
        }
        if let Some(v) = self.gamma {
            p.gamma = v;
        }
        if let Some(v) = self.gain {
            p.gain = v;
        }
    }

    /// True when applying this look to `p` would change nothing — how the UI
    /// knows which look is currently selected.
    pub fn is_applied(&self, p: &AdjustParams) -> bool {
        let mut q = *p;
        self.apply(&mut q);
        q == *p
    }
}

/// Per-colour adjustments from `(band name, [hue, sat, lum])` pairs; bands
/// not named stay at zero.
fn bands(hsl: &[(&str, [f32; 3])]) -> [[f32; 3]; 8] {
    let mut h = [[0.0; 3]; 8];
    for (name, v) in hsl {
        let i = crate::pipeline::HSL_BANDS.iter().position(|(n, _)| n == name).expect("band name");
        h[i] = *v;
    }
    h
}

/// White balance that makes sunlit asphalt neutral, measured on the Oct 2
/// 2026 frames. Daylight changes it a little; the eyedropper on a road or a
/// grey card refines it without changing the look's mixer.
const WB_FULL_SPECTRUM: [f32; 3] = [0.94, 1.0, 1.55];
const WB_ORANGE_16: [f32; 3] = [0.63, 1.0, 4.56];
/// For the #16 Aerochrome effects: neutral on *clouds* rather than asphalt.
/// Clouds reflect more IR than asphalt (raw R/G 1.24 against 1.58), so with
/// the asphalt balance they came out pink; real Aerochrome keeps them white.
const WB_ORANGE_16_CLOUDS: [f32; 3] = [0.72, 1.0, 4.56];

/// IR estimate behind the #16: blue minus its orange leak (see module docs).
const IR_16: [f32; 3] = [0.0, -1.5, 2.5];
/// Aerochrome mix behind the #16: IR → red, red → green, green → blue.
const AERO_16: [[f32; 3]; 3] = [IR_16, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// IR-ish signal with no filter: red's excess over green.
const IR_FULL: [f32; 3] = [3.0, -2.0, 0.0];

/// Names built-ins used to have, so a body look saved under an old name
/// still resolves.
pub const RENAMED: &[(&str, &str)] = &[
    ("Aerochrome classic", "IR Chrome — Aerochrome"),
    ("Aerochrome pink", "IR Chrome — Aerochrome pink"),
    ("Red–blue swap", "IR Chrome — red–blue swap"),
    ("IR mono", "IR Chrome — IR mono"),
];

/// The looks that ship with the editor, in picker order. The first, "None",
/// clears every look field (the linear rendering) and is always present.
pub fn builtin() -> Vec<Look> {
    let ir = Some(ColourMode::SensorRgb);
    // Gold foliage, blue sky: R = IR, G = between IR and green, B = green
    // pushed away from IR.
    let gold = [
        IR_16,
        [0.0, -0.25, 1.25],                       // 0.5·IR + 0.5·G
        [0.0, 2.25, -1.25],                       // 1.5·G − 0.5·IR
    ];
    use LookGroup::*;
    // Black-and-white weights: Rec. 709 luminance, and a red filter's
    // (dark blue skies, light skin and foliage).
    const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];
    const RED_FILTER: [f32; 3] = [0.70, 0.30, 0.0];
    let none = Look { colour_mode: None, tone_curve: Some(0.0), ..Look::general("None", Everyday) }
        .note("No look: the linear rendering, with only the photo's white balance and your own edits");
    vec![
        none,
        // --- Everyday -------------------------------------------------------
        Look::general("Everyday — Natural", Everyday)
            .sat(0.1)
            .note("What the camera's own JPEG would look like: its tone curve (measured on Sony and Panasonic bodies) and a touch of colour. New photos open in this"),
        Look::general("Everyday — Vivid", Everyday)
            .sat(0.3)
            .contrast(0.1)
            .bands(&[("Green", [0.0, 0.15, 0.0]), ("Blue", [0.0, 0.2, -0.1])])
            .note("Punchier colour and contrast: travel, flowers, bright days"),
        Look::general("Everyday — Portrait", Everyday)
            .contrast(-0.05)
            .tones(-0.1, 0.15, 0.0, 0.0)
            .warmth(0.04, 0.0)
            .bands(&[("Red", [0.0, -0.1, 0.05]), ("Orange", [-0.05, -0.1, 0.15])])
            .note("Softer contrast and gentle, slightly warm skin tones"),
        Look::general("Everyday — Landscape", Everyday)
            .sat(0.2)
            .contrast(0.1)
            .tones(-0.3, 0.1, 0.0, 0.0)
            .bands(&[("Green", [0.0, 0.2, -0.1]), ("Aqua", [0.0, 0.15, -0.1]), ("Blue", [0.0, 0.25, -0.2])])
            .note("Richer greens and deeper skies, with the highlights held back"),
        Look::general("Everyday — Soft", Everyday)
            .contrast(-0.2)
            .tones(-0.25, 0.25, 0.0, 0.05)
            .note("Low contrast with open shadows and calm highlights — a flat base to grade from"),
        Look::general("Everyday — Night", Everyday)
            .sat(-0.1)
            .contrast(0.1)
            .tones(-0.4, 0.0, 0.0, -0.1)
            .warmth(-0.1, 0.0)
            .note("City lights at night: tames bright signs and lamps, keeps the dark dark, cools the sodium glow"),
        // --- Film -----------------------------------------------------------
        Look::general("Film — Portra-style", Film)
            .sat(-0.05)
            .contrast(-0.1)
            .tones(-0.1, 0.1, 0.0, 0.0)
            .warmth(0.06, 0.0)
            .grade([0.006, 0.005, 0.004], [1.0; 3], [1.0; 3])
            .bands(&[("Orange", [-0.1, 0.1, 0.1]), ("Green", [-0.2, -0.2, 0.0]), ("Blue", [-0.1, -0.15, 0.0])])
            .note("Kodak Portra 400 as it is usually remembered: warm, flattering skin, soft contrast, muted greens"),
        Look::general("Film — Ektar-style", Film)
            .sat(0.3)
            .contrast(0.15)
            .bands(&[("Red", [0.0, 0.2, -0.1]), ("Orange", [0.0, 0.1, 0.0]), ("Blue", [0.1, 0.25, -0.15])])
            .note("Kodak Ektar 100: fine-grained and saturated, strong reds and blues"),
        Look::general("Film — Velvia-style", Film)
            .sat(0.45)
            .contrast(0.25)
            .tones(-0.1, 0.0, 0.0, -0.05)
            .warmth(0.0, 0.05)
            .bands(&[
                ("Green", [0.05, 0.25, -0.15]),
                ("Blue", [0.0, 0.3, -0.25]),
                ("Purple", [0.0, 0.2, 0.0]),
                ("Magenta", [0.0, 0.2, 0.0]),
            ])
            .note("Fujichrome Velvia 50 slide film: very saturated, deep blues and greens, a hint of magenta"),
        Look::general("Film — Kodachrome-style", Film)
            .sat(0.2)
            .contrast(0.25)
            .tones(0.0, 0.0, 0.0, -0.1)
            .warmth(0.04, 0.0)
            .bands(&[
                ("Red", [0.0, 0.25, -0.2]),
                ("Yellow", [-0.1, 0.15, 0.0]),
                ("Green", [-0.1, -0.1, -0.1]),
                ("Blue", [-0.15, 0.1, -0.2]),
            ])
            .note("Kodachrome: dense, contrasty slides with deep reds and cyan-leaning blues"),
        Look::general("Film — Fuji 400H-style", Film)
            .sat(-0.1)
            .contrast(-0.1)
            .warmth(-0.03, -0.04)
            .grade([0.004, 0.007, 0.008], [1.0; 3], [1.0; 3])
            .bands(&[("Orange", [0.0, -0.1, 0.1]), ("Green", [0.3, -0.1, 0.1]), ("Blue", [-0.1, -0.2, 0.1])])
            .note("Fuji Pro 400H: airy pastels, minty greens, cool lifted shadows"),
        Look::general("Film — Gold-style", Film)
            .sat(0.1)
            .contrast(0.05)
            .warmth(0.12, 0.0)
            .grade([0.006, 0.005, 0.002], [1.0; 3], [1.0; 3])
            .bands(&[("Yellow", [0.0, 0.2, 0.05]), ("Blue", [0.0, -0.1, 0.0])])
            .note("Kodak Gold 200: warm, golden consumer film"),
        Look::general("Film — CineStill 800T-style", Film)
            .contrast(0.1)
            .warmth(-0.25, 0.02)
            .grade([0.0, 0.004, 0.008], [1.0; 3], [1.03, 1.0, 0.97])
            .bands(&[("Red", [0.0, 0.2, 0.1]), ("Aqua", [0.0, 0.2, 0.0])])
            .note("Tungsten-balanced cinema film: cool cast, teal shadows, glowing reds — best for night scenes"),
        // --- Black & white --------------------------------------------------
        Look::general("B&W — Neutral", BlackWhite)
            .mono(LUMA)
            .note("Black and white from the photo's luminance"),
        Look::general("B&W — High contrast", BlackWhite)
            .mono(LUMA)
            .contrast(0.35)
            .tones(0.0, 0.0, 0.1, -0.15)
            .note("Punchy, grainy-film contrast (think Tri-X pushed a stop)"),
        Look::general("B&W — Red filter", BlackWhite)
            .mono(RED_FILTER)
            .contrast(0.15)
            .note("As if shot through a red filter: dark dramatic skies, light skin and foliage"),
        Look::general("B&W — Noir", BlackWhite)
            .mono(RED_FILTER)
            .contrast(0.5)
            .tones(-0.2, -0.2, 0.15, -0.3)
            .note("Hard light and deep blacks"),
        Look::general("B&W — Soft matte", BlackWhite)
            .mono(LUMA)
            .contrast(-0.15)
            .grade([0.01; 3], [1.0; 3], [0.97; 3])
            .note("Gentle black and white with faded blacks"),
        Look::general("B&W — Sepia", BlackWhite)
            .mono(LUMA)
            .contrast(0.05)
            .grade([0.006, 0.003, 0.0], [1.0; 3], [1.04, 0.97, 0.82])
            .note("Warm brown toning, like an old print"),
        Look::general("B&W — Selenium", BlackWhite)
            .mono(LUMA)
            .contrast(0.1)
            .grade([0.003, 0.0, 0.006], [1.0; 3], [0.99, 1.0, 1.02])
            .note("Cool purple-black shadows, like a selenium-toned print"),
        // --- Creative -------------------------------------------------------
        Look::general("Creative — Faded matte", Creative)
            .sat(-0.2)
            .contrast(-0.15)
            .tones(0.0, 0.0, -0.2, 0.0)
            .grade([0.012; 3], [1.0; 3], [0.97; 3])
            .note("Lifted blacks, soft highlights, quiet colour"),
        Look::general("Creative — Teal & orange", Creative)
            .sat(0.1)
            .contrast(0.1)
            .grade([0.0, 0.006, 0.01], [1.0; 3], [1.04, 1.0, 0.95])
            .bands(&[
                ("Orange", [0.0, 0.2, 0.05]),
                ("Yellow", [-0.3, 0.0, 0.0]),
                ("Green", [0.4, -0.3, 0.0]),
                ("Aqua", [0.0, 0.2, -0.1]),
                ("Blue", [-0.3, 0.1, -0.1]),
            ])
            .note("The blockbuster grade: warm skin against teal shadows and backgrounds"),
        Look::general("Creative — Bleach bypass", Creative)
            .sat(-0.55)
            .contrast(0.35)
            .tones(0.1, 0.0, 0.0, -0.1)
            .note("Silvery, desaturated and contrasty, as if the bleach step was skipped"),
        Look::general("Creative — Cross-process", Creative)
            .sat(0.25)
            .contrast(0.2)
            .grade([0.0, 0.003, 0.01], [1.0, 1.03, 0.96], [1.02, 1.03, 0.9])
            .note("Slide film in negative chemistry: yellow-green highlights, blue shadows"),
        Look::general("Creative — Warm vintage", Creative)
            .sat(-0.15)
            .contrast(-0.05)
            .warmth(0.15, 0.0)
            .grade([0.012, 0.009, 0.005], [1.0; 3], [1.0, 0.98, 0.92])
            .note("Warm, faded, like a print that has sat in an album"),
        Look::general("Creative — Cool & moody", Creative)
            .sat(-0.25)
            .contrast(0.1)
            .tones(-0.2, -0.15, 0.0, 0.0)
            .warmth(-0.12, 0.0)
            .grade([0.0, 0.003, 0.006], [1.0; 3], [1.0; 3])
            .note("Cold, muted and darker — overcast and rainy days"),
        Look::general("Creative — Golden hour", Creative)
            .sat(0.15)
            .tones(0.0, 0.1, 0.0, 0.0)
            .warmth(0.2, 0.03)
            .grade([0.0; 3], [1.0; 3], [1.04, 1.0, 0.94])
            .note("Late-afternoon warmth, even when the light was not"),
        // --- Infrared (full-spectrum GF1) -----------------------------------
        Look::builtin("Full spectrum — natural", IDENTITY3, Some(ColourMode::Camera))
            .wb(WB_FULL_SPECTRUM)
            .note("No filter. Normal colours with a warm IR glow on foliage; white balance set for daylight"),
        Look::builtin("Full spectrum — vivid", [[2.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], ir)
            .wb(WB_FULL_SPECTRUM)
            .note("No filter. Boosts red's IR excess: cyan sky, copper-red trees"),
        Look::builtin("Full spectrum — Aerochrome", [[2.5, -1.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], ir)
            .wb(WB_FULL_SPECTRUM)
            .finish(0.1, 0.25, 0.0, &[
                ("Red", [-0.2, 0.4, -0.2]),
                ("Orange", [-0.8, 0.3, -0.15]),
                ("Blue", [0.0, -0.2, 0.0]),
                ("Aqua", [0.5, 0.0, 0.0]),
            ])
            .note("No filter. Aerochrome-style: red foliage, natural blue sky. Weaker IR than with the #16, so foliage is red rather than crimson"),
        Look::builtin("Full spectrum — IR mono", [IR_FULL; 3], ir)
            .wb(WB_FULL_SPECTRUM)
            .note("No filter. Black and white from red minus green: dark sky, glowing trees"),
        // Aerochrome effects: the IR → red, red → green, green → blue mix,
        // cloud-neutral white balance, then slide-film contrast and per-colour
        // work — red pulled towards crimson and darkened, purple and aqua
        // pulled into blue so the sky is blue rather than violet or teal,
        // magenta nudged back to red so grass is not neon pink. Tuned by eye
        // on P1320278 (f/3.9) and P1320294 (f/20) through the real export.
        Look::builtin("#16 orange — Aerochrome", AERO_16, ir)
            .wb(WB_ORANGE_16_CLOUDS)
            .finish(0.3, 0.3, 0.0, &[
                ("Red", [-0.2, 0.4, -0.25]),
                ("Magenta", [0.4, 0.0, -0.2]),
                ("Purple", [-1.0, 0.2, -0.2]),
                ("Blue", [0.0, 0.5, -0.3]),
                ("Aqua", [0.8, 0.4, -0.25]),
            ])
            .note("Tiffen #16 orange. Kodak Aerochrome: crimson foliage, blue sky, white clouds. Set exposure per photo"),
        Look::builtin("#16 orange — Aerochrome deep", AERO_16, ir)
            .wb(WB_ORANGE_16_CLOUDS)
            .finish(0.45, 0.45, 0.0, &[
                ("Red", [-0.2, 0.5, -0.45]),
                ("Magenta", [0.5, 0.0, -0.3]),
                ("Purple", [-1.0, 0.3, -0.3]),
                ("Blue", [0.0, 0.7, -0.45]),
                ("Aqua", [0.9, 0.5, -0.4]),
            ])
            .note("Tiffen #16 orange. Aerochrome with more contrast: darker reds, deeper sky"),
        Look::builtin("#16 orange — Aerochrome faded", AERO_16, ir)
            .wb(WB_ORANGE_16_CLOUDS)
            .finish(0.15, 0.0, 0.35, &[
                ("Red", [-0.5, 0.1, 0.05]),
                ("Magenta", [0.2, 0.0, 0.0]),
                ("Purple", [-0.8, 0.0, 0.0]),
                ("Blue", [0.0, 0.2, -0.1]),
                ("Aqua", [0.6, 0.2, 0.0]),
            ])
            .note("Tiffen #16 orange. Aerochrome like an old scan: pink foliage, lifted blacks, soft"),
        Look::builtin("#16 orange — channel mix only", AERO_16, ir)
            .wb(WB_ORANGE_16)
            .note("Tiffen #16 orange. Just the Aerochrome channel mix, no tone or colour work — a base to build on"),
        Look::builtin("#16 orange — gold & blue", gold, ir)
            .wb(WB_ORANGE_16)
            .note("Tiffen #16 orange. Golden foliage under a blue sky"),
        Look::builtin("#16 orange — IR mono", [IR_16; 3], ir)
            .wb(WB_ORANGE_16)
            .note("Tiffen #16 orange. Black and white from the IR estimate: white foliage"),
        // IR → red, red → green, green → blue (see the module docs).
        Look::builtin("IR Chrome — Aerochrome", [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ir)
            .note("IR Chrome filter (untested — no shots yet). Classic Aerochrome"),
        // Same idea, but blue keeps half the IR, so foliage leans pink-magenta
        // instead of pure red — the look of later Aerochrome scans.
        Look::builtin("IR Chrome — Aerochrome pink", [[0.0, 0.0, 1.0], [0.7, 0.3, 0.0], [0.0, 0.5, 0.5]], ir)
            .note("IR Chrome filter (untested). Pink-magenta foliage"),
        // The common full-spectrum swap: blue skies, golden/white foliage.
        Look::builtin("IR Chrome — red–blue swap", [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], ir)
            .note("IR Chrome filter (untested). Blue sky, golden foliage"),
        // Black and white from the IR-only channel: white foliage, dark sky.
        Look::builtin("IR Chrome — IR mono", [[0.0, 0.0, 1.0]; 3], ir)
            .note("IR Chrome filter (untested). Black and white from the blue (IR) channel"),
    ]
}

/// A built-in by its current or former name.
pub fn find_builtin(name: &str) -> Option<Look> {
    let name = RENAMED.iter().find(|(old, _)| *old == name).map_or(name, |(_, new)| new);
    builtin().into_iter().find(|l| l.name == name)
}

/// A small sRGB picture of `look` applied on top of `p`, for the look
/// picker. `src` should be a box-downsampled copy of the photo. Detail is off
/// (noise reduction and sharpening mean nothing at thumbnail size). Returns
/// the size and RGBA8 pixels, row-major — the size follows the crop and
/// rotation, so it is not the source's.
pub fn thumbnail(src: &crate::develop::DevelopedImage, p: &AdjustParams, look: &Look) -> ([usize; 2], Vec<u8>) {
    let mut q = *p;
    look.apply(&mut q);
    q.nr_luma = 0.0;
    q.nr_chroma = 0.0;
    q.sharpen_amount = 0.0;
    let r = crate::export::render_cpu(src, &q);
    let rgba = r.data.chunks(3).flat_map(|c| [(c[0] >> 8) as u8, (c[1] >> 8) as u8, (c[2] >> 8) as u8, 255]).collect();
    ([r.width, r.height], rgba)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{Colour, ImageColour, to_working};

    #[test]
    fn builtin_looks_keep_neutrals_neutral() {
        for look in builtin() {
            for (i, row) in look.mixer.iter().enumerate() {
                let sum: f32 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-6, "{} row {i} sums to {sum}", look.name);
            }
        }
    }

    #[test]
    fn builtin_names_are_unique_and_none_comes_first() {
        let looks = builtin();
        assert_eq!(looks[0].name, "None");
        assert_eq!(looks[0].mixer, IDENTITY3);
        let mut names: Vec<_> = looks.iter().map(|l| l.name.clone()).collect();
        names.dedup();
        assert_eq!(names.len(), looks.len());
    }

    /// Foliage behind an IR Chrome filter: lots of IR (blue photosites),
    /// some red, little green. Classic Aerochrome must render it red.
    #[test]
    fn aerochrome_turns_ir_bright_foliage_red() {
        let mut p = AdjustParams::default();
        find_builtin("IR Chrome — Aerochrome").unwrap().apply(&mut p);
        let col = Colour::resolve(&ImageColour::LINEAR_SRGB, &p);
        let foliage = to_working(&col, [0.35, 0.2, 0.6], 0.0);
        assert!(foliage[0] > foliage[1] && foliage[0] > foliage[2], "{foliage:?}");
        // Sky: little IR, some blue-green visible light → stays cool.
        let sky = to_working(&col, [0.2, 0.35, 0.1], 0.0);
        assert!(sky[2] > sky[0], "{sky:?}");
    }

    #[test]
    fn ir_chrome_looks_keep_white_balance_and_saturation() {
        let mut p = AdjustParams { wb_gains: Some([0.6, 1.0, 3.0]), saturation: 0.3, ..Default::default() };
        find_builtin("IR Chrome — Aerochrome").unwrap().apply(&mut p);
        assert_eq!(p.wb_gains, Some([0.6, 1.0, 3.0]));
        assert_eq!(p.saturation, 0.3);
        assert_eq!(p.colour_mode, ColourMode::SensorRgb);
    }

    #[test]
    fn tuned_looks_set_white_balance_and_only_aerochrome_sets_tone() {
        for look in builtin().iter().filter(|l| l.name.starts_with("Full spectrum") || l.name.starts_with("#16")) {
            let mut p = AdjustParams { saturation: 0.3, exposure_ev: 0.7, ..Default::default() };
            look.apply(&mut p);
            assert!(p.wb_gains.is_some(), "{}", look.name);
            assert_eq!(p.exposure_ev, 0.7, "{}: exposure stays per photo", look.name);
            if !look.name.contains("Aerochrome") {
                assert_eq!(p.saturation, 0.3, "{}", look.name);
            }
            assert!(!look.note.is_empty(), "{} needs a note saying which filter it is for", look.name);
        }
    }

    #[test]
    fn thumbnail_shows_the_look_not_the_current_params() {
        // A reddish-IR pixel; Sensor RGB, identity: stays red. Through IR mono
        // it must come out grey.
        let img = crate::develop::DevelopedImage::linear_srgb(2, 1, vec![0.5, 0.2, 0.3, 0.5, 0.2, 0.3]);
        let p = AdjustParams { colour_mode: ColourMode::SensorRgb, ..Default::default() };
        let (_, plain) = thumbnail(&img, &p, &builtin()[0]);
        assert!(plain[0] > plain[1], "{plain:?}");
        let (_, mono) = thumbnail(&img, &p, &find_builtin("IR Chrome — IR mono").unwrap());
        assert_eq!((mono.len(), mono[0], mono[3]), (8, mono[1], 255));
        assert_eq!(mono[1], mono[2]);
    }

    /// The crash this caught: a straightened crop shrinks the render, and
    /// the GUI built its texture from the source size instead.
    #[test]
    fn thumbnail_size_follows_the_crop() {
        let img = crate::develop::DevelopedImage::linear_srgb(40, 30, vec![0.2; 40 * 30 * 3]);
        let p = AdjustParams { rotation: 1, crop: [0.0, 0.0, 0.5, 1.0], ..Default::default() };
        let ([w, h], rgba) = thumbnail(&img, &p, &builtin()[0]);
        assert_eq!((w, h), (15, 40));
        assert_eq!(rgba.len(), w * h * 4);
    }

    #[test]
    fn old_names_still_resolve() {
        for (old, new) in RENAMED {
            assert_eq!(find_builtin(old).map(|l| l.name), Some(new.to_string()));
        }
        assert!(find_builtin("No such look").is_none());
    }

    /// Region means (camera RGB, 1/8-scale averages) measured on real frames
    /// on Oct 2 2026, so the tuned looks are pinned to what they were tuned
    /// on. Order: sky, grass, maple, asphalt.
    const ORANGE_16: [[f32; 3]; 4] = [
        [0.6721, 0.5112, 0.1023], // P1320278 sky
        [0.4164, 0.2782, 0.0821], // grass
        [0.2641, 0.1605, 0.0574], // maple
        [0.3042, 0.1922, 0.0422], // asphalt
    ];
    const UNFILTERED: [[f32; 3]; 4] = [
        [0.7612, 0.9036, 0.8310], // P1320286 sky
        [0.5076, 0.4243, 0.1970], // grass
        [0.3186, 0.2346, 0.1325], // maple
        [0.3238, 0.3033, 0.1951], // asphalt
    ];

    fn render(name: &str, cam: [f32; 3]) -> [f32; 3] {
        let mut p = AdjustParams::default();
        find_builtin(name).unwrap().apply(&mut p);
        // Real camera matrix not needed: the tuned IR looks are Sensor RGB.
        to_working(&Colour::resolve(&ImageColour::LINEAR_SRGB, &p), cam, 0.0)
    }

    #[test]
    fn orange_16_aerochrome_turns_foliage_red_and_sky_blue() {
        let [sky, grass, maple, road] = ORANGE_16.map(|c| render("#16 orange — channel mix only", c));
        for f in [grass, maple] {
            assert!(f[0] > f[1] * 1.2 && f[0] > f[2] * 1.2, "foliage {f:?}");
        }
        assert!(sky[2] > sky[0], "sky {sky:?}");
        assert!(road.iter().all(|&v| (v - road[1]).abs() < road[1] * 0.25), "asphalt near neutral {road:?}");
    }

    /// The finished effect, end to end through `apply`: crimson foliage,
    /// blue sky, and clouds close to white.
    #[test]
    fn orange_16_aerochrome_effect_keeps_clouds_white() {
        let full = |name: &str, cam: [f32; 3]| {
            let mut p = AdjustParams::default();
            find_builtin(name).unwrap().apply(&mut p);
            crate::pipeline::apply(&p, to_working(&Colour::resolve(&ImageColour::LINEAR_SRGB, &p), cam, 0.0))
        };
        for name in ["#16 orange — Aerochrome", "#16 orange — Aerochrome deep"] {
            let [sky, grass, maple, _] = ORANGE_16.map(|c| full(name, c));
            for f in [grass, maple] {
                assert!(f[0] > 2.0 * f[1] && f[0] > 1.5 * f[2], "{name}: foliage should be crimson {f:?}");
            }
            assert!(sky[2] > sky[0], "{name}: sky {sky:?}");
            // Brightest 20% of the sky band in P1320278 — the clouds.
            let cloud = full(name, [0.9957, 0.8051, 0.173]);
            // Pink/violet = green well below both red and blue. The asphalt
            // white balance gave (1.01, 0.74, 1.17) here; a blue-white cloud
            // is fine.
            let pink = cloud[1] < 0.85 * cloud[0].min(cloud[2]);
            assert!(!pink, "{name}: clouds should be white, not pink {cloud:?}");
        }
    }

    #[test]
    fn orange_16_gold_and_blue_does_what_it_says() {
        let [sky, grass, maple, _] = ORANGE_16.map(|c| render("#16 orange — gold & blue", c));
        for f in [grass, maple] {
            assert!(f[0] > f[1] && f[1] > f[2], "foliage should be gold {f:?}");
        }
        assert!(sky[2] > sky[0] * 1.1, "sky should be blue {sky:?}");
    }

    #[test]
    fn ir_mono_looks_put_foliage_above_asphalt_and_sky() {
        for (name, regions) in [("#16 orange — IR mono", ORANGE_16), ("Full spectrum — IR mono", UNFILTERED)] {
            let [sky, grass, maple, road] = regions.map(|c| render(name, c)[0]);
            // Scale-free: compare against asphalt.
            assert!(grass > road * 1.3 && maple > road * 1.3, "{name}: grass {grass} maple {maple} road {road}");
            assert!(sky / road < grass / road, "{name}: sky {sky} should be darker than foliage");
        }
    }

    #[test]
    fn a_saved_look_round_trips_the_colour_settings_only() {
        let src = AdjustParams {
            wb_gains: Some([0.5, 1.0, 2.5]),
            colour_mode: ColourMode::SensorRgb,
            mixer: [[0.0, 0.2, 0.8], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            saturation: 0.25,
            hsl: [[0.2, -0.1, 0.3]; 8],
            exposure_ev: 1.5,
            ..Default::default()
        };
        let look = Look::from_params("  My IR  ", &src);
        assert_eq!(look.name, "My IR");
        let json = serde_json::to_string(&look).unwrap();
        let look: Look = serde_json::from_str(&json).unwrap();

        let mut dst = AdjustParams { exposure_ev: -0.5, ..Default::default() };
        look.apply(&mut dst);
        assert_eq!(dst.wb_gains, src.wb_gains);
        assert_eq!(dst.colour_mode, src.colour_mode);
        assert_eq!(dst.mixer, src.mixer);
        assert_eq!(dst.saturation, src.saturation);
        assert_eq!(dst.hsl, src.hsl);
        assert_eq!(dst.exposure_ev, -0.5, "exposure is per photo, not part of a look");
        assert!(look.is_applied(&dst));
    }

    #[test]
    fn saved_looks_from_before_hsl_still_load() {
        let json = r#"{"name":"Old","colour_mode":"sensor_rgb","wb":"keep","mixer":[[1,0,0],[0,1,0],[0,0,1]],"saturation":0.1}"#;
        let look: Look = serde_json::from_str(json).unwrap();
        assert_eq!(look.hsl, None);
        let mut p = AdjustParams { hsl: [[0.5; 3]; 8], ..Default::default() };
        look.apply(&mut p);
        assert_eq!(p.hsl, [[0.5; 3]; 8], "a look without HSL leaves the photo's alone");
    }

    #[test]
    fn as_shot_look_clears_custom_white_balance() {
        let look = Look::from_params("plain", &AdjustParams::default());
        assert_eq!(look.wb, LookWb::AsShot);
        let mut p = AdjustParams { wb_gains: Some([2.0, 1.0, 2.0]), ..Default::default() };
        look.apply(&mut p);
        assert_eq!(p.wb_gains, None);
    }

    #[test]
    fn is_applied_tracks_the_mixer() {
        let none = find_builtin("None").unwrap();
        let ir = find_builtin("IR Chrome — red–blue swap").unwrap();
        let mut p = AdjustParams::default();
        assert!(none.is_applied(&p));
        ir.apply(&mut p);
        assert!(ir.is_applied(&p) && !none.is_applied(&p));
        p.mixer[0][0] = 0.1;
        assert!(!ir.is_applied(&p));
    }

    fn general_looks() -> Vec<Look> {
        builtin().into_iter().filter(|l| l.group != LookGroup::Infrared && l.name != "None").collect()
    }

    /// Every look field moved off neutral, plus things looks never touch.
    fn messy() -> AdjustParams {
        AdjustParams {
            mixer: [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            colour_mode: ColourMode::SensorRgb,
            wb_gains: Some([0.7, 1.0, 2.0]),
            saturation: -0.6,
            hsl: [[0.3, 0.3, 0.3]; 8],
            contrast: 0.7,
            blacks: 0.4,
            tone_curve: 0.3,
            temp: -0.5,
            tint: 0.5,
            highlights: 0.6,
            shadows: -0.6,
            whites: -0.5,
            lift: [0.1, 0.0, -0.1],
            gamma: [1.3, 0.8, 1.0],
            gain: [0.7, 1.2, 1.0],
            exposure_ev: 0.8,
            sharpen_amount: 0.5,
            ..Default::default()
        }
    }

    /// Switching from any state to an everyday/film/B&W/creative look gives
    /// the same result as applying it to an untouched photo — nothing of the
    /// previous look (a split tone, an HSL shift) survives the switch.
    #[test]
    fn general_looks_replace_the_previous_look_completely() {
        for look in general_looks() {
            let (mut a, mut b) = (messy(), AdjustParams::default());
            look.apply(&mut a);
            look.apply(&mut b);
            let strip = |p: &AdjustParams| AdjustParams {
                wb_gains: None,
                exposure_ev: 0.0,
                sharpen_amount: 0.0,
                ..*p
            };
            assert_eq!(strip(&a), strip(&b), "{}", look.name);
            assert_eq!(a.colour_mode, ColourMode::Camera, "{}: camera colours", look.name);
            assert_eq!(a.wb_gains, messy().wb_gains, "{}: white balance is the photo's", look.name);
            assert_eq!(a.exposure_ev, 0.8, "{}: exposure is the photo's", look.name);
        }
    }

    #[test]
    fn none_clears_every_look_field_but_keeps_white_balance_and_mode() {
        let mut p = messy();
        find_builtin("None").unwrap().apply(&mut p);
        let d = AdjustParams::default();
        assert_eq!(
            (p.mixer, p.saturation, p.hsl, p.contrast, p.blacks, p.tone_curve),
            (d.mixer, d.saturation, d.hsl, d.contrast, d.blacks, d.tone_curve)
        );
        assert_eq!((p.temp, p.tint, p.highlights, p.shadows, p.whites), (0.0, 0.0, 0.0, 0.0, 0.0));
        assert_eq!((p.lift, p.gamma, p.gain), (d.lift, d.gamma, d.gain));
        assert_eq!((p.wb_gains, p.colour_mode), (messy().wb_gains, ColourMode::SensorRgb));
    }

    /// Render one linear-sRGB value through a look, from neutral params.
    fn through(look: &Look, rgb: [f32; 3]) -> [f32; 3] {
        let mut p = AdjustParams::default();
        look.apply(&mut p);
        crate::pipeline::apply(&p, to_working(&Colour::resolve(&ImageColour::LINEAR_SRGB, &p), rgb, 0.0))
    }

    fn luma(c: [f32; 3]) -> f32 {
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    }

    /// No built-in may produce NaN, and no look for ordinary photos may
    /// invert tones, from black to well past the clip level — a bad lift/
    /// gamma/curve combination shows up here. (The infrared looks are judged
    /// on real IR region means instead; on a plain grey ramp the #16
    /// Aerochrome effect dips a hair below zero next to black.)
    #[test]
    fn every_builtin_keeps_a_grey_ramp_ordered_and_finite() {
        for look in builtin() {
            let ordered = look.group != LookGroup::Infrared;
            let mut last = f32::NEG_INFINITY;
            for i in 0..=300 {
                let x = i as f32 / 200.0;
                let c = through(&look, [x; 3]);
                assert!(c.iter().all(|v| v.is_finite()), "{} at {x}: {c:?}", look.name);
                let y = luma(c);
                if ordered {
                    assert!(y >= last - 1e-6, "{}: tones invert at {x} ({y} < {last})", look.name);
                }
                last = y;
            }
        }
    }

    /// Looks with no warmth or colour grade leave greys grey: saturation,
    /// HSL and the B&W mixers never colour a neutral.
    #[test]
    fn untinted_looks_keep_greys_neutral() {
        let neutral_grade = |l: &Look| {
            l.temp == Some(0.0)
                && l.tint == Some(0.0)
                && l.lift.is_none_or(|v| v[0] == v[1] && v[1] == v[2])
                && l.gamma.is_none_or(|v| v[0] == v[1] && v[1] == v[2])
                && l.gain.is_none_or(|v| v[0] == v[1] && v[1] == v[2])
        };
        let looks: Vec<Look> = general_looks().into_iter().filter(neutral_grade).collect();
        assert!(looks.iter().any(|l| l.name == DEFAULT_LOOK));
        for look in looks {
            for x in [0.01, 0.1, 0.18, 0.5, 0.9] {
                let c = through(&look, [x; 3]);
                let spread = c.iter().fold(0f32, |m, v| m.max((v - c[1]).abs()));
                assert!(spread < 1e-4, "{} tints grey {x}: {c:?}", look.name);
            }
        }
    }

    #[test]
    fn black_and_white_looks_are_monochrome_or_evenly_toned() {
        let colours = [[0.4, 0.1, 0.05], [0.05, 0.3, 0.08], [0.06, 0.1, 0.5], [0.3, 0.3, 0.02]];
        for look in builtin().into_iter().filter(|l| l.group == LookGroup::BlackWhite) {
            let toned = look.lift != Some([0.0; 3]) || look.gain != Some([1.0; 3]);
            for c in colours {
                let o = through(&look, c);
                if toned {
                    // One tone everywhere: the channel order never changes.
                    let order = |v: [f32; 3]| (v[0] >= v[1], v[1] >= v[2]);
                    assert_eq!(order(o), order(through(&look, [0.18; 3])), "{}: {c:?} -> {o:?}", look.name);
                } else {
                    assert!((o[0] - o[1]).abs() < 1e-5 && (o[1] - o[2]).abs() < 1e-5, "{}: {c:?} -> {o:?}", look.name);
                }
            }
        }
        // The red filter darkens a blue sky more than plain luminance does.
        let sky = [0.08, 0.15, 0.45];
        let plain = through(&find_builtin("B&W — Neutral").unwrap(), sky)[0];
        let red = through(&find_builtin("B&W — Red filter").unwrap(), sky)[0];
        assert!(red < plain * 0.8, "red filter {red} vs neutral {plain}");
    }

    /// The default look puts a raw's mid-tones where the cameras' own JPEGs
    /// do (measured; see `pipeline::filmic`).
    #[test]
    fn natural_matches_the_camera_mid_tones() {
        let natural = find_builtin(DEFAULT_LOOK).unwrap();
        for (x, cam) in [(0.045, 0.067), (0.106, 0.238), (0.185, 0.426), (0.43, 0.735)] {
            let y = through(&natural, [x; 3])[1];
            assert!((y / cam).log2().abs() < 0.2, "{x}: {y} vs camera {cam}");
        }
    }

    #[test]
    fn every_builtin_has_a_note_and_a_group_in_picker_order() {
        let looks = builtin();
        for l in &looks {
            assert!(!l.note.is_empty(), "{} needs a note", l.name);
            assert_ne!(l.group, LookGroup::Saved, "{} is a built-in", l.name);
        }
        // Grouped contiguously, in `LookGroup::ALL` order.
        let order: Vec<usize> =
            looks.iter().map(|l| LookGroup::ALL.iter().position(|g| *g == l.group).unwrap()).collect();
        assert!(order.windows(2).all(|w| w[0] <= w[1]), "{order:?}");
        assert!(looks.len() >= 40, "{} looks", looks.len());
    }

    #[test]
    fn saved_looks_carry_tone_and_grade_and_load_without_them() {
        let src = AdjustParams { tone_curve: 0.7, temp: 0.2, lift: [0.01, 0.0, 0.02], highlights: -0.3, ..Default::default() };
        let look = Look::from_params("Mine", &src);
        let back: Look = serde_json::from_str(&serde_json::to_string(&look).unwrap()).unwrap();
        assert_eq!(back.group, LookGroup::Saved);
        let mut dst = messy();
        back.apply(&mut dst);
        assert_eq!((dst.tone_curve, dst.temp, dst.lift, dst.highlights), (0.7, 0.2, [0.01, 0.0, 0.02], -0.3));
        // A look saved before these fields existed leaves them alone.
        let old: Look = serde_json::from_str(
            r#"{"name":"Old","colour_mode":"camera","wb":"keep","mixer":[[1,0,0],[0,1,0],[0,0,1]],"saturation":0.1}"#,
        )
        .unwrap();
        let mut p = messy();
        old.apply(&mut p);
        assert_eq!((p.tone_curve, p.lift, p.temp), (messy().tone_curve, messy().lift, messy().temp));
    }
}
