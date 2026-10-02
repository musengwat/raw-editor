//! Colour looks: a channel mixer plus, optionally, the colour settings that go
//! with it (colour mode, white balance, saturation).
//!
//! Built-in looks are infrared false-colour starting points for the
//! full-spectrum GF1, grouped by the filter on the lens, because what each
//! photosite records depends entirely on the filter.
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Look {
    pub name: String,
    /// `None` leaves the photo's colour mode as it is.
    pub colour_mode: Option<ColourMode>,
    pub wb: LookWb,
    pub mixer: [[f32; 3]; 3],
    /// `None` leaves the photo's saturation as it is.
    pub saturation: Option<f32>,
    /// What the look is for (filter, what it does), shown as a tooltip.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
}

impl Look {
    fn builtin(name: &str, mixer: [[f32; 3]; 3], mode: Option<ColourMode>) -> Self {
        Self { name: name.into(), colour_mode: mode, wb: LookWb::Keep, mixer, saturation: None, note: String::new() }
    }

    fn wb(mut self, gains: [f32; 3]) -> Self {
        self.wb = LookWb::Gains(gains);
        self
    }

    fn note(mut self, note: &str) -> Self {
        self.note = note.into();
        self
    }

    /// Capture the current colour settings of `p` as a look. Exposure, tone
    /// and detail are per photo and deliberately left out.
    pub fn from_params(name: &str, p: &AdjustParams) -> Self {
        Self {
            name: name.trim().into(),
            colour_mode: Some(p.colour_mode),
            wb: p.wb_gains.map_or(LookWb::AsShot, LookWb::Gains),
            mixer: p.mixer,
            saturation: Some(p.saturation),
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
    }

    /// True when applying this look to `p` would change nothing — how the UI
    /// knows which look is currently selected.
    pub fn is_applied(&self, p: &AdjustParams) -> bool {
        let mut q = *p;
        self.apply(&mut q);
        q == *p
    }
}

/// White balance that makes sunlit asphalt neutral, measured on the Oct 2
/// 2026 frames. Daylight changes it a little; the eyedropper on a road or a
/// grey card refines it without changing the look's mixer.
const WB_FULL_SPECTRUM: [f32; 3] = [0.94, 1.0, 1.55];
const WB_ORANGE_16: [f32; 3] = [0.63, 1.0, 4.56];

/// IR estimate behind the #16: blue minus its orange leak (see module docs).
const IR_16: [f32; 3] = [0.0, -1.5, 2.5];
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

/// The looks that ship with the editor. The first, "None", turns the mixer
/// off and is always present.
pub fn builtin() -> Vec<Look> {
    let ir = Some(ColourMode::SensorRgb);
    // Gold foliage, blue sky: R = IR, G = between IR and green, B = green
    // pushed away from IR.
    let gold = [
        IR_16,
        [0.0, -0.25, 1.25],                       // 0.5·IR + 0.5·G
        [0.0, 2.25, -1.25],                       // 1.5·G − 0.5·IR
    ];
    vec![
        Look::builtin("None", IDENTITY3, None),
        Look::builtin("Full spectrum — natural", IDENTITY3, Some(ColourMode::Camera))
            .wb(WB_FULL_SPECTRUM)
            .note("No filter. Normal colours with a warm IR glow on foliage; white balance set for daylight"),
        Look::builtin("Full spectrum — vivid", [[2.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], ir)
            .wb(WB_FULL_SPECTRUM)
            .note("No filter. Boosts red's IR excess: cyan sky, copper-red trees"),
        Look::builtin("Full spectrum — IR mono", [IR_FULL; 3], ir)
            .wb(WB_FULL_SPECTRUM)
            .note("No filter. Black and white from red minus green: dark sky, glowing trees"),
        Look::builtin("#16 orange — Aerochrome", [IR_16, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ir)
            .wb(WB_ORANGE_16)
            .note("Tiffen #16 orange. IR → red, red → green, green → blue: red foliage, blue sky"),
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
    fn tuned_looks_set_their_white_balance_but_keep_saturation() {
        for look in builtin().iter().filter(|l| l.name.starts_with("Full spectrum") || l.name.starts_with("#16")) {
            let mut p = AdjustParams { saturation: 0.3, ..Default::default() };
            look.apply(&mut p);
            assert!(p.wb_gains.is_some(), "{}", look.name);
            assert_eq!(p.saturation, 0.3, "{}", look.name);
            assert!(!look.note.is_empty(), "{} needs a note saying which filter it is for", look.name);
        }
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
        let [sky, grass, maple, road] = ORANGE_16.map(|c| render("#16 orange — Aerochrome", c));
        for f in [grass, maple] {
            assert!(f[0] > f[1] * 1.2 && f[0] > f[2] * 1.2, "foliage {f:?}");
        }
        assert!(sky[2] > sky[0], "sky {sky:?}");
        assert!(road.iter().all(|&v| (v - road[1]).abs() < road[1] * 0.25), "asphalt near neutral {road:?}");
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
        assert_eq!(dst.exposure_ev, -0.5, "exposure is per photo, not part of a look");
        assert!(look.is_applied(&dst));
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
        let looks = builtin();
        let mut p = AdjustParams::default();
        assert!(looks[0].is_applied(&p));
        looks[3].apply(&mut p);
        assert!(looks[3].is_applied(&p) && !looks[0].is_applied(&p));
        p.mixer[0][0] = 0.1;
        assert!(!looks[3].is_applied(&p));
    }
}
