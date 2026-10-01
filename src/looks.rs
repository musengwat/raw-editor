//! Colour looks: a channel mixer plus, optionally, the colour settings that go
//! with it (colour mode, white balance, saturation).
//!
//! Built-in looks are infrared false-colour starting points for the
//! full-spectrum GF1. They are derived from what the sensor sees rather than
//! tuned against real photos — there are no daylight shots through the IR
//! Chrome filter yet — so expect to adjust them, and save the result as your
//! own look.
//!
//! The reasoning: an IR Chrome-style filter blocks blue light and passes
//! green, red and near-infrared. Bayer dyes are close to transparent in the
//! infrared, so behind that filter the **blue** photosites record almost pure
//! IR, red records red + IR and green records green + IR. Kodak Aerochrome
//! shifted its layers the same way: IR → red, red → green, green → blue. So
//! the classic look is the rotation `R ← B, G ← R, B ← G` on white-balanced
//! sensor RGB. Every built-in row sums to 1, so anything white-balanced to
//! neutral (clouds, a grey card) stays neutral.

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
}

impl Look {
    fn builtin(name: &str, mixer: [[f32; 3]; 3], mode: Option<ColourMode>) -> Self {
        Self { name: name.into(), colour_mode: mode, wb: LookWb::Keep, mixer, saturation: None }
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

/// The looks that ship with the editor. The first, "None", turns the mixer
/// off and is always present.
pub fn builtin() -> Vec<Look> {
    let ir = Some(ColourMode::SensorRgb);
    vec![
        Look::builtin("None", IDENTITY3, None),
        // IR → red, red → green, green → blue (see the module docs).
        Look::builtin("Aerochrome classic", [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ir),
        // Same idea, but blue keeps half the IR, so foliage leans pink-magenta
        // instead of pure red — the look of later Aerochrome scans.
        Look::builtin("Aerochrome pink", [[0.0, 0.0, 1.0], [0.7, 0.3, 0.0], [0.0, 0.5, 0.5]], ir),
        // The common full-spectrum swap: blue skies, golden/white foliage.
        Look::builtin("Red–blue swap", [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], ir),
        // Black and white from the IR-only channel: white foliage, dark sky.
        Look::builtin("IR mono", [[0.0, 0.0, 1.0]; 3], ir),
    ]
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
        builtin().iter().find(|l| l.name == "Aerochrome classic").unwrap().apply(&mut p);
        let col = Colour::resolve(&ImageColour::LINEAR_SRGB, &p);
        let foliage = to_working(&col, [0.35, 0.2, 0.6], 0.0);
        assert!(foliage[0] > foliage[1] && foliage[0] > foliage[2], "{foliage:?}");
        // Sky: little IR, some blue-green visible light → stays cool.
        let sky = to_working(&col, [0.2, 0.35, 0.1], 0.0);
        assert!(sky[2] > sky[0], "{sky:?}");
    }

    #[test]
    fn builtin_looks_keep_white_balance_and_saturation() {
        let mut p = AdjustParams { wb_gains: Some([0.6, 1.0, 3.0]), saturation: 0.3, ..Default::default() };
        builtin()[1].apply(&mut p);
        assert_eq!(p.wb_gains, Some([0.6, 1.0, 3.0]));
        assert_eq!(p.saturation, 0.3);
        assert_eq!(p.colour_mode, ColourMode::SensorRgb);
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
