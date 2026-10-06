//! JSON sidecar persistence. Edits live in `<name>.<ext>.edit.json` next to
//! the photo (`P1320071.rw2.edit.json`); the photo itself is never touched.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::Config;
use crate::metadata::CameraIds;
use crate::pipeline::AdjustParams;

/// Where a photo's starting adjustments came from.
#[derive(Clone, Debug, PartialEq)]
pub enum Origin {
    /// Its own sidecar.
    Sidecar,
    /// No sidecar; its camera body is set to start in this look.
    BodyLook(String),
    /// No sidecar and no body look: the default look for new photos.
    DefaultLook(String),
    /// No sidecar, no body look, and either the default is "None" or the
    /// file is a TIFF (already a finished image): neutral.
    Neutral,
}

/// What opening a photo with no sidecar gives, with the flat field on if its
/// lens has one: the look assigned to its camera body, else the default look
/// (`Config::default_look`, normally "Everyday — Natural"), else neutral.
/// TIFFs always start neutral — they are finished images, and a tone curve
/// on top of one would apply a second curve. Also the baseline the GUI
/// compares against to decide whether a sidecar is needed, and what "Reset
/// all" returns to.
pub fn baseline(config: &Config, ids: &CameraIds, photo: &Path) -> (AdjustParams, Origin) {
    let mut p = AdjustParams::default();
    if crate::formats::is_tiff(photo) {
        return (p, Origin::Neutral);
    }
    p.flat_field = config.flat_for(ids.lens.as_deref()).is_some();
    if let Some(look) = config.body_look(ids.serial.as_deref()) {
        look.apply(&mut p);
        return (p, Origin::BodyLook(look.name));
    }
    match config.default_look() {
        Some(look) => {
            look.apply(&mut p);
            (p, Origin::DefaultLook(look.name))
        }
        None => (p, Origin::Neutral),
    }
}

/// The adjustments a photo opens (or batch-exports) with: its sidecar if it
/// has one, otherwise [`baseline`]. Shared by the GUI, the batch export and
/// the CLI so all three agree.
pub fn initial(raw: &Path, config: &Config, ids: &CameraIds) -> Result<(AdjustParams, Origin)> {
    Ok(match load(raw)? {
        Some(p) => (p, Origin::Sidecar),
        None => baseline(config, ids, raw),
    })
}

/// Edits as clipboard text (Copy edits): the params, tagged so Paste can tell
/// them apart from whatever else is on the clipboard.
pub fn to_clipboard(p: &AdjustParams) -> String {
    serde_json::json!({ CLIPBOARD_TAG: p }).to_string()
}

/// Edits back from clipboard text, or `None` if the text is not ours.
pub fn from_clipboard(text: &str) -> Option<AdjustParams> {
    let v: serde_json::Value = serde_json::from_str(text.trim()).ok()?;
    serde_json::from_value(v.get(CLIPBOARD_TAG)?.clone()).ok()
}

const CLIPBOARD_TAG: &str = "raw_editor_edits";

/// Suffix appended to the *full* raw filename, so `P1320071.rw2` pairs with
/// `P1320071.rw2.edit.json` (keeping the raw extension visible avoids
/// collisions between `foo.rw2` and a hypothetical `foo.jpg`).
pub const SUFFIX: &str = ".edit.json";

pub fn sidecar_path(raw: &Path) -> PathBuf {
    let mut s = raw.as_os_str().to_os_string();
    s.push(SUFFIX);
    PathBuf::from(s)
}

/// Read the sidecar for `raw`. `Ok(None)` means "no sidecar yet", which is the
/// normal case for a file opened for the first time. A malformed sidecar is an
/// error rather than a silent reset, so the user finds out before editing over
/// it.
pub fn load(raw: &Path) -> Result<Option<AdjustParams>> {
    let path = sidecar_path(raw);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let params = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(Some(params))
}

/// Write the sidecar for `raw`, via a temp file + rename so an interrupted
/// save cannot leave a truncated JSON behind.
pub fn save(raw: &Path, params: &AdjustParams) -> Result<()> {
    let path = sidecar_path(raw);
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(params)?;
    std::fs::write(&tmp, json).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("renaming into {}", path.display()))?;
    Ok(())
}

/// Delete the sidecar if there is one (used when the user resets to neutral).
pub fn remove(raw: &Path) -> Result<()> {
    match std::fs::remove_file(sidecar_path(raw)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("raw-editor-test-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("P1320071.rw2")
    }

    #[test]
    fn path_appends_to_the_full_filename() {
        let p = sidecar_path(Path::new("/photos/P1320071.rw2"));
        assert_eq!(p, Path::new("/photos/P1320071.rw2.edit.json"));
    }

    #[test]
    fn roundtrip_params_through_a_sidecar() {
        let raw = scratch("roundtrip");
        assert!(load(&raw).unwrap().is_none(), "no sidecar yet");

        let params = AdjustParams {
            wb_gains: Some([0.64, 1.0, 2.35]),
            colour_mode: crate::pipeline::ColourMode::SensorRgb,
            mixer: [[0.0, 0.1, 0.9], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            mixer_strength: 0.7,
            hsl: std::array::from_fn(|i| [0.1 * i as f32 - 0.3, -0.2, 0.05 * i as f32]),
            exposure_ev: 1.25,
            contrast: -0.4,
            temp: 0.3,
            tint: -0.15,
            saturation: 0.2,
            tone_curve: 0.75,
            highlights: -0.8,
            shadows: 0.65,
            whites: 0.1,
            blacks: -0.35,
            lift: [0.02, -0.01, 0.05],
            gamma: [1.15, 0.95, 1.0],
            gain: [1.05, 1.0, 0.92],
            sharpen_amount: 1.3,
            sharpen_radius: 1.8,
            sharpen_threshold: 0.004,
            nr_luma: 0.45,
            nr_chroma: 0.7,
            hotspot_amount: 0.6,
            hotspot_radius: 0.22,
            hotspot_centre: [0.48, 0.55],
            hotspot_balance: [1.0, 0.7, 0.4],
            flat_field: true,
            rotation: 3,
            straighten: -1.5,
            crop: [0.05, 0.1, 0.95, 0.85],
        };
        save(&raw, &params).unwrap();
        assert_eq!(load(&raw).unwrap().unwrap(), params);

        // No stray temp file survives a successful save.
        let tmp = sidecar_path(&raw).with_extension("json.tmp");
        assert!(!tmp.exists(), "{} should have been renamed away", tmp.display());

        remove(&raw).unwrap();
        assert!(load(&raw).unwrap().is_none());
        remove(&raw).unwrap(); // idempotent
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        // Sidecars written by an older build must still load (serde(default)).
        let raw = scratch("older");
        std::fs::write(sidecar_path(&raw), r#"{"exposure_ev": 0.5}"#).unwrap();
        let p = load(&raw).unwrap().unwrap();
        assert_eq!(p.exposure_ev, 0.5);
        assert_eq!(p.gamma, [1.0; 3]);
        assert_eq!(p.gain, [1.0; 3]);
    }

    #[test]
    fn malformed_sidecar_is_an_error() {
        let raw = scratch("malformed");
        std::fs::write(sidecar_path(&raw), "{ not json").unwrap();
        assert!(load(&raw).is_err());
    }

    #[test]
    fn initial_params_prefer_sidecar_then_body_look_then_default_look() {
        let raw = scratch("initial");
        let mut config = Config::default();
        config.body_looks.insert("F92".into(), "Red–blue swap".into());

        let body = |s: &str| CameraIds { serial: Some(s.into()), lens: None };
        let natural = Origin::DefaultLook(crate::looks::DEFAULT_LOOK.into());
        let (p, origin) = initial(&raw, &config, &body("XEL")).unwrap();
        assert_eq!(origin, natural);
        assert_eq!(p.tone_curve, 1.0, "the default look brings the camera tone curve");
        assert_eq!(initial(&raw, &config, &CameraIds::default()).unwrap().1, natural);

        // A default of "None" opens neutral.
        let neutral = Config { default_look: Some("None".into()), ..config.clone() };
        assert_eq!(initial(&raw, &neutral, &body("XEL")).unwrap(), (AdjustParams::default(), Origin::Neutral));

        let (p, origin) = initial(&raw, &config, &body("F92")).unwrap();
        // Stored under the built-in's pre-rename name; resolves to the new one.
        assert_eq!(origin, Origin::BodyLook("IR Chrome — red–blue swap".into()));
        assert_eq!(p.mixer, [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]);
        assert_eq!(p.colour_mode, crate::pipeline::ColourMode::SensorRgb);

        // A sidecar always wins — including a neutral one, which is how a
        // single photo opts out of its body's look.
        save(&raw, &AdjustParams::default()).unwrap();
        assert_eq!(initial(&raw, &config, &body("F92")).unwrap(), (AdjustParams::default(), Origin::Sidecar));
    }

    #[test]
    fn a_lens_with_a_flat_field_starts_with_it_on() {
        let mut config = Config::default();
        let lens = "LUMIX G 20/F1.7";
        config.flat_fields.insert(
            lens.into(),
            crate::flatfield::FlatField { lens: lens.into(), width: 1, height: 1, gains: vec![[1.0; 3]] },
        );
        let ids = |l: &str| CameraIds { serial: None, lens: Some(l.into()) };
        let raw = Path::new("/photos/P1.rw2");
        assert!(baseline(&config, &ids(lens), raw).0.flat_field);
        assert!(!baseline(&config, &ids("LUMIX G VARIO 7-14/F4.0"), raw).0.flat_field);
    }

    #[test]
    fn tiffs_start_neutral_whatever_the_settings() {
        let mut config = Config::default();
        config.body_looks.insert("F92".into(), "Red–blue swap".into());
        let ids = CameraIds { serial: Some("F92".into()), lens: None };
        for name in ["scan.tif", "P1_edit.TIFF"] {
            assert_eq!(baseline(&config, &ids, Path::new(name)), (AdjustParams::default(), Origin::Neutral), "{name}");
        }
        assert_ne!(baseline(&config, &ids, Path::new("P1.ARW")).1, Origin::Neutral);
    }

    #[test]
    fn edits_round_trip_through_the_clipboard_and_other_text_is_ignored() {
        let p = AdjustParams { exposure_ev: 0.7, tone_curve: 1.0, hsl: [[0.1, 0.0, -0.2]; 8], ..Default::default() };
        assert_eq!(from_clipboard(&to_clipboard(&p)), Some(p));
        assert_eq!(from_clipboard(&format!("  {}\n", to_clipboard(&p))), Some(p), "whitespace is fine");
        for other in ["", "hello", "{\"exposure_ev\": 1}", "[1,2]", "{\"raw_editor_edits\": 5}"] {
            assert_eq!(from_clipboard(other), None, "{other:?}");
        }
    }

}
