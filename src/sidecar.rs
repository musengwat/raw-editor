//! JSON sidecar persistence. Edits live in `<name>.rw2.edit.json` next to the
//! raw file; the `.rw2` itself is never touched.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::Config;
use crate::pipeline::AdjustParams;

/// Where a photo's starting adjustments came from.
#[derive(Clone, Debug, PartialEq)]
pub enum Origin {
    /// Its own sidecar.
    Sidecar,
    /// No sidecar; its camera body is set to start in this look.
    BodyLook(String),
    /// No sidecar and no body look: neutral.
    Neutral,
}

/// What opening `raw` with no sidecar gives: neutral, or the look assigned to
/// its camera body (`serial`). Also the baseline the GUI compares against to
/// decide whether a sidecar is needed at all.
pub fn baseline(config: &Config, serial: Option<&str>) -> (AdjustParams, Origin) {
    match config.body_look(serial) {
        Some(look) => {
            let mut p = AdjustParams::default();
            look.apply(&mut p);
            (p, Origin::BodyLook(look.name))
        }
        None => (AdjustParams::default(), Origin::Neutral),
    }
}

/// The adjustments a photo opens (or batch-exports) with: its sidecar if it
/// has one, otherwise [`baseline`]. Shared by the GUI, the batch export and
/// the CLI so all three agree.
pub fn initial(raw: &Path, config: &Config, serial: Option<&str>) -> Result<(AdjustParams, Origin)> {
    Ok(match load(raw)? {
        Some(p) => (p, Origin::Sidecar),
        None => baseline(config, serial),
    })
}

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
            exposure_ev: 1.25,
            contrast: -0.4,
            temp: 0.3,
            tint: -0.15,
            saturation: 0.2,
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
    fn initial_params_prefer_sidecar_then_body_look_then_neutral() {
        let raw = scratch("initial");
        let mut config = Config::default();
        config.body_looks.insert("F92".into(), "Red–blue swap".into());

        assert_eq!(initial(&raw, &config, Some("XEL")).unwrap(), (AdjustParams::default(), Origin::Neutral));
        assert_eq!(initial(&raw, &config, None).unwrap().1, Origin::Neutral);

        let (p, origin) = initial(&raw, &config, Some("F92")).unwrap();
        assert_eq!(origin, Origin::BodyLook("Red–blue swap".into()));
        assert_eq!(p.mixer, [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]);
        assert_eq!(p.colour_mode, crate::pipeline::ColourMode::SensorRgb);

        // A sidecar always wins — including a neutral one, which is how a
        // single photo opts out of its body's look.
        save(&raw, &AdjustParams::default()).unwrap();
        assert_eq!(initial(&raw, &config, Some("F92")).unwrap(), (AdjustParams::default(), Origin::Sidecar));
    }
}
