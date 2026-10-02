//! Persistent settings: the Immich connection, saved looks, which look each
//! camera body starts in, and a flat field per lens.
//!
//! The API key is a credential. It is stored `0600` on Unix, redacted from
//! `Debug`, and must never reach a log line, an error message or the status
//! bar — see the `Debug` impl and `redacted()` below.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::flatfield::FlatField;
use crate::looks::Look;

#[derive(Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    /// Base URL of the Immich server, e.g. `https://photos.example.org`.
    pub immich_url: String,
    /// Immich API key (Account Settings → API Keys).
    pub immich_api_key: String,
    /// Whether a finished export is uploaded. Off unless deliberately enabled.
    pub upload_after_export: bool,
    /// Looks the user has saved, by name.
    pub looks: Vec<Look>,
    /// Body serial → name of the look a photo from that body starts in when
    /// it has no sidecar yet (e.g. the IR-converted GF1 → an Aerochrome look).
    pub body_looks: std::collections::BTreeMap<String, String>,
    /// Lens model → flat field made from a reference shot with that lens.
    pub flat_fields: std::collections::BTreeMap<String, FlatField>,
}

/// Redacts the key, so a stray `{:?}` cannot leak it.
impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("immich_url", &self.immich_url)
            .field("immich_api_key", &self.redacted())
            .field("upload_after_export", &self.upload_after_export)
            .field("looks", &self.looks.iter().map(|l| &l.name).collect::<Vec<_>>())
            .field("body_looks", &self.body_looks)
            .field("flat_fields", &self.flat_fields.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Config {
    /// A safe stand-in for the key in anything user-visible.
    pub fn redacted(&self) -> &'static str {
        if self.immich_api_key.is_empty() { "<unset>" } else { "<set>" }
    }

    /// Uploading needs both halves; without them the feature stays inert.
    pub fn immich_ready(&self) -> bool {
        !self.immich_url.trim().is_empty() && !self.immich_api_key.trim().is_empty()
    }

    /// Base URL without a trailing slash, so joining paths is predictable.
    pub fn base_url(&self) -> &str {
        self.immich_url.trim().trim_end_matches('/')
    }

    /// Every look by name: saved looks first, so a saved look can shadow a
    /// built-in of the same name.
    pub fn find_look(&self, name: &str) -> Option<Look> {
        self.looks.iter().find(|l| l.name == name).cloned().or_else(|| {
            crate::looks::find_builtin(name)
        })
    }

    /// Point body looks saved under a built-in's old name at its new name,
    /// unless a saved look of the old name exists (that one wins, as in
    /// [`Config::find_look`]).
    pub fn migrated(mut self) -> Self {
        for name in self.body_looks.values_mut() {
            let saved = self.looks.iter().any(|l| &l.name == name);
            if let (false, Some((_, new))) = (saved, crate::looks::RENAMED.iter().find(|(old, _)| old == name)) {
                *name = new.to_string();
            }
        }
        self
    }

    /// The look a photo from this body starts in, if one is set and still exists.
    pub fn body_look(&self, serial: Option<&str>) -> Option<Look> {
        self.find_look(self.body_looks.get(serial?)?)
    }

    /// The flat field for a lens, if one has been made.
    pub fn flat_for(&self, lens: Option<&str>) -> Option<std::sync::Arc<FlatField>> {
        self.flat_fields.get(lens?).cloned().map(std::sync::Arc::new)
    }

    /// Save a look, replacing any saved look with the same name.
    pub fn save_look(&mut self, look: Look) {
        match self.looks.iter_mut().find(|l| l.name == look.name) {
            Some(existing) => *existing = look,
            None => self.looks.push(look),
        }
    }

    /// Delete a saved look, and any body that started in it.
    pub fn delete_look(&mut self, name: &str) {
        self.looks.retain(|l| l.name != name);
        self.body_looks.retain(|_, v| v != name);
    }

    pub fn path() -> Result<PathBuf> {
        let dir = dirs::config_dir().context("no config directory for this platform")?;
        Ok(dir.join("raw-editor").join("config.json"))
    }

    /// Load the config, falling back to defaults when it does not exist yet.
    /// A corrupt file is reported rather than silently replaced.
    pub fn load() -> Result<Self> {
        let path = Self::path()?;
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str::<Self>(&text)
                .with_context(|| format!("parsing {}", path.display()))
                .map(Self::migrated),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
        restrict_permissions(&path)?;
        Ok(())
    }
}

/// Owner-only, because the file holds an API key.
#[cfg(unix)]
fn restrict_permissions(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("securing {}", path.display()))
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &std::path::Path) -> Result<()> {
    // Windows inherits the user profile's ACL, which is already per-user.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_looks_under_old_builtin_names_are_migrated() {
        let mut c = Config::default();
        c.body_looks.insert("A".into(), "Red–blue swap".into());
        c.body_looks.insert("B".into(), "IR mono".into());
        c.looks.push(look("IR mono", 0.0)); // a saved look by that name wins
        let c = c.migrated();
        assert_eq!(c.body_looks["A"], "IR Chrome — red–blue swap");
        assert_eq!(c.body_looks["B"], "IR mono");
    }

    #[test]
    fn debug_never_reveals_the_key() {
        let c = Config {
            immich_url: "https://photos.example.org".into(),
            immich_api_key: "super-secret-key-value".into(),
            upload_after_export: true,
            ..Default::default()
        };
        let text = format!("{c:?}");
        assert!(!text.contains("super-secret"), "the key leaked into Debug: {text}");
        assert!(text.contains("<set>"));
    }

    #[test]
    fn readiness_needs_both_halves() {
        let mut c = Config::default();
        assert!(!c.immich_ready());
        c.immich_url = "https://photos.example.org".into();
        assert!(!c.immich_ready(), "a URL alone is not enough");
        c.immich_api_key = "k".into();
        assert!(c.immich_ready());
        c.immich_url = "   ".into();
        assert!(!c.immich_ready(), "whitespace is not a URL");
    }

    #[test]
    fn base_url_drops_trailing_slashes() {
        let c = Config { immich_url: "https://x.org/// ".into(), ..Default::default() };
        assert_eq!(c.base_url(), "https://x.org");
    }

    #[test]
    fn config_round_trips_through_json() {
        let c = Config {
            immich_url: "https://photos.example.org".into(),
            immich_api_key: "k".into(),
            upload_after_export: true,
            ..Default::default()
        };
        let back: Config = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert!(back == c);
    }

    /// Old config files must keep loading as fields are added.
    #[test]
    fn missing_fields_fall_back_to_defaults() {
        let c: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(c, Config::default());
    }

    // --- looks (IR-3) ---------------------------------------------------

    fn look(name: &str, sat: f32) -> Look {
        let p = crate::pipeline::AdjustParams { saturation: sat, ..Default::default() };
        Look::from_params(name, &p)
    }

    #[test]
    fn saving_a_look_replaces_one_with_the_same_name() {
        let mut c = Config::default();
        c.save_look(look("Mine", 0.1));
        c.save_look(look("Other", 0.2));
        c.save_look(look("Mine", 0.5));
        assert_eq!(c.looks.len(), 2);
        assert_eq!(c.find_look("Mine").unwrap().saturation, Some(0.5));
    }

    #[test]
    fn find_look_prefers_saved_then_builtin() {
        let mut c = Config::default();
        assert!(c.find_look("Aerochrome classic").unwrap().saturation.is_none(), "built-in");
        c.save_look(look("Aerochrome classic", 0.4));
        assert_eq!(c.find_look("Aerochrome classic").unwrap().saturation, Some(0.4));
        assert!(c.find_look("nope").is_none());
    }

    #[test]
    fn body_look_resolves_by_serial_and_survives_a_round_trip() {
        let mut c = Config::default();
        c.save_look(look("GF1 IR", 0.3));
        c.body_looks.insert("F921004200220".into(), "GF1 IR".into());
        let c: Config = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(c.body_look(Some("F921004200220")).unwrap().name, "GF1 IR");
        assert!(c.body_look(Some("XEL2112060047")).is_none());
        assert!(c.body_look(None).is_none());
    }

    #[test]
    fn deleting_a_look_unassigns_it_from_bodies() {
        let mut c = Config::default();
        c.save_look(look("GF1 IR", 0.3));
        c.body_looks.insert("F92".into(), "GF1 IR".into());
        c.body_looks.insert("XEL".into(), "Red–blue swap".into());
        c.delete_look("GF1 IR");
        assert!(c.looks.is_empty());
        assert_eq!(c.body_looks.len(), 1, "{:?}", c.body_looks);
    }

    #[test]
    fn config_from_before_ir3_loads_without_looks() {
        let c: Config = serde_json::from_str(r#"{"immich_url":"https://x.org","upload_after_export":true}"#).unwrap();
        assert!(c.looks.is_empty() && c.body_looks.is_empty());
        assert!(c.upload_after_export);
    }

    #[test]
    fn flat_fields_are_found_by_lens() {
        let mut c = Config::default();
        let ff = FlatField { lens: "LUMIX G 20/F1.7".into(), width: 1, height: 1, gains: vec![[1.5; 3]] };
        c.flat_fields.insert(ff.lens.clone(), ff);
        let c: Config = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(c.flat_for(Some("LUMIX G 20/F1.7")).unwrap().gains, vec![[1.5; 3]]);
        assert!(c.flat_for(Some("LUMIX G VARIO 7-14/F4.0")).is_none());
        assert!(c.flat_for(None).is_none());
    }
}
