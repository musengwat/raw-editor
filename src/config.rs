//! Persistent settings, currently just the Immich connection.
//!
//! The API key is a credential. It is stored `0600` on Unix, redacted from
//! `Debug`, and must never reach a log line, an error message or the status
//! bar — see the `Debug` impl and `redacted()` below.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    /// Base URL of the Immich server, e.g. `https://photos.example.org`.
    pub immich_url: String,
    /// Immich API key (Account Settings → API Keys).
    pub immich_api_key: String,
    /// Whether a finished export is uploaded. Off unless deliberately enabled.
    pub upload_after_export: bool,
}

/// Redacts the key, so a stray `{:?}` cannot leak it.
impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("immich_url", &self.immich_url)
            .field("immich_api_key", &self.redacted())
            .field("upload_after_export", &self.upload_after_export)
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

    pub fn path() -> Result<PathBuf> {
        let dir = dirs::config_dir().context("no config directory for this platform")?;
        Ok(dir.join("raw-editor").join("config.json"))
    }

    /// Load the config, falling back to defaults when it does not exist yet.
    /// A corrupt file is reported rather than silently replaced.
    pub fn load() -> Result<Self> {
        let path = Self::path()?;
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text)
                .with_context(|| format!("parsing {}", path.display())),
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
    fn debug_never_reveals_the_key() {
        let c = Config {
            immich_url: "https://photos.example.org".into(),
            immich_api_key: "super-secret-key-value".into(),
            upload_after_export: true,
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
}
