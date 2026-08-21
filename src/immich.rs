//! Upload finished exports to an Immich server.
//!
//! This is the only module in the project that touches the network. Keep it
//! that way: nothing else should acquire the ability to make requests.
//!
//! The contract comes from a running server's own OpenAPI document (Immich
//! 3.1.0), not from the published guides, which are out of date — they list
//! `deviceAssetId`/`deviceId` as required and this version does not want them:
//!
//! ```text
//! POST {base}/api/assets    multipart/form-data    header: x-api-key
//! required: assetData, fileCreatedAt, fileModifiedAt
//! 201 -> { id, status }     status = "created" | "duplicate"
//! ```
//!
//! The server deduplicates by checksum, so re-uploading the same export is
//! harmless and comes back as `duplicate` rather than an error.
//!
//! Capture dates are deliberately *not* sent: `fileCreatedAt` is a filesystem
//! timestamp, and Immich reads the real capture time out of the EXIF that
//! `crate::metadata` now writes into every exported JPEG.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use ureq::unversioned::multipart::{Form, Part};

use crate::config::Config;

/// Uploads can be tens of megabytes, so this is generous rather than snappy.
const UPLOAD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
/// A connection test should fail fast — someone is waiting on the answer.
const PING_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Upload {
    /// The server stored a new asset.
    Created(String),
    /// The server already had this exact file. Not a failure.
    Duplicate(String),
}

impl Upload {
    pub fn describe(&self) -> String {
        match self {
            Self::Created(id) => format!("uploaded to Immich ({id})"),
            Self::Duplicate(_) => "already in Immich (duplicate)".to_string(),
        }
    }
}

fn agent(timeout: std::time::Duration) -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(timeout)).build().into()
}

/// Ask the server for its version — used by the settings dialog's connection
/// test, and the cheapest way to prove URL and key are both right.
pub fn ping(config: &Config) -> Result<String> {
    if !config.immich_ready() {
        bail!("set the Immich URL and API key first");
    }
    let url = format!("{}/api/server/version", config.base_url());
    let mut resp = agent(PING_TIMEOUT)
        .get(&url)
        .header("x-api-key", &config.immich_api_key)
        .header("accept", "application/json")
        .call()
        .with_context(|| format!("contacting {}", config.base_url()))?;

    let body = resp.body_mut().read_to_string().context("reading the version response")?;
    let v: serde_json::Value = serde_json::from_str(&body).context("parsing the version")?;
    let part = |k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0);
    Ok(format!("{}.{}.{}", part("major"), part("minor"), part("patch")))
}

/// Upload one already-written file.
pub fn upload(config: &Config, file: &Path) -> Result<Upload> {
    if !config.immich_ready() {
        bail!("set the Immich URL and API key first");
    }
    let meta = std::fs::metadata(file).with_context(|| format!("reading {}", file.display()))?;
    let modified = meta.modified().unwrap_or_else(|_| SystemTime::now());
    let stamp = iso8601_utc(modified);
    let name = file.file_name().unwrap_or_default().to_string_lossy().into_owned();

    let form = Form::new()
        .text("fileCreatedAt", &stamp)
        .text("fileModifiedAt", &stamp)
        .text("filename", &name)
        .part(
            "assetData",
            Part::file(file)
                .with_context(|| format!("opening {}", file.display()))?
                .file_name(&name),
        );

    let url = format!("{}/api/assets", config.base_url());
    let mut resp = agent(UPLOAD_TIMEOUT)
        .post(&url)
        .header("x-api-key", &config.immich_api_key)
        .header("accept", "application/json")
        .send(form)
        // Deliberately naming only the host: the key must never reach an error.
        .with_context(|| format!("uploading {} to {}", name, config.base_url()))?;

    let body = resp.body_mut().read_to_string().context("reading the upload response")?;
    let v: serde_json::Value = serde_json::from_str(&body)
        .with_context(|| format!("parsing the upload response: {body}"))?;
    let id = v.get("id").and_then(|x| x.as_str()).unwrap_or_default().to_string();
    match v.get("status").and_then(|x| x.as_str()) {
        Some("duplicate") => Ok(Upload::Duplicate(id)),
        Some("created") | None => Ok(Upload::Created(id)),
        Some(other) => bail!("unexpected upload status {other:?}"),
    }
}

/// Format a `SystemTime` as `YYYY-MM-DDTHH:MM:SS.000Z`.
///
/// Hand-rolled rather than pulling in a date library for one call: this is the
/// standard days-from-civil algorithm, and it is covered by tests below.
fn iso8601_utc(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    // civil_from_days, Howard Hinnant's algorithm.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}.000Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn at(secs: u64) -> String {
        iso8601_utc(UNIX_EPOCH + Duration::from_secs(secs))
    }

    #[test]
    fn formats_the_epoch() {
        assert_eq!(at(0), "1970-01-01T00:00:00.000Z");
    }

    #[test]
    fn formats_known_instants() {
        // 2001-09-09T01:46:40Z — the classic billennium check.
        assert_eq!(at(1_000_000_000), "2001-09-09T01:46:40.000Z");
        // A leap day, which the civil-from-days maths has to get right.
        assert_eq!(at(1_709_164_800), "2024-02-29T00:00:00.000Z");
        assert_eq!(at(1_735_689_599), "2024-12-31T23:59:59.000Z");
    }

    #[test]
    fn refuses_to_act_without_configuration() {
        let c = Config::default();
        assert!(upload(&c, Path::new("/tmp/x.jpg")).is_err());
        assert!(ping(&c).is_err());
    }

    #[test]
    fn upload_status_maps_to_the_right_variant() {
        assert_eq!(
            Upload::Duplicate("abc".into()).describe(),
            "already in Immich (duplicate)"
        );
        assert!(Upload::Created("abc".into()).describe().contains("abc"));
    }
}
