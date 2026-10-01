//! Optional checks against a real Immich server.
//!
//! These stay inert unless `RAW_EDITOR_IMMICH_TEST_URL` is set, so CI never
//! touches the network:
//!
//! ```text
//! RAW_EDITOR_IMMICH_TEST_URL=http://localhost:2283 cargo test --test immich_live -- --nocapture
//! ```
//!
//! The interesting case needs no credential. Posting a well-formed multipart
//! upload with a bogus key must come back **401 unauthorized**, not 400 bad
//! request — which tells us the endpoint, the headers and the multipart framing
//! are all acceptable to the server, and only the key is wrong. That is most of
//! the upload path verified without ever holding a real key.

use raw_editor::config::Config;
use raw_editor::immich;

fn test_url() -> Option<String> {
    std::env::var("RAW_EDITOR_IMMICH_TEST_URL").ok().filter(|s| !s.is_empty())
}

#[test]
fn a_bogus_key_is_rejected_as_unauthorized_not_malformed() {
    let Some(url) = test_url() else {
        eprintln!("SKIP: set RAW_EDITOR_IMMICH_TEST_URL to run this");
        return;
    };
    let dir = std::env::temp_dir().join(format!("raw-editor-immich-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("probe.jpg");
    // A tiny but real JPEG, so nothing can be rejected for being empty.
    let img = image::RgbImage::from_pixel(2, 2, image::Rgb([120, 130, 140]));
    img.save(&file).unwrap();

    let config = Config {
        immich_url: url.clone(),
        immich_api_key: "definitely-not-a-valid-key".into(),
        upload_after_export: false,
        ..Default::default()
    };

    let err = immich::upload(&config, &file).expect_err("a bogus key must not succeed");
    let text = format!("{err:#}");
    eprintln!("upload with a bogus key -> {text}");
    assert!(
        text.contains("401"),
        "expected 401 (request understood, key rejected); got: {text}"
    );
    assert!(
        !text.contains("definitely-not-a-valid-key"),
        "the API key leaked into an error message: {text}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn ping_reports_the_server_version() {
    let Some(url) = test_url() else {
        eprintln!("SKIP: set RAW_EDITOR_IMMICH_TEST_URL to run this");
        return;
    };
    // /api/server/version is unauthenticated on Immich, so any key works here.
    let config = Config {
        immich_url: url,
        immich_api_key: "unused".into(),
        upload_after_export: false,
        ..Default::default()
    };
    let version = immich::ping(&config).expect("version ping");
    eprintln!("immich version: {version}");
    assert!(version.split('.').count() == 3, "unexpected version shape: {version}");
}
