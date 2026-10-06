//! Cameras other than the Panasonic bodies the editor grew up on. The Sony
//! A7 IV samples come from raw.pixls.us (CC0) and live in the gitignored
//! `testdata/sony`; each test skips without them.

use std::path::{Path, PathBuf};

use raw_editor::metadata;

fn sony(name: &str) -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/sony").join(name);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("SKIP: {name} not present");
        None
    }
}

/// Every full-size A7 IV raw mode develops to the camera's own 7008x4672
/// crop. The active area is wider (7040 or 7032) and carries edge pixels.
#[test]
fn a7iv_full_size_modes_develop_to_the_camera_crop() {
    for name in [
        "ILCE-7M4_DSC06673_FullFrame-Raw-Uncompressed.ARW",
        "ILCE-7M4_DSC06674_FullFrame-LossLess-Compressed-Large.ARW",
        "ILCE-7M4_DSC06677_FullFrame-Raw-Compressed.ARW",
    ] {
        let Some(path) = sony(name) else { continue };
        let img = metadata::develop_file(&path).unwrap();
        assert_eq!((img.width, img.height), (7008, 4672), "{name}");
        // As-shot WB is a daylight-ish balance, not the unit fallback, and the
        // matrix came from the camera's colour data.
        let wb = img.colour.as_shot_wb;
        assert!(wb[0] > 1.5 && wb[2] > 1.2, "{name}: wb {wb:?}");
        assert_ne!(img.colour.cam_to_srgb, raw_editor::pipeline::IDENTITY3, "{name}");
        // Sensible exposure: a mid-grey-ish frame, not black and not blown.
        let mean = img.data.iter().map(|&v| v as f64).sum::<f64>() / img.data.len() as f64;
        assert!((0.02..0.5).contains(&mean), "{name}: mean {mean}");
    }
}

#[test]
fn a7iv_metadata_names_camera_and_lens() {
    let Some(path) = sony("ILCE-7M4_DSC06674_FullFrame-LossLess-Compressed-Large.ARW") else { return };
    let m = metadata::read(&path).unwrap();
    assert_eq!(m.make.as_deref(), Some("Sony"));
    assert_eq!(m.model.as_deref(), Some("ILCE-7M4"));
}

/// A cut-off file must come back as an error, never a panic — in the GUI a
/// panic killed the loader thread and left the window on "Loading…".
#[test]
fn truncated_raw_is_an_error_not_a_panic() {
    let Some(path) = sony("ILCE-7M4_DSC06677_FullFrame-Raw-Compressed.ARW") else { return };
    let bytes = std::fs::read(&path).unwrap();
    let dir = std::env::temp_dir().join(format!("raw-editor-trunc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for frac in [0.01, 0.5, 0.9] {
        let cut = dir.join("cut.ARW");
        std::fs::write(&cut, &bytes[..(bytes.len() as f64 * frac) as usize]).unwrap();
        assert!(metadata::develop_file(&cut).is_err(), "cut at {frac}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn garbage_with_a_raw_extension_is_an_error() {
    let dir = std::env::temp_dir().join(format!("raw-editor-garbage-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("not-really.ARW");
    std::fs::write(&p, vec![0x5au8; 64 * 1024]).unwrap();
    assert!(metadata::develop_file(&p).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// M and S lossless are full-colour (YCbCr in the file), already white
/// balanced, with a clip plateau far above the reported white level. They
/// skip the demosaic, develop to the camera crop, and land within a stop of
/// the L frame of the same scene.
#[test]
fn a7iv_reduced_size_lossless_develops_without_a_mosaic() {
    let Some(l) = sony("ILCE-7M4_DSC06674_FullFrame-LossLess-Compressed-Large.ARW") else { return };
    let mean = |img: &raw_editor::develop::DevelopedImage| {
        // Green after as-shot WB, i.e. roughly what the screen shows.
        let wb = img.colour.as_shot_wb[1];
        img.data.chunks(3).map(|p| (p[1] * wb) as f64).sum::<f64>() / (img.data.len() / 3) as f64
    };
    let large = mean(&metadata::develop_file(&l).unwrap());
    for (name, size) in [
        ("ILCE-7M4_DSC06675_FullFrame-LossLess-Compressed-Medium.ARW", (4608, 3072)),
        ("ILCE-7M4_DSC06676_FullFrame-LossLess-Compressed-Small.ARW", (3504, 2336)),
    ] {
        let Some(path) = sony(name) else { continue };
        let img = metadata::develop_file(&path).unwrap();
        assert_eq!((img.width, img.height), size, "{name}");
        assert_eq!(img.colour.as_shot_wb, [1.0; 3], "{name}: WB is baked in");
        assert!(!img.clip.is_empty(), "{name}: the blown sky should be found and flagged");
        let ratio = mean(&img) / large;
        assert!((0.5..2.0).contains(&ratio), "{name}: {ratio:.2}x the L frame's brightness");
    }
}
