//! End-to-end HDR merge of a real Sony A7 IV bracket (`testdata/hdr`,
//! DSC01195–97: 0 / −2 / +2 EV by auto ISO, handheld). Skips without it.

mod common;

use std::path::{Path, PathBuf};

use raw_editor::hdr::{self, Style};
use raw_editor::{bracket, formats, metadata};

fn set() -> Option<Vec<PathBuf>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/hdr");
    let list = formats::photos_in(&dir).ok()?;
    bracket::set_containing(&list, &dir.join("DSC01196.arw"))
}

#[test]
fn a_real_bracket_merges_and_reopens_like_a_raw() {
    let Some(set) = set() else { return };
    assert_eq!(set.len(), 3);
    let out = common::scratch_dir("hdr").join(hdr::output_path(&set).file_name().unwrap());
    let (path, report) = hdr::merge_files(&set, Style::Hdr, Some(&out), &|_| {}).unwrap();
    assert_eq!(path, out);

    // The 0 EV frame is the reference; the others are ±2 EV (ISO 640 and
    // 10000 against 2500), measured to within the cameras' rounding.
    assert_eq!(report.reference, 0);
    assert!((report.scales[1] - 4.0).abs() < 0.2 && (report.scales[2] - 0.25).abs() < 0.03, "{:?}", report.scales);
    assert!((report.extra_stops - 4.0).abs() < 0.2, "{}", report.extra_stops);
    // Handheld: the frames moved a few pixels and the merge found it.
    assert!(report.shifts[1] != [0, 0] && report.shifts[1].iter().all(|v| v.abs() < 40), "{:?}", report.shifts);
    assert!(report.rotation.iter().all(|r| r.abs() < 0.5), "{:?}", report.rotation);
    assert!(report.ghosted.iter().all(|&g| g < 0.1), "a still room: {:?}", report.ghosted);

    // The TIFF reopens full size with the camera's tags, and starts in the
    // default look like a raw.
    let img = metadata::develop_file(&path).unwrap();
    assert_eq!((img.width, img.height), (7008, 4672));
    let meta = metadata::read(&path).unwrap();
    assert_eq!(meta.model.as_deref(), Some("ILCE-7M4"));
    assert_eq!(meta.iso, Some(2500), "the reference frame's EXIF");
    let (_, origin) = raw_editor::sidecar::baseline(&Default::default(), &Default::default(), &path);
    assert!(matches!(origin, raw_editor::sidecar::Origin::DefaultLook(_)), "{origin:?}");

    // Tone mapped into range: almost nothing at either end.
    let n = img.width * img.height;
    let black = img.data.chunks(3).filter(|c| c.iter().all(|&v| v < 1e-4)).count();
    assert!(img.clip.iter().filter(|&&c| c > 0.0).count() < n / 1000, "highlights held");
    assert!(black < n / 1000, "shadows open: {black} black pixels");
}
