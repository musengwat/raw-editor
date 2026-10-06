//! Opening TIFFs: they come in as linear sRGB, go through the same pipeline
//! as a raw, and an untouched one exports back to (nearly) itself.

mod common;

use raw_editor::develop::DevelopedImage;
use raw_editor::export::{self, ExportFormat, Rendered};
use raw_editor::metadata::{self, ExportMetadata};
use raw_editor::pipeline::AdjustParams;

/// A ramp in all three channels with a colour twist, plus pure red and a
/// blown white, so the round trip covers dark, mid, bright and the clip rule.
fn source() -> Rendered {
    let (w, h) = (64, 16);
    let mut data = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let t = x as f32 / (w - 1) as f32;
            let px = match (x, y) {
                (0, 0) => [1.0, 0.0, 0.0],
                (1, 0) => [1.0, 1.0, 1.0],
                _ => [t, t * 0.8 + 0.1 * (y as f32 / h as f32), 1.0 - t],
            };
            data.extend(px.map(|v| (v * 65535.0).round() as u16));
        }
    }
    Rendered { width: w, height: h, data }
}

#[test]
fn own_16bit_export_round_trips_through_an_unedited_render() {
    let dir = common::scratch_dir("tiff_round_trip");
    let src = source();
    let tif = dir.join("photo_edit.tif");
    let meta = ExportMetadata {
        make: Some("Sony".into()),
        model: Some("ILCE-7M4".into()),
        lens_model: Some("FE 50mm F2.5 G".into()),
        iso: Some(400),
        ..Default::default()
    };
    export::write(&tif, &src, ExportFormat::Tiff16, 90, Some(&meta)).unwrap();

    let img = metadata::develop_file(&tif).unwrap();
    assert_eq!((img.width, img.height), (src.width, src.height));
    assert_eq!(img.colour.as_shot_wb, [1.0; 3], "TIFFs start neutral");

    // Pure red is a colour, not a clip; the white pixel is a clip.
    assert_eq!(img.clip_at(0), 0.0);
    assert_eq!(img.clip_at(1), 1.0);

    let back = export::render_cpu(&img, &AdjustParams::default());
    let worst = src.data.iter().zip(&back.data).map(|(&a, &b)| (a as i32 - b as i32).abs()).max().unwrap();
    assert!(worst <= 2, "an unedited TIFF should export as itself (worst {worst}/65535)");

    // The metadata we wrote comes back, so a re-export keeps it.
    let m = metadata::read(&tif).unwrap();
    assert_eq!(m.make.as_deref(), Some("Sony"));
    assert_eq!(m.model.as_deref(), Some("ILCE-7M4"));
    assert_eq!(m.lens_model.as_deref(), Some("FE 50mm F2.5 G"));
    assert_eq!(m.iso, Some(400));
}

#[test]
fn eight_bit_and_greyscale_tiffs_open_as_linear_srgb() {
    let dir = common::scratch_dir("tiff_kinds");

    let rgb8 = dir.join("rgb8.tif");
    image::RgbImage::from_raw(2, 1, vec![128, 128, 128, 255, 0, 0]).unwrap().save(&rgb8).unwrap();
    let img = metadata::develop_file(&rgb8).unwrap();
    // sRGB 128/255 ≈ 21.6% linear.
    assert!((img.data[0] - 0.2158).abs() < 1e-3, "{}", img.data[0]);
    assert_eq!(&img.data[3..6], &[1.0, 0.0, 0.0]);

    let grey16 = dir.join("grey16.tiff");
    image::ImageBuffer::<image::Luma<u16>, _>::from_raw(1, 1, vec![65535u16]).unwrap().save(&grey16).unwrap();
    let img = metadata::develop_file(&grey16).unwrap();
    assert_eq!(img.data, vec![1.0, 1.0, 1.0]);
    assert_eq!(img.clip_at(0), 1.0, "full white greyscale is clipped");
}

#[test]
fn float_tiffs_are_taken_as_linear_and_never_clip() {
    let dir = common::scratch_dir("tiff_float");
    let f = dir.join("hdr.tif");
    image::Rgb32FImage::from_raw(1, 1, vec![0.5, 2.0, 4.0]).unwrap().save(&f).unwrap();
    let img = metadata::develop_file(&f).unwrap();
    assert_eq!(img.data, vec![0.5, 2.0, 4.0], "float data is used as is");
    assert!(img.clip.is_empty());
}

#[test]
fn edits_apply_to_a_tiff_like_to_a_raw() {
    let dir = common::scratch_dir("tiff_edit");
    let tif = dir.join("in.tif");
    export::write(&tif, &source(), ExportFormat::Tiff16, 90, None).unwrap();
    let img: DevelopedImage = metadata::develop_file(&tif).unwrap();
    let plain = export::render_cpu(&img, &AdjustParams::default());
    let brighter = export::render_cpu(&img, &AdjustParams { exposure_ev: 1.0, ..Default::default() });
    let mean = |r: &Rendered| r.data.iter().map(|&v| v as f64).sum::<f64>() / r.data.len() as f64;
    assert!(mean(&brighter) > mean(&plain) * 1.1);
}

#[test]
fn a_corrupt_tiff_is_an_error() {
    let dir = common::scratch_dir("tiff_corrupt");
    let f = dir.join("bad.tif");
    std::fs::write(&f, b"II*\0garbage").unwrap();
    assert!(metadata::develop_file(&f).is_err());
}
