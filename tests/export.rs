//! Export smoke tests: render a small synthetic image through both the GPU
//! and CPU export paths, write JPEG + 16-bit TIFF, and read them back.
//!
//! The image is deliberately wider than 32px so the readback exercises the
//! 256-byte `bytes_per_row` padding (32 * 8 bytes = 256 exactly, so 37px
//! forces a padded row).

mod common;

use image::ImageDecoder;
use raw_editor::develop::DevelopedImage;
use raw_editor::export::{self, ExportFormat};
use raw_editor::pipeline::AdjustParams;
use raw_editor::sidecar;

const W: usize = 37;
const H: usize = 11;

/// A smooth ramp plus a few out-of-range highlights, so tonal masks, the
/// colour wheels and the sRGB transfer all have something to act on.
fn fixture() -> DevelopedImage {
    let mut data = Vec::with_capacity(W * H * 3);
    for y in 0..H {
        for x in 0..W {
            let t = x as f32 / (W - 1) as f32;
            let v = t * t * 1.8; // 0 .. 1.8 (above white on purpose)
            data.extend_from_slice(&[v, v * (0.5 + 0.5 * y as f32 / H as f32), v * 0.75]);
        }
    }
    DevelopedImage::linear_srgb(W, H, data)
}

fn params() -> AdjustParams {
    AdjustParams {
        exposure_ev: 0.4,
        contrast: 0.25,
        temp: 0.2,
        tint: -0.1,
        saturation: 0.3,
        highlights: -0.7,
        shadows: 0.5,
        whites: 0.15,
        blacks: -0.25,
        lift: [0.03, 0.0, -0.02],
        gamma: [1.1, 1.0, 0.95],
        gain: [1.05, 1.0, 0.95],
        // Detail on too, so the export path exercises the neighbourhood stage
        // and the GPU/CPU renders have to agree about it.
        sharpen_amount: 1.2,
        sharpen_radius: 1.4,
        sharpen_threshold: 0.003,
        nr_luma: 0.5,
        nr_chroma: 0.6,
        ..Default::default()
    }
}

#[test]
fn gpu_and_cpu_export_paths_agree() {
    let Some(ctx) = common::gpu() else {
        eprintln!("SKIP: no wgpu adapter available");
        return;
    };
    let img = fixture();
    let p = params();
    let gpu = export::render_gpu(&ctx.device, &ctx.queue, &img, &p).expect("gpu render");
    let cpu = export::render_cpu(&img, &p);

    assert_eq!((gpu.width, gpu.height), (W, H));
    assert_eq!(gpu.data.len(), cpu.data.len());

    // The GPU path goes through an rgba16float target, so ~11 bits of
    // mantissa: allow a few hundred counts out of 65535 (< 0.5%).
    let mut worst = 0i32;
    for (i, (g, c)) in gpu.data.iter().zip(cpu.data.iter()).enumerate() {
        let d = (*g as i32 - *c as i32).abs();
        worst = worst.max(d);
        assert!(d < 400, "sample {i}: gpu={g} cpu={c}");
    }
    eprintln!("worst gpu/cpu export delta: {worst}/65535");
}

#[test]
fn writes_a_readable_jpeg_and_16_bit_tiff() {
    let dir = common::scratch_dir("export");
    let rendered = export::render_cpu(&fixture(), &params());

    let jpg = dir.join("out.jpg");
    export::write(&jpg, &rendered, ExportFormat::Jpeg, 90, None).unwrap();
    let decoded = image::open(&jpg).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (W as u32, H as u32));

    let tif = dir.join("out.tif");
    export::write(&tif, &rendered, ExportFormat::Tiff16, 90, None).unwrap();
    let file = std::io::BufReader::new(std::fs::File::open(&tif).unwrap());
    let dec = image::codecs::tiff::TiffDecoder::new(file).unwrap();
    assert_eq!(dec.dimensions(), (W as u32, H as u32));
    assert_eq!(
        dec.color_type(),
        image::ColorType::Rgb16,
        "the TIFF must really be 16 bits per channel"
    );
}

#[test]
fn jpeg_quality_changes_the_file_size() {
    let dir = common::scratch_dir("quality");
    // A noisy image so quality actually matters.
    let mut data = Vec::with_capacity(64 * 64 * 3);
    for i in 0..64 * 64 * 3 {
        data.push(((i * 2654435761u64 as usize) % 997) as f32 / 997.0);
    }
    let img = DevelopedImage::linear_srgb(64, 64, data);
    let rendered = export::render_cpu(&img, &AdjustParams::default());

    let lo = dir.join("lo.jpg");
    let hi = dir.join("hi.jpg");
    export::write(&lo, &rendered, ExportFormat::Jpeg, 10, None).unwrap();
    export::write(&hi, &rendered, ExportFormat::Jpeg, 98, None).unwrap();
    let (lo_len, hi_len) =
        (std::fs::metadata(&lo).unwrap().len(), std::fs::metadata(&hi).unwrap().len());
    assert!(hi_len > lo_len, "q98 ({hi_len}) should be bigger than q10 ({lo_len})");
}

/// The full non-destructive loop the app performs: edit → sidecar → reopen →
/// export with the restored params.
#[test]
fn sidecar_round_trip_drives_the_export() {
    let dir = common::scratch_dir("roundtrip");
    let raw = dir.join("P1320071.rw2");
    std::fs::write(&raw, b"not a real raw, only the sidecar path matters").unwrap();

    let p = params();
    sidecar::save(&raw, &p).unwrap();
    let restored = sidecar::load(&raw).unwrap().expect("sidecar should exist");
    assert_eq!(restored, p);

    let img = fixture();
    let a = export::render_cpu(&img, &p);
    let b = export::render_cpu(&img, &restored);
    assert_eq!(a.data, b.data, "restored params must render identically");

    let out = export::output_path(&raw, ExportFormat::Jpeg);
    assert_eq!(out.file_name().unwrap(), "P1320071_edit.jpg");
    export::write(&out, &b, ExportFormat::Jpeg, 92, None).unwrap();
    assert!(out.exists());
}

/// Metadata must actually land in the file, and must not disturb the pixels.
#[test]
fn metadata_is_written_into_the_export() {
    use raw_editor::metadata::ExportMetadata;

    let dir = common::scratch_dir("metadata");
    let rendered = export::render_cpu(&fixture(), &AdjustParams::default());
    let meta = ExportMetadata {
        make: Some("Panasonic".into()),
        model: Some("DMC-G85".into()),
        lens_model: Some("LUMIX G 14-42".into()),
        date_time_original: Some("2026:05:02 00:53:16".into()),
        exposure_time: Some((1, 250)),
        fnumber: Some((4, 1)),
        iso: Some(100),
        orientation: Some(6),
        ..Default::default()
    };

    let jpg = dir.join("meta.jpg");
    export::write(&jpg, &rendered, ExportFormat::Jpeg, 92, Some(&meta)).unwrap();

    // Still a valid, correctly sized JPEG after the tags went in.
    let decoded = image::open(&jpg).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (W as u32, H as u32));

    let read = little_exif::metadata::Metadata::new_from_path(&jpg).expect("read back exif");
    let dump = read.into_iter().map(|t| format!("{t:?}")).collect::<Vec<_>>().join(" ");
    for want in ["Make", "Model", "ISO", "Orientation", "ColorSpace", "Software"] {
        assert!(dump.contains(want), "{want} did not survive the write: {dump}");
    }
}

/// TIFF compression must be lossless: every sample has to survive the
/// Deflate + horizontal-predictor round trip untouched. A silent corruption
/// here would be invisible until someone opened a 16-bit file months later.
#[test]
fn compressed_tiff_round_trips_exactly() {
    let dir = common::scratch_dir("tiff_lossless");
    let rendered = export::render_cpu(&fixture(), &params());

    let tif = dir.join("lossless.tif");
    export::write(&tif, &rendered, ExportFormat::Tiff16, 90, None).unwrap();

    let file = std::io::BufReader::new(std::fs::File::open(&tif).unwrap());
    let mut dec = tiff::decoder::Decoder::new(file).unwrap();
    let img = dec.read_image().unwrap();
    let tiff::decoder::DecodingResult::U16(samples) = img else {
        panic!("expected 16-bit samples back");
    };
    assert_eq!(samples.len(), rendered.data.len(), "sample count changed");
    assert_eq!(samples, rendered.data, "compression was not lossless");
}

/// ...and compression has to actually be doing something.
#[test]
fn compressed_tiff_is_smaller_than_raw_samples() {
    let dir = common::scratch_dir("tiff_size");
    // A smooth gradient, which is what the horizontal predictor is for.
    let (w, h) = (256usize, 64usize);
    let mut data = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let v = x as f32 / w as f32;
            data.extend_from_slice(&[v, v * 0.8, v * 0.6]);
        }
    }
    let img = DevelopedImage::linear_srgb(w, h, data);
    let rendered = export::render_cpu(&img, &AdjustParams::default());

    let tif = dir.join("gradient.tif");
    export::write(&tif, &rendered, ExportFormat::Tiff16, 90, None).unwrap();

    let uncompressed = (w * h * 6) as u64;
    let actual = std::fs::metadata(&tif).unwrap().len();
    assert!(
        actual < uncompressed / 2,
        "expected real compression: {actual} vs {uncompressed} raw"
    );
}

/// Metadata now goes into the TIFF during encoding, in a real EXIF sub-IFD.
#[test]
fn tiff_carries_metadata() {
    use raw_editor::metadata::ExportMetadata;

    let dir = common::scratch_dir("tiff_meta2");
    let rendered = export::render_cpu(&fixture(), &AdjustParams::default());
    let meta = ExportMetadata {
        make: Some("Panasonic".into()),
        model: Some("DMC-G85".into()),
        iso: Some(400),
        exposure_time: Some((1, 250)),
        orientation: Some(6),
        ..Default::default()
    };

    let tif = dir.join("meta.tif");
    export::write(&tif, &rendered, ExportFormat::Tiff16, 90, Some(&meta)).unwrap();

    let file = std::io::BufReader::new(std::fs::File::open(&tif).unwrap());
    let mut dec = tiff::decoder::Decoder::new(file).unwrap();
    use tiff::tags::Tag;
    assert_eq!(dec.get_tag_ascii_string(Tag::Make).unwrap(), "Panasonic");
    assert_eq!(dec.get_tag_ascii_string(Tag::Model).unwrap(), "DMC-G85");
    assert_eq!(dec.get_tag_ascii_string(Tag::Software).unwrap(), "raw-editor");
    assert_eq!(dec.get_tag_u32(Tag::Orientation).unwrap(), 6);
    // The EXIF sub-IFD must exist and be pointed at.
    assert!(dec.get_tag_u32(Tag::ExifDirectory).unwrap() > 0, "no EXIF IFD pointer");

    // ...and the image data must still decode after all that.
    let img = dec.read_image().unwrap();
    let tiff::decoder::DecodingResult::U16(samples) = img else { panic!("not 16-bit") };
    assert_eq!(samples, rendered.data, "metadata disturbed the pixels");
}
