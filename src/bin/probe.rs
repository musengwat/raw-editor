//! Phase 0 decode proof: verify rawler can decode our camera's RW2 files
//! and dump the metadata the develop stage will need. With a second argument,
//! develops the image and writes an sRGB PNG for visual verification.

use anyhow::{Context, Result};
use rawler::{RawImageData, decode_file};
use raw_editor::develop::develop;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: probe <file.rw2> [out.png]")?;
    let raw = decode_file(&path).with_context(|| format!("decoding {path}"))?;

    println!("camera:      {} {}", raw.clean_make, raw.clean_model);
    println!("dimensions:  {}x{}  cpp={}  bps={}", raw.width, raw.height, raw.cpp, raw.bps);
    println!(
        "cfa:         {} ({}x{} repeat)",
        raw.camera.cfa.name, raw.camera.cfa.width, raw.camera.cfa.height
    );
    println!("wb_coeffs:   {:?}  (RGBE)", raw.wb_coeffs);
    println!("blacklevel:  {:?}", raw.blacklevel.levels);
    println!("whitelevel:  {:?}", raw.whitelevel.0);
    for (illum, mat) in &raw.color_matrix {
        println!("color_matrix[{illum:?}]: {mat:?}");
    }
    println!("active_area: {:?}", raw.active_area);
    println!("crop_area:   {:?}", raw.crop_area);
    println!("orientation: {:?}", raw.orientation);

    match &raw.data {
        RawImageData::Integer(v) => {
            let (min, max) = v.iter().fold((u16::MAX, 0u16), |(lo, hi), &x| (lo.min(x), hi.max(x)));
            println!("data:        u16, {} samples, min={min} max={max}", v.len());
        }
        RawImageData::Float(v) => println!("data:        f32, {} samples", v.len()),
    }

    if let Some(out_path) = std::env::args().nth(2) {
        let t = std::time::Instant::now();
        let dev = develop(&raw)?;
        println!("developed:   {}x{} in {:.2?}", dev.width, dev.height, t.elapsed());
        let px: Vec<u8> = dev.data.iter().map(|&v| srgb_encode(v)).collect();
        image::RgbImage::from_raw(dev.width as u32, dev.height as u32, px)
            .context("bad buffer size")?
            .save(&out_path)?;
        println!("wrote:       {out_path}");
    }
    Ok(())
}

fn srgb_encode(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (e * 255.0 + 0.5) as u8
}
