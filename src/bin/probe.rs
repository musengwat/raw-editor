//! Phase 0 decode proof: verify rawler can decode our camera's RW2 files
//! and dump the metadata the develop stage will need.

use anyhow::{Context, Result};
use rawler::{RawImageData, decode_file};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: probe <file.rw2>")?;
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
    Ok(())
}
