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

            // Where does this sensor actually saturate? A hard ADC ceiling shows
            // up as a large spike of identical values at the very top, which is
            // the real clipping point regardless of the reported whitelevel.
            // Per-CFA-colour extremes. If one colour tops out well below the
            // reported whitelevel it has its own saturation ceiling.
            let mut cmax = [0u16; 4];
            let mut cmin = [u16::MAX; 4];
            let mut at_max = [0u64; 4];
            for row in 0..raw.height {
                for col in 0..raw.width {
                    let c = raw.camera.cfa.color_at(row, col).min(3);
                    let x = v[row * raw.width + col];
                    cmax[c] = cmax[c].max(x);
                    cmin[c] = cmin[c].min(x);
                }
            }
            for row in 0..raw.height {
                for col in 0..raw.width {
                    let c = raw.camera.cfa.color_at(row, col).min(3);
                    if v[row * raw.width + col] == cmax[c] {
                        at_max[c] += 1;
                    }
                }
            }
            for (c, name) in ["R", "G", "B", "E"].iter().enumerate() {
                if cmin[c] != u16::MAX {
                    println!(
                        "             {name}: min {} max {} ({} samples at max)",
                        cmin[c], cmax[c], at_max[c]
                    );
                }
            }

            let wl = raw.whitelevel.0.first().copied().unwrap_or(u16::MAX as u32) as f32;
            let floor = (wl * 0.6) as u16;
            let mut hist = std::collections::HashMap::new();
            for &x in v.iter().filter(|&&x| x >= floor) {
                *hist.entry(x).or_insert(0u64) += 1;
            }
            let mut top: Vec<_> = hist.into_iter().collect();
            top.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
            let total = v.len() as f64;
            let above = v.iter().filter(|&&x| x as f32 >= wl).count();
            println!(
                "saturation:  {} samples >= whitelevel {wl} ({:.2}% of frame)",
                above,
                above as f64 / total * 100.0
            );
            println!("most common values >= {floor} (value: count, % of frame):");
            for (val, n) in top.iter().take(6) {
                println!("             {val}: {n} ({:.2}%)", *n as f64 / total * 100.0);
            }
        }
        RawImageData::Float(v) => println!("data:        f32, {} samples", v.len()),
    }

    // `--sample fx,fy` reports mean raw value per CFA colour in a window around
    // a fractional image position — the way to see what the sensor actually
    // recorded somewhere the developed image looks wrong.
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--sample") {
        let spec = args.get(i + 1).context("--sample needs fx,fy")?;
        let (fx, fy) = spec.split_once(',').context("--sample wants fx,fy")?;
        let (fx, fy): (f32, f32) = (fx.parse()?, fy.parse()?);
        if let RawImageData::Integer(v) = &raw.data {
            let cx = (raw.width as f32 * fx) as usize;
            let cy = (raw.height as f32 * fy) as usize;
            let half = 32usize;
            let (mut sum, mut cnt) = ([0f64; 4], [0u64; 4]);
            let (mut wmin, mut wmax) = ([u16::MAX; 4], [0u16; 4]);
            for row in cy.saturating_sub(half)..(cy + half).min(raw.height) {
                for col in cx.saturating_sub(half)..(cx + half).min(raw.width) {
                    let c = raw.camera.cfa.color_at(row, col).min(3);
                    let x = v[row * raw.width + col];
                    sum[c] += x as f64;
                    cnt[c] += 1;
                    wmin[c] = wmin[c].min(x);
                    wmax[c] = wmax[c].max(x);
                }
            }
            let blk = raw.blacklevel.levels.first().map(|r| r.n as f64 / r.d as f64).unwrap_or(0.0);
            let wht = raw.whitelevel.0.first().copied().unwrap_or(65535) as f64;
            println!("sample at ({fx}, {fy}), {}x{} window:", half * 2, half * 2);
            for (c, name) in ["R", "G", "B", "E"].iter().enumerate() {
                if cnt[c] == 0 {
                    continue;
                }
                let mean = sum[c] / cnt[c] as f64;
                let norm = (mean - blk) / (wht - blk);
                println!(
                    "             {name}: raw {:5}..{:5} mean {mean:8.1}  normalized {norm:.4}  \
                     after WB gain {:.4}",
                    wmin[c],
                    wmax[c],
                    norm * raw.wb_coeffs[c].max(0.0) as f64
                        / raw.wb_coeffs[1].max(f32::MIN_POSITIVE) as f64
                );
            }
        }
    }

    if let Some(out_path) = std::env::args().nth(2).filter(|a| !a.starts_with("--")) {
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
