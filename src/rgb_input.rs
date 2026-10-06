//! Photos that arrive as full-colour pixels rather than a sensor mosaic:
//! reduced-size raws (Sony A7 IV M/S lossless, cpp = 3) and TIFFs. Both skip
//! the demosaic and land in the same [`DevelopedImage`] the rest of the
//! pipeline already understands — the colour data just says what space the
//! pixels are in.

use std::path::Path;

use anyhow::{Context, Result, bail};
use rawler::{RawImage, RawImageData};
use rayon::prelude::*;

use crate::develop::DevelopedImage;
use crate::pipeline::ImageColour;

/// A pixel counts as clipped when every channel is within this fraction of
/// the ceiling. All three, not any: these files are already white balanced
/// (or finished), so a single channel at the top is usually a saturated
/// colour, not a blown highlight, and forcing it neutral would turn a pure red
/// into white.
const CLIP_FRACTION: f32 = 0.995;

fn clip_mask(data: &[f32]) -> Vec<f32> {
    let clip: Vec<f32> = data
        .par_chunks(3)
        .map(|p| if p.iter().all(|&v| v >= CLIP_FRACTION) { 1.0 } else { 0.0 })
        .collect();
    if clip.iter().any(|&c| c > 0.0) { clip } else { Vec::new() }
}

/// Where a reduced-size raw really saturates. Sony's M/S lossless files
/// report a white level (15360) that is neither their scale nor their ceiling:
/// the camera writes white-balanced YCbCr, rawler converts it, and blown
/// areas sit on one plateau shared by all three channels (~18877 on the
/// A7 IV samples) with only conversion overshoot above it. Find that plateau:
/// the most common value in the top part of each channel's histogram,
/// corroborated by at least two channels. Without one, fall back to the
/// largest value present — nothing clipped, so nothing should normalize past 1.
pub fn rgb_ceiling(data: &[u16], black: f32) -> f32 {
    let max = data.iter().copied().max().unwrap_or(0) as usize;
    if max == 0 {
        return black + 1.0;
    }
    let mut hist = vec![vec![0u32; max + 1]; 3];
    for p in data.chunks_exact(3) {
        for c in 0..3 {
            hist[c][p[c] as usize] += 1;
        }
    }
    let pixels = (data.len() / 3) as u32;
    let b = (black.max(0.0) as usize).min(max);
    let floor = b + (max - b) / 2;
    let peaks: Vec<usize> = hist
        .iter()
        .filter_map(|h| {
            let (v, &n) = h.iter().enumerate().skip(floor).max_by_key(|&(_, n)| *n)?;
            // A real plateau stands far above the values around it. It can be
            // a tiny share of the frame (a patch of blown sky was 0.04% on
            // the A7 IV M sample), so the share bar is low; the neighbour
            // test does the work. The plateau itself spreads over a few
            // values, hence the 30..60 gap.
            let around = |lo: usize, hi: usize| -> f64 {
                let (lo, hi) = (lo.min(h.len()), hi.min(h.len()));
                if hi > lo { h[lo..hi].iter().map(|&x| x as f64).sum::<f64>() / (hi - lo) as f64 } else { 0.0 }
            };
            let near = around(v.saturating_sub(60), v.saturating_sub(30)).max(around(v + 30, v + 60));
            (n > pixels / 100_000 && n > 8 && n as f64 > 8.0 * near + 8.0).then_some(v)
        })
        .collect();
    let mut best = None;
    for &a in &peaks {
        let agree = peaks.iter().filter(|&&b| (a as f32 - b as f32).abs() <= a as f32 * 0.01).count();
        if agree >= 2 {
            best = Some(best.map_or(a, |b: usize| b.min(a)));
        }
    }
    best.map_or(max as f32, |v| v as f32)
}

/// A raw with three samples per pixel (rawler's `LinearRaw`): already
/// demosaiced and, for Sony's M/S files, already white balanced (rawler
/// reports unit WB coefficients for them). Camera RGB still needs the
/// camera matrix, which the rest of the pipeline applies as usual.
pub fn develop_linear_raw(raw: &RawImage, cam_to_srgb: [[f32; 3]; 3], turns: u8) -> Result<DevelopedImage> {
    if raw.cpp != 3 {
        bail!("expected three samples per pixel, got {}", raw.cpp);
    }
    let RawImageData::Integer(data) = &raw.data else { bail!("float raw data not supported yet") };
    let black = raw.blacklevel.levels.first().map_or(0.0, |r| r.n as f32 / r.d as f32);
    let white = rgb_ceiling(data, black).max(black + 1.0);
    if std::env::var_os("RAW_EDITOR_DEBUG").is_some() {
        eprintln!("linear raw: black {black}, ceiling {white} (reported {:?})", raw.whitelevel.0);
    }

    let area = raw.crop_area.or(raw.active_area).unwrap_or(rawler::imgop::Rect {
        p: rawler::imgop::Point { x: 0, y: 0 },
        d: rawler::imgop::Dim2 { w: raw.width, h: raw.height },
    });
    let (ax, ay, aw, ah) = (area.p.x, area.p.y, area.d.w, area.d.h);
    let scale = 1.0 / (white - black);
    let mut out = vec![0f32; aw * ah * 3];
    out.par_chunks_mut(aw * 3).enumerate().for_each(|(row, o)| {
        let src = &data[((row + ay) * raw.width + ax) * 3..][..aw * 3];
        for (d, &s) in o.iter_mut().zip(src) {
            *d = ((s as f32 - black) * scale).max(0.0);
        }
    });
    let clip = clip_mask(&out);
    let wb = raw.wb_coeffs;
    let g = wb[1];
    let as_shot = std::array::from_fn(|i| if wb[i].is_finite() && wb[i] > 0.0 && g > 0.0 { wb[i] / g } else { 1.0 });
    let colour = ImageColour { as_shot_wb: as_shot, cam_to_srgb, turns };
    Ok(DevelopedImage { width: aw, height: ah, data: out, clip, colour, flat: None })
}

/// sRGB transfer function, encoded → linear.
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// Open a TIFF as linear sRGB. 8- and 16-bit files are taken to be
/// sRGB-encoded (which is what this editor's own 16-bit exports are); float
/// files are taken to be linear already, which is the convention for
/// scene-referred float TIFFs. An embedded ICC profile is not applied — see
/// [`tiff_profile_note`].
pub fn open_tiff(path: &Path) -> Result<DevelopedImage> {
    let reader = image::ImageReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .with_guessed_format()?;
    let img = reader.decode().with_context(|| format!("decoding {}", path.display()))?;
    let float = matches!(img.color(), image::ColorType::Rgb32F | image::ColorType::Rgba32F);
    let rgb = img.into_rgb32f();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut data = rgb.into_raw();
    if !float {
        data.par_iter_mut().for_each(|v| *v = srgb_to_linear(*v));
    }
    let clip = if float { Vec::new() } else { clip_mask(&data) };
    Ok(DevelopedImage { width: w, height: h, data, clip, colour: ImageColour::LINEAR_SRGB, flat: None })
}

/// A warning to show when a TIFF carries a colour profile the editor will not
/// honour. Profiles that are sRGB (by their description) are fine.
pub fn tiff_profile_note(path: &Path) -> Option<String> {
    use image::ImageDecoder;
    let file = std::io::BufReader::new(std::fs::File::open(path).ok()?);
    let mut dec = image::codecs::tiff::TiffDecoder::new(file).ok()?;
    let icc = dec.icc_profile().ok()??;
    let text = String::from_utf8_lossy(&icc).to_ascii_lowercase();
    let srgb = text.contains("srgb") || text.contains("s\0r\0g\0b") || text.contains("iec61966");
    (!srgb).then(|| "this TIFF has a non-sRGB colour profile, shown as if it were sRGB".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_transfer_endpoints_and_mid_grey() {
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        // sRGB 0.5 is ~21.4% linear.
        assert!((srgb_to_linear(0.5) - 0.214).abs() < 1e-3);
    }

    #[test]
    fn clip_needs_all_three_channels() {
        let data = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5];
        assert_eq!(clip_mask(&data), vec![0.0, 1.0, 0.0], "pure red is a colour, not a clip");
        assert!(clip_mask(&[0.5; 6]).is_empty(), "nothing clipped → no mask");
    }

    #[test]
    fn ceiling_finds_a_shared_plateau() {
        // A ramp of scene values, plus 5% of the frame blown to 18877 in all
        // channels, plus a little conversion overshoot above it.
        let mut d = Vec::new();
        for i in 0..20_000u32 {
            let v = 1024 + (i % 12_000) as u16;
            d.extend_from_slice(&[v, v + 10, v / 2 + 600]);
        }
        for _ in 0..1000 {
            d.extend_from_slice(&[18877, 18877, 18876]);
        }
        d.extend_from_slice(&[21000, 18000, 19500]);
        assert_eq!(rgb_ceiling(&d, 1024.0), 18876.0);
    }

    #[test]
    fn ceiling_without_a_plateau_is_the_maximum() {
        let d: Vec<u16> = (0..30_000u32).map(|i| 1024 + (i % 9000) as u16).collect();
        assert_eq!(rgb_ceiling(&d, 1024.0), d.iter().copied().max().unwrap() as f32);
    }

    #[test]
    fn single_colour_peak_is_not_a_ceiling() {
        // Lots of one blue value (a sky), but red and green spread out.
        let mut d = Vec::new();
        for i in 0..20_000u32 {
            d.extend_from_slice(&[1024 + (i % 9000) as u16, 1100 + (i % 8000) as u16, 9000]);
        }
        assert_eq!(rgb_ceiling(&d, 1024.0), 10023.0);
    }
}
