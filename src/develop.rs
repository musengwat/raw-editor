//! Develop stage: CFA sensor data → demosaiced, scene-referred camera RGB f32.
//!
//! black-level subtract → normalize → bilinear demosaic → clip mask. White
//! balance and the camera→sRGB matrix are *not* applied here: they travel with
//! the image as [`ImageColour`] and are applied per pixel by
//! `pipeline::to_working` (and its shader twin), so both can be changed live.
//! Output is unclamped.

use anyhow::{Context, Result, bail};
use rawler::imgop::xyz::Illuminant;
use rawler::{RawImage, RawImageData};
use rayon::prelude::*;

use crate::pipeline::{Colour, ImageColour, WB_GAIN_MAX, WB_GAIN_MIN, to_working};

/// Half-width of the eyedropper's averaging window (9x9 pixels), so one noisy
/// pixel cannot set the white balance.
pub const PICK_RADIUS: usize = 4;

/// Why the eyedropper could not produce a white balance.
#[derive(Debug, PartialEq)]
pub enum PickError {
    /// Every pixel in the window is clipped, so its colour is unknown.
    AllClipped,
    /// A channel is essentially black; no gain can make that neutral.
    TooDark,
}

impl std::fmt::Display for PickError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::AllClipped => "that spot is blown out — pick something neutral that isn't clipped",
            Self::TooDark => "that spot is too dark to judge — pick something brighter",
        })
    }
}

/// Result of a successful pick.
#[derive(Debug, PartialEq)]
pub struct Pick {
    /// `[r, 1, b]`, within `WB_GAIN_MIN..=WB_GAIN_MAX`.
    pub gains: [f32; 3],
    /// True when the ideal gains fell outside the allowed range.
    pub clamped: bool,
}

/// Developed image: tightly packed RGB f32, active-area size.
pub struct DevelopedImage {
    pub width: usize,
    pub height: usize,
    /// width*height*3 camera RGB samples, normalized but not white balanced,
    /// unclamped above 1.0.
    pub data: Vec<f32>,
    /// width*height, 1.0 where the pixel's demosaic neighbourhood held a
    /// saturated photosite, else 0.0. Empty means nothing clipped.
    pub clip: Vec<f32>,
    pub colour: ImageColour,
}

impl DevelopedImage {
    /// An image whose data is already linear sRGB (tests, synthetic input).
    pub fn linear_srgb(width: usize, height: usize, data: Vec<f32>) -> Self {
        Self { width, height, data, clip: Vec::new(), colour: ImageColour::LINEAR_SRGB }
    }

    /// Eyedropper: white-balance gains that make the area around `(x, y)`
    /// neutral. Averages camera RGB over a `PICK_RADIUS` window, skipping
    /// clipped pixels (their ratios are meaningless). Neutral camera RGB stays
    /// neutral through the matrix (its rows sum to 1), so the same gains work
    /// in both colour modes.
    pub fn neutral_gains(&self, x: usize, y: usize) -> Result<Pick, PickError> {
        let r = PICK_RADIUS;
        let mut sum = [0f64; 3];
        let mut n = 0u32;
        for py in y.saturating_sub(r)..(y + r + 1).min(self.height) {
            for px in x.saturating_sub(r)..(x + r + 1).min(self.width) {
                let i = py * self.width + px;
                if self.clip_at(i) > 0.0 {
                    continue;
                }
                for c in 0..3 {
                    sum[c] += self.data[i * 3 + c] as f64;
                }
                n += 1;
            }
        }
        if n == 0 {
            return Err(PickError::AllClipped);
        }
        let mean = sum.map(|v| (v / n as f64) as f32);
        if mean.iter().any(|&v| v < 1e-4) {
            return Err(PickError::TooDark);
        }
        let ideal = [mean[1] / mean[0], 1.0, mean[1] / mean[2]];
        let gains = ideal.map(|g| g.clamp(WB_GAIN_MIN, WB_GAIN_MAX));
        Ok(Pick { gains, clamped: gains != ideal })
    }

    pub fn clip_at(&self, i: usize) -> f32 {
        self.clip.get(i).copied().unwrap_or(0.0)
    }

    /// The whole image through [`to_working`] — the CPU path's equivalent of
    /// what the shader does to every texture sample.
    pub fn working(&self, col: &Colour) -> Vec<f32> {
        let mut out = vec![0f32; self.data.len()];
        out.par_chunks_mut(3).zip(self.data.par_chunks(3)).enumerate().for_each(|(i, (o, c))| {
            o.copy_from_slice(&to_working(col, [c[0], c[1], c[2]], self.clip_at(i)));
        });
        out
    }
}

/// Bradford-adapted sRGB D65 → XYZ D65 matrix (row-major).
const SRGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

pub fn develop(raw: &RawImage) -> Result<DevelopedImage> {
    if raw.cpp != 1 {
        bail!("expected bayer CFA data (cpp=1), got cpp={}", raw.cpp);
    }
    let data = match &raw.data {
        RawImageData::Integer(v) => v,
        RawImageData::Float(_) => bail!("float raw data not supported yet"),
    };

    let raw_w = raw.width;
    let raw_h = raw.height;
    let area = raw.active_area.unwrap_or(rawler::imgop::Rect {
        p: rawler::imgop::Point { x: 0, y: 0 },
        d: rawler::imgop::Dim2 { w: raw_w, h: raw_h },
    });
    let (ax, ay, aw, ah) = (area.p.x, area.p.y, area.d.w, area.d.h);

    // Per-CFA-color black and white levels (RGBE order, E falls back to G).
    let black: Vec<f32> = raw.blacklevel.levels.iter().map(|r| r.n as f32 / r.d as f32).collect();
    let white: Vec<f32> = raw.whitelevel.0.iter().map(|&w| w as f32).collect();
    let level = |arr: &[f32], c: usize, dflt: f32| -> f32 {
        *arr.get(c).or_else(|| arr.first()).unwrap_or(&dflt)
    };

    // WB gains normalized so green = 1. NaN entries (unused E slot) fall back to green.
    let g = raw.wb_coeffs[1];
    let wb: [f32; 4] = std::array::from_fn(|i| {
        let c = raw.wb_coeffs[i];
        if c.is_finite() && g.is_finite() && g > 0.0 { c / g } else { 1.0 }
    });

    let cfa = &raw.camera.cfa;

    // Where each colour actually saturates, which is not always the reported
    // white level (see `saturation_levels`).
    let sat = saturation_levels(data, raw_w, raw_h, cfa, &black, &white, &level);

    // Plane of normalized values in full-raw coordinates. Values are NOT
    // clamped at 1.0: clipped highlights are flagged after demosaicing, where
    // all three channels are known, and made neutral after white balance.
    let mut plane = vec![0f32; raw_w * raw_h];
    plane
        .par_chunks_mut(raw_w)
        .enumerate()
        .for_each(|(row, out_row)| {
            for col in 0..raw_w {
                let c = cfa.color_at(row, col);
                let blk = level(&black, c, 0.0);
                let wht = level(&white, c, u16::MAX as f32).max(blk + 1.0);
                let v = (data[row * raw_w + col] as f32 - blk) / (wht - blk);
                out_row[col] = v.max(0.0);
            }
        });

    // Per-colour saturation expressed in the same normalized units as
    // `plane`, so the demosaic loop can spot clipped photosites cheaply.
    let sat_plane: [f32; 4] = std::array::from_fn(|c| {
        let blk = level(&black, c, 0.0);
        let wht = level(&white, c, u16::MAX as f32).max(blk + 1.0);
        (sat[c] - blk) / (wht - blk)
    });

    // Camera → linear sRGB matrix from the D65 XYZ→cam matrix.
    let cam_to_srgb = srgb_from_cam_matrix(raw)?;

    // Bilinear demosaic (3x3 neighborhood average per missing channel).
    // Output cropped to the active area.
    let mut out = vec![0f32; aw * ah * 3];
    let mut clip = vec![0f32; aw * ah];
    out.par_chunks_mut(aw * 3)
        .zip(clip.par_chunks_mut(aw))
        .enumerate()
        .for_each(|(orow, (out_row, clip_row))| {
            let row = orow + ay;
            for ocol in 0..aw {
                let col = ocol + ax;
                let mut sum = [0f32; 3];
                let mut cnt = [0u32; 3];
                let mut clipped = false;
                for dr in -1isize..=1 {
                    for dc in -1isize..=1 {
                        let r = (row as isize + dr).clamp(0, raw_h as isize - 1) as usize;
                        let cl = (col as isize + dc).clamp(0, raw_w as isize - 1) as usize;
                        let c = cfa.color_at(r, cl).min(2);
                        let v = plane[r * raw_w + cl];
                        sum[c] += v;
                        cnt[c] += 1;
                        clipped |= v >= sat_plane[c];
                    }
                }
                let own = cfa.color_at(row, col).min(2);
                let mut cam = [0f32; 3];
                for c in 0..3 {
                    cam[c] = if c == own {
                        plane[row * raw_w + col]
                    } else if cnt[c] > 0 {
                        sum[c] / cnt[c] as f32
                    } else {
                        0.0
                    };
                }

                // Flag only: making the pixel neutral has to wait until white
                // balance is known (see `pipeline::to_working`).
                clip_row[ocol] = if clipped { 1.0 } else { 0.0 };
                out_row[ocol * 3..ocol * 3 + 3].copy_from_slice(&cam);
            }
        });

    let colour = ImageColour { as_shot_wb: [wb[0], wb[1], wb[2]], cam_to_srgb };
    Ok(DevelopedImage { width: aw, height: ah, data: out, clip, colour })
}

/// Detect, per CFA colour, the raw value at which this exposure actually
/// saturates.
///
/// The reported white level is not always where clipping happens. Panasonic's
/// extended low ISO (100 on a G85, below its base of 200) overexposes the sensor
/// and scales the result back down, so the analog ceiling lands far below the
/// reported level — around 2880 rather than 4095 — and highlights clip without
/// ever approaching it. Normalizing against the reported level then leaves
/// clipped greens and blues sitting at ~0.7 while red stays linear, and the
/// white-balance gains turn that into a magenta cast across every blown area.
///
/// Real saturation looks like a large plateau with almost nothing above it, so
/// that is what we look for: the tallest histogram bin in the upper range, kept
/// only if it dwarfs the typical bin and hardly any samples exceed it.
/// Otherwise the reported level stands.
pub fn saturation_levels(
    data: &[u16],
    raw_w: usize,
    raw_h: usize,
    cfa: &rawler::CFA,
    black: &[f32],
    white: &[f32],
    level: &impl Fn(&[f32], usize, f32) -> f32,
) -> [f32; 4] {
    /// A bin must hold at least this fraction of the colour's samples to count
    /// as populated — keeps stray hot pixels from setting the ceiling.
    const BIN_FLOOR: f64 = 0.0001;
    /// At most this fraction of samples may sit above the plateau.
    const MAX_ABOVE: f64 = 0.02;
    /// The plateau must be this many times taller than the run just below it.
    const PEAK_RATIO: f64 = 3.0;
    /// A genuine ceiling spans at most this fraction of the working range.
    const MAX_WIDTH: f32 = 0.10;
    /// Detected levels must agree within this ratio across colours.
    const COHERENCE: f32 = 1.10;

    let mut hists: Vec<Vec<u32>> = (0..4)
        .map(|c| {
            let wht = level(white, c, u16::MAX as f32);
            vec![0u32; wht.max(1.0) as usize + 1]
        })
        .collect();
    let mut totals = [0u64; 4];
    let mut over = [0u64; 4];

    for row in 0..raw_h {
        for col in 0..raw_w {
            let c = cfa.color_at(row, col).min(3);
            let x = data[row * raw_w + col] as usize;
            totals[c] += 1;
            if x < hists[c].len() {
                hists[c][x] += 1;
            } else {
                over[c] += 1;
            }
        }
    }

    let candidates: [Option<f32>; 4] = std::array::from_fn(|c| {
        let blk = level(black, c, 0.0);
        let reported = level(white, c, u16::MAX as f32);
        let hist = &hists[c];
        if totals[c] == 0 || hist.len() < 16 {
            return None;
        }
        // Only look above the shadows; a big flat dark area is not saturation.
        let lo = (blk + (reported - blk) * 0.3).max(1.0) as usize;
        if lo + 8 >= hist.len() {
            return None;
        }

        // The clipping plateau is the mode of the upper range. In a normally
        // exposed frame the mode sits mid-histogram with plenty of brighter
        // pixels above it, which is what the `above` test below rejects.
        let (peak_off, &peak_n) = hist[lo..]
            .iter()
            .enumerate()
            .max_by_key(|&(_, &n)| n)
            .expect("non-empty range");
        let top = lo + peak_off;
        if peak_n < (totals[c] as f64 * BIN_FLOOR).max(8.0) as u32 || top <= lo + 8 {
            if std::env::var_os("RAW_EDITOR_DEBUG").is_some() {
                eprintln!("sat[{c}]: bail FLOOR peak={top} n={peak_n} floor={:.0} lo={lo}", (totals[c] as f64 * BIN_FLOOR).max(8.0));
            }
            return None;
        }

        // The plateau has width — the analog ceiling smears across several
        // digital values — so walk out from the peak while bins stay tall to
        // find its edges. Measuring "above" from the peak itself would count
        // the plateau's own upper half.
        let shoulder = peak_n / 5;
        let mut hi = top;
        while hi + 1 < hist.len() && hist[hi + 1] >= shoulder {
            hi += 1;
        }
        let mut band_lo = top;
        while band_lo > lo && hist[band_lo - 1] >= shoulder {
            band_lo -= 1;
        }

        // Anything meaningfully above a genuine clipping plateau is noise only.
        // `over` counts samples beyond the reported level, which the histogram
        // does not have bins for.
        let above: u64 = hist[hi + 1..].iter().map(|&n| n as u64).sum::<u64>() + over[c];
        if above as f64 >= totals[c] as f64 * MAX_ABOVE {
            if std::env::var_os("RAW_EDITOR_DEBUG").is_some() {
                eprintln!("sat[{c}]: bail ABOVE peak={top} n={peak_n} band={band_lo}..{hi} above={above} limit={:.0}", totals[c] as f64 * MAX_ABOVE);
            }
            return None;
        }

        // A real ceiling is narrow — the analog limit smears over a handful of
        // digital values, not a broad swathe. A flat-ish histogram would
        // otherwise let the walk above swallow the whole range and call it a
        // plateau.
        if (hi - band_lo) as f32 > (reported - lo as f32) * MAX_WIDTH {
            if std::env::var_os("RAW_EDITOR_DEBUG").is_some() {
                eprintln!("sat[{c}]: bail WIDTH band={band_lo}..{hi}");
            }
            return None;
        }

        // A plateau piles many samples into a narrow band; a scene that simply
        // had no bright tones tapers off smoothly. Compare the band against the
        // run just below it.
        let width = (hi - band_lo + 1).max(4);
        let ref_lo = band_lo.saturating_sub(width * 6).max(lo);
        if band_lo <= ref_lo + 2 {
            return None;
        }
        let mean = |r: std::ops::Range<usize>| -> f64 {
            let n = r.len().max(1);
            hist[r].iter().map(|&x| x as f64).sum::<f64>() / n as f64
        };
        let plateau = mean(band_lo..(hi + 1).min(hist.len()));
        let below = mean(ref_lo..band_lo).max(1.0);
        let top = band_lo;

        let is_plateau = plateau >= below * PEAK_RATIO && (top as f32) < reported * 0.98;

        if std::env::var_os("RAW_EDITOR_DEBUG").is_some() {
            eprintln!(
                "sat[{c}]: reported {reported} top {top} above {above} \
                 plateau {plateau:.0} below {below:.0} ratio {:.2} -> {}",
                plateau / below,
                if is_plateau { "PLATEAU" } else { "reported" }
            );
        }

        // Sit just under the plateau so its whole width counts as clipped.
        if is_plateau { Some(top as f32 * 0.995) } else { None }
    });

    // Analog saturation is a property of the sensor, so a genuine ceiling shows
    // up in more than one colour at close to the same level. A single colour
    // peaking on its own is just a big flat area in the scene — a bright sky or
    // a beach — and forcing that neutral would drain the colour out of it.
    let found: Vec<f32> = candidates.iter().flatten().copied().collect();
    let coherent = found.len() >= 2
        && found.iter().cloned().fold(f32::MIN, f32::max)
            <= found.iter().cloned().fold(f32::MAX, f32::min) * COHERENCE;

    std::array::from_fn(|c| {
        let reported = level(white, c, u16::MAX as f32);
        match candidates[c] {
            Some(v) if coherent => v,
            _ => reported,
        }
    })
}

/// Build the camera→linear-sRGB matrix: row-normalize (XYZ→cam · sRGB→XYZ),
/// then invert. Row normalization makes camera white (1,1,1) map to sRGB
/// white after white balance — the dcraw convention.
fn srgb_from_cam_matrix(raw: &RawImage) -> Result<[[f32; 3]; 3]> {
    let flat = raw
        .color_matrix
        .get(&Illuminant::D65)
        .or_else(|| raw.color_matrix.values().next())
        .context("no color matrix in raw file")?;
    if flat.len() < 9 {
        bail!("unexpected color matrix size {}", flat.len());
    }
    let mut xyz_to_cam = [[0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            xyz_to_cam[i][j] = flat[i * 3 + j];
        }
    }
    let mut cam_from_srgb = mat_mul(&xyz_to_cam, &SRGB_TO_XYZ);
    for row in &mut cam_from_srgb {
        let s: f32 = row.iter().sum();
        if s.abs() > f32::EPSILON {
            row.iter_mut().for_each(|v| *v /= s);
        }
    }
    mat_inv(&cam_from_srgb).context("camera matrix not invertible")
}

pub fn mat_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut m = [[0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

pub fn mat_inv(m: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut inv = [[0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            // Cofactor expansion; note transposed indices (adjugate).
            let a = m[(j + 1) % 3][(i + 1) % 3] * m[(j + 2) % 3][(i + 2) % 3];
            let b = m[(j + 1) % 3][(i + 2) % 3] * m[(j + 2) % 3][(i + 1) % 3];
            inv[i][j] = (a - b) * inv_det;
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic value spread in 0..1000, decorrelated from the CFA phase
    /// (plain `i % n` lines up with the Bayer parity and skews each colour).
    fn spread(i: usize) -> usize {
        (i.wrapping_mul(2_654_435_761) >> 13) % 1000
    }

    /// Build an RGGB frame from a per-colour value generator.
    fn synth(w: usize, h: usize, mut f: impl FnMut(usize, usize) -> u16) -> (Vec<u16>, rawler::CFA) {
        let cfa = rawler::CFA::new("RGGB");
        let mut data = vec![0u16; w * h];
        for row in 0..h {
            for col in 0..w {
                data[row * w + col] = f(cfa.color_at(row, col).min(2), row * w + col);
            }
        }
        (data, cfa)
    }

    fn levels() -> (Vec<f32>, Vec<f32>, impl Fn(&[f32], usize, f32) -> f32) {
        (vec![0.0], vec![4095.0], |arr: &[f32], c: usize, d: f32| {
            *arr.get(c).or_else(|| arr.first()).unwrap_or(&d)
        })
    }

    /// Panasonic extended-ISO case: green and blue pile up on an analog ceiling
    /// far below the reported white level, red stays linear.
    #[test]
    fn saturation_found_when_colours_agree() {
        let (black, white, level) = levels();
        let (data, cfa) = synth(256, 256, |c, i| {
            let r = spread(i);
            match c {
                0 => 1300 + r as u16 / 2,               // red: linear, no ceiling
                _ if r < 700 => 2870 + (r % 5) as u16,  // green/blue: on the plateau
                _ => 1300 + r as u16,                   // ...and well below it
            }
        });
        let sat = saturation_levels(&data, 256, 256, &cfa, &black, &white, &level);
        assert!(sat[1] < 2900.0, "green ceiling not detected: {}", sat[1]);
        assert!(sat[2] < 2900.0, "blue ceiling not detected: {}", sat[2]);
        assert!(sat[1] > 2700.0, "green ceiling too low: {}", sat[1]);
    }

    /// A smoothly exposed frame has no ceiling below the reported white level;
    /// inventing one would flatten real highlight colour.
    #[test]
    fn no_saturation_on_smooth_data() {
        let (black, white, level) = levels();
        let (data, cfa) = synth(256, 256, |c, i| {
            let base = (spread(i) * 35 / 10) as u16;
            match c {
                0 => base / 2,
                1 => base,
                _ => base * 3 / 4,
            }
        });
        let sat = saturation_levels(&data, 256, 256, &cfa, &black, &white, &level);
        assert_eq!(sat[..3], [4095.0, 4095.0, 4095.0], "invented a ceiling: {sat:?}");
    }

    /// One colour peaking alone is a scene feature (a big flat bright area), not
    /// a sensor ceiling, so it must be ignored.
    #[test]
    fn single_colour_peak_is_ignored() {
        let (black, white, level) = levels();
        let (data, cfa) = synth(256, 256, |c, i| {
            let r = spread(i);
            match c {
                0 if r < 700 => 2800 + (r % 5) as u16, // only red piles up
                0 => 1300 + r as u16,
                1 => (r * 39 / 10) as u16,
                _ => (r * 35 / 10) as u16,
            }
        });
        let sat = saturation_levels(&data, 256, 256, &cfa, &black, &white, &level);
        assert_eq!(sat[0], 4095.0, "single-colour peak was treated as saturation");
    }

    #[test]
    fn mat_inv_roundtrip() {
        let m = [[0.9f32, 0.1, 0.0], [0.05, 1.1, -0.15], [0.02, -0.3, 1.28]];
        let inv = mat_inv(&m).unwrap();
        let id = mat_mul(&m, &inv);
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((id[i][j] - expect).abs() < 1e-5, "id[{i}][{j}] = {}", id[i][j]);
            }
        }
    }

    #[test]
    fn mat_inv_singular_is_none() {
        let m = [[1.0f32, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 0.0]];
        assert!(mat_inv(&m).is_none());
    }

    // --- eyedropper -------------------------------------------------------

    fn flat(w: usize, h: usize, rgb: [f32; 3]) -> DevelopedImage {
        DevelopedImage::linear_srgb(w, h, (0..w * h).flat_map(|_| rgb).collect())
    }

    #[test]
    fn eyedropper_gains_make_the_spot_neutral() {
        // IR-ish camera RGB: red above green, blue far below.
        let img = flat(20, 20, [0.30, 0.25, 0.11]);
        let pick = img.neutral_gains(10, 10).unwrap();
        assert!(!pick.clamped);
        let col = Colour { wb: pick.gains, matrix: crate::pipeline::IDENTITY3 };
        let out = to_working(&col, [0.30, 0.25, 0.11], 0.0);
        for v in out {
            assert!((v - 0.25).abs() < 1e-5, "{out:?}");
        }
    }

    #[test]
    fn eyedropper_ignores_clipped_pixels() {
        let mut img = flat(20, 20, [0.2, 0.4, 0.1]);
        // A blown, wrongly-coloured blob covering most of the window.
        for y in 6..14 {
            for x in 6..14 {
                let i = y * 20 + x;
                img.data[i * 3..i * 3 + 3].copy_from_slice(&[0.9, 0.5, 0.9]);
            }
        }
        img.clip = (0..400)
            .map(|i| if (6..14).contains(&(i % 20)) && (6..14).contains(&(i / 20)) { 1.0 } else { 0.0 })
            .collect();
        let pick = img.neutral_gains(10, 10).unwrap();
        assert!((pick.gains[0] - 2.0).abs() < 1e-5 && (pick.gains[2] - 4.0).abs() < 1e-5, "{pick:?}");
    }

    #[test]
    fn eyedropper_refuses_all_clipped_or_black() {
        let mut img = flat(10, 10, [0.9, 0.9, 0.9]);
        img.clip = vec![1.0; 100];
        assert_eq!(img.neutral_gains(5, 5), Err(PickError::AllClipped));
        assert_eq!(flat(10, 10, [0.2, 0.2, 0.0]).neutral_gains(5, 5), Err(PickError::TooDark));
    }

    #[test]
    fn eyedropper_clamps_to_the_slider_range_and_says_so() {
        let pick = flat(10, 10, [0.5, 0.5, 0.01]).neutral_gains(0, 0).unwrap();
        assert!(pick.clamped);
        assert_eq!(pick.gains, [1.0, 1.0, WB_GAIN_MAX]);
    }

    #[test]
    fn eyedropper_window_is_clipped_at_the_image_edge() {
        let mut img = flat(12, 12, [0.2, 0.2, 0.2]);
        // Only the corner window's pixels should count.
        img.data[0..3].copy_from_slice(&[0.1, 0.2, 0.4]);
        let far = img.neutral_gains(11, 11).unwrap();
        assert_eq!(far.gains, [1.0, 1.0, 1.0]);
        let corner = img.neutral_gains(0, 0).unwrap();
        assert!(corner.gains[0] > 1.0 && corner.gains[2] < 1.0, "{corner:?}");
    }
}

