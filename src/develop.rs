//! Develop stage: CFA sensor data → demosaiced, scene-referred camera RGB f32.
//!
//! black-level subtract → normalize → Malvar-He-Cutler demosaic → clip mask. White
//! balance and the camera→sRGB matrix are *not* applied here: they travel with
//! the image as [`ImageColour`] and are applied per pixel by
//! `pipeline::to_working` (and its shader twin), so both can be changed live.
//! Output is unclamped.

use anyhow::{Context, Result, bail};
use rawler::imgop::xyz::Illuminant;
use rawler::{RawImage, RawImageData};
use rayon::prelude::*;

use crate::flatfield::{FlatField, Optics};
use crate::pipeline::{AdjustParams, Colour, ImageColour, WB_GAIN_MAX, WB_GAIN_MIN, to_working};

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
#[derive(Clone)]
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
    /// The flat field saved for this photo's lens, if any. Set by whoever
    /// opened the file (it comes from the settings, keyed by lens); used only
    /// when `AdjustParams::flat_field` is on.
    pub flat: Option<std::sync::Arc<FlatField>>,
}

impl DevelopedImage {
    /// An image whose data is already linear sRGB (tests, synthetic input).
    pub fn linear_srgb(width: usize, height: usize, data: Vec<f32>) -> Self {
        Self { width, height, data, clip: Vec::new(), colour: ImageColour::LINEAR_SRGB, flat: None }
    }

    /// Eyedropper: white-balance gains that make the area around `(x, y)`
    /// neutral. Averages camera RGB over a `PICK_RADIUS` window, skipping
    /// clipped pixels (their ratios are meaningless). Neutral camera RGB stays
    /// neutral through the matrix (its rows sum to 1), so the same gains work
    /// in both colour modes. The optics correction in `p` comes first, as it
    /// does in the pipeline, so a pick inside a coloured hot spot is judged
    /// on corrected colour.
    pub fn neutral_gains(&self, x: usize, y: usize, p: &AdjustParams) -> Result<Pick, PickError> {
        let optics = self.optics(p);
        let r = PICK_RADIUS;
        let mut sum = [0f64; 3];
        let mut n = 0u32;
        for py in y.saturating_sub(r)..(y + r + 1).min(self.height) {
            for px in x.saturating_sub(r)..(x + r + 1).min(self.width) {
                let i = py * self.width + px;
                if self.clip_at(i) > 0.0 {
                    continue;
                }
                let k = optics.gain_at(px, py, self.width, self.height);
                for c in 0..3 {
                    sum[c] += (self.data[i * 3 + c] * k[c]) as f64;
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

    /// Optics correction (flat field + hot spot) for one render of this image.
    pub fn optics(&self, p: &AdjustParams) -> Optics<'_> {
        Optics::new(p, self.flat.as_deref(), self.width, self.height)
    }

    /// Camera RGB of pixel `i` with the optics gain applied.
    pub fn camera_rgb(&self, optics: &Optics, i: usize) -> [f32; 3] {
        let c = [self.data[i * 3], self.data[i * 3 + 1], self.data[i * 3 + 2]];
        if optics.is_identity() {
            return c;
        }
        let k = optics.gain_at(i % self.width, i / self.width, self.width, self.height);
        [c[0] * k[0], c[1] * k[1], c[2] * k[2]]
    }

    /// The whole image through optics and [`to_working`] — the CPU path's
    /// equivalent of what the shader does to every texture sample.
    pub fn working(&self, p: &AdjustParams) -> Vec<f32> {
        let col = Colour::resolve(&self.colour, p);
        let optics = self.optics(p);
        let mut out = vec![0f32; self.data.len()];
        out.par_chunks_mut(3).enumerate().for_each(|(i, o)| {
            o.copy_from_slice(&to_working(&col, self.camera_rgb(&optics, i), self.clip_at(i)));
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

/// Clockwise quarter turns that show a frame upright, from its EXIF
/// orientation (6 = turn 90° clockwise, 8 = 90° anticlockwise). Mirrored
/// orientations do not come out of cameras held normally and are ignored.
pub fn orientation_turns(o: rawler::decoders::Orientation) -> u8 {
    use rawler::decoders::Orientation::*;
    match o {
        Rotate90 => 1,
        Rotate180 => 2,
        Rotate270 => 3,
        _ => 0,
    }
}

/// Panasonic RW2 black-level tags (0x1C–0x1E) are 15 short of the real
/// black: the GF1 reports 0 but its darkest values pile up at 16, the G85
/// reports 128 but its floor sits at ~143 (dcraw/LibRaw add the same 15).
/// rawler uses the tag as is, so without this every pixel carried a +15
/// pedestal — invisible in bright channels, but behind a #16 orange filter
/// the blue (IR) channel's median is ~95, so it was a 17% error that the
/// Aerochrome mix turned into maroon skies and red shadows.
pub fn black_pedestal(make: &str) -> f32 {
    if make.to_ascii_lowercase().contains("panasonic") { 15.0 } else { 0.0 }
}

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
    // The camera's own crop where there is one (Sony frames carry a 12x8 px
    // border of edge pixels inside the active area), else the active area.
    let area = raw.crop_area.or(raw.active_area).unwrap_or(rawler::imgop::Rect {
        p: rawler::imgop::Point { x: 0, y: 0 },
        d: rawler::imgop::Dim2 { w: raw_w, h: raw_h },
    });
    let (ax, ay, aw, ah) = (area.p.x, area.p.y, area.d.w, area.d.h);

    // Per-CFA-color black and white levels (RGBE order, E falls back to G).
    let pedestal = black_pedestal(&raw.make);
    let black: Vec<f32> = raw.blacklevel.levels.iter().map(|r| r.n as f32 / r.d as f32 + pedestal).collect();
    let white: Vec<f32> = raw.whitelevel.0.iter().map(|&w| w as f32).collect();
    let level = |arr: &[f32], c: usize, dflt: f32| -> f32 {
        *arr.get(c).or_else(|| arr.first()).unwrap_or(&dflt)
    };

    // WB gains normalized so green = 1. NaN entries (unused E slot) fall back to green.
    let g = raw.wb_coeffs[1];
    let wb: [f32; 4] = std::array::from_fn(|i| {
        let c = raw.wb_coeffs[i];
        if c.is_finite() && c > 0.0 && g.is_finite() && g > 0.0 { c / g } else { 1.0 }
    });

    let cfa = &raw.camera.cfa;

    // Where each colour actually saturates, which is not always the reported
    // white level (see `saturation_levels`).
    let sat = saturation_levels(data, raw_w, raw_h, cfa, &black, &white, &level);

    // Plane of normalized values in full-raw coordinates. Values are NOT
    // clamped at 1.0: clipped highlights are flagged after demosaicing, where
    // all three channels are known, and made neutral after white balance.
    // `saturated` marks photosites at or above their colour's real ceiling.
    let mut plane = vec![0f32; raw_w * raw_h];
    let mut saturated = vec![false; raw_w * raw_h];
    plane
        .par_chunks_mut(raw_w)
        .zip(saturated.par_chunks_mut(raw_w))
        .enumerate()
        .for_each(|(row, (out_row, sat_row))| {
            for col in 0..raw_w {
                let c = cfa.color_at(row, col);
                let blk = level(&black, c, 0.0);
                let wht = level(&white, c, u16::MAX as f32).max(blk + 1.0);
                let raw_v = data[row * raw_w + col] as f32;
                out_row[col] = ((raw_v - blk) / (wht - blk)).max(0.0);
                sat_row[col] = raw_v >= sat[c.min(3)];
            }
        });

    // Camera → linear sRGB matrix from the D65 XYZ→cam matrix.
    let cam_to_srgb = srgb_from_cam_matrix(raw)?;

    let mosaic = Mosaic {
        plane: &plane,
        saturated: &saturated,
        width: raw_w,
        height: raw_h,
        color_at: &|r, c| cfa.color_at(r, c).min(2),
        balance: [wb[0], wb[1], wb[2]],
    };
    let (out, clip) = demosaic(&mosaic, (ax, ay, aw, ah), Demosaic::Mhc);

    let colour = ImageColour { as_shot_wb: [wb[0], wb[1], wb[2]], cam_to_srgb, turns: orientation_turns(raw.orientation) };
    Ok(DevelopedImage { width: aw, height: ah, data: out, clip, colour, flat: None })
}

/// Demosaic algorithm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Demosaic {
    /// 3x3 average of each missing colour. Kept as the reference the MHC tests
    /// measure against, and as the fallback next to clipped photosites.
    Bilinear,
    /// Malvar-He-Cutler: bilinear plus a cross-channel Laplacian correction
    /// over a 5x5 window. See `docs/phase5-6-algorithms.md`.
    Mhc,
}

/// A normalized single-plane CFA frame, in full-raw coordinates.
pub struct Mosaic<'a, F: Fn(usize, usize) -> usize + Sync> {
    pub plane: &'a [f32],
    /// Photosites at or above their colour's saturation level.
    pub saturated: &'a [bool],
    pub width: usize,
    pub height: usize,
    /// CFA colour at `(row, col)`: 0 = R, 1 = G, 2 = B.
    pub color_at: &'a F,
    /// Per-channel gains that bring the plane to roughly neutral balance
    /// (the as-shot white balance). MHC runs in these units and the result is
    /// divided back out, so the output is still camera RGB. See [`Mosaic::mhc`].
    pub balance: [f32; 3],
}

/// Mirror an index into `0..n` around the edge sample (-1 → 1, n → n-2).
/// Unlike clamping, mirroring by whole pixels keeps the Bayer parity, so the
/// sample that comes back is the same colour the filter expects.
fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    let i = if i < 0 { -i } else { i };
    let i = if i >= n { 2 * (n - 1) - i } else { i };
    i.clamp(0, n - 1) as usize
}

impl<F: Fn(usize, usize) -> usize + Sync> Mosaic<'_, F> {
    fn idx(&self, row: usize, col: usize, dr: isize, dc: isize) -> usize {
        // Every offset used is within ±2, so away from the border no mirroring
        // can happen; skipping it is most of the demosaic's speed.
        if row >= 2 && col >= 2 && row + 2 < self.height && col + 2 < self.width {
            ((row as isize + dr) as usize) * self.width + (col as isize + dc) as usize
        } else {
            reflect(row as isize + dr, self.height) * self.width + reflect(col as isize + dc, self.width)
        }
    }

    fn at(&self, row: usize, col: usize, dr: isize, dc: isize) -> f32 {
        self.plane[self.idx(row, col, dr, dc)]
    }

    fn color(&self, row: usize, col: usize, dr: isize, dc: isize) -> usize {
        (self.color_at)(
            reflect(row as isize + dr, self.height),
            reflect(col as isize + dc, self.width),
        )
    }

    /// Any saturated photosite among the given offsets.
    fn any_saturated(&self, row: usize, col: usize, offsets: &[(isize, isize)]) -> bool {
        offsets.iter().any(|&(dr, dc)| self.saturated[self.idx(row, col, dr, dc)])
    }

    fn bilinear(&self, row: usize, col: usize) -> [f32; 3] {
        let mut sum = [0f32; 3];
        let mut cnt = [0u32; 3];
        for &(dr, dc) in &NEAR {
            let c = self.color(row, col, dr, dc);
            sum[c] += self.at(row, col, dr, dc);
            cnt[c] += 1;
        }
        let own = self.color(row, col, 0, 0);
        std::array::from_fn(|c| {
            if c == own {
                self.at(row, col, 0, 0)
            } else if cnt[c] > 0 {
                sum[c] / cnt[c] as f32
            } else {
                0.0
            }
        })
    }

    /// `plane` with `balance` applied, built once per demosaic so the filter
    /// taps don't each look up their colour.
    fn balanced(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.plane.len()];
        out.par_chunks_mut(self.width).enumerate().for_each(|(row, out_row)| {
            let src = &self.plane[row * self.width..(row + 1) * self.width];
            for (col, (o, &v)) in out_row.iter_mut().zip(src).enumerate() {
                *o = v * self.balance[(self.color_at)(row, col)];
            }
        });
        out
    }

    /// `balanced` is [`Mosaic::balanced`].
    fn mhc(&self, balanced: &[f32], row: usize, col: usize) -> [f32; 3] {
        // MHC adds green's Laplacian to red and blue unscaled, which assumes
        // the three channels sit at the same level on a neutral edge. Camera
        // RGB does not (GF1 red runs ~2.3x below green), so in raw units the
        // correction overshoots: dark sides of edges go teal and bright sides
        // pink. Filtering white-balanced values and undoing the gains after
        // keeps the texture in camera RGB without that fringe.
        let p = |dr, dc| balanced[self.idx(row, col, dr, dc)];
        let own = self.color(row, col, 0, 0);
        let centre = p(0, 0);
        let mut out = [0f32; 3];
        out[own] = centre;
        if own == 1 {
            // Green site: one of R/B sits left and right, the other above and
            // below. The orientation comes from the CFA itself, never from the
            // paper's figure (whose labels are easy to transpose).
            let diag = p(-1, -1) + p(-1, 1) + p(1, -1) + p(1, 1);
            let along_row = (5.0 * centre + 4.0 * (p(0, -1) + p(0, 1))
                - (p(0, -2) + p(0, 2))
                - diag
                + 0.5 * (p(-2, 0) + p(2, 0)))
                / 8.0;
            let along_col = (5.0 * centre + 4.0 * (p(-1, 0) + p(1, 0))
                - (p(-2, 0) + p(2, 0))
                - diag
                + 0.5 * (p(0, -2) + p(0, 2)))
                / 8.0;
            let row_colour = self.color(row, col, 0, 1);
            out[row_colour] = along_row;
            out[2 - row_colour] = along_col;
        } else {
            // Red or blue site: green from the 4 axial neighbours, the
            // opposite colour from the 4 diagonals.
            let axial1 = p(-1, 0) + p(1, 0) + p(0, -1) + p(0, 1);
            let axial2 = p(-2, 0) + p(2, 0) + p(0, -2) + p(0, 2);
            let diag = p(-1, -1) + p(-1, 1) + p(1, -1) + p(1, 1);
            out[1] = (4.0 * centre + 2.0 * axial1 - axial2) / 8.0;
            out[2 - own] = (6.0 * centre + 2.0 * diag - 1.5 * axial2) / 8.0;
        }
        // The Laplacian correction can overshoot below zero at hard edges;
        // light can't be negative, and the plane it came from never is.
        std::array::from_fn(|c| (out[c] / self.balance[c]).max(0.0))
    }
}

/// The 3x3 neighbourhood: bilinear's support, and the clip flag's.
const NEAR: [(isize, isize); 9] =
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)];

/// The part of MHC's support outside the 3x3: the four axial ±2 samples.
const FAR: [(isize, isize); 4] = [(-2, 0), (2, 0), (0, -2), (0, 2)];

/// Demosaic the `(x, y, w, h)` region of `m` into camera RGB, plus the clip
/// mask (1.0 where the pixel's 3x3 neighbourhood holds a saturated photosite).
///
/// MHC's Laplacian term trusts the ±2 samples to carry real detail; a
/// saturated one carries a ceiling instead and would ring. Pixels whose only
/// saturated neighbours are those outer samples therefore fall back to
/// bilinear — they are not clipped themselves, so they keep their colour, and
/// the clip mask stays the same 3x3 shape regardless of the method.
pub fn demosaic<F: Fn(usize, usize) -> usize + Sync>(
    m: &Mosaic<'_, F>,
    (ax, ay, aw, ah): (usize, usize, usize, usize),
    method: Demosaic,
) -> (Vec<f32>, Vec<f32>) {
    let balanced = if method == Demosaic::Mhc { m.balanced() } else { Vec::new() };
    let mut out = vec![0f32; aw * ah * 3];
    let mut clip = vec![0f32; aw * ah];
    out.par_chunks_mut(aw * 3)
        .zip(clip.par_chunks_mut(aw))
        .enumerate()
        .for_each(|(orow, (out_row, clip_row))| {
            let row = orow + ay;
            for ocol in 0..aw {
                let col = ocol + ax;
                // Flag only: making the pixel neutral has to wait until white
                // balance is known (see `pipeline::to_working`).
                let clipped = m.any_saturated(row, col, &NEAR);
                let rgb = match method {
                    Demosaic::Mhc if !clipped && !m.any_saturated(row, col, &FAR) => m.mhc(&balanced, row, col),
                    _ => m.bilinear(row, col),
                };
                clip_row[ocol] = if clipped { 1.0 } else { 0.0 };
                out_row[ocol * 3..ocol * 3 + 3].copy_from_slice(&rgb);
            }
        });
    (out, clip)
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
        let pick = img.neutral_gains(10, 10, &AdjustParams::default()).unwrap();
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
        let pick = img.neutral_gains(10, 10, &AdjustParams::default()).unwrap();
        assert!((pick.gains[0] - 2.0).abs() < 1e-5 && (pick.gains[2] - 4.0).abs() < 1e-5, "{pick:?}");
    }

    #[test]
    fn panasonic_black_level_gets_its_pedestal() {
        assert_eq!(black_pedestal("Panasonic"), 15.0);
        assert_eq!(black_pedestal("PANASONIC"), 15.0);
        assert_eq!(black_pedestal("Canon"), 0.0);
    }

    #[test]
    fn eyedropper_refuses_all_clipped_or_black() {
        let mut img = flat(10, 10, [0.9, 0.9, 0.9]);
        img.clip = vec![1.0; 100];
        assert_eq!(img.neutral_gains(5, 5, &AdjustParams::default()), Err(PickError::AllClipped));
        assert_eq!(flat(10, 10, [0.2, 0.2, 0.0]).neutral_gains(5, 5, &AdjustParams::default()), Err(PickError::TooDark));
    }

    #[test]
    fn eyedropper_clamps_to_the_slider_range_and_says_so() {
        let pick = flat(10, 10, [0.5, 0.5, 0.01]).neutral_gains(0, 0, &AdjustParams::default()).unwrap();
        assert!(pick.clamped);
        assert_eq!(pick.gains, [1.0, 1.0, WB_GAIN_MAX]);
    }

    #[test]
    fn eyedropper_window_is_clipped_at_the_image_edge() {
        let mut img = flat(12, 12, [0.2, 0.2, 0.2]);
        // Only the corner window's pixels should count.
        img.data[0..3].copy_from_slice(&[0.1, 0.2, 0.4]);
        let far = img.neutral_gains(11, 11, &AdjustParams::default()).unwrap();
        assert_eq!(far.gains, [1.0, 1.0, 1.0]);
        let corner = img.neutral_gains(0, 0, &AdjustParams::default()).unwrap();
        assert!(corner.gains[0] > 1.0 && corner.gains[2] < 1.0, "{corner:?}");
    }

    // --- demosaic ---------------------------------------------------------

    /// Mosaic an RGB image through `pattern`, then demosaic the whole frame.
    fn round_trip(w: usize, h: usize, rgb: &[f32], pattern: &str, method: Demosaic) -> Vec<f32> {
        round_trip_sat(w, h, rgb, pattern, method, &vec![false; w * h]).0
    }

    fn round_trip_sat(
        w: usize,
        h: usize,
        rgb: &[f32],
        pattern: &str,
        method: Demosaic,
        saturated: &[bool],
    ) -> (Vec<f32>, Vec<f32>) {
        let cfa = rawler::CFA::new(pattern);
        let color_at = |r: usize, c: usize| cfa.color_at(r, c).min(2);
        let plane: Vec<f32> = (0..w * h).map(|i| rgb[i * 3 + color_at(i / w, i % w)]).collect();
        let m = Mosaic { plane: &plane, saturated, width: w, height: h, color_at: &color_at, balance: [1.0; 3] };
        demosaic(&m, (0, 0, w, h), method)
    }

    /// A test scene shaped like a photograph as far as MHC is concerned: the
    /// detail (a hard diagonal edge, horizontal and vertical bars, a ring
    /// pattern) is shared by all three channels, and colour varies slowly on
    /// top of it. MHC's gain comes entirely from that inter-channel
    /// correlation; with independent per-channel detail it has nothing to
    /// borrow and barely beats bilinear.
    fn scene(w: usize, h: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
                let edge = if x + y / 2 > w / 2 { 0.3 } else { 0.0 };
                let bars = if (x / 5) % 2 == 0 { 0.12 } else { 0.0 } + if (y / 7) % 2 == 0 { 0.1 } else { 0.0 };
                let r2 = (fx - 0.7).powi(2) + (fy - 0.3).powi(2);
                let rings = 0.1 * (r2 * 300.0).sin();
                let luma = 0.25 + edge + bars + rings;
                v.push(luma * (0.7 + 0.6 * fx));
                v.push(luma * (0.9 + 0.2 * fy));
                v.push(luma * (1.2 - 0.5 * fx * fy));
            }
        }
        v
    }

    /// PSNR over the interior (a 2-pixel border is excluded; the mirrored
    /// edge is a different, smaller question tested separately).
    fn psnr(w: usize, h: usize, a: &[f32], b: &[f32]) -> f64 {
        let (mut se, mut n) = (0f64, 0usize);
        for y in 2..h - 2 {
            for x in 2..w - 2 {
                for c in 0..3 {
                    let i = (y * w + x) * 3 + c;
                    se += ((a[i] - b[i]) as f64).powi(2);
                    n += 1;
                }
            }
        }
        10.0 * (1.0 / (se / n as f64)).log10()
    }

    /// The test the algorithm notes ask for: a transposed G-site filter or a
    /// hardcoded RGGB phase would drop MHC below bilinear on at least one of
    /// these layouts. GF1 is GBRG; the others prove nothing is phase-specific.
    #[test]
    fn mhc_beats_bilinear_on_every_bayer_layout() {
        let (w, h) = (96, 80);
        let img = scene(w, h);
        for pattern in ["GBRG", "RGGB", "BGGR", "GRBG"] {
            let bil = psnr(w, h, &img, &round_trip(w, h, &img, pattern, Demosaic::Bilinear));
            let mhc = psnr(w, h, &img, &round_trip(w, h, &img, pattern, Demosaic::Mhc));
            assert!(mhc > bil + 1.0, "{pattern}: MHC {mhc:.2} dB vs bilinear {bil:.2} dB");
        }
    }

    /// Each filter's weights sum to one and its Laplacian terms cancel on a
    /// plane, so a per-channel linear ramp comes back exactly.
    #[test]
    fn mhc_reproduces_linear_ramps_exactly() {
        let (w, h) = (24, 20);
        let img: Vec<f32> = (0..w * h)
            .flat_map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                [0.01 * x + 0.02 * y, 0.5 - 0.01 * y, 0.2 + 0.015 * x]
            })
            .collect();
        for pattern in ["GBRG", "RGGB"] {
            let out = round_trip(w, h, &img, pattern, Demosaic::Mhc);
            for y in 2..h - 2 {
                for x in 2..w - 2 {
                    for c in 0..3 {
                        let i = (y * w + x) * 3 + c;
                        assert!((out[i] - img[i]).abs() < 1e-5, "{pattern} ({x},{y}) c{c}: {} vs {}", out[i], img[i]);
                    }
                }
            }
        }
    }

    /// Mirroring at the edges keeps the Bayer phase, so a flat colour is exact
    /// all the way to the border (clamping would mix in the wrong colour).
    #[test]
    fn flat_colour_is_exact_to_the_edge() {
        let (w, h) = (9, 7);
        let img: Vec<f32> = (0..w * h).flat_map(|_| [0.3, 0.6, 0.15]).collect();
        for pattern in ["GBRG", "RGGB", "BGGR", "GRBG"] {
            let out = round_trip(w, h, &img, pattern, Demosaic::Mhc);
            for (i, (a, b)) in out.iter().zip(&img).enumerate() {
                assert!((a - b).abs() < 1e-6, "{pattern} sample {i}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn reflect_mirrors_without_repeating_the_edge() {
        assert_eq!([-2, -1, 0, 4, 5, 6].map(|i| reflect(i, 5)), [2, 1, 0, 4, 3, 2]);
        assert_eq!(reflect(3, 1), 0);
    }

    /// The Laplacian overshoot at a hard edge must not produce negative light.
    #[test]
    fn mhc_output_is_never_negative() {
        let (w, h) = (16, 16);
        let img: Vec<f32> = (0..w * h)
            .flat_map(|i| if (i % w) < 8 { [0.0, 0.0, 0.0] } else { [1.0, 0.0, 1.0] })
            .collect();
        let out = round_trip(w, h, &img, "GBRG", Demosaic::Mhc);
        assert!(out.iter().all(|&v| v >= 0.0));
    }

    /// A saturated photosite two pixels away flags nothing (the clip mask is
    /// 3x3) but switches that pixel to bilinear; one pixel away flags it.
    #[test]
    fn saturated_neighbours_flag_near_and_fall_back_far() {
        let (w, h) = (16, 16);
        let img = scene(w, h);
        let mut sat = vec![false; w * h];
        sat[8 * w + 8] = true;
        let (mhc, clip) = round_trip_sat(w, h, &img, "GBRG", Demosaic::Mhc, &sat);
        let (bil, _) = round_trip_sat(w, h, &img, "GBRG", Demosaic::Bilinear, &sat);
        let clean = round_trip(w, h, &img, "GBRG", Demosaic::Mhc);
        let px = |v: &[f32], x: usize, y: usize| v[(y * w + x) * 3..(y * w + x) * 3 + 3].to_vec();
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x.abs_diff(8), y.abs_diff(8));
                let near = dx <= 1 && dy <= 1;
                let far = (dx == 2 && dy == 0) || (dx == 0 && dy == 2);
                assert_eq!(clip[y * w + x] > 0.0, near, "clip flag at ({x},{y})");
                let expect = if near || far { px(&bil, x, y) } else { px(&clean, x, y) };
                assert_eq!(px(&mhc, x, y), expect, "pixel ({x},{y})");
            }
        }
    }

    /// The trap the algorithm notes warn about: a green-site filter whose
    /// Laplacian runs along the wrong axis. The mixed scene above averages it
    /// away, so test stripes along each axis on their own. A correct filter
    /// gains ~5 dB on both and is symmetric; the transposed one gains ~2 dB
    /// along one axis and nothing along the other (checked by planting it).
    #[test]
    fn mhc_gain_holds_along_both_axes() {
        let (w, h) = (64, 64);
        for pattern in ["GBRG", "RGGB", "BGGR", "GRBG"] {
            let gain = |axis: usize| {
                let img: Vec<f32> = (0..w * h)
                    .flat_map(|i| {
                        let t = if axis == 0 { i % w } else { i / w } as f32;
                        let l = 0.4 + 0.25 * (t * std::f32::consts::TAU / 8.0).sin();
                        [l * 0.8, l, l * 1.3]
                    })
                    .collect();
                psnr(w, h, &img, &round_trip(w, h, &img, pattern, Demosaic::Mhc))
                    - psnr(w, h, &img, &round_trip(w, h, &img, pattern, Demosaic::Bilinear))
            };
            let (across, down) = (gain(0), gain(1));
            assert!(across > 4.0 && down > 4.0, "{pattern}: gain {across:.2} / {down:.2} dB");
            assert!((across - down).abs() < 0.1, "{pattern}: asymmetric {across:.2} / {down:.2} dB");
        }
    }

    /// Camera RGB demosaiced with its white balance passed as `balance` must
    /// equal the neutral image demosaiced directly, once the gains are put
    /// back; without the balance, MHC overshoots and fringes edges (the teal
    /// branches against a white sky that first showed this on P1050113).
    #[test]
    fn mhc_runs_in_white_balanced_units() {
        let (w, h) = (64, 48);
        let neutral = scene(w, h);
        let wb = [2.3f32, 1.0, 1.5];
        let camera: Vec<f32> = neutral.iter().enumerate().map(|(i, v)| v / wb[i % 3]).collect();
        let cfa = rawler::CFA::new("GBRG");
        let color_at = |r: usize, c: usize| cfa.color_at(r, c).min(2);
        let plane: Vec<f32> = (0..w * h).map(|i| camera[i * 3 + color_at(i / w, i % w)]).collect();
        let sat = vec![false; w * h];
        let run = |balance| {
            let m = Mosaic { plane: &plane, saturated: &sat, width: w, height: h, color_at: &color_at, balance };
            let (out, _) = demosaic(&m, (0, 0, w, h), Demosaic::Mhc);
            out.iter().enumerate().map(|(i, v)| v * wb[i % 3]).collect::<Vec<f32>>()
        };
        let balanced = run(wb);
        let direct = round_trip(w, h, &neutral, "GBRG", Demosaic::Mhc);
        for (i, (a, b)) in balanced.iter().zip(&direct).enumerate() {
            assert!((a - b).abs() < 1e-5, "sample {i}: {a} vs {b}");
        }
        let unbalanced = run([1.0; 3]);
        assert!(psnr(w, h, &neutral, &balanced) > psnr(w, h, &neutral, &unbalanced) + 1.0);
    }
}

