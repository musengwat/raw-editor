//! Merging an exposure-bracketed set into one HDR image, and tone mapping it
//! so the whole range fits on screen.
//!
//! The steps, all automatic:
//!
//! 1. **Reference.** The frame with the middle exposure (the camera's 0 EV
//!    frame in a −2/0/+2 set). The merge keeps its geometry, colour and
//!    brightness scale.
//! 2. **Alignment.** Each other frame is shifted onto the reference by a
//!    whole-pixel translation found with Ward's median threshold bitmaps
//!    (MTB). A frame's MTB marks pixels above its own median, which is the
//!    same picture whatever the exposure, so differently exposed frames can
//!    be compared directly. Searched coarse to fine over a pyramid.
//! 3. **Exposure ratios.** Nominal from EXIF (shutter × ISO / f²), then
//!    measured: the median ratio of block means where both frames are well
//!    exposed. Cameras round their settings; the merge needs the true ratio
//!    or highlight transitions band.
//! 4. **Ghosts.** Where a frame disagrees with the reference by more than
//!    noise allows (something moved), it is left out and the reference is
//!    used.
//! 5. **Merge.** Per pixel, a weighted mean in the reference's units. Brighter
//!    frames weigh more (more light, less noise) until they near clipping,
//!    where they fade out; the darkest frame always remains as the fallback.
//! 6. **Tone mapping.** Local, on log luminance: an edge-aware base layer (a
//!    guided filter) is compressed to fit, and the detail on top of it is
//!    kept or boosted. That is what makes "HDR": shadows and sky both
//!    readable, local contrast intact.
//!
//! The result is written as a 16-bit sRGB TIFF (`<first frame>_hdr.tif`)
//! carrying the reference frame's EXIF; the editor opens it like a raw, in
//! the default look (see `sidecar::baseline`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::develop::DevelopedImage;
use crate::pipeline::{AdjustParams, luma, smoothstep, srgb_encode};

/// Pixels whose brightest channel is above this start fading out of the
/// merge; at `FADE_HI` they are gone. Below clipping on purpose: sensors are
/// slightly non-linear in their top few percent.
const FADE_LO: f32 = 0.75;
const FADE_HI: f32 = 0.95;
/// A shifted frame fades in over this many pixels from its own edge, so
/// where it runs out the merge hands over to the others gradually instead of
/// leaving a seam (frames never agree perfectly in deep shadow).
const EDGE_FEATHER: f32 = 64.0;
/// Side of the blocks used for exposure measurement and ghost detection.
const BLOCK: usize = 16;
/// Ghost detection: disagreement (in stops) that starts and completes the
/// exclusion of a frame. Block means are 256 pixels each, so noise sits well
/// below this even at ISO 12800.
const GHOST_LO: f32 = 0.3;
const GHOST_HI: f32 = 0.6;
/// Below this (reference units) a block is too dark to judge.
const GHOST_FLOOR: f32 = 0.004;
/// MTB: pixels this close to the median (as a fraction of it) are noise and
/// left out of the comparison.
const MTB_EXCLUDE: f32 = 0.08;
/// The coarsest pyramid level is at least this big on its short side.
const MTB_MIN_SIDE: usize = 24;

/// How strong an HDR look to give the merge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Style {
    /// Range compressed just enough; looks like a well-exposed photo.
    Natural,
    /// The usual HDR look: open shadows, held highlights, extra local contrast.
    #[default]
    Hdr,
    /// Heavy: strong local contrast, flat overall tone.
    Dramatic,
}

impl Style {
    pub const ALL: [Style; 3] = [Style::Natural, Style::Hdr, Style::Dramatic];

    pub fn label(self) -> &'static str {
        match self {
            Style::Natural => "Natural",
            Style::Hdr => "HDR",
            Style::Dramatic => "Dramatic",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|st| st.label().eq_ignore_ascii_case(s))
    }

    fn tone(self) -> ToneParams {
        match self {
            Style::Natural => ToneParams {
                mid: None,
                shadow_stops: 6.0,
                detail: 1.0,
                radius: 1.0 / 25.0,
                eps: 0.5,
                saturation: 0.0,
            },
            Style::Hdr => ToneParams {
                mid: Some(MID_GREY),
                shadow_stops: 4.5,
                detail: 1.6,
                radius: 1.0 / 45.0,
                eps: 0.3,
                saturation: 0.15,
            },
            Style::Dramatic => ToneParams {
                mid: Some(MID_GREY),
                shadow_stops: 3.0,
                detail: 2.3,
                radius: 1.0 / 80.0,
                eps: 0.15,
                saturation: 0.25,
            },
        }
    }
}

/// Tone-mapping settings behind a [`Style`].
#[derive(Clone, Copy, Debug)]
struct ToneParams {
    /// Where the base layer's median goes, in stops; `None` keeps the
    /// reference frame's exposure.
    mid: Option<f32>,
    /// The darkest areas end up no more than this far below the mid-tones.
    shadow_stops: f32,
    /// Local contrast multiplier (1 = as shot) for detail between the fine
    /// and base scales ("clarity"). The finest detail — texture and noise —
    /// stays at 1, or the +2 EV frame's ISO 12800 grain would be amplified.
    detail: f32,
    /// Base-layer filter radius, as a fraction of the long side.
    radius: f32,
    /// Guided-filter edge threshold, in stops²: variations smaller than
    /// about √eps stops count as detail, larger ones as edges.
    eps: f32,
    /// Extra saturation (0 = none), around each pixel's luminance.
    saturation: f32,
}

/// What the merge found, for the status line and the tests.
#[derive(Clone, Debug, PartialEq)]
pub struct MergeReport {
    /// Index of the reference frame.
    pub reference: usize,
    /// Per frame: where each reference pixel is found in that frame.
    pub warps: Vec<Warp>,
    /// Per frame: the warp's shift at the image centre, in whole pixels.
    pub shifts: Vec<[i32; 2]>,
    /// Per frame: the warp's rotation, in degrees.
    pub rotation: Vec<f32>,
    /// Per frame: the factor that brings it to the reference's brightness.
    pub scales: Vec<f32>,
    /// Per frame: share of the image where it was left out as a ghost.
    pub ghosted: Vec<f32>,
    /// Dynamic range gained over the reference alone, in stops.
    pub extra_stops: f32,
}

// --- whole-pipeline entry points -----------------------------------------

/// Where the merge of `set` is written: next to its first frame,
/// `<stem>_hdr.tif`.
pub fn output_path(set: &[PathBuf]) -> PathBuf {
    let first = &set[0];
    let stem = first.file_stem().unwrap_or_default().to_string_lossy();
    first.with_file_name(format!("{stem}_hdr.tif"))
}

/// Develop, merge, tone map and write one bracketed set. `progress` hears
/// about each step. Returns the written path and the report.
pub fn merge_files(
    set: &[PathBuf],
    style: Style,
    out: Option<&Path>,
    progress: &dyn Fn(&str),
) -> Result<(PathBuf, MergeReport)> {
    if set.len() < 2 {
        bail!("an HDR merge needs at least two frames");
    }
    let mut frames = Vec::with_capacity(set.len());
    let mut evs = Vec::with_capacity(set.len());
    let mut metas = Vec::with_capacity(set.len());
    for (i, path) in set.iter().enumerate() {
        progress(&format!("developing frame {}/{}", i + 1, set.len()));
        frames.push(crate::metadata::develop_file(path)?);
        let meta = crate::metadata::read(path).ok();
        evs.push(meta.as_ref().and_then(|m| crate::bracket::Shot::from_metadata(path, m).ev));
        metas.push(meta);
    }
    progress("aligning and merging");
    let (merged, report) = merge(&frames, &evs)?;
    drop(frames);
    progress("tone mapping");
    let rendered = tone_map(&merged, style);
    let out = out.map_or_else(|| output_path(set), Path::to_path_buf);
    progress("writing");
    // The reference frame's EXIF, orientation included: the pixels are not
    // turned upright here, so whoever opens the TIFF must turn them.
    let meta = metas[report.reference].clone();
    crate::export::write(&out, &rendered, crate::export::ExportFormat::Tiff16, 100, meta.as_ref())
        .with_context(|| format!("writing {}", out.display()))?;
    Ok((out, report))
}

// --- merge ----------------------------------------------------------------

/// Merge developed frames (same camera and size) into one image in the
/// reference frame's camera RGB, values unclamped above 1. `evs` are the
/// frames' EXIF exposures in stops ([`crate::bracket::exposure_ev`]); `None`
/// where unknown, in which case brightness is measured.
pub fn merge(frames: &[DevelopedImage], evs: &[Option<f32>]) -> Result<(DevelopedImage, MergeReport)> {
    let n = frames.len();
    if n < 2 || evs.len() != n {
        bail!("an HDR merge needs at least two frames");
    }
    let (w, h) = (frames[0].width, frames[0].height);
    if frames.iter().any(|f| f.width != w || f.height != h) {
        bail!("the frames are different sizes — not one bracketed set");
    }

    // 1. Brightness order and reference.
    let known = evs.iter().all(Option::is_some);
    let keys: Vec<f32> = if known {
        evs.iter().map(|e| e.unwrap()).collect()
    } else {
        frames.iter().map(|f| mean_green(f).max(1e-6).log2()).collect()
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
    let reference = order[n / 2];
    let darkest = order[0];

    // 2. Alignment.
    let greys: Vec<Grey> = frames.iter().map(Grey::half_res).collect();
    let warps: Vec<Warp> = (0..n)
        .map(|i| if i == reference { Warp::IDENTITY } else { align_warp(&greys[reference], &greys[i]).to_full_res() })
        .collect();
    drop(greys);

    // 3. Exposure ratios, measured on block means of the aligned frames.
    let blocks: Vec<Blocks> = (0..n).map(|i| Blocks::new(&frames[i], &warps[i])).collect();
    let scales: Vec<f32> = (0..n)
        .map(|i| {
            if i == reference {
                return 1.0;
            }
            let nominal = (keys[reference] - keys[i]).exp2();
            match measured_ratio(&blocks[reference], &blocks[i]) {
                // EXIF is right to within rounding; a measurement far from it
                // means the frames do not overlap enough to trust it.
                Some(r) if known => r.clamp(nominal * 0.7, nominal * 1.4),
                Some(r) => r,
                None => nominal,
            }
        })
        .collect();

    // 4. Ghost maps, per block.
    let ghosts: Vec<Option<Vec<f32>>> = (0..n)
        .map(|i| (i != reference).then(|| ghost_map(&blocks[reference], &blocks[i], scales[i])))
        .collect();
    let ghosted: Vec<f32> = ghosts
        .iter()
        .map(|g| g.as_ref().map_or(0.0, |g| g.iter().sum::<f32>() / g.len().max(1) as f32))
        .collect();
    let (bw, bh) = (blocks[0].bw, blocks[0].bh);
    drop(blocks);

    // 5. Merge.
    let mut data = vec![0f32; w * h * 3];
    let mut clip = vec![0f32; w * h];
    data.par_chunks_mut(w * 3).zip(clip.par_chunks_mut(w)).enumerate().for_each(|(y, (row, crow))| {
        for x in 0..w {
            let mut sum = [0f32; 3];
            let mut wsum = 0f32;
            for i in 0..n {
                let q = warps[i].map(x as f32, y as f32);
                let Some((v, clipped)) = sample(&frames[i], q, false) else {
                    continue;
                };
                if clipped {
                    continue;
                }
                let m = v[0].max(v[1]).max(v[2]);
                let mut wt = (1.0 - smoothstep(FADE_LO, FADE_HI, m)) / scales[i];
                if i != reference {
                    let edge = q[0].min(q[1]).min((w - 1) as f32 - q[0]).min((h - 1) as f32 - q[1]);
                    wt *= smoothstep(0.0, EDGE_FEATHER, edge);
                }
                if let Some(g) = &ghosts[i] {
                    wt *= 1.0 - sample_grid(g, bw, bh, x, y);
                }
                if wt <= 0.0 {
                    continue;
                }
                for c in 0..3 {
                    sum[c] += wt * v[c] * scales[i];
                }
                wsum += wt;
            }
            let o = &mut row[x * 3..x * 3 + 3];
            if wsum > 1e-6 {
                for c in 0..3 {
                    o[c] = sum[c] / wsum;
                }
            } else {
                // Too bright for every frame's comfort: the darkest frame is
                // the best there is, clipped or not.
                let q = warps[darkest].map(x as f32, y as f32);
                let (v, clipped) = sample(&frames[darkest], q, true).unwrap_or_default();
                for c in 0..3 {
                    o[c] = v[c] * scales[darkest];
                }
                crow[x] = if clipped { 1.0 } else { 0.0 };
            }
        }
    });
    let clip = if clip.iter().any(|&c| c > 0.0) { clip } else { Vec::new() };
    let extra_stops = scales[darkest].log2().max(0.0) - scales[order[n - 1]].log2().min(0.0);
    let merged = DevelopedImage {
        width: w,
        height: h,
        data,
        clip,
        colour: frames[reference].colour,
        flat: None,
    };
    let centre = [w as f32 / 2.0, h as f32 / 2.0];
    let shifts = warps
        .iter()
        .map(|wp| {
            let q = wp.map(centre[0], centre[1]);
            [(q[0] - centre[0]).round() as i32, (q[1] - centre[1]).round() as i32]
        })
        .collect();
    let rotation = warps.iter().map(|wp| wp.m[1][0].atan2(wp.m[0][0]).to_degrees()).collect();
    Ok((merged, MergeReport { reference, warps, shifts, rotation, scales, ghosted, extra_stops }))
}

/// Bilinear sample of a frame at pixel coordinates `q` (pixel centres on
/// integers), with its clip flag: clipped if any of the four contributing
/// pixels is. `None` outside the frame, unless `clamp`.
fn sample(f: &DevelopedImage, q: [f32; 2], clamp: bool) -> Option<([f32; 3], bool)> {
    let (w, h) = (f.width, f.height);
    let (mut x, mut y) = (q[0], q[1]);
    if !(0.0..=(w - 1) as f32).contains(&x) || !(0.0..=(h - 1) as f32).contains(&y) {
        if !clamp {
            return None;
        }
        x = x.clamp(0.0, (w - 1) as f32);
        y = y.clamp(0.0, (h - 1) as f32);
    }
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    let (i00, i10, i01, i11) = (y0 * w + x0, y0 * w + x1, y1 * w + x0, y1 * w + x1);
    let clipped = [i00, i10, i01, i11].iter().any(|&i| f.clip_at(i) > 0.0);
    let v = std::array::from_fn(|c| {
        let at = |i: usize| f.data[i * 3 + c];
        let top = at(i00) + (at(i10) - at(i00)) * tx;
        let bot = at(i01) + (at(i11) - at(i01)) * tx;
        top + (bot - top) * ty
    });
    Some((v, clipped))
}

/// An affine map from reference pixel coordinates to another frame's:
/// `q = m · p + t`. Handheld brackets move *and* turn a little between
/// frames; a shift alone left visible seams in the corners of a 33 MP frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Warp {
    pub m: [[f32; 2]; 2],
    pub t: [f32; 2],
}

impl Warp {
    pub const IDENTITY: Warp = Warp { m: [[1.0, 0.0], [0.0, 1.0]], t: [0.0, 0.0] };

    pub fn translation(dx: f32, dy: f32) -> Self {
        Warp { t: [dx, dy], ..Self::IDENTITY }
    }

    pub fn map(&self, x: f32, y: f32) -> [f32; 2] {
        [self.m[0][0] * x + self.m[0][1] * y + self.t[0], self.m[1][0] * x + self.m[1][1] * y + self.t[1]]
    }

    /// The same warp for full-resolution pixels, from one found on the
    /// half-resolution `Grey` (half-res pixel i covers full-res 2i and 2i+1,
    /// so its centre is at full-res 2i + 0.5).
    fn to_full_res(self) -> Self {
        let m = self.m;
        let t = [
            2.0 * self.t[0] + 0.5 - 0.5 * (m[0][0] + m[0][1]),
            2.0 * self.t[1] + 0.5 - 0.5 * (m[1][0] + m[1][1]),
        ];
        Warp { m, t }
    }
}

fn mean_green(f: &DevelopedImage) -> f32 {
    let step = (f.width * f.height / 100_000).max(1);
    let (mut s, mut k) = (0f64, 0usize);
    for i in (0..f.width * f.height).step_by(step) {
        s += f.data[i * 3 + 1] as f64;
        k += 1;
    }
    (s / k.max(1) as f64) as f32
}

/// Bilinear sample of a per-block grid at full-resolution pixel `(x, y)`.
fn sample_grid(g: &[f32], bw: usize, bh: usize, x: usize, y: usize) -> f32 {
    let fx = ((x as f32 + 0.5) / BLOCK as f32 - 0.5).clamp(0.0, (bw - 1) as f32);
    let fy = ((y as f32 + 0.5) / BLOCK as f32 - 0.5).clamp(0.0, (bh - 1) as f32);
    let (x0, y0) = (fx as usize, fy as usize);
    let (x1, y1) = ((x0 + 1).min(bw - 1), (y0 + 1).min(bh - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let top = g[y0 * bw + x0] * (1.0 - tx) + g[y0 * bw + x1] * tx;
    let bot = g[y1 * bw + x0] * (1.0 - tx) + g[y1 * bw + x1] * tx;
    top * (1.0 - ty) + bot * ty
}

// --- block statistics -------------------------------------------------------

/// Per-block means of one frame, sampled through its alignment shift.
struct Blocks {
    bw: usize,
    bh: usize,
    /// Mean camera RGB summed over the channels, per block.
    mean: Vec<f32>,
    /// Brightest channel of any pixel in the block (clipped pixels count as
    /// 1), or `None` when part of the block falls outside the frame.
    peak: Vec<Option<f32>>,
}

impl Blocks {
    fn new(f: &DevelopedImage, warp: &Warp) -> Self {
        let (w, h) = (f.width, f.height);
        let (bw, bh) = (w.div_ceil(BLOCK), h.div_ceil(BLOCK));
        let rows: Vec<(Vec<f32>, Vec<Option<f32>>)> = (0..bh)
            .into_par_iter()
            .map(|by| {
                let mut mean = vec![0f32; bw];
                let mut peak = vec![Some(0f32); bw];
                for bx in 0..bw {
                    let (mut s, mut k, mut pk) = (0f32, 0u32, 0f32);
                    for y in by * BLOCK..((by + 1) * BLOCK).min(h) {
                        for x in bx * BLOCK..((bx + 1) * BLOCK).min(w) {
                            let q = warp.map(x as f32, y as f32);
                            let (sx, sy) = (q[0].round() as i64, q[1].round() as i64);
                            if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
                                peak[bx] = None;
                                continue;
                            }
                            let j = sy as usize * w + sx as usize;
                            let v = &f.data[j * 3..j * 3 + 3];
                            s += v[0] + v[1] + v[2];
                            k += 1;
                            let m = if f.clip_at(j) > 0.0 { 1.0 } else { v[0].max(v[1]).max(v[2]) };
                            pk = pk.max(m);
                        }
                    }
                    mean[bx] = s / k.max(1) as f32;
                    peak[bx] = peak[bx].map(|_| pk);
                }
                (mean, peak)
            })
            .collect();
        let mut mean = Vec::with_capacity(bw * bh);
        let mut peak = Vec::with_capacity(bw * bh);
        for (m, p) in rows {
            mean.extend(m);
            peak.extend(p);
        }
        Self { bw, bh, mean, peak }
    }

    /// Whether block `i` is cleanly exposed: inside the frame, nothing near
    /// clipping, and well above the noise floor.
    fn usable(&self, i: usize, floor: f32) -> bool {
        self.peak[i].is_some_and(|p| p < FADE_LO) && self.mean[i] > floor
    }
}

/// The factor that brings `other` to `reference`'s brightness: the median
/// ratio of block means over blocks both expose cleanly. `None` with too
/// few such blocks to say.
fn measured_ratio(reference: &Blocks, other: &Blocks) -> Option<f32> {
    let mut ratios: Vec<f32> = (0..reference.mean.len())
        .filter(|&i| reference.usable(i, 0.03) && other.usable(i, 0.03))
        .map(|i| reference.mean[i] / other.mean[i])
        .collect();
    if ratios.len() < 50 {
        return None;
    }
    let mid = ratios.len() / 2;
    Some(*ratios.select_nth_unstable_by(mid, f32::total_cmp).1)
}

/// Per block, 0 where `other` (scaled by `scale`) agrees with the
/// reference, rising to 1 where it disagrees by more than [`GHOST_HI`]
/// stops — something moved. Only judged where both blocks are cleanly
/// exposed; elsewhere one of them is not used anyway. Grown by three blocks
/// and then blurred over as many, so a moving edge is still fully covered
/// (a block only partly crossed by it can stay under the threshold) and
/// the hand-over between frames is a gradient rather than a seam.
fn ghost_map(reference: &Blocks, other: &Blocks, scale: f32) -> Vec<f32> {
    let (bw, bh) = (reference.bw, reference.bh);
    // Clipping is judged by peak, so a block that is merely bright still
    // counts; the floor is in the reference's units (sum of three channels).
    let floor = 3.0 * GHOST_FLOOR;
    let raw: Vec<f32> = (0..bw * bh)
        .map(|i| {
            let ok_ref = reference.peak[i].is_some_and(|p| p < FADE_HI) && reference.mean[i] > floor;
            let ok_other = other.peak[i].is_some_and(|p| p < FADE_HI) && other.mean[i] * scale > floor;
            if !(ok_ref && ok_other) {
                return 0.0;
            }
            let d = ((other.mean[i] * scale + floor) / (reference.mean[i] + floor)).log2().abs();
            smoothstep(GHOST_LO, GHOST_HI, d)
        })
        .collect();
    // Dilate (3x3 max) three times, then blur.
    let at = |g: &[f32], x: i64, y: i64| g[y.clamp(0, bh as i64 - 1) as usize * bw + x.clamp(0, bw as i64 - 1) as usize];
    let pass = |g: &[f32], max: bool| -> Vec<f32> {
        (0..bw * bh)
            .map(|i| {
                let (x, y) = ((i % bw) as i64, (i / bw) as i64);
                let mut acc = if max { 0f32 } else { 0.0 };
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let v = at(g, x + dx, y + dy);
                        acc = if max { acc.max(v) } else { acc + v / 9.0 };
                    }
                }
                acc
            })
            .collect()
    };
    let grown = pass(&pass(&pass(&raw, true), true), true);
    // Saturate the core: a brighter frame outweighs the reference several
    // times over, so even a few percent of a moved frame would show.
    box_blur(&box_blur(&grown, bw, bh, 2), bw, bh, 1).into_iter().map(|g| smoothstep(0.0, 0.5, g)).collect()
}

// --- alignment ---------------------------------------------------------------

/// A single-channel image for alignment: green at half resolution (2x2 box).
struct Grey {
    w: usize,
    h: usize,
    px: Vec<f32>,
}

impl Grey {
    fn half_res(f: &DevelopedImage) -> Self {
        let (w, h) = (f.width / 2, f.height / 2);
        let px = (0..w * h)
            .into_par_iter()
            .map(|i| {
                let (x, y) = (i % w * 2, i / w * 2);
                let g = |x: usize, y: usize| f.data[(y * f.width + x) * 3 + 1];
                (g(x, y) + g(x + 1, y) + g(x, y + 1) + g(x + 1, y + 1)) * 0.25
            })
            .collect();
        Self { w, h, px }
    }

    fn half(&self) -> Self {
        let (w, h) = (self.w / 2, self.h / 2);
        let px = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w * 2, i / w * 2);
                let g = |x: usize, y: usize| self.px[y * self.w + x];
                (g(x, y) + g(x + 1, y) + g(x, y + 1) + g(x + 1, y + 1)) * 0.25
            })
            .collect();
        Self { w, h, px }
    }

    fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> Self {
        let px = (y..y + h).flat_map(|r| self.px[r * self.w + x..r * self.w + x + w].iter().copied()).collect();
        Self { w, h, px }
    }

    /// The value below which a fraction `p` of the pixels lie.
    fn percentile(&self, p: f32) -> f32 {
        let mut sorted = self.px.clone();
        let i = ((sorted.len() - 1) as f32 * p) as usize;
        *sorted.select_nth_unstable_by(i, f32::total_cmp).1
    }

    /// Threshold bitmap at percentile `p` and its exclusion mask (true =
    /// comparable).
    fn bitmaps(&self, p: f32) -> (Vec<bool>, Vec<bool>) {
        let t = self.percentile(p);
        let band = (t * MTB_EXCLUDE).max(1e-5);
        let bits = self.px.iter().map(|&v| v > t).collect();
        let keep = self.px.iter().map(|&v| (v - t).abs() > band).collect();
        (bits, keep)
    }
}

/// Count of disagreeing comparable pixels with `other` shifted by `(dx, dy)`.
fn mtb_error(a: &(Vec<bool>, Vec<bool>), b: &(Vec<bool>, Vec<bool>), w: usize, h: usize, dx: i32, dy: i32) -> u64 {
    let (x0, x1) = ((-dx).max(0) as usize, (w as i32 - dx.max(0)) as usize);
    let (y0, y1) = ((-dy).max(0) as usize, (h as i32 - dy.max(0)) as usize);
    (y0..y1)
        .into_par_iter()
        .map(|y| {
            let mut e = 0u64;
            let sy = (y as i32 + dy) as usize;
            for x in x0..x1 {
                let (i, j) = (y * w + x, sy * w + (x as i32 + dx) as usize);
                e += ((a.0[i] != b.0[j]) & a.1[i] & b.1[j]) as u64;
            }
            e
        })
        .sum()
}

/// The percentile both frames' bitmaps are cut at. Ward uses the median,
/// but a +2 EV frame can have more than half its pixels clipped, and its
/// median bitmap is then blank. Same percentile in both, so the bitmaps show
/// the same scene contour; the median when both expose it well, else the
/// nearest one that both do.
fn shared_percentile(a: &Grey, b: &Grey) -> f32 {
    const CANDIDATES: [f32; 9] = [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9];
    let fine = |g: &Grey, p: f32| (0.002..FADE_HI).contains(&g.percentile(p));
    CANDIDATES.into_iter().find(|&p| fine(a, p) && fine(b, p)).unwrap_or(0.5)
}

/// Translation that lines `other` up with `reference`, in `Grey` pixels:
/// `other(x + dx, y + dy)` ≈ `reference(x, y)`.
fn align(reference: &Grey, other: &Grey) -> [i32; 2] {
    let p = shared_percentile(reference, other);
    let mut pyr = vec![(Grey { w: reference.w, h: reference.h, px: reference.px.clone() }, Grey {
        w: other.w,
        h: other.h,
        px: other.px.clone(),
    })];
    while pyr.last().is_some_and(|(r, _)| r.w.min(r.h) / 2 >= MTB_MIN_SIDE) {
        let (r, o) = pyr.last().unwrap();
        let next = (r.half(), o.half());
        pyr.push(next);
    }
    let mut shift = [0i32; 2];
    for (r, o) in pyr.iter().rev() {
        shift = [shift[0] * 2, shift[1] * 2];
        let (a, b) = (r.bitmaps(p), o.bitmaps(p));
        let mut best = (u64::MAX, shift);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let s = [shift[0] + dx, shift[1] + dy];
                let e = mtb_error(&a, &b, r.w, r.h, s[0], s[1]);
                // On a tie, the smaller shift wins.
                let smaller = s[0].abs() + s[1].abs() < best.1[0].abs() + best.1[1].abs();
                if e < best.0 || (e == best.0 && smaller) {
                    best = (e, s);
                }
            }
        }
        shift = best.1;
    }
    shift
}

/// Tiles per side for the local alignment measurements.
const TILES: usize = 4;
/// A tile whose shift is further than this (half-res pixels) from the fitted
/// warp is a bad measurement (sky, moving water) and is dropped.
const TILE_TOLERANCE: f32 = 1.5;
/// Largest rotation/scale a fitted warp may have (|m − I| summed). Brackets
/// are fractions of a second apart; anything more is a bad fit.
const MAX_WARP: f32 = 0.03;

/// Warp from `reference` to `other`, in `Grey` pixels: a global shift
/// (see [`align`]), then each tile aligned on its own around it, and an
/// affine map fitted through the tile shifts. Falls back to the global
/// shift when too few tiles agree.
fn align_warp(reference: &Grey, other: &Grey) -> Warp {
    let [gx, gy] = align(reference, other);
    let global = Warp::translation(gx as f32, gy as f32);
    let (tw, th) = (reference.w / TILES, reference.h / TILES);
    if tw.min(th) < 4 * MTB_MIN_SIDE {
        return global;
    }
    // (tile centre in the reference, where it is found in `other`)
    let mut pairs: Vec<([f32; 2], [f32; 2])> = (0..TILES * TILES)
        .into_par_iter()
        .filter_map(|k| {
            let (x0, y0) = (k % TILES * tw, k / TILES * th);
            let ox = (x0 as i64 + gx as i64).clamp(0, (other.w - tw) as i64) as usize;
            let oy = (y0 as i64 + gy as i64).clamp(0, (other.h - th) as i64) as usize;
            let (a, b) = (reference.crop(x0, y0, tw, th), other.crop(ox, oy, tw, th));
            // A tile with nothing to lock on to (open sky) gives noise.
            let (_, keep) = a.bitmaps(0.5);
            if keep.iter().filter(|&&k| k).count() < keep.len() / 4 {
                return None;
            }
            let [lx, ly] = align(&a, &b);
            let p = [x0 as f32 + tw as f32 / 2.0, y0 as f32 + th as f32 / 2.0];
            let q = [p[0] + (ox as f32 - x0 as f32) + lx as f32, p[1] + (oy as f32 - y0 as f32) + ly as f32];
            Some((p, q))
        })
        .collect();
    loop {
        if pairs.len() < 4 {
            return global;
        }
        let Some(fit) = fit_affine(&pairs) else { return global };
        let resid = |(p, q): &([f32; 2], [f32; 2])| {
            let r = fit.map(p[0], p[1]);
            (r[0] - q[0]).hypot(r[1] - q[1])
        };
        let (worst, err) = pairs
            .iter()
            .enumerate()
            .map(|(i, pq)| (i, resid(pq)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        if err <= TILE_TOLERANCE {
            let off = (fit.m[0][0] - 1.0).abs() + fit.m[0][1].abs() + fit.m[1][0].abs() + (fit.m[1][1] - 1.0).abs();
            return if off <= MAX_WARP { fit } else { global };
        }
        pairs.swap_remove(worst);
    }
}

/// Least-squares affine map through point pairs `p → q`.
fn fit_affine(pairs: &[([f32; 2], [f32; 2])]) -> Option<Warp> {
    // Centred sums, for a well-conditioned 2x2 solve.
    let n = pairs.len() as f64;
    let mean = |f: &dyn Fn(&([f32; 2], [f32; 2])) -> f32| pairs.iter().map(|pq| f(pq) as f64).sum::<f64>() / n;
    let (px, py) = (mean(&|pq| pq.0[0]), mean(&|pq| pq.0[1]));
    let (qx, qy) = (mean(&|pq| pq.1[0]), mean(&|pq| pq.1[1]));
    let (mut sxx, mut sxy, mut syy) = (0f64, 0f64, 0f64);
    let mut sq = [[0f64; 2]; 2]; // Σ q_r · p_c
    for (p, q) in pairs {
        let (dx, dy) = (p[0] as f64 - px, p[1] as f64 - py);
        let (ex, ey) = (q[0] as f64 - qx, q[1] as f64 - qy);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
        sq[0][0] += ex * dx;
        sq[0][1] += ex * dy;
        sq[1][0] += ey * dx;
        sq[1][1] += ey * dy;
    }
    let det = sxx * syy - sxy * sxy;
    if det.abs() < 1e-9 {
        return None;
    }
    let inv = [[syy / det, -sxy / det], [-sxy / det, sxx / det]];
    let m: [[f32; 2]; 2] =
        std::array::from_fn(|r| std::array::from_fn(|c| (sq[r][0] * inv[0][c] + sq[r][1] * inv[1][c]) as f32));
    let t = [
        (qx - m[0][0] as f64 * px - m[0][1] as f64 * py) as f32,
        (qy - m[1][0] as f64 * px - m[1][1] as f64 * py) as f32,
    ];
    Some(Warp { m, t })
}

// --- tone mapping -------------------------------------------------------------

/// Tone map a merged image (camera RGB, any range) to a 16-bit sRGB render,
/// ready to write. Colour goes through the frame's own white balance and
/// matrix, as a neutral edit of the reference frame would.
pub fn tone_map(img: &DevelopedImage, style: Style) -> crate::export::Rendered {
    let lin = local_tone_map(img.width, img.height, &img.working(&AdjustParams::default()), style);
    let data = lin.par_iter().map(|&v| (srgb_encode(v).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).collect();
    crate::export::Rendered { width: img.width, height: img.height, data }
}

/// The tone mapping itself, on linear sRGB, returning linear sRGB with every
/// value in 0..=1. Mid-tones keep the reference frame's exposure; the base
/// layer above them is compressed until the brightest areas sit just under
/// white, and the base below them until the deepest shadows are no more than
/// `shadow_stops` down. Detail (log luminance minus base) is kept, times
/// `detail`.
pub fn local_tone_map(w: usize, h: usize, rgb: &[f32], style: Style) -> Vec<f32> {
    let tp = style.tone();
    let lum: Vec<f32> = rgb.par_chunks(3).map(|c| luma([c[0], c[1], c[2]]).max(1e-6).log2()).collect();

    // Base layer: a guided filter on log luminance, computed at a quarter of
    // the resolution and applied at full (He & Sun's "fast guided filter").
    let k = if w.max(h) >= 1024 { 4 } else { 1 };
    let (sw, sh) = (w.div_ceil(k), h.div_ceil(k));
    let small = box_downsample(&lum, w, h, k);
    let r = ((sw.max(sh) as f32 * tp.radius).round() as usize).max(1);
    let (a, b) = guided(&small, sw, sh, r, tp.eps);
    // A second, fine-scale layer splits the detail into clarity (boosted)
    // and texture (kept). On small images it would be the pixel scale.
    let rf = ((sw.max(sh) as f32 * FINE_RADIUS).round() as usize).max(1);
    let fine = (k > 1 && rf < r).then(|| guided(&small, sw, sh, rf, tp.eps));

    // Where the base layer sits, from the small image.
    let mut base_small: Vec<f32> = small.iter().zip(a.iter().zip(&b)).map(|(l, (a, b))| a * l + b).collect();
    let pct = |v: &mut Vec<f32>, p: f32| {
        let i = ((v.len() - 1) as f32 * p) as usize;
        *v.select_nth_unstable_by(i, f32::total_cmp).1
    };
    let mid = pct(&mut base_small, 0.5);
    let hi = pct(&mut base_small, 0.995);
    let lo = pct(&mut base_small, 0.01);
    let curve = BaseCurve::new(lo, mid, hi, tp.shadow_stops, tp.mid);
    if std::env::var_os("RAW_EDITOR_DEBUG").is_some() {
        eprintln!("hdr tone map: base lo {lo:.2} mid {mid:.2} hi {hi:.2} stops, radius {r}, {curve:?}");
    }

    let mut out = vec![0f32; rgb.len()];
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = y * w + x;
            let (ga, gb) = (upsample(&a, sw, sh, k, x, y), upsample(&b, sw, sh, k, x, y));
            let l = lum[i];
            let base = ga * l + gb;
            let fine_base = fine.as_ref().map_or(l, |(fa, fb)| upsample(fa, sw, sh, k, x, y) * l + upsample(fb, sw, sh, k, x, y));
            let target = curve.map(base) + tp.detail * (fine_base - base) + (l - fine_base);
            let gain = (target - l).exp2();
            let mut c = [rgb[i * 3] * gain, rgb[i * 3 + 1] * gain, rgb[i * 3 + 2] * gain];
            if tp.saturation != 0.0 {
                let y = luma(c);
                c = c.map(|v| y + (v - y) * (1.0 + tp.saturation));
            }
            let o = soft_limit(c);
            row[x * 3..x * 3 + 3].copy_from_slice(&o);
        }
    });
    out
}

/// Guided filter of `px` guided by itself (He et al.), returning the
/// per-pixel linear coefficients `(a, b)`, already box-averaged: the
/// filtered value at a pixel with value `v` is `a·v + b`.
fn guided(px: &[f32], w: usize, h: usize, r: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let mean_i = box_blur(px, w, h, r);
    let sq: Vec<f32> = px.iter().map(|v| v * v).collect();
    let mean_ii = box_blur(&sq, w, h, r);
    let a: Vec<f32> = mean_i
        .iter()
        .zip(&mean_ii)
        .map(|(m, mm)| {
            let var = (mm - m * m).max(0.0);
            var / (var + eps)
        })
        .collect();
    let b: Vec<f32> = a.iter().zip(&mean_i).map(|(a, m)| m * (1.0 - a)).collect();
    (box_blur(&a, w, h, r), box_blur(&b, w, h, r))
}

/// Radius of the fine layer that separates clarity from texture, as a
/// fraction of the long side (~25 px on a 33 MP frame).
const FINE_RADIUS: f32 = 1.0 / 280.0;

/// Where the base layer's tones go, in stops: mid-tones go to `target` or
/// stay where the reference frame put them, unless that would crowd the
/// highlights; the
/// top of the range lands at `TOP` and the bottom within `shadow_stops` of
/// the mid-tones. A shoulder and a toe rather than a straight squeeze: slope
/// 1 at the mid-tones, so a white wall keeps its brightness and only the
/// extremes are pulled in.
#[derive(Clone, Copy, Debug)]
struct BaseCurve {
    mid: f32,
    new_mid: f32,
    /// [`roll_off`] strengths above and below `mid` (0 = untouched).
    k_hi: f32,
    k_lo: f32,
}

/// log2 of 18% grey.
const MID_GREY: f32 = -2.47;
/// log2 of where the brightest areas' base layer ends up.
const TOP: f32 = -0.25;
/// Mid-tones stay at least this far below `TOP`.
const MID_HEADROOM: f32 = 2.0;

impl BaseCurve {
    fn new(lo: f32, mid: f32, hi: f32, shadow_stops: f32, target: Option<f32>) -> Self {
        let new_mid = target.unwrap_or(mid).min(TOP - MID_HEADROOM);
        Self { mid, new_mid, k_hi: roll_off_strength(hi - mid, TOP - new_mid), k_lo: roll_off_strength(mid - lo, shadow_stops) }
    }

    fn map(&self, base: f32) -> f32 {
        let d = base - self.mid;
        self.new_mid + if d > 0.0 { roll_off(d, self.k_hi) } else { -roll_off(-d, self.k_lo) }
    }
}

/// `d / (1 + k·d)`: slope 1 at 0, bending over for large `d`.
fn roll_off(d: f32, k: f32) -> f32 {
    d / (1.0 + k * d)
}

/// The [`roll_off`] strength that brings `range` stops into `room` stops
/// (0 when it already fits).
fn roll_off_strength(range: f32, room: f32) -> f32 {
    if range <= room || room <= 0.0 { 0.0 } else { 1.0 / room - 1.0 / range }
}

/// Bring anything above a knee smoothly under 1, scaling all three channels
/// together so hue holds.
fn soft_limit(c: [f32; 3]) -> [f32; 3] {
    const KNEE: f32 = 0.8;
    let m = c[0].max(c[1]).max(c[2]);
    if m <= KNEE {
        return c.map(|v| v.max(0.0));
    }
    let t = KNEE + (1.0 - KNEE) * (1.0 - (-(m - KNEE) / (1.0 - KNEE)).exp());
    c.map(|v| (v * t / m).max(0.0))
}

/// Mean of each `k`×`k` block.
fn box_downsample(px: &[f32], w: usize, h: usize, k: usize) -> Vec<f32> {
    if k == 1 {
        return px.to_vec();
    }
    let (sw, sh) = (w.div_ceil(k), h.div_ceil(k));
    (0..sw * sh)
        .into_par_iter()
        .map(|i| {
            let (bx, by) = (i % sw, i / sw);
            let (mut s, mut n) = (0f32, 0u32);
            for y in by * k..((by + 1) * k).min(h) {
                for x in bx * k..((bx + 1) * k).min(w) {
                    s += px[y * w + x];
                    n += 1;
                }
            }
            s / n as f32
        })
        .collect()
}

/// Bilinear sample of a `k`-times-smaller grid at full-resolution `(x, y)`.
fn upsample(g: &[f32], sw: usize, sh: usize, k: usize, x: usize, y: usize) -> f32 {
    if k == 1 {
        return g[y * sw + x];
    }
    let fx = ((x as f32 + 0.5) / k as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
    let fy = ((y as f32 + 0.5) / k as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
    let (x0, y0) = (fx as usize, fy as usize);
    let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let top = g[y0 * sw + x0] * (1.0 - tx) + g[y0 * sw + x1] * tx;
    let bot = g[y1 * sw + x0] * (1.0 - tx) + g[y1 * sw + x1] * tx;
    top * (1.0 - ty) + bot * ty
}

/// Mean over a (2r+1)² window, shrunk at the edges (only pixels inside
/// count). Separable running sums, so the cost does not grow with `r`.
pub fn box_blur(px: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let run = |src: &[f32], n: usize, stride: usize, dst: &mut [f32]| {
        // Prefix sums along one line, in f64 so long lines stay exact.
        let mut pre = vec![0f64; n + 1];
        for i in 0..n {
            pre[i + 1] = pre[i] + src[i * stride] as f64;
        }
        for i in 0..n {
            let (a, b) = (i.saturating_sub(r), (i + r + 1).min(n));
            dst[i * stride] = ((pre[b] - pre[a]) / (b - a) as f64) as f32;
        }
    };
    let mut rows = vec![0f32; w * h];
    rows.par_chunks_mut(w).zip(px.par_chunks(w)).for_each(|(d, s)| run(s, w, 1, d));
    // Columns: transpose, blur as rows, transpose back.
    let mut t = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            t[x * h + y] = rows[y * w + x];
        }
    }
    let mut tb = vec![0f32; w * h];
    tb.par_chunks_mut(h).zip(t.par_chunks(h)).for_each(|(d, s)| run(s, h, 1, d));
    for y in 0..h {
        for x in 0..w {
            rows[y * w + x] = tb[x * h + y];
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::ImageColour;

    /// Smooth random texture: bilinear value noise over a `cell`-pixel
    /// lattice, -1..1. Not periodic, so alignment has one right answer.
    fn noise(x: f32, y: f32, cell: f32, seed: u32) -> f32 {
        let hash = |i: i32, j: i32| {
            let mut v = (i as u32).wrapping_mul(0x9E37_79B1) ^ (j as u32).wrapping_mul(0x85EB_CA77) ^ seed;
            v ^= v >> 15;
            v = v.wrapping_mul(0x2C1B_3C6D);
            v ^= v >> 12;
            (v & 0xFFFF) as f32 / 32767.5 - 1.0
        };
        let (gx, gy) = (x / cell, y / cell);
        let (i, j) = (gx.floor() as i32, gy.floor() as i32);
        let (tx, ty) = (smoothstep(0.0, 1.0, gx.fract()), smoothstep(0.0, 1.0, gy.fract()));
        let top = hash(i, j) * (1.0 - tx) + hash(i + 1, j) * tx;
        let bot = hash(i, j + 1) * (1.0 - tx) + hash(i + 1, j + 1) * tx;
        top * (1.0 - ty) + bot * ty
    }

    /// A synthetic scene with structure everywhere (so alignment has
    /// something to lock on to) and a ~9-stop range: linear radiance.
    fn scene(w: usize, h: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32, y as f32);
                let texture = noise(fx, fy, 23.0, 1) * 0.8 + noise(fx, fy, 7.0, 2) * 0.4;
                let ramp = fx / w as f32 * 7.0 - 5.0;
                v.push((ramp + texture).exp2());
            }
        }
        v
    }

    /// One exposure of a radiance map: scaled by `2^ev`, shifted so the
    /// frame's `(x + dx, y + dy)` shows the scene's `(x, y)`, clipped at 1.
    fn expose(radiance: &[f32], w: usize, h: usize, ev: f32, shift: [i32; 2]) -> DevelopedImage {
        let k = ev.exp2();
        let mut data = vec![0f32; w * h * 3];
        let mut clip = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = (x as i32 - shift[0], y as i32 - shift[1]);
                let r = radiance[sy.clamp(0, h as i32 - 1) as usize * w + sx.clamp(0, w as i32 - 1) as usize] * k;
                let i = y * w + x;
                for c in 0..3 {
                    data[i * 3 + c] = (r * [0.9, 1.0, 0.8][c]).min(1.0);
                }
                clip[i] = if r >= 1.0 { 1.0 } else { 0.0 };
            }
        }
        DevelopedImage { width: w, height: h, data, clip, colour: ImageColour::LINEAR_SRGB, flat: None }
    }

    #[test]
    fn box_blur_is_a_windowed_mean() {
        let (w, h) = (7, 5);
        let px: Vec<f32> = (0..w * h).map(|i| (i * 7 % 11) as f32).collect();
        let out = box_blur(&px, w, h, 2);
        for y in 0..h {
            for x in 0..w {
                let (mut s, mut n) = (0.0, 0);
                for yy in y.saturating_sub(2)..(y + 3).min(h) {
                    for xx in x.saturating_sub(2)..(x + 3).min(w) {
                        s += px[yy * w + xx];
                        n += 1;
                    }
                }
                assert!((out[y * w + x] - s / n as f32).abs() < 1e-4, "({x},{y})");
            }
        }
    }

    #[test]
    fn mtb_finds_a_translation_across_exposures() {
        let (w, h) = (480, 360);
        let rad = scene(w, h);
        let a = Grey::half_res(&expose(&rad, w, h, 0.0, [0, 0]));
        for shift in [[6, -4], [-10, 2], [0, 0], [14, 12]] {
            // Two stops darker, so the raw values differ everywhere.
            let b = Grey::half_res(&expose(&rad, w, h, -2.0, shift));
            assert_eq!(align(&a, &b), [shift[0] / 2, shift[1] / 2], "shift {shift:?}");
        }
    }

    #[test]
    fn merge_recovers_the_scene_from_three_shifted_exposures() {
        let (w, h) = (320, 240);
        let rad = scene(w, h);
        // The scene peaks around 2^2.8; the 0 EV frame clips a good part of it.
        let frames = vec![expose(&rad, w, h, 0.0, [0, 0]), expose(&rad, w, h, -3.0, [4, -2]), expose(&rad, w, h, 2.0, [-6, 2])];
        let evs = [Some(0.0), Some(-3.0), Some(2.0)];
        let (m, report) = merge(&frames, &evs).unwrap();
        assert_eq!(report.reference, 0);
        assert_eq!(report.shifts, vec![[0, 0], [4, -2], [-6, 2]]);
        for (s, want) in report.scales.iter().zip([1.0, 8.0, 0.25]) {
            assert!((s / want - 1.0).abs() < 0.02, "scales {:?}", report.scales);
        }
        assert!((report.extra_stops - 5.0).abs() < 0.05);
        assert!(report.ghosted.iter().all(|&g| g < 0.01), "nothing moved: {:?}", report.ghosted);
        assert!(m.clip.is_empty(), "the −3 EV frame holds every highlight");
        // Away from the borders the shifted frames could not fill, the merge
        // is the radiance, highlights included.
        let mut worst = 0f32;
        for y in 8..h - 8 {
            for x in 8..w - 8 {
                let i = y * w + x;
                let got = m.data[i * 3 + 1];
                worst = worst.max((got / rad[i]).log2().abs());
            }
        }
        assert!(worst < 0.01, "worst error {worst} stops");
        assert!(rad.iter().cloned().fold(0.0, f32::max) > 4.0, "the test scene really exceeds the 0 EV frame");
    }

    #[test]
    fn merge_measures_exposure_when_exif_is_off() {
        let (w, h) = (320, 240);
        let rad = scene(w, h);
        let frames = vec![expose(&rad, w, h, 0.0, [0, 0]), expose(&rad, w, h, -2.0, [0, 0]), expose(&rad, w, h, 1.9, [0, 0])];
        // EXIF says ±2, but the bright frame was really +1.9.
        let (_, report) = merge(&frames, &[Some(0.0), Some(-2.0), Some(2.0)]).unwrap();
        assert!((report.scales[2] - (-1.9f32).exp2()).abs() < 0.005, "{:?}", report.scales);
        // And with no EXIF at all, brightness order and ratios are measured.
        let (_, report) = merge(&frames, &[None, None, None]).unwrap();
        assert_eq!(report.reference, 0);
        assert!((report.scales[1] - 4.0).abs() < 0.05, "{:?}", report.scales);
    }

    #[test]
    fn a_moving_object_does_not_ghost() {
        let (w, h) = (320, 240);
        let rad = scene(w, h);
        let mut frames = vec![expose(&rad, w, h, 0.0, [0, 0]), expose(&rad, w, h, -2.0, [0, 0]), expose(&rad, w, h, 2.0, [0, 0])];
        // A dark object in the bright frame only, in a mid-tone area.
        let (ox, oy) = (60, 100);
        for y in oy..oy + 40 {
            for x in ox..ox + 40 {
                let i = y * w + x;
                for c in 0..3 {
                    frames[2].data[i * 3 + c] *= 0.1;
                }
            }
        }
        let (m, report) = merge(&frames, &[Some(0.0), Some(-2.0), Some(2.0)]).unwrap();
        assert!(report.ghosted[2] > 0.01 && report.ghosted[2] < 0.4, "{:?}", report.ghosted);
        assert_eq!(report.ghosted[1], 0.0);
        for y in oy..oy + 40 {
            for x in ox..ox + 40 {
                let i = y * w + x;
                let err = (m.data[i * 3 + 1] / rad[i]).log2().abs();
                assert!(err < 0.05, "ghost at ({x},{y}): {err} stops");
            }
        }
    }

    #[test]
    fn affine_fit_recovers_a_rotation() {
        let truth = Warp { m: [[0.9998, -0.0175], [0.0175, 0.9998]], t: [12.0, -7.0] };
        let pairs: Vec<_> = (0..16)
            .map(|k| {
                let p = [(k % 4) as f32 * 300.0 + 150.0, (k / 4) as f32 * 200.0 + 100.0];
                (p, truth.map(p[0], p[1]))
            })
            .collect();
        let fit = fit_affine(&pairs).unwrap();
        for r in 0..2 {
            assert!((fit.t[r] - truth.t[r]).abs() < 0.01, "{fit:?}");
            for c in 0..2 {
                assert!((fit.m[r][c] - truth.m[r][c]).abs() < 1e-4, "{fit:?}");
            }
        }
    }

    #[test]
    fn half_res_warps_scale_to_full_res() {
        // A pure shift of 3 half-res pixels is 6 full-res pixels.
        let w = Warp::translation(3.0, -1.0).to_full_res();
        assert_eq!(w.map(10.0, 10.0), [16.0, 8.0]);
    }

    #[test]
    fn a_rotated_frame_is_aligned_everywhere() {
        // Turn the scene 0.5° about an off-centre point, as a handheld frame
        // does; a shift alone would leave the corners ~5 px off.
        let (w, h) = (1280, 960);
        let rad = scene(w, h);
        let a = 0.5f32.to_radians();
        let truth = Warp { m: [[a.cos(), -a.sin()], [a.sin(), a.cos()]], t: [9.0, -4.0] };
        let reference = expose(&rad, w, h, 0.0, [0, 0]);
        // The other frame's pixel truth(p) shows what the reference has at p.
        let mut other = expose(&rad, w, h, -1.0, [0, 0]);
        for y in 0..h {
            for x in 0..w {
                let (qx, qy) = (x as f32 - truth.t[0], y as f32 - truth.t[1]);
                let p = [a.cos() * qx + a.sin() * qy, -a.sin() * qx + a.cos() * qy];
                let px = p[0].round().clamp(0.0, (w - 1) as f32) as usize;
                let py = p[1].round().clamp(0.0, (h - 1) as f32) as usize;
                for c in 0..3 {
                    other.data[(y * w + x) * 3 + c] = reference.data[(py * w + px) * 3 + c] * 0.5;
                }
            }
        }
        let warp = align_warp(&Grey::half_res(&reference), &Grey::half_res(&other)).to_full_res();
        for p in [[40.0, 40.0], [1240.0, 40.0], [40.0, 920.0], [1240.0, 920.0], [640.0, 480.0]] {
            let (got, want) = (warp.map(p[0], p[1]), truth.map(p[0], p[1]));
            let err = (got[0] - want[0]).hypot(got[1] - want[1]);
            assert!(err < 2.0, "at {p:?}: {got:?} vs {want:?} ({warp:?})");
        }
    }

    #[test]
    fn merge_rejects_mismatched_frames() {
        let a = DevelopedImage::linear_srgb(4, 4, vec![0.5; 48]);
        let b = DevelopedImage::linear_srgb(4, 3, vec![0.5; 36]);
        assert!(merge(&[a.clone(), b], &[Some(0.0), Some(1.0)]).is_err());
        assert!(merge(&[a], &[Some(0.0)]).is_err());
    }

    #[test]
    fn tone_map_fits_the_range_and_keeps_order() {
        let (w, h) = (300, 200);
        let rad = scene(w, h);
        let rgb: Vec<f32> = rad.iter().flat_map(|&v| [v * 0.18, v * 0.18, v * 0.18]).collect();
        for style in Style::ALL {
            let out = local_tone_map(w, h, &rgb, style);
            assert!(out.iter().all(|v| (0.0..=1.0).contains(v)), "{style:?} out of range");
            // The ramp still runs dark to bright: column means increase.
            let col = |x: usize| (0..h).map(|y| out[(y * w + x) * 3]).sum::<f32>() / h as f32;
            assert!(col(10) < col(w / 2) && col(w / 2) < col(w - 10), "{style:?}");
            // And the range is compressed: the scene spans 9+ stops.
            let (a, b) = (col(10).log2(), col(w - 10).log2());
            assert!(b - a < 7.5, "{style:?} kept {} stops", b - a);
        }
    }

    #[test]
    fn stronger_styles_compress_more() {
        let (w, h) = (300, 200);
        let rad = scene(w, h);
        let rgb: Vec<f32> = rad.iter().flat_map(|&v| [v * 0.18; 3]).collect();
        let span = |s: Style| {
            let out = local_tone_map(w, h, &rgb, s);
            let col = |x: usize| (0..h).map(|y| out[(y * w + x) * 3]).sum::<f32>() / h as f32;
            col(w - 10).log2() - col(10).log2()
        };
        let (n, hd, d) = (span(Style::Natural), span(Style::Hdr), span(Style::Dramatic));
        assert!(n > hd && hd > d, "{n} {hd} {d}");
    }

    #[test]
    fn a_low_range_image_is_left_nearly_alone() {
        // 2 stops of range around mid grey: nothing needs compressing.
        let (w, h) = (200, 100);
        let rgb: Vec<f32> = (0..w * h)
            .flat_map(|i| {
                let v = 0.09 * ((i % w) as f32 / w as f32 * 2.0).exp2();
                [v; 3]
            })
            .collect();
        let out = local_tone_map(w, h, &rgb, Style::Natural);
        let worst = rgb.iter().zip(&out).map(|(a, b)| (b / a).log2().abs()).fold(0f32, f32::max);
        assert!(worst < 0.05, "moved by {worst} stops");
    }

    #[test]
    fn base_curve_holds_the_mid_tones_and_fits_the_ends() {
        let c = BaseCurve::new(-12.0, -3.0, 3.0, 4.0, None);
        assert_eq!(c.map(-3.0), -3.0);
        assert!((c.map(3.0) - TOP).abs() < 1e-4, "{}", c.map(3.0));
        assert!((c.map(-12.0) - -7.0).abs() < 1e-4, "{}", c.map(-12.0));
        // Slope ~1 near the mid-tones, monotonic throughout.
        assert!((c.map(-2.9) - c.map(-3.0) - 0.1).abs() < 0.01);
        let ys: Vec<f32> = (-150..=60).map(|i| c.map(i as f32 * 0.1)).collect();
        assert!(ys.windows(2).all(|p| p[1] > p[0]));
        // A range that already fits is left alone.
        let id = BaseCurve::new(-5.0, -3.0, -1.0, 4.0, None);
        assert_eq!(id.map(-1.0), -1.0);
        assert_eq!(id.map(-5.0), -5.0);
        // A target moves the mid-tones, and the ends still fit.
        let t = BaseCurve::new(-12.0, -4.0, 0.0, 4.0, Some(MID_GREY));
        assert_eq!(t.map(-4.0), MID_GREY);
        assert!((t.map(0.0) - TOP).abs() < 1e-4 && (t.map(-12.0) - (MID_GREY - 4.0)).abs() < 1e-4);
    }

    #[test]
    fn soft_limit_keeps_hue_and_stays_below_one() {
        let c = soft_limit([4.0, 2.0, 1.0]);
        assert!(c[0] <= 1.0 && c[0] > 0.9 && (c[0] / c[1] - 2.0).abs() < 1e-5 && (c[1] / c[2] - 2.0).abs() < 1e-5);
        assert_eq!(soft_limit([0.5, 0.2, -0.1]), [0.5, 0.2, 0.0]);
    }

    #[test]
    fn styles_parse_by_label() {
        for s in Style::ALL {
            assert_eq!(Style::parse(&s.label().to_lowercase()), Some(s));
        }
        assert_eq!(Style::parse("nope"), None);
    }

    #[test]
    fn output_sits_next_to_the_first_frame() {
        let set = [PathBuf::from("/p/DSC01195.ARW"), PathBuf::from("/p/DSC01196.ARW")];
        assert_eq!(output_path(&set), Path::new("/p/DSC01195_hdr.tif"));
    }
}
