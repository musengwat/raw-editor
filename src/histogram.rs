//! Output histogram and clipping statistics, computed on the CPU from a small
//! copy of the image.
//!
//! It runs the same `to_working` → `pipeline::apply` chain the shader does, so
//! it shows what the screen shows (parity tests keep the two in step). The
//! detail stage is skipped: noise reduction and sharpening act on source
//! pixels, which a 1/8-scale copy no longer has, and they barely move a
//! histogram anyway.

use rayon::prelude::*;

use crate::develop::DevelopedImage;
use crate::pipeline::{AdjustParams, Colour, apply, srgb_encode, to_working};

pub const BINS: usize = 256;
/// Long side of the copy the histogram is computed from. ~190k pixels on a
/// 4:3 frame: enough for a smooth histogram, a few ms per update.
pub const PREVIEW_LONG_SIDE: usize = 512;

pub struct Histogram {
    /// Per channel, counts over display-encoded (sRGB) output values.
    pub bins: [[u32; BINS]; 3],
    /// Per channel, fraction of pixels whose output is at or above white.
    pub clipped: [f32; 3],
}

/// Point-sampled copy whose long side is at most `long_side`. Keeps the clip
/// mask and colour info, so it renders exactly like the full image.
pub fn downsample(img: &DevelopedImage, long_side: usize) -> DevelopedImage {
    let step = img.width.max(img.height).div_ceil(long_side).max(1);
    let (w, h) = (img.width.div_ceil(step), img.height.div_ceil(step));
    let mut data = Vec::with_capacity(w * h * 3);
    let mut clip = Vec::with_capacity(if img.clip.is_empty() { 0 } else { w * h });
    for y in 0..h {
        for x in 0..w {
            let i = (y * step) * img.width + x * step;
            data.extend_from_slice(&img.data[i * 3..i * 3 + 3]);
            if !img.clip.is_empty() {
                clip.push(img.clip[i]);
            }
        }
    }
    DevelopedImage { width: w, height: h, data, clip, colour: img.colour, flat: img.flat.clone() }
}

/// Box-averaged copy whose long side is at most `long_side` — for pictures
/// (look thumbnails), where point sampling would show full-resolution noise.
/// The clip mask is averaged too, which is what the GPU's filtering does.
pub fn box_downsample(img: &DevelopedImage, long_side: usize) -> DevelopedImage {
    let step = img.width.max(img.height).div_ceil(long_side).max(1);
    let (w, h) = (img.width / step, img.height / step);
    let n = (step * step) as f32;
    let px: Vec<([f32; 3], f32)> = (0..w * h)
        .into_par_iter()
        .map(|o| {
            let (bx, by) = ((o % w) * step, (o / w) * step);
            let (mut c, mut k) = ([0f32; 3], 0f32);
            for y in by..by + step {
                for x in bx..bx + step {
                    let i = y * img.width + x;
                    for ch in 0..3 {
                        c[ch] += img.data[i * 3 + ch];
                    }
                    k += img.clip_at(i);
                }
            }
            (c.map(|v| v / n), k / n)
        })
        .collect();
    let data = px.iter().flat_map(|(c, _)| *c).collect();
    let clip = if img.clip.is_empty() { Vec::new() } else { px.iter().map(|(_, k)| *k).collect() };
    DevelopedImage { width: w, height: h, data, clip, colour: img.colour, flat: img.flat.clone() }
}

pub fn compute(img: &DevelopedImage, p: &AdjustParams) -> Histogram {
    let col = Colour::resolve(&img.colour, p);
    // Optics are evaluated at the small copy's own pixel centres, a fraction
    // of a source pixel off the point-sampled positions — invisible in a
    // histogram.
    let optics = img.optics(p);
    let n = img.width * img.height;
    let (bins, over) = (0..n)
        .into_par_iter()
        .fold(
            || (Box::new([[0u32; BINS]; 3]), [0u32; 3]),
            |(mut bins, mut over), i| {
                let c = apply(p, to_working(&col, img.camera_rgb(&optics, i), img.clip_at(i)));
                for ch in 0..3 {
                    let b = (srgb_encode(c[ch]) * (BINS - 1) as f32 + 0.5) as usize;
                    bins[ch][b.min(BINS - 1)] += 1;
                    over[ch] += (c[ch] >= 1.0) as u32;
                }
                (bins, over)
            },
        )
        .reduce(
            || (Box::new([[0u32; BINS]; 3]), [0u32; 3]),
            |(mut a, mut ao), (b, bo)| {
                for ch in 0..3 {
                    for k in 0..BINS {
                        a[ch][k] += b[ch][k];
                    }
                    ao[ch] += bo[ch];
                }
                (a, ao)
            },
        );
    let total = n.max(1) as f32;
    Histogram { bins: *bins, clipped: over.map(|o| o as f32 / total) }
}

impl Histogram {
    /// Bar heights, 0..1, for drawing. Square-root scaled so small tails stay
    /// visible, and normalised against the tallest *interior* bin across all
    /// channels: a clipped frame piles thousands of pixels into the end bins,
    /// and scaling to that spike would flatten everything else. End bins are
    /// capped at 1 (the clip percentages report them exactly).
    pub fn heights(&self) -> [[f32; BINS]; 3] {
        let max = self
            .bins
            .iter()
            .flat_map(|b| b[1..BINS - 1].iter())
            .copied()
            .max()
            .unwrap_or(0)
            .max(1) as f32;
        self.bins.map(|b| b.map(|c| (c as f32 / max).sqrt().min(1.0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_downsample_averages_blocks() {
        // 4x2, values 0..8 in red; long side 2 => step 2, a 2x1 result.
        let data: Vec<f32> = (0..8).flat_map(|i| [i as f32, 1.0, 0.0]).collect();
        let img = DevelopedImage::linear_srgb(4, 2, data);
        let small = box_downsample(&img, 2);
        assert_eq!((small.width, small.height), (2, 1));
        assert_eq!(small.data, vec![2.5, 1.0, 0.0, 4.5, 1.0, 0.0]); // mean(0,1,4,5), mean(2,3,6,7)
    }
    use crate::pipeline::ImageColour;

    fn img(w: usize, h: usize, px: impl Fn(usize, usize) -> [f32; 3]) -> DevelopedImage {
        let data = (0..w * h).flat_map(|i| px(i % w, i / w)).collect();
        DevelopedImage::linear_srgb(w, h, data)
    }

    #[test]
    fn counts_every_pixel_once_per_channel() {
        let h = compute(&img(7, 5, |x, _| [x as f32 / 6.0; 3]), &AdjustParams::default());
        for ch in 0..3 {
            assert_eq!(h.bins[ch].iter().sum::<u32>(), 35);
        }
        // Black and white land in the end bins.
        assert_eq!(h.bins[0][0], 5);
        assert_eq!(h.bins[0][BINS - 1], 5);
    }

    #[test]
    fn clipping_is_per_channel_and_follows_the_adjustments() {
        // A quarter of the pixels are hot in red only.
        let src = img(8, 8, |x, y| if x < 4 && y < 4 { [1.4, 0.5, 0.3] } else { [0.3; 3] });
        let h = compute(&src, &AdjustParams::default());
        assert_eq!(h.clipped, [0.25, 0.0, 0.0]);
        // Pull exposure down two stops and nothing clips any more.
        let h = compute(&src, &AdjustParams { exposure_ev: -2.0, ..Default::default() });
        assert_eq!(h.clipped, [0.0; 3]);
    }

    #[test]
    fn white_balance_moves_the_histogram() {
        // Camera RGB with blue a third of green: as shot (unit gains) blue is
        // dark; a 3x blue gain lines it up with green.
        let src = img(4, 4, |_, _| [0.2, 0.2, 0.2 / 3.0]);
        let dark = compute(&src, &AdjustParams::default());
        let fixed = compute(&src, &AdjustParams { wb_gains: Some([1.0, 1.0, 3.0]), ..Default::default() });
        let peak = |h: &Histogram, ch: usize| h.bins[ch].iter().position(|&c| c > 0).unwrap();
        assert!(peak(&dark, 2) < peak(&dark, 1));
        assert!(peak(&fixed, 2).abs_diff(peak(&fixed, 1)) <= 1);
    }

    #[test]
    fn downsample_keeps_colour_info_and_clip_mask() {
        let mut src = img(1000, 600, |x, _| [x as f32 / 1000.0; 3]);
        src.clip = (0..600_000).map(|i| ((i % 1000) >= 900) as u8 as f32).collect();
        src.colour = ImageColour { as_shot_wb: [2.0, 1.0, 1.5], ..ImageColour::LINEAR_SRGB };
        let small = downsample(&src, 512);
        assert_eq!((small.width, small.height), (500, 300));
        assert_eq!(small.clip.len(), 500 * 300);
        assert_eq!(small.colour, src.colour);
        // Pixel (450, 10) of the copy is source pixel (900, 20): clipped.
        assert_eq!(small.clip[10 * 500 + 450], 1.0);
        assert_eq!(small.clip[10 * 500 + 449], 0.0);
        // Already small: returned at the same size.
        let tiny = downsample(&img(10, 10, |_, _| [0.1; 3]), 512);
        assert_eq!((tiny.width, tiny.height), (10, 10));
        assert!(tiny.clip.is_empty());
    }

    #[test]
    fn heights_ignore_the_clipped_spike_when_scaling() {
        let mut bins = [[0u32; BINS]; 3];
        bins[0][BINS - 1] = 10_000; // a big clipped spike in red
        bins[1][100] = 400;
        bins[2][50] = 100;
        let h = Histogram { bins, clipped: [0.0; 3] }.heights();
        assert_eq!(h[1][100], 1.0, "tallest interior bin fills the height");
        assert!((h[2][50] - 0.5).abs() < 1e-6, "sqrt scaling: 100/400 -> 0.5");
        assert_eq!(h[0][BINS - 1], 1.0, "the spike is capped, not scaled against");
        assert_eq!(h[0][10], 0.0);
    }
}

