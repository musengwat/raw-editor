//! Optics correction (IR-4): a radial hot-spot model and a per-lens flat
//! field, both multiplying camera RGB before white balance.
//!
//! Many lenses throw a bright, often coloured blob into the centre of an
//! infrared frame (internal reflections that their coatings were never
//! designed to suppress at those wavelengths). The radial model divides out a
//! Gaussian bump. The flat field goes further: photograph something evenly lit
//! (a defocused white wall or overcast sky) through the same lens and filter,
//! and the editor stores the inverse of that frame's shading as a coarse grid
//! of gains. It corrects hot spot, vignetting and colour shading in one go.
//!
//! The grid is small and smooth, so the GPU samples it as a bilinear-filtered
//! texture and [`FlatField::sample`] mirrors that filtering exactly, on values
//! pre-rounded to f16 (the texture's format) so CPU and GPU agree.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::develop::DevelopedImage;
use crate::pipeline::AdjustParams;

/// Grid cells across the long side. ~125 px per cell on a 4000 px frame:
/// averaging ~15k pixels per cell removes noise, and the cells are still fine
/// enough for a hot spot spanning a fifth of the frame. There is deliberately
/// no extra smoothing on top: one 3x3 blur pass left 5.6% of a tight hot spot
/// behind, against 1.9% without (see `flat_field_evens_out_its_own_frame`).
/// Defocusing the flat frame is what keeps surface texture out.
pub const GRID_LONG_SIDE: usize = 32;
/// A cell with more than this share of clipped pixels is not trusted.
const MAX_CLIPPED: f32 = 0.05;
/// A flat frame darker than this (mean camera RGB) has too little signal.
const MIN_LEVEL: f32 = 0.02;
/// Gains are limited to this range, so a bad cell cannot blow a region up.
const GAIN_LIMIT: f32 = 4.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlatField {
    /// The lens this was made with, as `metadata::CameraIds::lens` reports it.
    pub lens: String,
    pub width: usize,
    pub height: usize,
    /// Row-major camera-RGB gains, f16-exact.
    pub gains: Vec<[f32; 3]>,
}

fn f16_round(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

impl FlatField {
    /// Build a flat field from a photo of an evenly lit surface.
    ///
    /// Each grid cell averages its unclipped camera RGB; the gain for a cell is
    /// the frame's mean over the cell's value, per channel. That keeps overall
    /// exposure and colour where they were and only removes the *shading*.
    pub fn from_image(img: &DevelopedImage, lens: &str) -> Result<Self> {
        let (w, h) = (img.width, img.height);
        let (gw, gh) = if w >= h {
            (GRID_LONG_SIDE, (GRID_LONG_SIDE * h).div_ceil(w).max(2))
        } else {
            ((GRID_LONG_SIDE * w).div_ceil(h).max(2), GRID_LONG_SIDE)
        };
        let mut cells = vec![[0f64; 3]; gw * gh];
        let mut counts = vec![0u32; gw * gh];
        let mut clipped = vec![0u32; gw * gh];
        for y in 0..h {
            let gy = y * gh / h;
            for x in 0..w {
                let g = gy * gw + x * gw / w;
                let i = y * w + x;
                if img.clip_at(i) > 0.0 {
                    clipped[g] += 1;
                    continue;
                }
                for c in 0..3 {
                    cells[g][c] += img.data[i * 3 + c] as f64;
                }
                counts[g] += 1;
            }
        }
        let mut valid = vec![false; gw * gh];
        let mut mean = vec![[0f32; 3]; gw * gh];
        for g in 0..gw * gh {
            let total = counts[g] + clipped[g];
            if counts[g] > 0 && (clipped[g] as f32) <= MAX_CLIPPED * total as f32 {
                valid[g] = true;
                mean[g] = cells[g].map(|v| (v / counts[g] as f64) as f32);
            }
        }
        let n_valid = valid.iter().filter(|&&v| v).count();
        if n_valid * 2 < gw * gh {
            bail!("too much of this frame is blown out to use as a flat field — shoot it darker");
        }
        fill_invalid(&mut mean, &mut valid, gw, gh);
        let overall: [f32; 3] =
            std::array::from_fn(|c| mean.iter().map(|m| m[c]).sum::<f32>() / mean.len() as f32);
        if overall.iter().any(|&v| v < MIN_LEVEL) {
            bail!("this frame is too dark to use as a flat field — shoot it brighter");
        }
        let gains = mean
            .iter()
            .map(|m| {
                std::array::from_fn(|c| {
                    f16_round((overall[c] / m[c].max(1e-6)).clamp(1.0 / GAIN_LIMIT, GAIN_LIMIT))
                })
            })
            .collect();
        Ok(Self { lens: lens.to_string(), width: gw, height: gh, gains })
    }

    /// Bilinear sample at `uv` (0..1), texel centres at `(i + 0.5) / n`,
    /// clamped at the edges — what a linear, clamp-to-edge GPU sampler does.
    pub fn sample(&self, uv: [f32; 2]) -> [f32; 3] {
        let fx = (uv[0] * self.width as f32 - 0.5).clamp(0.0, (self.width - 1) as f32);
        let fy = (uv[1] * self.height as f32 - 0.5).clamp(0.0, (self.height - 1) as f32);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let at = |x: usize, y: usize| self.gains[y * self.width + x];
        let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
        std::array::from_fn(|k| {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bot = c[k] + (d[k] - c[k]) * tx;
            top + (bot - top) * ty
        })
    }
}

/// Replace untrusted cells with the mean of their trusted neighbours, growing
/// inward until every cell has a value.
fn fill_invalid(mean: &mut [[f32; 3]], valid: &mut [bool], w: usize, h: usize) {
    while valid.iter().any(|v| !v) {
        let snapshot = valid.to_vec();
        for y in 0..h {
            for x in 0..w {
                if snapshot[y * w + x] {
                    continue;
                }
                let mut sum = [0f32; 3];
                let mut n = 0;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)] {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if snapshot[j] {
                        (0..3).for_each(|c| sum[c] += mean[j][c]);
                        n += 1;
                    }
                }
                if n > 0 {
                    mean[y * w + x] = sum.map(|v| v / n as f32);
                    valid[y * w + x] = true;
                }
            }
        }
    }
}

/// Position-dependent camera-RGB gain for one render: flat field and radial
/// hot-spot model together. Mirrors `optics()` in the shader.
pub struct Optics<'a> {
    flat: Option<&'a FlatField>,
    amount: [f32; 3],
    sigma: f32,
    centre: [f32; 2],
    /// Image width / height, so the hot spot is round in pixels.
    aspect: f32,
}

impl<'a> Optics<'a> {
    /// `flat` is the open image's lens flat field, used only when the params
    /// switch it on.
    pub fn new(p: &AdjustParams, flat: Option<&'a FlatField>, width: usize, height: usize) -> Self {
        Self {
            flat: flat.filter(|_| p.flat_field),
            amount: hotspot_amounts(p),
            sigma: p.hotspot_radius.max(MIN_HOTSPOT_RADIUS),
            centre: p.hotspot_centre,
            aspect: width as f32 / height.max(1) as f32,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.flat.is_none() && self.amount == [0.0; 3]
    }

    /// Gain at `uv` (pixel centre, 0..1).
    pub fn gain(&self, uv: [f32; 2]) -> [f32; 3] {
        let mut k = self.flat.map_or([1.0; 3], |f| f.sample(uv));
        if self.amount != [0.0; 3] {
            let hd = 0.5 * (self.aspect * self.aspect + 1.0).sqrt();
            let dx = (uv[0] - self.centre[0]) * self.aspect / hd;
            let dy = (uv[1] - self.centre[1]) / hd;
            let g = (-(dx * dx + dy * dy) / (2.0 * self.sigma * self.sigma)).exp();
            for c in 0..3 {
                k[c] /= 1.0 + self.amount[c] * g;
            }
        }
        k
    }

    /// Gain for pixel `(x, y)` of a `w`-wide, `h`-tall image.
    pub fn gain_at(&self, x: usize, y: usize, w: usize, h: usize) -> [f32; 3] {
        self.gain([(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32])
    }
}

/// Smallest hot-spot radius the model will use, keeping the Gaussian finite.
pub const MIN_HOTSPOT_RADIUS: f32 = 0.02;

/// Per-channel hot-spot strength: the amount times each channel's weight.
pub fn hotspot_amounts(p: &AdjustParams) -> [f32; 3] {
    p.hotspot_balance.map(|b| (p.hotspot_amount * b).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat frame with a centred, reddish hot spot and darker corners.
    fn shaded(w: usize, h: usize) -> DevelopedImage {
        let mut data = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let (u, v) = ((x as f32 + 0.5) / w as f32 - 0.5, (y as f32 + 0.5) / h as f32 - 0.5);
                let r2 = u * u + v * v;
                let spot = (-r2 / 0.02).exp();
                let vignette = 1.0 - 0.6 * r2;
                data.extend_from_slice(&[
                    0.30 * vignette * (1.0 + 0.8 * spot),
                    0.40 * vignette * (1.0 + 0.4 * spot),
                    0.20 * vignette * (1.0 + 0.2 * spot),
                ]);
            }
        }
        DevelopedImage::linear_srgb(w, h, data)
    }

    #[test]
    fn flat_field_evens_out_its_own_frame() {
        let img = shaded(320, 240);
        let ff = FlatField::from_image(&img, "LUMIX G 20/F1.7").unwrap();
        assert_eq!((ff.width, ff.height), (32, 24));
        // Corrected, the centre and a corner should agree within a few percent
        // per channel (cell averaging softens the very peak a little).
        let at = |x: usize, y: usize| {
            let g = ff.sample([(x as f32 + 0.5) / 320.0, (y as f32 + 0.5) / 240.0]);
            let i = (y * 320 + x) * 3;
            [0, 1, 2].map(|c| img.data[i + c] * g[c])
        };
        let (centre, corner) = (at(160, 120), at(20, 20));
        for c in 0..3 {
            let ratio = centre[c] / corner[c];
            assert!((ratio - 1.0).abs() < 0.03, "channel {c}: centre/corner {ratio:.3}");
        }
        // Raw, they differ a lot — the test is meaningful.
        let raw = |x: usize, y: usize| img.data[(y * 320 + x) * 3];
        assert!(raw(160, 120) / raw(20, 20) > 1.5);
    }

    #[test]
    fn flat_field_keeps_overall_level_and_colour() {
        let ff = FlatField::from_image(&shaded(320, 240), "x").unwrap();
        for c in 0..3 {
            let mean: f32 = ff.gains.iter().map(|g| g[c]).sum::<f32>() / ff.gains.len() as f32;
            assert!((mean - 1.0).abs() < 0.1, "channel {c} mean gain {mean}");
        }
    }

    #[test]
    fn gains_are_f16_exact_and_survive_json() {
        let ff = FlatField::from_image(&shaded(64, 48), "x").unwrap();
        assert!(ff.gains.iter().flatten().all(|&g| half::f16::from_f32(g).to_f32() == g));
        let back: FlatField = serde_json::from_str(&serde_json::to_string(&ff).unwrap()).unwrap();
        assert_eq!(back, ff);
    }

    #[test]
    fn blown_or_black_frames_are_refused() {
        let mut img = shaded(64, 48);
        img.clip = vec![1.0; 64 * 48];
        assert!(FlatField::from_image(&img, "x").is_err());
        let dark = DevelopedImage::linear_srgb(64, 48, vec![0.001; 64 * 48 * 3]);
        assert!(FlatField::from_image(&dark, "x").is_err());
    }

    #[test]
    fn a_few_clipped_cells_are_filled_from_their_neighbours() {
        let mut img = shaded(320, 240);
        img.clip = (0..320 * 240).map(|i| if (i % 320) < 30 && i / 320 < 30 { 1.0 } else { 0.0 }).collect();
        let ff = FlatField::from_image(&img, "x").unwrap();
        assert!(ff.gains.iter().flatten().all(|g| g.is_finite() && *g > 0.0));
    }

    #[test]
    fn sample_hits_texel_centres_and_clamps() {
        let ff = FlatField {
            lens: "x".into(),
            width: 2,
            height: 1,
            gains: vec![[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        };
        assert_eq!(ff.sample([0.25, 0.5])[0], 1.0);
        assert_eq!(ff.sample([0.75, 0.5])[0], 3.0);
        assert_eq!(ff.sample([0.5, 0.5])[0], 2.0);
        assert_eq!(ff.sample([0.0, 0.0])[0], 1.0);
        assert_eq!(ff.sample([1.0, 1.0])[0], 3.0);
    }

    #[test]
    fn hotspot_model_divides_out_a_centred_bump() {
        let p = AdjustParams { hotspot_amount: 0.5, hotspot_radius: 0.3, ..Default::default() };
        let o = Optics::new(&p, None, 400, 300);
        let centre = o.gain([0.5, 0.5]);
        assert!((centre[0] - 1.0 / 1.5).abs() < 1e-6, "{centre:?}");
        let corner = o.gain([0.0, 0.0]);
        assert!(corner[0] > 0.97, "barely touches the corners: {corner:?}");
        // Round in pixels: equal pixel distances right and down give equal gain.
        let right = o.gain_at(200 + 60, 150, 400, 300);
        let down = o.gain_at(200, 150 + 60, 400, 300);
        assert!((right[0] - down[0]).abs() < 1e-3, "{right:?} vs {down:?}");
    }

    #[test]
    fn hotspot_balance_weights_each_channel() {
        let p = AdjustParams { hotspot_amount: 1.0, hotspot_balance: [1.0, 0.5, 0.0], ..Default::default() };
        let k = Optics::new(&p, None, 10, 10).gain([0.5, 0.5]);
        assert!((k[0] - 0.5).abs() < 1e-6 && (k[1] - 1.0 / 1.5).abs() < 1e-6 && k[2] == 1.0, "{k:?}");
    }

    #[test]
    fn optics_are_identity_by_default_and_flat_needs_the_switch() {
        let ff = FlatField { lens: "x".into(), width: 1, height: 1, gains: vec![[2.0; 3]] };
        let p = AdjustParams::default();
        assert!(Optics::new(&p, Some(&ff), 10, 10).is_identity());
        let on = AdjustParams { flat_field: true, ..Default::default() };
        let o = Optics::new(&on, Some(&ff), 10, 10);
        assert!(!o.is_identity());
        assert_eq!(o.gain([0.3, 0.7]), [2.0; 3]);
        assert!(Optics::new(&on, None, 10, 10).is_identity(), "switch on but no flat for this lens");
    }
}
