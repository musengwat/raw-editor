//! Develop stage: CFA sensor data → scene-referred linear sRGB f32.
//!
//! black-level subtract → normalize (clamped at whitelevel) → white-balance
//! gains → bilinear demosaic → camera→linear-sRGB matrix. Output is unclamped
//! (highlights may exceed 1.0 after matrixing); the GPU pipeline handles the
//! final transfer.

use anyhow::{Context, Result, bail};
use rawler::imgop::xyz::Illuminant;
use rawler::{RawImage, RawImageData};
use rayon::prelude::*;

/// Developed image: tightly packed RGB f32, active-area size.
pub struct DevelopedImage {
    pub width: usize,
    pub height: usize,
    /// width*height*3 linear sRGB samples, unclamped above 1.0.
    pub data: Vec<f32>,
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

    // Plane of normalized, WB'd values in full-raw coordinates.
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
                out_row[col] = v.clamp(0.0, 1.0) * wb[c.min(3)];
            }
        });

    // Camera → linear sRGB matrix from the D65 XYZ→cam matrix.
    let srgb_from_cam = srgb_from_cam_matrix(raw)?;

    // Bilinear demosaic (3x3 neighborhood average per missing channel),
    // then matrix to linear sRGB. Output cropped to the active area.
    let mut out = vec![0f32; aw * ah * 3];
    out.par_chunks_mut(aw * 3)
        .enumerate()
        .for_each(|(orow, out_row)| {
            let row = orow + ay;
            for ocol in 0..aw {
                let col = ocol + ax;
                let mut sum = [0f32; 3];
                let mut cnt = [0u32; 3];
                for dr in -1isize..=1 {
                    for dc in -1isize..=1 {
                        let r = (row as isize + dr).clamp(0, raw_h as isize - 1) as usize;
                        let cl = (col as isize + dc).clamp(0, raw_w as isize - 1) as usize;
                        let c = cfa.color_at(r, cl).min(2);
                        sum[c] += plane[r * raw_w + cl];
                        cnt[c] += 1;
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
                let o = ocol * 3;
                for i in 0..3 {
                    out_row[o + i] = srgb_from_cam[i][0] * cam[0]
                        + srgb_from_cam[i][1] * cam[1]
                        + srgb_from_cam[i][2] * cam[2];
                }
            }
        });

    Ok(DevelopedImage { width: aw, height: ah, data: out })
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
}
