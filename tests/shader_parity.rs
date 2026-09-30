//! Headless GPU test: render a tiny image through the real WGSL pipeline and
//! compare every output pixel against the CPU mirror in `pipeline::apply`.
//! Skips (with a loud message) when no wgpu adapter is available.
//!
//! The single shared wgpu device lives in `tests/common` — see the note there
//! about why every GPU test must go through it.

mod common;

use eframe::wgpu;
use raw_editor::develop::DevelopedImage;
use raw_editor::gpu::ImageRenderer;
use raw_editor::detail;
use raw_editor::pipeline::{AdjustParams, Colour, ImageColour, Uniforms, View, apply, srgb_encode};

const W: usize = 2;
const H: usize = 2;
/// One pixel per quadrant: saturated red-ish, neutral grey, deep blue,
/// above-white highlight (tests unclamped linear input).
const PIXELS: [[f32; 3]; 4] = [
    [0.7, 0.05, 0.02],
    [0.18, 0.18, 0.18],
    [0.01, 0.02, 0.5],
    [1.8, 1.5, 1.2],
];

fn render_gpu(device: &wgpu::Device, queue: &wgpu::Queue, params: &AdjustParams) -> Vec<[u8; 4]> {
    let img = DevelopedImage::linear_srgb(W, H, PIXELS.iter().flatten().copied().collect());
    render_image(device, queue, &img, params)
}

/// Render one image through the real pipeline at 1:1 and read back sRGB bytes.
fn render_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img: &DevelopedImage,
    params: &AdjustParams,
) -> Vec<[u8; 4]> {
    let (w, h) = (img.width, img.height);
    let mut renderer = ImageRenderer::from_device(device, wgpu::TextureFormat::Rgba8Unorm);
    renderer.set_image(device, queue, img);
    let view = View::one_to_one([1.0 / w as f32, 1.0 / h as f32]);
    renderer.write_uniforms(queue, &Uniforms::new(params, &img.colour, view, false));

    let size = wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 };
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());

    // bytes_per_row must be a multiple of 256 for texture->buffer copies.
    let padded_row = (w * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded_row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("parity"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        assert!(renderer.draw(&mut pass), "renderer had no image bound");
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row as u32),
                rows_per_image: Some(h as u32),
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

    let mapped = slice.get_mapped_range().expect("map readback buffer");
    let mut out = Vec::with_capacity(w * h);
    for row in 0..h {
        for col in 0..w {
            let o = row * padded_row + col * 4;
            out.push([mapped[o], mapped[o + 1], mapped[o + 2], mapped[o + 3]]);
        }
    }
    out
}

fn cpu_reference(params: &AdjustParams) -> Vec<[u8; 3]> {
    PIXELS
        .iter()
        .map(|&px| {
            let c = apply(params, px);
            [
                (srgb_encode(c[0]) * 255.0 + 0.5) as u8,
                (srgb_encode(c[1]) * 255.0 + 0.5) as u8,
                (srgb_encode(c[2]) * 255.0 + 0.5) as u8,
            ]
        })
        .collect()
}

fn assert_parity(params: AdjustParams) {
    let Some(ctx) = common::gpu() else {
        eprintln!("SKIP: no wgpu adapter available");
        return;
    };
    let gpu = render_gpu(&ctx.device, &ctx.queue, &params);
    let cpu = cpu_reference(&params);
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        for ch in 0..3 {
            let diff = (g[ch] as i32 - c[ch] as i32).abs();
            assert!(
                diff <= 2,
                "pixel {i} channel {ch}: gpu={} cpu={} (params {params:?}, gpu row {gpu:?})",
                g[ch],
                c[ch]
            );
        }
    }
}

#[test]
fn parity_identity() {
    assert_parity(AdjustParams::default());
}

#[test]
fn parity_all_controls_active() {
    assert_parity(AdjustParams {
        exposure_ev: 0.8,
        contrast: 0.4,
        temp: -0.3,
        tint: 0.2,
        saturation: 0.5,
        ..Default::default()
    });
}

#[test]
fn parity_extremes() {
    assert_parity(AdjustParams {
        exposure_ev: -2.0,
        contrast: -0.8,
        temp: 1.0,
        tint: -1.0,
        saturation: -1.0,
        ..Default::default()
    });
}

// --- Phase 3 ------------------------------------------------------------

#[test]
fn parity_tonal_ranges() {
    assert_parity(AdjustParams {
        highlights: -0.7,
        shadows: 0.6,
        whites: 0.35,
        blacks: -0.4,
        ..Default::default()
    });
}

#[test]
fn parity_tonal_ranges_reversed() {
    assert_parity(AdjustParams {
        highlights: 1.0,
        shadows: -1.0,
        whites: -1.0,
        blacks: 1.0,
        ..Default::default()
    });
}

#[test]
fn parity_lift_gamma_gain() {
    assert_parity(AdjustParams {
        lift: [0.08, -0.05, 0.12],
        gamma: [1.2, 0.85, 1.05],
        gain: [1.1, 1.0, 0.9],
        ..Default::default()
    });
}

#[test]
fn parity_lift_gamma_gain_extremes() {
    assert_parity(AdjustParams {
        lift: [-0.5, 0.5, 0.0],
        gamma: [0.5, 1.5, 1.0],
        gain: [1.5, 0.5, 1.25],
        ..Default::default()
    });
}

/// Everything at once — the case most likely to expose an ordering mismatch
/// between `fs_main` and `pipeline::apply`.
#[test]
fn parity_full_chain() {
    assert_parity(AdjustParams {
        exposure_ev: 0.5,
        contrast: 0.3,
        temp: 0.25,
        tint: -0.2,
        saturation: 0.4,
        highlights: -0.6,
        shadows: 0.45,
        whites: 0.2,
        blacks: -0.3,
        lift: [0.05, 0.0, -0.04],
        gamma: [1.1, 1.0, 0.95],
        gain: [1.05, 0.98, 1.02],
        ..Default::default()
    });
}

// --- Phase 5: detail (noise reduction + sharpening) ---------------------

/// A 32x32 test frame with an edge, a colour patch and per-pixel noise — the
/// three things the detail filters have to treat differently.
fn detail_source() -> DevelopedImage {
    let (w, h) = (32usize, 32usize);
    let mut data = vec![0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let n = if (x + y) % 2 == 0 { 0.015 } else { -0.015 };
            let base = if x < w / 2 { 0.20 } else { 0.55 };
            let o = (y * w + x) * 3;
            data[o] = base + n + if y < h / 2 { 0.10 } else { 0.0 };
            data[o + 1] = base + n;
            data[o + 2] = base - n + if y >= h / 2 { 0.08 } else { 0.0 };
        }
    }
    DevelopedImage::linear_srgb(w, h, data)
}

/// The neighbourhood filters have to agree between `shader.wgsl::detail` and
/// `detail::pixel`, edge clamping included.
fn assert_detail_parity(params: AdjustParams) {
    assert_image_parity(&detail_source(), params);
}

/// Render `img` on the GPU at 1:1 and compare with the CPU path: camera RGB →
/// working (white balance, clip, matrix) → detail → tone chain.
fn assert_image_parity(img: &DevelopedImage, params: AdjustParams) {
    let Some(ctx) = common::gpu() else {
        eprintln!("SKIP: no wgpu adapter available");
        return;
    };
    let gpu = render_image(&ctx.device, &ctx.queue, img, &params);

    let working = img.working(&Colour::resolve(&img.colour, &params));
    let detailed = detail::apply(img.width, img.height, &working, &params);
    let mut worst = 0i32;
    for (i, px) in detailed.chunks(3).enumerate() {
        let c = apply(&params, [px[0], px[1], px[2]]);
        for ch in 0..3 {
            let want = (srgb_encode(c[ch]) * 255.0 + 0.5) as u8;
            let got = gpu[i][ch];
            worst = worst.max((want as i32 - got as i32).abs());
        }
    }
    // One step of slack for f16 texture storage plus 8-bit quantisation.
    assert!(worst <= 3, "detail parity off by {worst} (params {params:?})");
}

#[test]
fn parity_noise_reduction() {
    assert_detail_parity(AdjustParams { nr_luma: 0.8, nr_chroma: 0.6, ..Default::default() });
}

#[test]
fn parity_sharpening() {
    assert_detail_parity(AdjustParams {
        sharpen_amount: 1.4,
        sharpen_radius: 1.2,
        sharpen_threshold: 0.004,
        ..Default::default()
    });
}

/// Detail plus the whole tone chain — the case most likely to expose a
/// mismatch in where the detail stage sits in the order.
#[test]
fn parity_detail_with_full_chain() {
    assert_detail_parity(AdjustParams {
        exposure_ev: 0.4,
        contrast: 0.25,
        temp: 0.2,
        saturation: 0.3,
        highlights: -0.4,
        shadows: 0.3,
        nr_luma: 0.5,
        nr_chroma: 0.7,
        sharpen_amount: 1.0,
        sharpen_radius: 1.5,
        sharpen_threshold: 0.002,
        ..Default::default()
    });
}

// --- IR-1: white balance, clip mask and matrix on the GPU -----------------

/// Camera-RGB frame with a real white balance and matrix, a fully clipped
/// block, and a half-weight clip column (what bilinear filtering produces at
/// the edge of a blown area). Values are what a sensor sees before WB: green
/// highest, red and blue lower.
fn camera_source() -> DevelopedImage {
    let (w, h) = (24usize, 24usize);
    let mut img = detail_source();
    img = DevelopedImage::linear_srgb(w, h, {
        let mut d = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let o = (y * 32 + x) * 3;
                let s = &img.data[o..o + 3];
                d.extend_from_slice(&[s[0] * 0.45, s[1] * 0.9, s[2] * 0.6]);
            }
        }
        d
    });
    img.clip = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            if (4..10).contains(&x) && (4..10).contains(&y) {
                1.0
            } else if x == 16 {
                0.5
            } else {
                0.0
            }
        })
        .collect();
    img.colour = ImageColour {
        as_shot_wb: [2.1, 1.0, 1.6],
        cam_to_srgb: [[1.7, -0.55, -0.15], [-0.25, 1.45, -0.2], [0.02, -0.5, 1.48]],
    };
    img
}

#[test]
fn parity_camera_rgb_white_balance_and_clip() {
    assert_image_parity(&camera_source(), AdjustParams::default());
}

/// The detail stage reads neighbours through the same camera→working step, so
/// clipped neighbours and the matrix must agree inside the 9x9 window too.
#[test]
fn parity_camera_rgb_with_detail_and_tone() {
    assert_image_parity(
        &camera_source(),
        AdjustParams {
            exposure_ev: 0.3,
            contrast: 0.2,
            saturation: 0.25,
            highlights: -0.5,
            nr_luma: 0.6,
            nr_chroma: 0.7,
            sharpen_amount: 1.2,
            sharpen_radius: 1.3,
            sharpen_threshold: 0.003,
            ..Default::default()
        },
    );
}

