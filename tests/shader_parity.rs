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
use raw_editor::pipeline::{ColourMode, 
    AdjustParams, ImageColour, Uniforms, View, apply, clip_view, srgb_encode,
};

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
    render_image_with(device, queue, img, params, false)
}

fn render_image_with(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img: &DevelopedImage,
    params: &AdjustParams,
    show_clip: bool,
) -> Vec<[u8; 4]> {
    render_image_sized(device, queue, img, params, show_clip, img.width, img.height)
}

/// Render into a `w`x`h` target. Smaller than the image = the fit-to-window
/// case, where the GPU picks a mip level.
fn render_image_sized(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img: &DevelopedImage,
    params: &AdjustParams,
    show_clip: bool,
    w: usize,
    h: usize,
) -> Vec<[u8; 4]> {
    let mut renderer = ImageRenderer::from_device(device, wgpu::TextureFormat::Rgba8Unorm);
    renderer.set_image(device, queue, img);
    let view = View::one_to_one([1.0 / img.width as f32, 1.0 / img.height as f32]);
    let mut uniforms = Uniforms::new(params, &img.colour, view, false);
    uniforms.show_clip = if show_clip { 1.0 } else { 0.0 };
    renderer.write_uniforms(queue, &uniforms);

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

    let working = img.working(&params);
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
        turns: 0,
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

// --- IR-2: channel mixer --------------------------------------------------

/// Every built-in look, in both colour modes, with the clip block and detail
/// active: the mixer is folded into the matrix uniform, so this proves the
/// GPU sees exactly the composed matrix the CPU uses.
#[test]
fn parity_channel_mixer_looks() {
    for look in raw_editor::looks::builtin() {
        for mode in [ColourMode::Camera, ColourMode::SensorRgb] {
            let mut p = AdjustParams { saturation: 0.3, nr_chroma: 0.5, sharpen_amount: 0.8, ..Default::default() };
            look.apply(&mut p);
            p.colour_mode = mode;
            assert_image_parity(&camera_source(), p);
        }
    }
    // An arbitrary non-neutral mixer, negative weights included.
    assert_image_parity(
        &camera_source(),
        AdjustParams { mixer: [[1.2, -0.4, 0.3], [0.1, 0.6, 0.2], [-0.2, 0.5, 0.9]], ..Default::default() },
    );
}

// --- Crop, rotate, straighten ----------------------------------------------

/// GPU render at the geometry's output size against the CPU export path.
fn assert_geometry_parity(img: &DevelopedImage, params: AdjustParams, tolerance: i32) {
    let Some(ctx) = common::gpu() else { return };
    let cpu = raw_editor::export::render_cpu(img, &params);
    let gpu = render_image_sized(&ctx.device, &ctx.queue, img, &params, false, cpu.width, cpu.height);
    let mut worst = 0i32;
    for (i, px) in gpu.iter().enumerate() {
        for ch in 0..3 {
            let want = (cpu.data[i * 3 + ch] / 257) as i32;
            worst = worst.max((want - px[ch] as i32).abs());
        }
    }
    assert!(worst <= tolerance, "geometry parity off by {worst} (params {params:?})");
}

/// Quarter turns and a crop resample nothing: exact, detail included.
#[test]
fn parity_rotation_and_crop() {
    for rotation in 0..4u8 {
        let p = AdjustParams {
            rotation,
            crop: [0.125, 0.25, 0.75, 0.875],
            sharpen_amount: 0.8,
            nr_chroma: 0.4,
            saturation: 0.2,
            ..Default::default()
        };
        assert_geometry_parity(&camera_source(), p, 3);
    }
}

/// Straightening resamples. The GPU filters camera RGB, the CPU filters the
/// processed working image, so allow a little more at sharp edges.
#[test]
fn parity_straighten() {
    let p = AdjustParams {
        straighten: 7.0,
        crop: raw_editor::pipeline::straighten_crop(7.0, 24.0, 24.0),
        ..Default::default()
    };
    assert_geometry_parity(&camera_source(), p, 6);
}

// --- Fit-to-window preview -------------------------------------------------

/// Noise rendered at 1/8 size must come out as the 8x8 block averages (mip
/// level 3), not as one picked source pixel per output pixel — the bug that
/// made the fit view look far noisier than the photo.
#[test]
fn downscaled_view_averages_instead_of_point_sampling() {
    let Some(gpu) = common::gpu() else { return };
    let (sw, sh, k) = (64usize, 64usize, 8usize);
    let mut seed = 12345u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        0.05 + 0.5 * (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    let data: Vec<f32> = (0..sw * sh).flat_map(|_| { let v = rnd(); [v, v, v] }).collect();
    let img = DevelopedImage::linear_srgb(sw, sh, data.clone());
    let out = render_image_sized(&gpu.device, &gpu.queue, &img, &AdjustParams::default(), false, sw / k, sh / k);
    let mut worst = 0i32;
    for by in 0..sh / k {
        for bx in 0..sw / k {
            let mean: f32 = (0..k * k).map(|i| data[((by * k + i / k) * sw + bx * k + i % k) * 3]).sum::<f32>() / (k * k) as f32;
            let want = (srgb_encode(mean) * 255.0).round() as i32;
            worst = worst.max((out[by * (sw / k) + bx][0] as i32 - want).abs());
        }
    }
    assert!(worst <= 2, "downscaled output is {worst} levels off the block average");
}

// --- IR-6: per-colour HSL and look strength --------------------------------

/// Every band moved, each differently, on top of saturation and a half-strength
/// mixer — the hue interpolation, the HSV round trip and the luminance gain all
/// have to agree with the CPU mirror.
#[test]
fn parity_hsl_and_look_strength() {
    let hsl = std::array::from_fn(|i| {
        let f = i as f32 / 7.0;
        [1.0 - 2.0 * f, 0.8 * f - 0.4, 0.6 - f]
    });
    let p = AdjustParams {
        hsl,
        saturation: 0.3,
        colour_mode: ColourMode::SensorRgb,
        mixer: [[0.0, -1.5, 2.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        mixer_strength: 0.5,
        ..Default::default()
    };
    assert_image_parity(&camera_source(), p);
}

// --- IR-4: hot spot and flat field --------------------------------------

/// A flat field with real structure: brighter gain towards one corner and a
/// coloured bump, so bilinear sampling and per-channel gains both show.
fn test_flat() -> raw_editor::flatfield::FlatField {
    let (w, h) = (6usize, 5usize);
    let gains = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            [0.7 + 0.12 * x, 1.0 + 0.05 * y, 1.3 - 0.06 * x - 0.04 * y]
                .map(|v| half::f16::from_f32(v).to_f32())
        })
        .collect();
    raw_editor::flatfield::FlatField { lens: "test".into(), width: w, height: h, gains }
}

#[test]
fn parity_hotspot() {
    // Off-centre, coloured, on a non-square image — exercises the aspect
    // handling and the per-channel balance.
    let mut img = camera_source();
    img = DevelopedImage { width: 24, height: 24, ..img };
    for p in [
        AdjustParams { hotspot_amount: 0.8, hotspot_radius: 0.25, ..Default::default() },
        AdjustParams {
            hotspot_amount: 1.5,
            hotspot_radius: 0.15,
            hotspot_centre: [0.3, 0.65],
            hotspot_balance: [1.0, 0.6, 0.2],
            ..Default::default()
        },
    ] {
        assert_image_parity(&img, p);
    }
    assert_image_parity(&detail_source(), AdjustParams { hotspot_amount: 1.0, hotspot_centre: [0.8, 0.2], ..Default::default() });
}

#[test]
fn parity_flat_field() {
    let mut img = camera_source();
    img.flat = Some(std::sync::Arc::new(test_flat()));
    assert_image_parity(&img, AdjustParams { flat_field: true, ..Default::default() });
    // Switched off, the bound flat field must have no effect at all.
    assert_image_parity(&img, AdjustParams::default());
}

/// Optics + detail: the 9x9 neighbourhood reads past the image edge, where
/// the shader must clamp the optics position just as the CPU clamps pixels.
/// (Checked by deleting that clamp from the shader: this test then fails.)
#[test]
fn parity_optics_with_detail_at_the_edges() {
    let mut img = detail_source();
    img.flat = Some(std::sync::Arc::new(test_flat()));
    assert_image_parity(
        &img,
        AdjustParams {
            flat_field: true,
            // Tight, strong and coloured, centred on a corner, so the optics
            // change steeply across the out-of-image neighbours.
            hotspot_amount: 3.0,
            hotspot_radius: 0.06,
            hotspot_centre: [0.0, 1.0],
            hotspot_balance: [1.0, 0.3, 0.0],
            nr_luma: 0.7,
            nr_chroma: 1.0,
            // Strong, wide sharpening magnifies any disagreement in the far
            // neighbours, which is where the clamp matters.
            sharpen_amount: 3.0,
            sharpen_radius: 2.5,
            ..Default::default()
        },
    );
}

/// The clipping overlay must mark the same pixels, with the same channel
/// colours, on the GPU as `pipeline::clip_view` does on the CPU.
#[test]
fn parity_clipping_overlay() {
    let Some(ctx) = common::gpu() else {
        eprintln!("SKIP: no wgpu adapter available");
        return;
    };
    // Red-only over white, all-three over white, and ordinary pixels.
    let px: [[f32; 3]; 4] = [[1.6, 0.4, 0.2], [0.2, 0.3, 0.4], [1.3, 1.2, 1.1], [0.7, 0.6, 0.1]];
    let img = DevelopedImage::linear_srgb(2, 2, px.iter().flatten().copied().collect());
    let params = AdjustParams { exposure_ev: 0.3, ..Default::default() };
    let gpu = render_image_with(&ctx.device, &ctx.queue, &img, &params, true);
    for (i, p) in px.iter().enumerate() {
        let c = clip_view(apply(&params, *p));
        for ch in 0..3 {
            let want = (srgb_encode(c[ch]) * 255.0 + 0.5) as u8;
            let got = gpu[i][ch];
            assert!(want.abs_diff(got) <= 2, "pixel {i} ch {ch}: gpu {got} cpu {want}");
        }
    }
    // And it really is an overlay: pixel 0 shows pure red, pixel 2 white.
    assert_eq!(&gpu[0][..3], &[255, 0, 0]);
    assert_eq!(&gpu[2][..3], &[255, 255, 255]);
}

