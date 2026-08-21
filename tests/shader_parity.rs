//! Headless GPU test: render a tiny image through the real WGSL pipeline and
//! compare every output pixel against the CPU mirror in `pipeline::apply`.
//! Skips (with a loud message) when no wgpu adapter is available.
//!
//! All tests in this file share **one** wgpu instance/device, created lazily in
//! a `OnceLock`, and take a `Mutex` around every render. Cargo runs the tests
//! in this binary on parallel threads; building one wgpu instance per thread
//! segfaults the software (llvmpipe) driver, so sharing + serializing is what
//! keeps a plain `cargo test --release` green.

use std::sync::{Mutex, MutexGuard, OnceLock};

use eframe::wgpu;
use raw_editor::develop::DevelopedImage;
use raw_editor::gpu::ImageRenderer;
use raw_editor::pipeline::{AdjustParams, Uniforms, apply, srgb_encode};

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

/// The process-wide GPU context. `_instance` is held so the driver stays
/// loaded for the lifetime of the test binary.
struct Gpu {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn make_gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .ok()?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some(Gpu { _instance: instance, device, queue })
}

/// Lazily built once, then locked for the duration of each GPU test.
static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();

/// `None` when this machine has no usable adapter (tests then skip loudly).
/// Otherwise the returned guard serializes all GPU work across test threads.
fn gpu() -> Option<MutexGuard<'static, Gpu>> {
    let cell = GPU.get_or_init(|| make_gpu().map(Mutex::new));
    // A panicking test would poison the mutex; recover so the rest still run.
    cell.as_ref()
        .map(|m| m.lock().unwrap_or_else(|poisoned| poisoned.into_inner()))
}

fn render_gpu(device: &wgpu::Device, queue: &wgpu::Queue, params: &AdjustParams) -> Vec<[u8; 4]> {
    let mut renderer = ImageRenderer::from_device(device, wgpu::TextureFormat::Rgba8Unorm);
    let img = DevelopedImage {
        width: W,
        height: H,
        data: PIXELS.iter().flatten().copied().collect(),
    };
    renderer.set_image(device, queue, &img);
    renderer.write_uniforms(queue, &Uniforms::new(params, [1.0, 1.0], false));

    let size = wgpu::Extent3d { width: W as u32, height: H as u32, depth_or_array_layers: 1 };
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
    let padded_row = 256usize;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded_row * H) as u64,
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
                rows_per_image: Some(H as u32),
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

    let mapped = slice.get_mapped_range().expect("map readback buffer");
    let mut out = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
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
    let Some(ctx) = gpu() else {
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
    });
}
