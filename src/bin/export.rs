//! Headless full-resolution export, using the same GPU pipeline as the GUI.
//!
//! ```text
//! export <file.rw2> [jpg|tif] [quality]
//! ```
//!
//! Adjustments come from the `<file>.rw2.edit.json` sidecar if there is one,
//! so this doubles as a batch-apply tool and as the end-to-end check that a
//! full-size render + readback really works.

use std::path::PathBuf;

use anyhow::{Context, Result};
use eframe::wgpu;
use raw_editor::develop::develop;
use raw_editor::export::{self, ExportFormat};
use raw_editor::pipeline::AdjustParams;
use raw_editor::sidecar;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().context("usage: export <file.rw2> [jpg|tif] [quality]")?);
    let format = match args.next().as_deref() {
        None | Some("jpg") | Some("jpeg") => ExportFormat::Jpeg,
        Some("tif") | Some("tiff") => ExportFormat::Tiff16,
        Some(other) => anyhow::bail!("unknown format {other:?} (want jpg or tif)"),
    };
    let quality: u8 = match args.next() {
        Some(q) => q.parse().context("quality must be 1..=100")?,
        None => 92,
    };

    let params = sidecar::load(&path)?.unwrap_or_default();
    if params == AdjustParams::default() {
        println!("params:    neutral (no sidecar)");
    } else {
        println!("params:    {params:?}");
    }

    let t = std::time::Instant::now();
    let raw = rawler::decode_file(&path).with_context(|| format!("decoding {}", path.display()))?;
    let img = develop(&raw)?;
    println!("developed: {}x{} in {:.2?}", img.width, img.height, t.elapsed());

    let t = std::time::Instant::now();
    let rendered = match headless_gpu() {
        Some((device, queue)) => {
            let r = export::render_gpu(&device, &queue, &img, &params)?;
            println!("rendered:  GPU in {:.2?}", t.elapsed());
            r
        }
        None => {
            let r = export::render_cpu(&img, &params);
            println!("rendered:  CPU (no adapter) in {:.2?}", t.elapsed());
            r
        }
    };

    let out = export::output_path(&path, format);
    export::write(&out, &rendered, format, quality)?;
    let bytes = std::fs::metadata(&out)?.len();
    println!("wrote:     {} ({:.1} MiB)", out.display(), bytes as f64 / (1024.0 * 1024.0));
    Ok(())
}

fn headless_gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
}
