//! Headless full-resolution export, using the same GPU pipeline as the GUI.
//!
//! ```text
//! export <file.rw2 | dir> [jpg|tif] [quality] [--out DIR] [--upload]
//! ```
//!
//! Adjustments come from each file's `<name>.rw2.edit.json` sidecar, so
//! pointing this at a folder applies every saved edit in one go. It is also the
//! end-to-end check that a full-size render, readback and encode really work.
//!
//! `--upload` sends each finished file to the Immich server configured in the
//! app's settings; it refuses to run rather than silently skipping if that is
//! not set up.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use eframe::wgpu;
use raw_editor::develop::develop;
use raw_editor::export::{self, ExportFormat};
use raw_editor::metadata;
use raw_editor::sidecar;

const USAGE: &str =
    "usage: export <file.rw2 | dir> [jpg|tif] [quality] [--out DIR] [--upload]";

struct Args {
    input: PathBuf,
    format: ExportFormat,
    quality: u8,
    out_dir: Option<PathBuf>,
    upload: bool,
}

fn parse_args() -> Result<Args> {
    let mut input = None;
    let mut format = ExportFormat::Jpeg;
    let mut quality = 92u8;
    let mut out_dir = None;
    let mut upload = false;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => out_dir = Some(PathBuf::from(it.next().context("--out needs a directory")?)),
            "--upload" => upload = true,
            "jpg" | "jpeg" => format = ExportFormat::Jpeg,
            "tif" | "tiff" => format = ExportFormat::Tiff16,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other if other.starts_with('-') => anyhow::bail!("unknown option {other:?}\n{USAGE}"),
            other => match other.parse::<u8>() {
                Ok(q) if input.is_some() => quality = q,
                _ => input = Some(PathBuf::from(other)),
            },
        }
    }
    Ok(Args {
        input: input.context(USAGE)?,
        format,
        quality,
        out_dir,
        upload,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // Uploading only happens when asked for explicitly *and* configured.
    let config = if args.upload {
        let c = raw_editor::config::Config::load()?;
        if !c.immich_ready() {
            anyhow::bail!(
                "--upload needs an Immich URL and API key in {}",
                raw_editor::config::Config::path()?.display()
            );
        }
        println!("upload:    {} (after each export)", c.base_url());
        Some(c)
    } else {
        None
    };

    let files = if args.input.is_dir() {
        let found = export::raw_files_in(&args.input)?;
        if found.is_empty() {
            anyhow::bail!("no .rw2 files in {}", args.input.display());
        }
        println!("found {} raw files in {}", found.len(), args.input.display());
        found
    } else {
        vec![args.input.clone()]
    };

    if let Some(dir) = &args.out_dir {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }

    // One device for the whole run — creating an adapter per file would dwarf
    // the actual work.
    let gpu = headless_gpu();
    if gpu.is_none() {
        eprintln!("note: no GPU adapter, falling back to the CPU path");
    }

    let (mut ok, mut failed) = (0usize, 0usize);
    for (i, path) in files.iter().enumerate() {
        if files.len() > 1 {
            println!("\n[{}/{}] {}", i + 1, files.len(), path.display());
        }
        // One bad file must not abandon the rest of the batch.
        match export_one(path, &args, gpu.as_ref(), config.as_ref()) {
            Ok(out) => {
                let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
                println!("wrote:     {} ({:.1} MiB)", out.display(), bytes as f64 / (1024.0 * 1024.0));
                ok += 1;
            }
            Err(e) => {
                eprintln!("failed:    {}: {e:#}", path.display());
                failed += 1;
            }
        }
    }

    if files.len() > 1 {
        println!("\n{ok} exported, {failed} failed");
    }
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn export_one(
    path: &Path,
    args: &Args,
    gpu: Option<&(wgpu::Device, wgpu::Queue)>,
    config: Option<&raw_editor::config::Config>,
) -> Result<PathBuf> {
    let params = sidecar::load(path)?.unwrap_or_default();
    println!(
        "params:    {}",
        if params == Default::default() { "neutral (no sidecar)".into() } else { format!("{params:?}") }
    );

    let t = std::time::Instant::now();
    let raw = rawler::decode_file(path).with_context(|| format!("decoding {}", path.display()))?;
    let img = develop(&raw)?;
    println!("developed: {}x{} in {:.2?}", img.width, img.height, t.elapsed());

    let t = std::time::Instant::now();
    let rendered = match gpu {
        Some((device, queue)) => {
            let r = export::render_gpu(device, queue, &img, &params)?;
            println!("rendered:  GPU in {:.2?}", t.elapsed());
            r
        }
        None => {
            let r = export::render_cpu(&img, &params);
            println!("rendered:  CPU in {:.2?}", t.elapsed());
            r
        }
    };

    let meta = match metadata::read(path) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("warning:   no metadata from {}: {e:#}", path.display());
            None
        }
    };

    let out = export::batch_output_path(path, args.out_dir.as_deref(), args.format);
    export::write(&out, &rendered, args.format, args.quality, meta.as_ref())?;

    if let Some(config) = config {
        // A failed upload must not lose an export that is already on disk.
        match raw_editor::immich::upload(config, &out) {
            Ok(r) => println!("immich:    {}", r.describe()),
            Err(e) => eprintln!("immich:    upload failed: {e:#}"),
        }
    }
    Ok(out)
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
