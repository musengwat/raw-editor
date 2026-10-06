//! Shared headless-GPU harness for the integration tests.
//!
//! Cargo runs the tests inside one binary on parallel threads, and building a
//! `wgpu::Instance` per thread segfaults the software (llvmpipe) driver. So
//! every GPU test goes through [`gpu()`], which builds **one** instance/device
//! lazily in a `OnceLock` and hands out a `Mutex` guard that serializes the
//! actual GPU work. That is what keeps a plain `cargo test --release` green
//! without `--test-threads=1`.

use std::sync::{Mutex, MutexGuard, OnceLock};

use eframe::wgpu;

/// The process-wide GPU context. `_instance` is held so the driver stays
/// loaded for the lifetime of the test binary.
pub struct Gpu {
    _instance: wgpu::Instance,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

fn make_gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .ok()?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&raw_editor::gpu::device_descriptor(&adapter))).ok()?;
    Some(Gpu { _instance: instance, device, queue })
}

static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();

/// `None` when this machine has no usable adapter (tests then skip loudly).
/// Otherwise the returned guard serializes all GPU work across test threads.
pub fn gpu() -> Option<MutexGuard<'static, Gpu>> {
    let cell = GPU.get_or_init(|| make_gpu().map(Mutex::new));
    // A panicking test would poison the mutex; recover so the rest still run.
    cell.as_ref().map(|m| m.lock().unwrap_or_else(|poisoned| poisoned.into_inner()))
}

/// A scratch directory under the system temp dir, wiped on creation.
pub fn scratch_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("raw-editor-it-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
