//! raw-editor: a basic GPU-accelerated RAW editor for Panasonic RW2 files.

// Release builds are a GUI app: without this Windows opens a console window
// behind it. Debug builds keep the console so stdout/stderr stay visible.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use eframe::egui;
use raw_editor::develop::DevelopedImage;
use raw_editor::export::{self, ExportFormat};
use raw_editor::gpu::{ImageCallback, ImageRenderer};
use raw_editor::pipeline::{
    AdjustParams, ColourMode, Geometry, HSL_BANDS, IDENTITY3, ImageColour, Section, Uniforms, WB_GAIN_MAX,
    WB_GAIN_MIN, straighten_crop,
};
use raw_editor::config::Config;
use raw_editor::history::History;
use raw_editor::immich;
use raw_editor::looks::Look;
use raw_editor::metadata::CameraIds;
use raw_editor::sidecar::{self, Origin};
use raw_editor::widgets::{ColorWheel, CropGrab, drag_crop, fit_aspect, resettable, section_header};

/// Hold this key to bypass every adjustment and see the original.
const COMPARE_KEY: egui::Key = egui::Key::B;
/// Toggles between fit-to-window and 1:1.
const ZOOM_KEY: egui::Key = egui::Key::Z;
/// Arms the white-balance eyedropper; the next click on the image picks.
const PICK_WB_KEY: egui::Key = egui::Key::W;
/// Toggles the clipping overlay.
const CLIP_KEY: egui::Key = egui::Key::C;

/// Long side of a look thumbnail, in pixels.
const THUMB_LONG_SIDE: usize = 120;

/// Ctrl+Z / Ctrl+Shift+Z (⌘ on macOS); Ctrl+Y also redoes, as on Windows.
const UNDO: egui::KeyboardShortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);
const REDO: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT), egui::Key::Z);
const REDO_ALT: egui::KeyboardShortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y);

/// Single-key shortcuts must not fire while the user is typing into a text
/// field (the Immich settings, say), nor as part of a Ctrl/⌘ chord — Ctrl+Z
/// is undo, not zoom.
fn shortcut(ctx: &egui::Context, key: egui::Key, held: bool) -> bool {
    !ctx.egui_wants_keyboard_input()
        && ctx.input(|i| !i.modifiers.command && if held { i.key_down(key) } else { i.key_pressed(key) })
}
/// How long the params must sit still before the sidecar is written.
const AUTOSAVE_DEBOUNCE: Duration = Duration::from_millis(600);

fn main() -> eframe::Result {
    let path = std::env::args().nth(1).map(PathBuf::from);
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(eframe::egui_wgpu::WgpuSetupCreateNew {
                device_descriptor: std::sync::Arc::new(raw_editor::gpu::device_descriptor),
                ..eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle()
            }),
            ..Default::default()
        },
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "raw-editor",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, path)?))),
    )
}

type LoadResult = (PathBuf, anyhow::Result<(DevelopedImage, CameraIds)>);
/// Messages from an export or batch-export worker.
enum ExportMsg {
    /// A single-file export finished, with an optional upload note.
    Done(anyhow::Result<(PathBuf, Option<String>)>),
    /// A batch is part way through.
    Progress { done: usize, total: usize, name: String },
    /// A batch finished.
    BatchDone { ok: usize, failed: usize, dir: PathBuf },
}
type ExportResult = ExportMsg;

/// A flat-field change for a lens, requested from the side panel.
enum OpticsAction {
    /// Make (or replace) the lens's flat field from the open photo.
    Make(String),
    Remove(String),
}

/// A change to saved looks requested from the side panel, applied after the
/// panel closure has released its borrows.
enum LookAction {
    Save(Look),
    Delete(String),
    /// Body serial, look name.
    Assign(String, String),
    Unassign(String),
}

struct App {
    render_state: eframe::egui_wgpu::RenderState,
    /// Live values the shader is driven from.
    params: AdjustParams,
    /// What is currently on disk in the sidecar.
    saved_params: AdjustParams,
    /// What this photo opens with when it has no sidecar: neutral, or its
    /// camera body's look. Params equal to this need no sidecar.
    baseline: AdjustParams,
    /// Body serial and lens of the open photo.
    camera: CameraIds,
    /// Name typed into the "save look" field.
    look_name: String,
    /// Previous frame's values, used to spot edits and restart the debounce.
    prev_params: AdjustParams,
    /// Undo / redo of `params` for the open photo; reset on every load.
    history: History<AdjustParams>,
    dirty_since: Option<Instant>,

    image: Option<Arc<DevelopedImage>>,
    current: Option<PathBuf>,
    status: String,

    loader_tx: mpsc::Sender<LoadResult>,
    loader_rx: mpsc::Receiver<LoadResult>,
    loading: bool,

    export_tx: mpsc::Sender<ExportResult>,
    export_rx: mpsc::Receiver<ExportResult>,
    exporting: bool,
    export_format: ExportFormat,
    jpeg_quality: u8,

    /// 1:1 view — required to judge sharpening and noise reduction, which act
    /// on source pixels a fit-to-window view has already thrown away.
    zoom_100: bool,
    /// Pan offset in NDC, only meaningful while zoomed.
    pan: [f32; 2],
    /// The white-balance eyedropper is armed: the next click on the image
    /// sets white balance from that spot.
    picking_wb: bool,
    /// Clipping overlay on the image (a viewing aid; never exported).
    show_clip: bool,
    /// Small copy of the image the histogram is computed from.
    preview: Option<DevelopedImage>,
    /// Box-averaged copy the look thumbnails render from.
    thumb_src: Option<DevelopedImage>,
    /// Look thumbnails and the params they were rendered for.
    thumbs: Option<(AdjustParams, Vec<(Look, egui::TextureHandle)>)>,
    /// Histogram and the params it was computed with, so it is only redone
    /// when something changes.
    histogram: Option<(AdjustParams, raw_editor::histogram::Histogram)>,

    config: Config,
    show_settings: bool,
    /// Which property the per-colour panel shows: 0 hue, 1 saturation, 2 luminance.
    hsl_tab: usize,
    /// Crop mode: the whole frame is shown with the crop as a draggable box.
    cropping: bool,
    /// The crop when crop mode was entered, for Esc.
    crop_before: [f32; 4],
    /// Locked crop ratio (width / height in pixels), `None` = free.
    crop_aspect: Option<f32>,
    /// Drag in progress: what was grabbed, the crop and pointer at its start.
    crop_drag: Option<(CropGrab, [f32; 4], egui::Pos2)>,
    /// Result of the last "Test connection", shown in the settings dialog.
    ping_rx: Option<mpsc::Receiver<anyhow::Result<String>>>,
    ping_status: String,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, path: Option<PathBuf>) -> anyhow::Result<Self> {
        let render_state = cc
            .wgpu_render_state
            .clone()
            .ok_or_else(|| anyhow::anyhow!("wgpu render state unavailable"))?;
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(ImageRenderer::new(&render_state));

        let (loader_tx, loader_rx) = mpsc::channel();
        let (export_tx, export_rx) = mpsc::channel();
        let mut app = Self {
            render_state,
            params: AdjustParams::default(),
            saved_params: AdjustParams::default(),
            baseline: AdjustParams::default(),
            camera: CameraIds::default(),
            look_name: String::new(),
            prev_params: AdjustParams::default(),
            history: History::new(AdjustParams::default()),
            dirty_since: None,
            image: None,
            current: None,
            status: "Open a raw file: pass a path or drop a file here".into(),
            loader_tx,
            loader_rx,
            loading: false,
            export_tx,
            export_rx,
            exporting: false,
            export_format: ExportFormat::Jpeg,
            jpeg_quality: 92,
            zoom_100: false,
            pan: [0.0, 0.0],
            picking_wb: false,
            show_clip: false,
            preview: None,
            thumb_src: None,
            thumbs: None,
            histogram: None,
            config: Config::load().unwrap_or_else(|e| {
                eprintln!("config: {e:#} — starting with defaults");
                Config::default()
            }),
            show_settings: false,
            hsl_tab: 0,
            cropping: false,
            crop_before: [0.0, 0.0, 1.0, 1.0],
            crop_aspect: None,
            crop_drag: None,
            ping_rx: None,
            ping_status: String::new(),
        };
        if let Some(p) = path {
            app.start_load(p, &cc.egui_ctx);
        }
        Ok(app)
    }

    // --- loading ---------------------------------------------------------

    fn start_load(&mut self, path: PathBuf, ctx: &egui::Context) {
        // Don't lose pending edits to the file we're leaving.
        self.flush_sidecar();
        self.loading = true;
        self.status = format!("Loading {}…", path.display());
        let tx = self.loader_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = raw_editor::metadata::develop_file(&path)
                .map(|img| {
                    let ids = raw_editor::metadata::camera_ids(&path).unwrap_or_else(|e| {
                        eprintln!("camera ids: {e:#}");
                        CameraIds::default()
                    });
                    (img, ids)
                });
            let _ = tx.send((path, result));
            ctx.request_repaint();
        });
    }

    fn poll_loader(&mut self) {
        while let Ok((path, result)) = self.loader_rx.try_recv() {
            self.loading = false;
            match result {
                Ok((img, _)) if let Err(e) = raw_editor::gpu::check_fits(&self.render_state.device, img.width, img.height) => {
                    self.status = format!("Can't open {}: {e:#}", path.display());
                }
                Ok((mut img, camera)) => {
                    img.flat = self.config.flat_for(camera.lens.as_deref());
                    let mut renderer = self.render_state.renderer.write();
                    if let Some(r) = renderer.callback_resources.get_mut::<ImageRenderer>() {
                        r.set_image(&self.render_state.device, &self.render_state.queue, &img);
                    }
                    drop(renderer);

                    let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
                    let (baseline, _) = sidecar::baseline(&self.config, &camera);
                    let (params, note) = match sidecar::initial(&path, &self.config, &camera) {
                        Ok((p, Origin::Sidecar)) => (p, " — edits restored".to_string()),
                        Ok((p, Origin::BodyLook(name))) => (p, format!(" — this camera starts in \"{name}\"")),
                        Ok((p, Origin::Neutral)) => (p, String::new()),
                        Err(e) => {
                            eprintln!("sidecar: {e:#}");
                            (baseline, " — sidecar unreadable, starting fresh".to_string())
                        }
                    };
                    let profile = raw_editor::formats::is_tiff(&path)
                        .then(|| raw_editor::rgb_input::tiff_profile_note(&path))
                        .flatten()
                        .map_or(String::new(), |n| format!(" — {n}"));
                    self.status = format!("{name} — {}x{}{note}{profile}", img.width, img.height);
                    self.preview =
                        Some(raw_editor::histogram::downsample(&img, raw_editor::histogram::PREVIEW_LONG_SIDE));
                    self.thumb_src = Some(raw_editor::histogram::box_downsample(&img, THUMB_LONG_SIDE));
                    self.thumbs = None;
                    self.histogram = None;
                    self.picking_wb = false;
                    self.image = Some(Arc::new(img));
                    self.current = Some(path);
                    self.params = params;
                    self.saved_params = params;
                    self.baseline = baseline;
                    self.camera = camera;
                    self.prev_params = params;
                    self.history = History::new(params);
                    self.cropping = false;
                    self.crop_drag = None;
                    self.dirty_since = None;
                }
                Err(e) => self.status = format!("Failed to load {}: {e:#}", path.display()),
            }
        }
    }

    fn undo(&mut self) {
        match self.history.undo(self.params) {
            Some(p) => self.params = p,
            None => self.status = "Nothing to undo".into(),
        }
    }

    fn redo(&mut self) {
        match self.history.redo(self.params) {
            Some(p) => self.params = p,
            None => self.status = "Nothing to redo".into(),
        }
    }

    // --- sidecar ---------------------------------------------------------

    /// Write the sidecar once the params have been still for a moment.
    fn autosave_tick(&mut self, ctx: &egui::Context) {
        if self.params != self.prev_params {
            self.prev_params = self.params;
            self.dirty_since = Some(Instant::now());
        }
        if self.params == self.saved_params {
            self.dirty_since = None;
            return;
        }
        let Some(since) = self.dirty_since else { return };
        let waited = since.elapsed();
        if waited < AUTOSAVE_DEBOUNCE {
            // Make sure a frame happens when the debounce expires, even if the
            // user stops moving the mouse.
            ctx.request_repaint_after(AUTOSAVE_DEBOUNCE - waited);
            return;
        }
        self.flush_sidecar();
    }

    /// Write pending edits now (also called when switching files and on exit).
    fn flush_sidecar(&mut self) {
        let Some(path) = self.current.clone() else { return };
        if self.params == self.saved_params {
            return;
        }
        // Params equal to what the photo opens with anyway mean "no edits" —
        // drop the sidecar instead of leaving a redundant file next to the
        // raw. With a body look that baseline is the look, so resetting such
        // a photo to neutral *does* write a (neutral) sidecar: that is how one
        // photo opts out of its camera's look.
        let result = if self.params == self.baseline {
            sidecar::remove(&path)
        } else {
            sidecar::save(&path, &self.params)
        };
        match result {
            Ok(()) => {
                self.saved_params = self.params;
                self.dirty_since = None;
            }
            Err(e) => self.status = format!("Sidecar save failed: {e:#}"),
        }
    }

    // --- export ----------------------------------------------------------

    fn start_export_to(&mut self, ctx: &egui::Context, out: PathBuf) {
        let (Some(img), Some(raw)) = (self.image.clone(), self.current.clone()) else { return };
        self.flush_sidecar();

        let device = self.render_state.device.clone();
        let queue = self.render_state.queue.clone();
        let params = self.params;
        let format = self.export_format;
        let quality = self.jpeg_quality;
        let tx = self.export_tx.clone();
        let ctx = ctx.clone();
        let config = self.config.clone();

        self.exporting = true;
        self.status = format!("Exporting {}…", out.display());
        std::thread::spawn(move || {
            let rendered = match export::render_gpu(&device, &queue, &img, &params) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("GPU export failed ({e:#}) — falling back to the CPU path");
                    export::render_cpu(&img, &params)
                }
            };
            // The camera's own metadata, read straight from the raw.
            let meta = raw_editor::metadata::read(&raw).ok();
            let result = export::write(&out, &rendered, format, quality, meta.as_ref())
                .map(|()| maybe_upload(&config, out));
            let _ = tx.send(ExportMsg::Done(result));
            ctx.request_repaint();
        });
    }

    fn poll_export(&mut self) {
        while let Ok(msg) = self.export_rx.try_recv() {
            match msg {
                ExportMsg::Done(result) => {
                    self.exporting = false;
                    self.status = match result {
                        Ok((path, None)) => format!("Exported {}", path.display()),
                        Ok((path, Some(note))) => format!("Exported {} — {note}", path.display()),
                        Err(e) => format!("Export failed: {e:#}"),
                    };
                }
                ExportMsg::Progress { done, total, name } => {
                    self.status = format!("Exporting {done}/{total}: {name}");
                }
                ExportMsg::BatchDone { ok, failed, dir } => {
                    self.exporting = false;
                    self.status = if failed == 0 {
                        format!("Exported {ok} files to {}", dir.display())
                    } else {
                        format!("Exported {ok} to {}, {failed} failed", dir.display())
                    };
                }
            }
        }
    }

    /// Export every raw in one folder using each file's own sidecar.
    ///
    /// Files are processed one at a time on purpose: a developed 16MP frame is
    /// ~190 MB, and running them in parallel would spike memory for no real
    /// gain when the GPU pass is the bottleneck anyway.
    fn start_batch_export(&mut self, ctx: &egui::Context, files: Vec<PathBuf>, out_dir: PathBuf) {
        let device = self.render_state.device.clone();
        let queue = self.render_state.queue.clone();
        let format = self.export_format;
        let quality = self.jpeg_quality;
        let tx = self.export_tx.clone();
        let ctx = ctx.clone();
        let config = self.config.clone();

        self.exporting = true;
        self.status = format!("Exporting {} files…", files.len());
        std::thread::spawn(move || {
            let total = files.len();
            let (mut ok, mut failed) = (0usize, 0usize);
            for (i, path) in files.iter().enumerate() {
                let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
                let _ = tx.send(ExportMsg::Progress { done: i, total, name });
                ctx.request_repaint();

                // One bad file must not abandon the rest of the batch.
                match batch_one(&device, &queue, path, &out_dir, format, quality, &config) {
                    Ok(note) => {
                        if let Some(note) = note {
                            eprintln!("batch: {} — {note}", path.display());
                        }
                        ok += 1;
                    }
                    Err(e) => {
                        eprintln!("batch: {} failed: {e:#}", path.display());
                        failed += 1;
                    }
                }
            }
            let _ = tx.send(ExportMsg::BatchDone { ok, failed, dir: out_dir });
            ctx.request_repaint();
        });
    }

    /// Non-blocking result of the settings dialog's connection test.
    fn poll_ping(&mut self) {
        let Some(rx) = &self.ping_rx else { return };
        match rx.try_recv() {
            Ok(Ok(version)) => {
                self.ping_status = format!("Connected — Immich {version}");
                self.ping_rx = None;
            }
            Ok(Err(e)) => {
                self.ping_status = format!("Failed: {e:#}");
                self.ping_rx = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => self.ping_rx = None,
        }
    }

    fn settings_window(&mut self, ui: &mut egui::Ui) {
        if !self.show_settings {
            return;
        }
        let mut open = true;
        let mut save = false;
        egui::Window::new("Settings")
            .open(&mut open)
            .resizable(false)
            .show(ui.ctx(), |ui| {
                ui.label(egui::RichText::new("Immich").strong());
                ui.horizontal(|ui| {
                    ui.label("Server URL");
                    ui.text_edit_singleline(&mut self.config.immich_url);
                });
                ui.horizontal(|ui| {
                    ui.label("API key   ");
                    // password() keeps the credential off the screen.
                    ui.add(egui::TextEdit::singleline(&mut self.config.immich_api_key).password(true));
                });
                ui.label(
                    egui::RichText::new("Immich → Account Settings → API Keys")
                        .weak()
                        .small(),
                );

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    let testing = self.ping_rx.is_some();
                    let ready = self.config.immich_ready();
                    if ui
                        .add_enabled(ready && !testing, egui::Button::new("Test connection"))
                        .clicked()
                    {
                        let (tx, rx) = mpsc::channel();
                        let config = self.config.clone();
                        let ctx = ui.ctx().clone();
                        self.ping_rx = Some(rx);
                        self.ping_status = "Testing…".into();
                        std::thread::spawn(move || {
                            let _ = tx.send(immich::ping(&config));
                            ctx.request_repaint();
                        });
                    }
                    save = ui.button("Save").clicked();
                });
                if !self.ping_status.is_empty() {
                    ui.label(&self.ping_status);
                }
            });

        if save {
            match self.config.save() {
                Ok(()) => self.status = "Settings saved".into(),
                Err(e) => self.status = format!("Could not save settings: {e:#}"),
            }
        }
        self.show_settings = open && !save;
    }

    // --- ui --------------------------------------------------------------

    /// Ask for a raw to open. The dialog blocks the UI thread while it is up,
    /// which is the usual arrangement for a small egui app.
    fn pick_and_open(&mut self, ctx: &egui::Context) {
        let mut dialog =
            rfd::FileDialog::new().add_filter("Photos", &raw_editor::formats::dialog_extensions()).add_filter("All files", &["*"]);
        if let Some(dir) = self.current.as_ref().and_then(|p| p.parent()) {
            dialog = dialog.set_directory(dir);
        }
        if let Some(path) = dialog.pick_file() {
            self.start_load(path, ctx);
        }
    }

    /// Ask where to write an export, defaulting to the name we would have used
    /// automatically.
    fn pick_and_export(&mut self, ctx: &egui::Context, format: ExportFormat) {
        let Some(raw) = self.current.clone() else { return };
        let suggested = export::output_path(&raw, format);
        let mut dialog = rfd::FileDialog::new().add_filter(format.label(), &[format.extension()]);
        if let Some(dir) = suggested.parent() {
            dialog = dialog.set_directory(dir);
        }
        if let Some(name) = suggested.file_name().and_then(|n| n.to_str()) {
            dialog = dialog.set_file_name(name);
        }
        if let Some(out) = dialog.save_file() {
            self.export_format = format;
            self.start_export_to(ctx, out);
        }
    }

    /// Pick a folder of raws and a destination, then export the lot.
    fn pick_and_batch_export(&mut self, ctx: &egui::Context) {
        let mut dialog = rfd::FileDialog::new();
        if let Some(dir) = self.current.as_ref().and_then(|p| p.parent()) {
            dialog = dialog.set_directory(dir);
        }
        let Some(input) = dialog.pick_folder() else { return };

        let files = match raw_editor::formats::photos_in(&input) {
            Ok(f) if !f.is_empty() => f,
            Ok(_) => {
                self.status = format!("No supported photos in {}", input.display());
                return;
            }
            Err(e) => {
                self.status = format!("Could not read {}: {e:#}", input.display());
                return;
            }
        };

        let Some(out_dir) = rfd::FileDialog::new()
            .set_title(format!("Where should {} exports go?", files.len()))
            .set_directory(&input)
            .pick_folder()
        else {
            return;
        };
        self.start_batch_export(ctx, files, out_dir);
    }

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open…").clicked() {
                    ui.close();
                    self.pick_and_open(&ctx);
                }
                ui.separator();
                let has_image = self.image.is_some() && !self.exporting;
                for format in [ExportFormat::Jpeg, ExportFormat::Tiff16] {
                    let label = format!("Export as {}…", format.label());
                    if ui.add_enabled(has_image, egui::Button::new(label)).clicked() {
                        ui.close();
                        self.pick_and_export(&ctx, format);
                    }
                }
                if ui.button("Settings…").clicked() {
                    ui.close();
                    self.show_settings = true;
                }
                if ui.add_enabled(!self.exporting, egui::Button::new("Batch export folder…")).clicked() {
                    ui.close();
                    self.pick_and_batch_export(&ctx);
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("View", |ui| {
                let fit = ui.selectable_label(!self.zoom_100, "Fit to window").clicked();
                let one = ui.selectable_label(self.zoom_100, "1:1 (100%)").clicked();
                if fit || one {
                    self.zoom_100 = one;
                    ui.close();
                }
            });
            ui.menu_button("Edit", |ui| {
                let undo = ui.ctx().format_shortcut(&UNDO);
                let redo = ui.ctx().format_shortcut(&REDO);
                let p = self.params;
                if ui.add_enabled(self.history.can_undo(p), egui::Button::new("Undo").shortcut_text(undo)).clicked() {
                    self.undo();
                    ui.close();
                }
                if ui.add_enabled(self.history.can_redo(p), egui::Button::new("Redo").shortcut_text(redo)).clicked() {
                    self.redo();
                    ui.close();
                }
                ui.separator();
                let changed = !self.params.is_neutral();
                if ui.add_enabled(changed, egui::Button::new("Reset adjustments")).clicked() {
                    self.params = AdjustParams::default();
                    ui.close();
                }
            });
        });
    }

    /// Re-render the look thumbnails once the params have settled (not on
    /// every frame of a drag). ~12k pixels per look, a few ms in all.
    fn update_thumbs(&mut self, ctx: &egui::Context) {
        let Some(src) = &self.thumb_src else { return };
        if self.thumbs.as_ref().is_some_and(|(p, _)| *p == self.params) || ctx.input(|i| i.pointer.any_down()) {
            return;
        }
        let looks: Vec<Look> = raw_editor::looks::builtin().into_iter().chain(self.config.looks.iter().cloned()).collect();
        let thumbs = looks
            .into_iter()
            .map(|look| {
                let (size, rgba) = raw_editor::looks::thumbnail(src, &self.params, &look);
                let img = egui::ColorImage::from_rgba_unmultiplied(size, &rgba);
                let tex = ctx.load_texture(format!("look:{}", look.name), img, egui::TextureOptions::LINEAR);
                (look, tex)
            })
            .collect();
        self.thumbs = Some((self.params, thumbs));
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        self.update_thumbs(ui.ctx());
        ui.heading("Adjust");
        ui.add_space(6.0);

        // Histogram of the output as it looks now; recomputed only when the
        // params have changed (a few ms on the 1/8-scale copy).
        if let Some(preview) = &self.preview {
            if self.histogram.as_ref().is_none_or(|(p, _)| *p != self.params) {
                let h = raw_editor::histogram::compute(preview, &self.params);
                self.histogram = Some((self.params, h));
            }
        }
        if let Some((_, h)) = &self.histogram {
            raw_editor::widgets::histogram(ui, h);
            ui.checkbox(&mut self.show_clip, format!("Show clipping ({})", CLIP_KEY.name()))
                .on_hover_text("Clipped pixels show which channels clipped; the rest go grey");
            ui.add_space(8.0);
        }

        let mut reset = false;
        let mut do_export = false;
        let has_image = self.image.is_some();
        let exporting = self.exporting;
        let as_shot = self.image.as_ref().map_or([1.0; 3], |img| img.colour.as_shot_wb);
        let (export_format, jpeg_quality) = (&mut self.export_format, &mut self.jpeg_quality);
        let picking_wb = &mut self.picking_wb;
        let builtin_looks = raw_editor::looks::builtin();
        let saved_looks = self.config.looks.clone();
        let serial = self.camera.serial.clone();
        let body_look = serial.as_ref().and_then(|s| self.config.body_looks.get(s).cloned());
        let look_name = &mut self.look_name;
        let mut look_action: Option<LookAction> = None;
        let lens = self.camera.lens.clone();
        let has_flat = lens.as_ref().is_some_and(|l| self.config.flat_fields.contains_key(l));
        let mut optics_action: Option<OpticsAction> = None;
        let hsl_tab = &mut self.hsl_tab;
        let (cropping, crop_before, crop_aspect) = (&mut self.cropping, &mut self.crop_before, &mut self.crop_aspect);
        let source = self.image.as_ref().map(|img| (img.width as f32, img.height as f32, img.colour.turns));
        let thumbs: Vec<(Look, egui::TextureHandle)> = self.thumbs.as_ref().map(|(_, t)| t.clone()).unwrap_or_default();
        let p = &mut self.params;
        let d = AdjustParams::default();
        let mut reset_section: Option<Section> = None;
        let mut header = |ui: &mut egui::Ui, title: &str, section: Section, p: &AdjustParams| {
            if section_header(ui, title, !p.section_is_neutral(section)) {
                reset_section = Some(section);
            }
        };

        egui::ScrollArea::vertical().show(ui, |ui| {
            header(ui, "Colour", Section::Colour, p);
            ui.horizontal(|ui| {
                ui.selectable_value(&mut p.colour_mode, ColourMode::Camera, "Camera")
                    .on_hover_text("The camera's colour matrix — right for visible light");
                ui.selectable_value(&mut p.colour_mode, ColourMode::SensorRgb, "Sensor RGB")
                    .on_hover_text("No colour matrix — for infrared, where the camera's matrix scrambles hues");
            });
            ui.horizontal(|ui| {
                ui.add_enabled_ui(has_image, |ui| {
                    ui.toggle_value(picking_wb, format!("Pick neutral ({})", PICK_WB_KEY.name()))
                        .on_hover_text("Then click something that should be neutral: foliage in IR, a wall, a grey card");
                });
                if ui.add_enabled(p.wb_gains.is_some(), egui::Button::new("As shot")).clicked() {
                    p.wb_gains = None;
                }
                ui.label(if p.wb_gains.is_some() { "custom" } else { "as shot" });
            });
            // The sliders show the as-shot gains until one is moved.
            let mut g = p.wb_gains.unwrap_or(as_shot);
            let r = ui.add(
                egui::Slider::new(&mut g[0], WB_GAIN_MIN..=WB_GAIN_MAX).logarithmic(true).text("Red gain"),
            );
            let b = ui.add(
                egui::Slider::new(&mut g[2], WB_GAIN_MIN..=WB_GAIN_MAX).logarithmic(true).text("Blue gain"),
            );
            if r.changed() || b.changed() {
                p.wb_gains = Some(g);
            }

            ui.add_space(6.0);
            // Saved looks first: a saved look may share a built-in's name and
            // is the one `Config::find_look` resolves to.
            let current = saved_looks
                .iter()
                .chain(&builtin_looks)
                .find(|l| l.is_applied(p))
                .map(|l| l.name.clone());
            let current_saved = saved_looks.iter().any(|l| Some(&l.name) == current.as_ref());
            egui::ComboBox::from_label("Look")
                .selected_text(current.as_deref().unwrap_or("Custom"))
                .show_ui(ui, |ui| {
                    for look in &builtin_looks {
                        let item = ui.selectable_label(look.is_applied(p), &look.name);
                        let item = if look.note.is_empty() { item } else { item.on_hover_text(&look.note) };
                        if item.clicked() {
                            look.apply(p);
                        }
                    }
                    if !saved_looks.is_empty() {
                        ui.separator();
                        for look in &saved_looks {
                            if ui.selectable_label(look.is_applied(p), &look.name).clicked() {
                                look.apply(p);
                            }
                        }
                    }
                });
            if !thumbs.is_empty() {
                egui::CollapsingHeader::new("Look previews").default_open(true).show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        for (look, tex) in &thumbs {
                            // Scale by the long side so portrait crops fit too.
                            let size = tex.size_vec2() * (92.0 / tex.size_vec2().max_elem());
                            let selected = look.is_applied(p);
                            let tip = if look.note.is_empty() { look.name.clone() } else { format!("{}\n{}", look.name, look.note) };
                            let resp = ui
                                .add(egui::Button::image(egui::Image::new((tex.id(), size))).selected(selected))
                                .on_hover_text(tip);
                            if resp.clicked() {
                                look.apply(p);
                            }
                        }
                    });
                });
            }
            ui.add_enabled_ui(p.mixer != IDENTITY3, |ui| {
                resettable(ui, &mut p.mixer_strength, d.mixer_strength, |ui, v| {
                    ui.add(egui::Slider::new(v, 0.0..=1.5).text("Look strength"))
                        .on_hover_text("Blend the look's channel mix with the plain image; past 1 exaggerates it")
                });
            });
            if let Some(note) = builtin_looks
                .iter()
                .find(|l| Some(&l.name) == current.as_ref() && !current_saved)
                .map(|l| &l.note)
                .filter(|n| !n.is_empty())
            {
                ui.label(egui::RichText::new(note).weak().small());
            }
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(look_name).hint_text("Look name").desired_width(120.0));
                let name = look_name.trim().to_string();
                if ui
                    .add_enabled(!name.is_empty(), egui::Button::new("Save look"))
                    .on_hover_text("Saves white balance, colour mode, mixer and saturation; replaces a saved look of the same name")
                    .clicked()
                {
                    look_action = Some(LookAction::Save(Look::from_params(&name, p)));
                }
                if let (true, Some(name)) = (current_saved, &current) {
                    if ui.button("Delete").on_hover_text(format!("Delete the saved look \"{name}\"")).clicked() {
                        look_action = Some(LookAction::Delete(name.clone()));
                    }
                }
            });
            if let Some(serial) = &serial {
                let starts = body_look.as_deref().unwrap_or("no look");
                ui.label(egui::RichText::new(format!("Camera {serial} starts in: {starts}")).weak().small());
                ui.horizontal(|ui| {
                    let assignable = current.as_ref().filter(|n| n.as_str() != "None");
                    let label = match assignable {
                        Some(n) => format!("Start this camera in \"{n}\""),
                        None => "Start this camera in this look".into(),
                    };
                    if ui
                        .add_enabled(assignable.is_some() && assignable != body_look.as_ref(), egui::Button::new(label))
                        .on_hover_text("Photos from this body open in this look until they have edits of their own")
                        .on_disabled_hover_text("Pick or save a named look first")
                        .clicked()
                    {
                        look_action = assignable.map(|n| LookAction::Assign(serial.clone(), n.clone()));
                    }
                    if body_look.is_some() && ui.button("Clear").clicked() {
                        look_action = Some(LookAction::Unassign(serial.clone()));
                    }
                });
            }
            egui::CollapsingHeader::new("Channel mixer").show(ui, |ui| {
                egui::Grid::new("mixer").num_columns(5).spacing([6.0, 4.0]).show(ui, |ui| {
                    ui.label("");
                    for h in ["R in", "G in", "B in", "Sum"] {
                        ui.label(egui::RichText::new(h).weak());
                    }
                    ui.end_row();
                    for (i, out) in ["R out", "G out", "B out"].iter().enumerate() {
                        ui.label(*out);
                        for j in 0..3 {
                            ui.add(
                                egui::DragValue::new(&mut p.mixer[i][j])
                                    .speed(0.01)
                                    .range(-2.0..=2.0)
                                    .fixed_decimals(2),
                            );
                        }
                        let sum: f32 = p.mixer[i].iter().sum();
                        let text = egui::RichText::new(format!("{sum:.2}"));
                        let text = if (sum - 1.0).abs() > 0.005 { text.color(ui.visuals().warn_fg_color) } else { text.weak() };
                        ui.label(text).on_hover_text("Rows that sum to 1 keep whites neutral");
                        ui.end_row();
                    }
                });
                if ui.add_enabled(p.mixer != IDENTITY3, egui::Button::new("Reset mixer")).clicked() {
                    p.mixer = IDENTITY3;
                }
            });

            ui.add_space(10.0);
            header(ui, "Basic", Section::Basic, p);
            resettable(ui, &mut p.exposure_ev, d.exposure_ev, |ui, v| ui.add(egui::Slider::new(v, -5.0..=5.0).text("Exposure (EV)")));
            resettable(ui, &mut p.contrast, d.contrast, |ui, v| ui.add(egui::Slider::new(v, -1.0..=1.0).text("Contrast")));
            resettable(ui, &mut p.temp, d.temp, |ui, v| ui.add(egui::Slider::new(v, -1.0..=1.0).text("Temp")));
            resettable(ui, &mut p.tint, d.tint, |ui, v| ui.add(egui::Slider::new(v, -1.0..=1.0).text("Tint")));
            resettable(ui, &mut p.saturation, d.saturation, |ui, v| ui.add(egui::Slider::new(v, -1.0..=1.0).text("Saturation")));

            ui.add_space(10.0);
            header(ui, "Per colour", Section::Hsl, p);
            ui.horizontal(|ui| {
                for (i, name) in ["Hue", "Saturation", "Luminance"].iter().enumerate() {
                    ui.selectable_value(hsl_tab, i, *name);
                }
            });
            for (b, (name, centre)) in HSL_BANDS.iter().enumerate() {
                ui.horizontal(|ui| {
                    let swatch = egui::ecolor::Hsva::new(centre / 360.0, 0.8, 0.9, 1.0);
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 2.0, egui::Color32::from(swatch));
                    let t = *hsl_tab;
                    resettable(ui, &mut p.hsl[b][t], d.hsl[b][t], |ui, v| ui.add(egui::Slider::new(v, -1.0..=1.0).text(*name)));
                });
            }

            ui.add_space(10.0);
            header(ui, "Tonal range", Section::Tonal, p);
            resettable(ui, &mut p.highlights, d.highlights, |ui, v| ui.add(egui::Slider::new(v, -1.0..=1.0).text("Highlights")));
            resettable(ui, &mut p.shadows, d.shadows, |ui, v| ui.add(egui::Slider::new(v, -1.0..=1.0).text("Shadows")));
            resettable(ui, &mut p.whites, d.whites, |ui, v| ui.add(egui::Slider::new(v, -1.0..=1.0).text("Whites")));
            resettable(ui, &mut p.blacks, d.blacks, |ui, v| ui.add(egui::Slider::new(v, -1.0..=1.0).text("Blacks")));

            ui.add_space(10.0);
            header(ui, "Lift / Gamma / Gain", Section::Wheels, p);
            ui.horizontal(|ui| {
                ColorWheel::new("Lift", &mut p.lift, 0.0, 0.5).show(ui);
                ColorWheel::new("Gamma", &mut p.gamma, 1.0, 0.5).show(ui);
                ColorWheel::new("Gain", &mut p.gain, 1.0, 0.5).show(ui);
            });

            ui.add_space(10.0);
            header(ui, "Detail", Section::Detail, p);
            resettable(ui, &mut p.nr_luma, d.nr_luma, |ui, v| ui.add(egui::Slider::new(v, 0.0..=1.0).text("NR luminance")));
            resettable(ui, &mut p.nr_chroma, d.nr_chroma, |ui, v| ui.add(egui::Slider::new(v, 0.0..=1.0).text("NR colour")));
            resettable(ui, &mut p.sharpen_amount, d.sharpen_amount, |ui, v| ui.add(egui::Slider::new(v, 0.0..=3.0).text("Sharpen")));
            resettable(ui, &mut p.sharpen_radius, d.sharpen_radius, |ui, v| ui.add(egui::Slider::new(v, 0.3..=3.0).text("Radius")));
            resettable(ui, &mut p.sharpen_threshold, d.sharpen_threshold, |ui, v| {
                ui.add(egui::Slider::new(v, 0.0..=0.05).text("Threshold").fixed_decimals(3))
            });
            ui.label(
                egui::RichText::new("Detail acts on source pixels — judge it at 1:1")
                    .weak()
                    .small(),
            );

            ui.add_space(10.0);
            header(ui, "Optics", Section::Optics, p);
            resettable(ui, &mut p.hotspot_amount, d.hotspot_amount, |ui, v| ui.add(egui::Slider::new(v, 0.0..=2.0).text("Hot spot")))
                .on_hover_text("How much brighter the lens's central hot spot is than its surroundings; 0 = off");
            ui.add_enabled_ui(p.hotspot_amount > 0.0, |ui| {
                resettable(ui, &mut p.hotspot_radius, d.hotspot_radius, |ui, v| ui.add(egui::Slider::new(v, 0.05..=1.0).text("Spot radius")));
                resettable(ui, &mut p.hotspot_centre[0], d.hotspot_centre[0], |ui, v| ui.add(egui::Slider::new(v, 0.0..=1.0).text("Spot X")));
                resettable(ui, &mut p.hotspot_centre[1], d.hotspot_centre[1], |ui, v| ui.add(egui::Slider::new(v, 0.0..=1.0).text("Spot Y")));
                egui::CollapsingHeader::new("Spot colour").show(ui, |ui| {
                    for (i, name) in ["Red", "Green", "Blue"].iter().enumerate() {
                        resettable(ui, &mut p.hotspot_balance[i], d.hotspot_balance[i], |ui, v| {
                            ui.add(egui::Slider::new(v, 0.0..=2.0).text(*name))
                        });
                    }
                });
            });
            match &lens {
                Some(lens) => {
                    ui.add_enabled(has_flat, egui::Checkbox::new(&mut p.flat_field, format!("Flat field ({lens})")))
                        .on_hover_text("Divides out this lens's hot spot, vignetting and colour shading")
                        .on_disabled_hover_text("No flat field for this lens yet");
                    ui.horizontal(|ui| {
                        let label = if has_flat { "Replace from this photo" } else { "Make from this photo" };
                        if ui
                            .add_enabled(has_image, egui::Button::new(label))
                            .on_hover_text(
                                "Use this photo as the lens's flat field: an evenly lit wall or overcast sky, \
                                 defocused, same lens, filter and aperture, nothing blown out",
                            )
                            .clicked()
                        {
                            optics_action = Some(OpticsAction::Make(lens.clone()));
                        }
                        if has_flat && ui.button("Remove").clicked() {
                            optics_action = Some(OpticsAction::Remove(lens.clone()));
                        }
                    });
                }
                None => {
                    ui.label(egui::RichText::new("Lens unknown — flat fields need the lens model").weak().small());
                }
            }

            ui.add_space(10.0);
            header(ui, "Crop & rotate", Section::Geometry, p);
            // The rotated frame's size, for straightening and aspect ratios.
            let frame = |p: &AdjustParams| {
                source.map(|(w, h, turns)| if (turns + p.rotation) % 2 == 1 { (h, w) } else { (w, h) })
            };
            ui.add_enabled_ui(source.is_some(), |ui| {
                ui.horizontal(|ui| {
                    let mut turned = false;
                    if ui.button("⟲ 90°").on_hover_text("Rotate anticlockwise").clicked() {
                        p.rotation = (p.rotation + 3) % 4;
                        turned = true;
                    }
                    if ui.button("⟳ 90°").on_hover_text("Rotate clockwise").clicked() {
                        p.rotation = (p.rotation + 1) % 4;
                        turned = true;
                    }
                    // The old crop was for the other orientation; start over.
                    if turned && let Some((w, h)) = frame(p) {
                        p.crop = straighten_crop(p.straighten, w, h);
                    }
                    if ui.toggle_value(cropping, "Crop").on_hover_text("Drag the box on the photo; Enter when done").changed()
                        && *cropping
                    {
                        *crop_before = p.crop;
                    }
                });
                let before = p.straighten;
                resettable(ui, &mut p.straighten, d.straighten, |ui, v| {
                    ui.add(egui::Slider::new(v, -45.0..=45.0).step_by(0.1).suffix("°").text("Straighten"))
                        .on_hover_text("Fine rotation; the crop shrinks so no empty corners show")
                });
                if p.straighten != before && let Some((w, h)) = frame(p) {
                    p.crop = straighten_crop(p.straighten, w, h);
                    if let Some(a) = *crop_aspect {
                        p.crop = fit_aspect(p.crop, a * h / w);
                    }
                }
                const RATIOS: [(&str, Option<f32>); 9] = [
                    ("Free", None),
                    ("Original", Some(0.0)),
                    ("1:1", Some(1.0)),
                    ("4:5", Some(0.8)),
                    ("5:4", Some(1.25)),
                    ("3:2", Some(1.5)),
                    ("2:3", Some(2.0 / 3.0)),
                    ("16:9", Some(16.0 / 9.0)),
                    ("9:16", Some(9.0 / 16.0)),
                ];
                let label = RATIOS
                    .iter()
                    .find(|(_, r)| *r == *crop_aspect || (r == &Some(0.0) && crop_aspect.is_some_and(|a| frame(p).is_some_and(|(w, h)| (a - w / h).abs() < 1e-4))))
                    .map_or("Custom", |(n, _)| n);
                egui::ComboBox::from_label("Aspect").selected_text(label).show_ui(ui, |ui| {
                    for (name, ratio) in RATIOS {
                        if ui.selectable_label(name == label, name).clicked() {
                            let Some((w, h)) = frame(p) else { continue };
                            // "Original" = the frame's own ratio.
                            *crop_aspect = ratio.map(|r| if r == 0.0 { w / h } else { r });
                            if let Some(a) = *crop_aspect {
                                p.crop = fit_aspect(p.crop, a * h / w);
                            }
                        }
                    }
                });
            });

            ui.add_space(12.0);
            reset = ui
                .add_enabled(!p.is_neutral(), egui::Button::new("Reset all"))
                .on_hover_text("Each slider also has its own ↺, or double-click a slider to reset it")
                .clicked();
            ui.label(
                egui::RichText::new(format!("Hold {} to compare with the original", COMPARE_KEY.name()))
                    .weak()
                    .small(),
            );

            ui.add_space(14.0);
            ui.separator();
            ui.label(egui::RichText::new("Export").strong());
            egui::ComboBox::from_id_salt("export_format")
                .selected_text(export_format.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(export_format, ExportFormat::Jpeg, ExportFormat::Jpeg.label());
                    ui.selectable_value(
                        export_format,
                        ExportFormat::Tiff16,
                        ExportFormat::Tiff16.label(),
                    );
                });
            if *export_format == ExportFormat::Jpeg {
                ui.add(egui::Slider::new(jpeg_quality, 1..=100).text("Quality"));
            }
            let immich_ready = self.config.immich_ready();
            let upload = &mut self.config.upload_after_export;
            ui.add_enabled(
                immich_ready,
                egui::Checkbox::new(upload, "Upload to Immich after export"),
            )
            .on_disabled_hover_text("Set the Immich server and API key in File → Settings…");

            do_export = ui
                .add_enabled(has_image && !exporting, egui::Button::new("Export full resolution…"))
                .clicked();
            ui.add_space(8.0);
        });

        if reset {
            self.params = AdjustParams::default();
        } else if let Some(section) = reset_section {
            self.params.reset_section(section);
        }
        if let Some(action) = look_action {
            self.apply_look_action(action);
        }
        if let Some(action) = optics_action {
            self.apply_optics_action(action);
        }
        if do_export {
            // Always ask where to save: the dialog also gives us overwrite
            // confirmation for free, which the old silent
            // `<stem>_edit.jpg`-beside-the-raw behaviour did not.
            let ctx = ui.ctx().clone();
            let format = self.export_format;
            self.pick_and_export(&ctx, format);
        }
    }

    fn apply_optics_action(&mut self, action: OpticsAction) {
        let Some(img) = self.image.as_mut() else { return };
        let flat = match &action {
            OpticsAction::Make(lens) => match raw_editor::flatfield::FlatField::from_image(img, lens) {
                Ok(ff) => {
                    self.config.flat_fields.insert(lens.clone(), ff.clone());
                    self.status = format!("Flat field saved for {lens} — now applied to photos taken with it");
                    Some(Arc::new(ff))
                }
                Err(e) => {
                    self.status = format!("Can't use this photo as a flat field: {e:#}");
                    return;
                }
            },
            OpticsAction::Remove(lens) => {
                self.config.flat_fields.remove(lens);
                self.status = format!("Flat field for {lens} removed");
                None
            }
        };
        // The open photo picks the change up at once: re-upload it with the
        // new (or no) flat field. `make_mut` only copies if an export holds it.
        Arc::make_mut(img).flat = flat.clone();
        if let Some(preview) = &mut self.preview {
            preview.flat = flat;
        }
        self.histogram = None;
        let mut renderer = self.render_state.renderer.write();
        if let Some(r) = renderer.callback_resources.get_mut::<ImageRenderer>() {
            r.set_image(&self.render_state.device, &self.render_state.queue, img);
        }
        drop(renderer);
        self.baseline = sidecar::baseline(&self.config, &self.camera).0;
        if let Err(e) = self.config.save() {
            self.status = format!("Couldn't save settings: {e:#}");
        }
    }

    fn apply_look_action(&mut self, action: LookAction) {
        self.status = match &action {
            LookAction::Save(look) => format!("Saved look \"{}\"", look.name),
            LookAction::Delete(name) => format!("Deleted look \"{name}\""),
            LookAction::Assign(serial, name) => format!("Camera {serial} now starts in \"{name}\""),
            LookAction::Unassign(serial) => format!("Camera {serial} now starts neutral"),
        };
        match action {
            LookAction::Save(look) => {
                self.look_name.clear();
                self.config.save_look(look);
            }
            LookAction::Delete(name) => self.config.delete_look(&name),
            LookAction::Assign(serial, name) => {
                self.config.body_looks.insert(serial, name);
            }
            LookAction::Unassign(serial) => {
                self.config.body_looks.remove(&serial);
            }
        }
        // The open photo's no-sidecar baseline may have changed with it.
        self.baseline = sidecar::baseline(&self.config, &self.camera).0;
        if let Err(e) = self.config.save() {
            self.status = format!("Couldn't save settings: {e:#}");
        }
    }

    /// Eyedropper armed: crosshair over the image, a hint badge, and on click
    /// the white balance that makes that spot neutral. Esc disarms.
    /// The crop box over the whole frame: drag edges, corners or the inside.
    /// Enter finishes, Esc puts the crop back as it was.
    fn crop_overlay(
        &mut self,
        ui: &egui::Ui,
        rect: egui::Rect,
        response: &egui::Response,
        view: raw_editor::pipeline::View,
        frame: [usize; 2],
    ) {
        if ui.ctx().input(|i| i.key_pressed(egui::Key::Escape)) {
            self.params.crop = self.crop_before;
            self.cropping = false;
            return;
        }
        if ui.ctx().input(|i| i.key_pressed(egui::Key::Enter)) {
            self.cropping = false;
            return;
        }
        let quad = egui::Rect::from_center_size(
            rect.center() + egui::vec2(view.offset[0] * rect.width() / 2.0, -view.offset[1] * rect.height() / 2.0),
            egui::vec2(view.scale[0] * rect.width(), view.scale[1] * rect.height()),
        );
        let at = |x: f32, y: f32| quad.min + egui::vec2(x * quad.width(), y * quad.height());
        let [x0, y0, x1, y1] = self.params.crop;
        let crop = egui::Rect::from_min_max(at(x0, y0), at(x1, y1));

        // Interaction.
        let slop = 10.0;
        // Hit-test where the button went down: egui only reports a drag once
        // the pointer has already moved a few points, which on a fast drag
        // is enough to miss a corner.
        if response.drag_started()
            && let Some(pos) = ui.ctx().input(|i| i.pointer.press_origin()).or(response.interact_pointer_pos())
        {
            let grab = CropGrab::at(crop, pos, slop);
            if !grab.is_none() {
                self.crop_drag = Some((grab, self.params.crop, pos));
            }
        }
        if let (Some((grab, start, from)), Some(pos)) = (self.crop_drag, response.interact_pointer_pos())
            && response.dragged()
        {
            let d = pos - from;
            // Ratio in normalised units: pixel ratio × frame height / width.
            let aspect = self.crop_aspect.map(|a| a * frame[1] as f32 / frame[0] as f32);
            self.params.crop = drag_crop(start, grab, d.x / quad.width(), d.y / quad.height(), aspect);
        }
        if response.drag_stopped() {
            self.crop_drag = None;
        }
        if let Some(pos) = response.hover_pos() {
            let grab = self.crop_drag.map_or_else(|| CropGrab::at(crop, pos, slop), |(g, _, _)| g);
            ui.ctx().set_cursor_icon(grab.cursor());
        }

        // Drawing (with the crop as it is after this frame's drag).
        let [x0, y0, x1, y1] = self.params.crop;
        let crop = egui::Rect::from_min_max(at(x0, y0), at(x1, y1));
        let painter = ui.painter_at(rect);
        let shade = egui::Color32::from_black_alpha(150);
        for r in [
            egui::Rect::from_min_max(quad.min, egui::pos2(quad.max.x, crop.min.y)),
            egui::Rect::from_min_max(egui::pos2(quad.min.x, crop.max.y), quad.max),
            egui::Rect::from_min_max(egui::pos2(quad.min.x, crop.min.y), egui::pos2(crop.min.x, crop.max.y)),
            egui::Rect::from_min_max(egui::pos2(crop.max.x, crop.min.y), egui::pos2(quad.max.x, crop.max.y)),
        ] {
            painter.rect_filled(r, 0.0, shade);
        }
        let thin = egui::Stroke::new(1.0, egui::Color32::from_white_alpha(90));
        for t in [1.0 / 3.0, 2.0 / 3.0] {
            let x = crop.min.x + crop.width() * t;
            let y = crop.min.y + crop.height() * t;
            painter.line_segment([egui::pos2(x, crop.min.y), egui::pos2(x, crop.max.y)], thin);
            painter.line_segment([egui::pos2(crop.min.x, y), egui::pos2(crop.max.x, y)], thin);
        }
        painter.rect_stroke(crop, 0.0, egui::Stroke::new(1.5, egui::Color32::WHITE), egui::StrokeKind::Middle);
        for c in [crop.left_top(), crop.right_top(), crop.left_bottom(), crop.right_bottom()] {
            painter.rect_filled(egui::Rect::from_center_size(c, egui::vec2(10.0, 10.0)), 1.0, egui::Color32::WHITE);
        }
        let (pw, ph) = (((x1 - x0) * frame[0] as f32).round(), ((y1 - y0) * frame[1] as f32).round());
        let hint = format!("{pw} × {ph}  ·  drag the box  ·  Enter done  ·  Esc cancel");
        let badge = egui::Rect::from_min_size(rect.center_top() + egui::vec2(-170.0, 12.0), egui::vec2(340.0, 24.0));
        painter.rect_filled(badge, 4.0, egui::Color32::from_black_alpha(180));
        painter.text(badge.center(), egui::Align2::CENTER_CENTER, hint, egui::FontId::proportional(14.0), egui::Color32::WHITE);
    }

    fn handle_wb_pick(
        &mut self,
        ui: &egui::Ui,
        rect: egui::Rect,
        response: &egui::Response,
        view: raw_editor::pipeline::View,
        geom: Geometry,
        size: [usize; 2],
    ) {
        if ui.ctx().input(|i| i.key_pressed(egui::Key::Escape)) {
            self.picking_wb = false;
            return;
        }
        if response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        }
        let hint = "Click something neutral  ·  Esc cancels";
        let badge = egui::Rect::from_min_size(
            rect.center_top() + egui::vec2(-150.0, 12.0),
            egui::vec2(300.0, 24.0),
        );
        ui.painter().rect_filled(badge, 4.0, egui::Color32::from_black_alpha(180));
        ui.painter().text(
            badge.center(),
            egui::Align2::CENTER_CENTER,
            hint,
            egui::FontId::proportional(14.0),
            egui::Color32::WHITE,
        );

        if !response.clicked() {
            return;
        }
        let (Some(pos), Some(img)) = (response.interact_pointer_pos(), self.image.clone()) else {
            return;
        };
        let ndc = [
            (pos.x - rect.left()) / rect.width() * 2.0 - 1.0,
            1.0 - (pos.y - rect.top()) / rect.height() * 2.0,
        ];
        let Some([x, y]) = view.quad_uv(ndc).and_then(|uv| geom.source_pixel(uv, size[0], size[1])) else {
            self.status = "Missed the image — click on the photo".into();
            return;
        };
        match img.neutral_gains(x, y, &self.params) {
            Ok(pick) => {
                self.params.wb_gains = Some(pick.gains);
                self.picking_wb = false;
                let [r, _, b] = pick.gains;
                self.status = format!(
                    "White balance from ({x}, {y}): red ×{r:.2}, blue ×{b:.2}{}",
                    if pick.clamped { " — limited to the slider range" } else { "" }
                );
            }
            // Stay armed so the user can simply click somewhere better.
            Err(e) => self.status = format!("Can't pick there: {e}"),
        }
    }

    fn image_panel(&mut self, ui: &mut egui::Ui) {
        let (image_size, target_is_srgb) = {
            let renderer = self.render_state.renderer.read();
            match renderer.callback_resources.get::<ImageRenderer>() {
                Some(r) => (r.image_size, r.target_is_srgb),
                None => (None, false),
            }
        };

        let avail = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());
        let Some([iw, ih]) = image_size else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                if self.loading { "Loading…" } else { &self.status },
                egui::FontId::proportional(16.0),
                ui.visuals().text_color(),
            );
            return;
        };

        let texel = [1.0 / iw as f32, 1.0 / ih as f32];
        let colour = self.image.as_ref().map_or(ImageColour::LINEAR_SRGB, |img| img.colour);
        // In crop mode the whole (rotated, straightened) frame is shown and
        // the crop is drawn on top; otherwise the output is the cropped frame.
        let mut shown = self.params;
        if self.cropping {
            shown.crop = [0.0, 0.0, 1.0, 1.0];
            self.zoom_100 = false;
        }
        let geom = Geometry::resolve(&shown, &colour, iw, ih);
        let (ow, oh) = (geom.width, geom.height);
        // Viewport size in physical pixels — NDC [-1,1] spans exactly this.
        let ppp = ui.ctx().pixels_per_point();
        let (vw, vh) = (rect.width() * ppp, rect.height() * ppp);

        let view = if self.zoom_100 {
            // One source pixel per screen pixel: the quad runs off the edges
            // and the visible part is a crop the user pans around. Without
            // this the detail filters are invisible, because a 4592px image
            // shown in a ~1000px viewport throws away exactly the frequencies
            // sharpening and noise reduction act on.
            let scale = [ow as f32 / vw, oh as f32 / vh];
            if response.dragged() {
                let d = response.drag_delta();
                self.pan[0] += 2.0 * d.x * ppp / vw;
                self.pan[1] -= 2.0 * d.y * ppp / vh;
            }
            // Never drag the image completely off the viewport.
            for i in 0..2 {
                let slack = (scale[i] - 1.0).max(0.0);
                self.pan[i] = self.pan[i].clamp(-slack, slack);
            }
            raw_editor::pipeline::View { scale, offset: self.pan, texel }
        } else {
            self.pan = [0.0, 0.0];
            let img_aspect = ow as f32 / oh as f32;
            let rect_aspect = rect.width() / rect.height();
            let scale = if img_aspect > rect_aspect {
                [1.0, rect_aspect / img_aspect]
            } else {
                [img_aspect / rect_aspect, 1.0]
            };
            raw_editor::pipeline::View::fit(scale, texel)
        };

        // Hold-to-compare: feed the shader neutral params instead of bypassing
        // it, so the two views differ only by the adjustments themselves.
        // The geometry stays, so before and after line up.
        let comparing = shortcut(ui.ctx(), COMPARE_KEY, true);
        let mut uniforms = if comparing {
            let before = AdjustParams { rotation: shown.rotation, straighten: shown.straighten, crop: shown.crop, ..Default::default() };
            Uniforms::new(&before, &colour, view, target_is_srgb)
        } else {
            Uniforms::new(&shown, &colour, view, target_is_srgb)
        };
        uniforms.show_clip = if self.show_clip { 1.0 } else { 0.0 };

        ui.painter()
            .add(eframe::egui_wgpu::Callback::new_paint_callback(rect, ImageCallback { uniforms }));

        if self.picking_wb {
            self.handle_wb_pick(ui, rect, &response, view, geom, [iw, ih]);
        } else if self.cropping {
            self.crop_overlay(ui, rect, &response, view, [ow, oh]);
        }

        if comparing {
            let badge = egui::Rect::from_min_size(
                rect.left_top() + egui::vec2(12.0, 12.0),
                egui::vec2(76.0, 24.0),
            );
            ui.painter().rect_filled(badge, 4.0, egui::Color32::from_black_alpha(180));
            ui.painter().text(
                badge.center(),
                egui::Align2::CENTER_CENTER,
                "BEFORE",
                egui::FontId::proportional(14.0),
                egui::Color32::WHITE,
            );
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_loader();
        self.poll_ping();
        self.poll_export();

        // Leave Ctrl+Z alone while a text field has focus: it has its own undo.
        if !ui.ctx().egui_wants_keyboard_input() {
            // Redo first: Ctrl+Shift+Z would otherwise also match plain Ctrl+Z.
            let (redo, undo) = ui.ctx().input_mut(|i| {
                let redo = i.consume_shortcut(&REDO) || i.consume_shortcut(&REDO_ALT);
                (redo, !redo && i.consume_shortcut(&UNDO))
            });
            if redo {
                self.redo();
            } else if undo {
                self.undo();
            }
        }

        // Z flips between fit and 1:1.
        if shortcut(ui.ctx(), PICK_WB_KEY, false) && self.image.is_some() {
            self.picking_wb = !self.picking_wb;
        }
        if shortcut(ui.ctx(), CLIP_KEY, false) {
            self.show_clip = !self.show_clip;
        }
        if shortcut(ui.ctx(), ZOOM_KEY, false) {
            self.zoom_100 = !self.zoom_100;
        }

        // Load a dropped file (last one wins).
        let dropped: Option<PathBuf> = ui.ctx().input(|i| {
            i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).next_back()
        });
        if let Some(path) = dropped {
            if raw_editor::formats::is_supported(&path) {
                let ctx = ui.ctx().clone();
                self.start_load(path, &ctx);
            } else {
                self.status = format!("Can't open {}: not a supported photo format", path.display());
            }
        }

        egui::Panel::top("menu").show(ui, |ui| {
            self.menu_bar(ui);
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.label(&self.status);
        });
        egui::Panel::right("controls").min_size(320.0).show(ui, |ui| {
            self.side_panel(ui);
        });
        egui::CentralPanel::default().show(ui, |ui| {
            self.image_panel(ui);
        });
        self.settings_window(ui);

        // A drag is one undo step: record only once every button is up.
        let settled = !ui.ctx().input(|i| i.pointer.any_down());
        self.history.observe(self.params, settled);
        self.autosave_tick(ui.ctx());
    }

    fn on_exit(&mut self) {
        self.flush_sidecar();
    }
}

/// Develop, render and write one file of a batch.
fn batch_one(
    device: &eframe::wgpu::Device,
    queue: &eframe::wgpu::Queue,
    path: &std::path::Path,
    out_dir: &std::path::Path,
    format: ExportFormat,
    quality: u8,
    config: &Config,
) -> anyhow::Result<Option<String>> {
    let ids = raw_editor::metadata::camera_ids(path).unwrap_or_default();
    let (params, _) = sidecar::initial(path, config, &ids)?;
    let mut img = raw_editor::metadata::develop_file(path)?;
    img.flat = config.flat_for(ids.lens.as_deref());
    let rendered = match export::render_gpu(device, queue, &img, &params) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("batch: GPU render failed ({e:#}) — using the CPU path");
            export::render_cpu(&img, &params)
        }
    };
    let meta = raw_editor::metadata::read(path).ok();
    let out = export::batch_output_path(path, Some(out_dir), format);
    export::write(&out, &rendered, format, quality, meta.as_ref())?;
    Ok(maybe_upload(config, out).1)
}

/// Upload a finished export when the user has asked for it and Immich is
/// configured. Never fails the export: the file is already safely on disk, and
/// losing it because a server was unreachable would be a bad trade.
fn maybe_upload(config: &Config, out: PathBuf) -> (PathBuf, Option<String>) {
    if !(config.upload_after_export && config.immich_ready()) {
        return (out, None);
    }
    let note = match immich::upload(config, &out) {
        Ok(result) => result.describe(),
        Err(e) => format!("upload failed: {e:#}"),
    };
    (out, Some(note))
}
