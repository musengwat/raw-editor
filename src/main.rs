//! raw-editor: a basic GPU-accelerated RAW editor for Panasonic RW2 files.

// Release builds are a GUI app: without this Windows opens a console window
// behind it. Debug builds keep the console so stdout/stderr stay visible.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use eframe::egui;
use raw_editor::develop::{DevelopedImage, develop};
use raw_editor::export::{self, ExportFormat};
use raw_editor::gpu::{ImageCallback, ImageRenderer};
use raw_editor::pipeline::{AdjustParams, Uniforms};
use raw_editor::sidecar;
use raw_editor::widgets::ColorWheel;

/// Hold this key to bypass every adjustment and see the original.
const COMPARE_KEY: egui::Key = egui::Key::B;
/// Toggles between fit-to-window and 1:1.
const ZOOM_KEY: egui::Key = egui::Key::Z;
/// How long the params must sit still before the sidecar is written.
const AUTOSAVE_DEBOUNCE: Duration = Duration::from_millis(600);

fn main() -> eframe::Result {
    let path = std::env::args().nth(1).map(PathBuf::from);
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "raw-editor",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, path)?))),
    )
}

type LoadResult = (PathBuf, anyhow::Result<DevelopedImage>);
type ExportResult = anyhow::Result<PathBuf>;

struct App {
    render_state: eframe::egui_wgpu::RenderState,
    /// Live values the shader is driven from.
    params: AdjustParams,
    /// What is currently on disk in the sidecar.
    saved_params: AdjustParams,
    /// Previous frame's values, used to spot edits and restart the debounce.
    prev_params: AdjustParams,
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
            prev_params: AdjustParams::default(),
            dirty_since: None,
            image: None,
            current: None,
            status: "Open an .rw2: pass a path or drop a file here".into(),
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
            let result = rawler::decode_file(&path)
                .map_err(anyhow::Error::from)
                .and_then(|raw| develop(&raw));
            let _ = tx.send((path, result));
            ctx.request_repaint();
        });
    }

    fn poll_loader(&mut self) {
        while let Ok((path, result)) = self.loader_rx.try_recv() {
            self.loading = false;
            match result {
                Ok(img) => {
                    let mut renderer = self.render_state.renderer.write();
                    if let Some(r) = renderer.callback_resources.get_mut::<ImageRenderer>() {
                        r.set_image(&self.render_state.device, &self.render_state.queue, &img);
                    }
                    drop(renderer);

                    let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
                    let (params, note) = match sidecar::load(&path) {
                        Ok(Some(p)) => (p, " — edits restored"),
                        Ok(None) => (AdjustParams::default(), ""),
                        Err(e) => {
                            eprintln!("sidecar: {e:#}");
                            (AdjustParams::default(), " — sidecar unreadable, starting neutral")
                        }
                    };
                    self.status = format!("{name} — {}x{}{note}", img.width, img.height);
                    self.image = Some(Arc::new(img));
                    self.current = Some(path);
                    self.params = params;
                    self.saved_params = params;
                    self.prev_params = params;
                    self.dirty_since = None;
                }
                Err(e) => self.status = format!("Failed to load {}: {e:#}", path.display()),
            }
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
        // Neutral params mean "no edits" — drop the sidecar instead of
        // leaving a file full of zeroes next to the raw.
        let result = if self.params.is_neutral() {
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

    /// Export to the default `<stem>_edit.<ext>` next to the raw.
    fn start_export(&mut self, ctx: &egui::Context) {
        let Some(raw) = self.current.clone() else { return };
        let out = export::output_path(&raw, self.export_format);
        self.start_export_to(ctx, out);
    }

    fn start_export_to(&mut self, ctx: &egui::Context, out: PathBuf) {
        let (Some(img), Some(_raw)) = (self.image.clone(), self.current.clone()) else { return };
        self.flush_sidecar();

        let device = self.render_state.device.clone();
        let queue = self.render_state.queue.clone();
        let params = self.params;
        let format = self.export_format;
        let quality = self.jpeg_quality;
        let tx = self.export_tx.clone();
        let ctx = ctx.clone();

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
            let result = export::write(&out, &rendered, format, quality).map(|()| out);
            let _ = tx.send(result);
            ctx.request_repaint();
        });
    }

    fn poll_export(&mut self) {
        while let Ok(result) = self.export_rx.try_recv() {
            self.exporting = false;
            self.status = match result {
                Ok(path) => format!("Exported {}", path.display()),
                Err(e) => format!("Export failed: {e:#}"),
            };
        }
    }

    // --- ui --------------------------------------------------------------

    /// Ask for a raw to open. The dialog blocks the UI thread while it is up,
    /// which is the usual arrangement for a small egui app.
    fn pick_and_open(&mut self, ctx: &egui::Context) {
        let mut dialog = rfd::FileDialog::new().add_filter("Panasonic raw", &["rw2", "RW2"]);
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
                let changed = !self.params.is_neutral();
                if ui.add_enabled(changed, egui::Button::new("Reset adjustments")).clicked() {
                    self.params = AdjustParams::default();
                    ui.close();
                }
            });
        });
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Adjust");
        ui.add_space(6.0);

        let mut reset = false;
        let mut do_export = false;
        let has_image = self.image.is_some();
        let exporting = self.exporting;
        let (export_format, jpeg_quality) = (&mut self.export_format, &mut self.jpeg_quality);
        let p = &mut self.params;

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.label(egui::RichText::new("Basic").strong());
            ui.add(egui::Slider::new(&mut p.exposure_ev, -5.0..=5.0).text("Exposure (EV)"));
            ui.add(egui::Slider::new(&mut p.contrast, -1.0..=1.0).text("Contrast"));
            ui.add(egui::Slider::new(&mut p.temp, -1.0..=1.0).text("Temp"));
            ui.add(egui::Slider::new(&mut p.tint, -1.0..=1.0).text("Tint"));
            ui.add(egui::Slider::new(&mut p.saturation, -1.0..=1.0).text("Saturation"));

            ui.add_space(10.0);
            ui.label(egui::RichText::new("Tonal range").strong());
            ui.add(egui::Slider::new(&mut p.highlights, -1.0..=1.0).text("Highlights"));
            ui.add(egui::Slider::new(&mut p.shadows, -1.0..=1.0).text("Shadows"));
            ui.add(egui::Slider::new(&mut p.whites, -1.0..=1.0).text("Whites"));
            ui.add(egui::Slider::new(&mut p.blacks, -1.0..=1.0).text("Blacks"));

            ui.add_space(10.0);
            ui.label(egui::RichText::new("Lift / Gamma / Gain").strong());
            ui.horizontal(|ui| {
                ColorWheel::new("Lift", &mut p.lift, 0.0, 0.5).show(ui);
                ColorWheel::new("Gamma", &mut p.gamma, 1.0, 0.5).show(ui);
                ColorWheel::new("Gain", &mut p.gain, 1.0, 0.5).show(ui);
            });

            ui.add_space(10.0);
            ui.label(egui::RichText::new("Detail").strong());
            ui.add(egui::Slider::new(&mut p.nr_luma, 0.0..=1.0).text("NR luminance"));
            ui.add(egui::Slider::new(&mut p.nr_chroma, 0.0..=1.0).text("NR colour"));
            ui.add(egui::Slider::new(&mut p.sharpen_amount, 0.0..=3.0).text("Sharpen"));
            ui.add(
                egui::Slider::new(&mut p.sharpen_radius, 0.3..=3.0).text("Radius"),
            );
            ui.add(
                egui::Slider::new(&mut p.sharpen_threshold, 0.0..=0.05)
                    .text("Threshold")
                    .fixed_decimals(3),
            );
            ui.label(
                egui::RichText::new("Detail acts on source pixels — judge it at 1:1")
                    .weak()
                    .small(),
            );

            ui.add_space(12.0);
            reset = ui.button("Reset").clicked();
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
            do_export = ui
                .add_enabled(has_image && !exporting, egui::Button::new("Export full resolution"))
                .clicked();
            ui.add_space(8.0);
        });

        if reset {
            self.params = AdjustParams::default();
        }
        if do_export {
            let ctx = ui.ctx().clone();
            self.start_export(&ctx);
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
        let (rect, response) = ui.allocate_exact_size(avail, egui::Sense::drag());
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
        // Viewport size in physical pixels — NDC [-1,1] spans exactly this.
        let ppp = ui.ctx().pixels_per_point();
        let (vw, vh) = (rect.width() * ppp, rect.height() * ppp);

        let view = if self.zoom_100 {
            // One source pixel per screen pixel: the quad runs off the edges
            // and the visible part is a crop the user pans around. Without
            // this the detail filters are invisible, because a 4592px image
            // shown in a ~1000px viewport throws away exactly the frequencies
            // sharpening and noise reduction act on.
            let scale = [iw as f32 / vw, ih as f32 / vh];
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
            let img_aspect = iw as f32 / ih as f32;
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
        let comparing = ui.ctx().input(|i| i.key_down(COMPARE_KEY));
        let uniforms = if comparing {
            Uniforms::passthrough(view, target_is_srgb)
        } else {
            Uniforms::new(&self.params, view, target_is_srgb)
        };

        ui.painter()
            .add(eframe::egui_wgpu::Callback::new_paint_callback(rect, ImageCallback { uniforms }));

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
        self.poll_export();

        // Z flips between fit and 1:1.
        if ui.ctx().input(|i| i.key_pressed(ZOOM_KEY)) {
            self.zoom_100 = !self.zoom_100;
        }

        // Load a dropped file (last one wins).
        let dropped: Option<PathBuf> = ui.ctx().input(|i| {
            i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).next_back()
        });
        if let Some(path) = dropped {
            let ctx = ui.ctx().clone();
            self.start_load(path, &ctx);
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

        self.autosave_tick(ui.ctx());
    }

    fn on_exit(&mut self) {
        self.flush_sidecar();
    }
}
