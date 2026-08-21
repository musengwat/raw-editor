//! raw-editor: a basic GPU-accelerated RAW editor for Panasonic RW2 files.

use std::path::PathBuf;
use std::sync::mpsc;

use eframe::egui;
use raw_editor::develop::{DevelopedImage, develop};
use raw_editor::gpu::{ImageCallback, ImageRenderer};
use raw_editor::pipeline::{AdjustParams, Uniforms};
use raw_editor::widgets::ColorWheel;

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

struct App {
    render_state: eframe::egui_wgpu::RenderState,
    params: AdjustParams,
    current: Option<PathBuf>,
    status: String,
    loader_tx: mpsc::Sender<LoadResult>,
    loader_rx: mpsc::Receiver<LoadResult>,
    loading: bool,
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
        let mut app = Self {
            render_state,
            params: AdjustParams::default(),
            current: None,
            status: "Open an .rw2: pass a path or drop a file here".into(),
            loader_tx,
            loader_rx,
            loading: false,
        };
        if let Some(p) = path {
            app.start_load(p, &cc.egui_ctx);
        }
        Ok(app)
    }

    fn start_load(&mut self, path: PathBuf, ctx: &egui::Context) {
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
                    self.status = format!(
                        "{} — {}x{}",
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        img.width,
                        img.height
                    );
                    self.current = Some(path);
                    self.params = AdjustParams::default();
                }
                Err(e) => self.status = format!("Failed to load {}: {e:#}", path.display()),
            }
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Adjust");
        ui.add_space(6.0);
        let mut reset = false;
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

            ui.add_space(12.0);
            reset = ui.button("Reset").clicked();
        });
        if reset {
            self.params = AdjustParams::default();
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
        let (rect, _) = ui.allocate_exact_size(avail, egui::Sense::hover());
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

        // Aspect-fit the quad inside the callback rect (NDC scale).
        let img_aspect = iw as f32 / ih as f32;
        let rect_aspect = rect.width() / rect.height();
        let scale = if img_aspect > rect_aspect {
            [1.0, rect_aspect / img_aspect]
        } else {
            [img_aspect / rect_aspect, 1.0]
        };

        ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(
            rect,
            ImageCallback { uniforms: Uniforms::new(&self.params, scale, target_is_srgb) },
        ));
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_loader();

        // Load a dropped file (last one wins).
        let dropped: Option<PathBuf> = ui.ctx().input(|i| {
            i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).next_back()
        });
        if let Some(path) = dropped {
            let ctx = ui.ctx().clone();
            self.start_load(path, &ctx);
        }

        egui::Panel::bottom("status").show(ui, |ui| {
            ui.label(&self.status);
        });
        egui::Panel::right("controls").min_size(320.0).show(ui, |ui| {
            self.side_panel(ui);
        });
        egui::CentralPanel::default().show(ui, |ui| {
            self.image_panel(ui);
        });
    }
}
