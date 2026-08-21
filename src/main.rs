//! raw-editor: a basic GPU-accelerated RAW editor for Panasonic RW2 files.
//!
//! Phase 0: hello-world eframe window on the wgpu backend.

use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "raw-editor",
        options,
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}

#[derive(Default)]
struct App {}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading("raw-editor");
            ui.label("Phase 0: wgpu backend up. Open an RW2 coming in Phase 1.");
        });
    }
}
