//! Bespoke egui widgets: the Resolve-style colour wheel used to drive the
//! lift / gamma / gain params, and the RGB histogram.

use eframe::egui::{self, Color32, Pos2, Sense, Stroke, StrokeKind, Vec2};

/// Vectorscope-style channel directions: R up, G at 210°, B at 330°.
/// They sum to zero, so a knob offset is *pure chroma* — overall lightness is
/// controlled separately by the master value under each wheel, exactly like
/// Resolve's wheel + master-ring pairing.
const DIRS: [Vec2; 3] =
    [Vec2::new(0.0, 1.0), Vec2::new(-0.866_025_4, -0.5), Vec2::new(0.866_025_4, -0.5)];

/// Split a per-channel triple into (master, chroma) around `neutral`.
fn split(value: &[f32; 3], neutral: f32) -> (f32, [f32; 3]) {
    let master = (value[0] + value[1] + value[2]) / 3.0 - neutral;
    (master, std::array::from_fn(|i| value[i] - neutral - master))
}

fn combine(master: f32, chroma: [f32; 3], neutral: f32) -> [f32; 3] {
    std::array::from_fn(|i| neutral + master + chroma[i])
}

/// Chroma triple → knob position in the unit disc. Inverse of
/// [`chroma_from_knob`] because the three directions are unit vectors 120°
/// apart, for which `Σ (u·dᵢ) dᵢ = 1.5 u`.
fn knob_from_chroma(chroma: [f32; 3], range: f32) -> Vec2 {
    let mut u = Vec2::ZERO;
    for i in 0..3 {
        u += DIRS[i] * (chroma[i] / range);
    }
    u * (2.0 / 3.0)
}

fn chroma_from_knob(u: Vec2, range: f32) -> [f32; 3] {
    std::array::from_fn(|i| range * (u.x * DIRS[i].x + u.y * DIRS[i].y))
}

/// The colour shown at the rim of the wheel for a given angle (radians, y-up).
fn rim_color(angle: f32) -> Color32 {
    let u = Vec2::new(angle.cos(), angle.sin());
    let ch = |d: Vec2| (((u.x * d.x + u.y * d.y) * 0.5 + 0.5) * 255.0) as u8;
    Color32::from_rgb(ch(DIRS[0]), ch(DIRS[1]), ch(DIRS[2]))
}

/// A circular chroma pad plus a master slider, writing a per-channel triple.
///
/// `neutral` is the value that means "no change" (0.0 for lift, 1.0 for gamma
/// and gain) and `range` is the full-deflection excursion from it. Returns
/// `true` when the user changed the value this frame. Double-click the wheel
/// to reset it.
pub struct ColorWheel<'a> {
    label: &'a str,
    value: &'a mut [f32; 3],
    neutral: f32,
    range: f32,
    diameter: f32,
}

impl<'a> ColorWheel<'a> {
    pub fn new(label: &'a str, value: &'a mut [f32; 3], neutral: f32, range: f32) -> Self {
        Self { label, value, neutral, range, diameter: 84.0 }
    }

    pub fn diameter(mut self, d: f32) -> Self {
        self.diameter = d;
        self
    }

    pub fn show(self, ui: &mut egui::Ui) -> bool {
        let Self { label, value, neutral, range, diameter } = self;
        let mut changed = false;

        ui.vertical(|ui| {
            ui.set_width(diameter);
            ui.label(label);

            let (rect, response) =
                ui.allocate_exact_size(Vec2::splat(diameter), Sense::click_and_drag());
            let center = rect.center();
            let radius = diameter * 0.5 - 2.0;

            let (mut master, mut chroma) = split(value, neutral);

            // --- interaction -------------------------------------------------
            if response.double_clicked() {
                chroma = [0.0; 3];
                master = 0.0;
                changed = true;
            } else if (response.dragged() || response.is_pointer_button_down_on())
                && let Some(p) = ui.ctx().pointer_interact_pos()
            {
                // egui's y grows downward; the wheel's maths is y-up.
                let mut u = Vec2::new((p.x - center.x) / radius, (center.y - p.y) / radius);
                if u.length() > 1.0 {
                    u /= u.length();
                }
                chroma = chroma_from_knob(u, range);
                changed = true;
            }

            // --- painting ----------------------------------------------------
            if ui.is_rect_visible(rect) {
                let painter = ui.painter();
                let mut mesh = egui::Mesh::default();
                mesh.colored_vertex(center, Color32::from_gray(128));
                const SEGMENTS: usize = 48;
                for i in 0..=SEGMENTS {
                    let a = i as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
                    // y-up for the colour maths, y-down for the screen position.
                    let pos = Pos2::new(center.x + a.cos() * radius, center.y - a.sin() * radius);
                    mesh.colored_vertex(pos, rim_color(a));
                    if i > 0 {
                        mesh.add_triangle(0, i as u32, i as u32 + 1);
                    }
                }
                painter.add(mesh);

                let faint = Stroke::new(1.0, Color32::from_black_alpha(90));
                painter.circle_stroke(center, radius, ui.visuals().widgets.inactive.fg_stroke);
                painter.line_segment(
                    [Pos2::new(center.x - radius, center.y), Pos2::new(center.x + radius, center.y)],
                    faint,
                );
                painter.line_segment(
                    [Pos2::new(center.x, center.y - radius), Pos2::new(center.x, center.y + radius)],
                    faint,
                );

                let u = knob_from_chroma(chroma, range);
                let knob = Pos2::new(center.x + u.x * radius, center.y - u.y * radius);
                painter.circle_filled(knob, 5.0, Color32::WHITE);
                painter.circle_stroke(knob, 5.0, Stroke::new(1.5, Color32::BLACK));

                if response.hovered() {
                    painter.rect_stroke(
                        rect,
                        2.0,
                        ui.visuals().widgets.hovered.fg_stroke,
                        StrokeKind::Inside,
                    );
                }
            }

            // --- master value -------------------------------------------------
            let before = master;
            ui.add(
                egui::DragValue::new(&mut master)
                    .speed(range / 200.0)
                    .range(-range..=range)
                    .fixed_decimals(2),
            );
            changed |= before != master;

            if changed {
                *value = combine(master, chroma, neutral);
            }
        });

        changed
    }
}

const RESET_GLYPH: &str = "↺";

/// Wrap one control with a way back to `default`: a ↺ button beside it that
/// shows once the value has moved (its space is always reserved, so nothing
/// shifts), or double-click the control itself.
/// `add` draws the control against the value, e.g. a `Slider`.
pub fn resettable<T: PartialEq + Copy + std::fmt::Debug>(
    ui: &mut egui::Ui,
    value: &mut T,
    default: T,
    add: impl FnOnce(&mut egui::Ui, &mut T) -> egui::Response,
) -> egui::Response {
    ui.horizontal(|ui| {
        let mut response = add(ui, value);
        let changed = *value != default;
        let button = ui
            .add_visible(changed, egui::Button::new(RESET_GLYPH).small().frame(false))
            .on_hover_text(format!("Reset to {default:?}"));
        // Sliders only sense drags, so their response never reports a
        // double-click; read it off the pointer instead.
        let double_clicked = ui.input(|i| {
            i.pointer.button_double_clicked(egui::PointerButton::Primary)
                && i.pointer.interact_pos().is_some_and(|p| response.rect.contains(p))
        });
        // No right-click menu: egui sliders drag with any button, so a
        // right-click would move the value before the menu opened.
        if button.clicked() || (changed && double_clicked) {
            *value = default;
            response.mark_changed();
        }
        response
    })
    .inner
}

/// A bold section heading with a Reset button on the right, enabled while
/// anything in the section differs from neutral. Returns true when clicked.
pub fn section_header(ui: &mut egui::Ui, title: &str, changed: bool) -> bool {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(title).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_enabled(changed, egui::Button::new("Reset").small())
                .on_hover_text(format!("Reset everything under {title}"))
                .clicked()
        })
        .inner
    })
    .inner
}

/// Which parts of a crop rectangle a drag moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CropGrab {
    pub left: bool,
    pub right: bool,
    pub top: bool,
    pub bottom: bool,
    /// Inside the rectangle, away from the edges: move the whole thing.
    pub inside: bool,
}

impl CropGrab {
    pub fn is_none(&self) -> bool {
        *self == Self::default()
    }

    /// What a pointer at `pos` grabs on the crop `r` (screen space), with
    /// edges `slop` points thick.
    pub fn at(r: egui::Rect, pos: Pos2, slop: f32) -> Self {
        let near_x = pos.x > r.left() - slop && pos.x < r.right() + slop;
        let near_y = pos.y > r.top() - slop && pos.y < r.bottom() + slop;
        let g = Self {
            left: near_y && (pos.x - r.left()).abs() < slop,
            right: near_y && (pos.x - r.right()).abs() < slop,
            top: near_x && (pos.y - r.top()).abs() < slop,
            bottom: near_x && (pos.y - r.bottom()).abs() < slop,
            inside: false,
        };
        if g.is_none() && r.contains(pos) { Self { inside: true, ..g } } else { g }
    }

    pub fn cursor(&self) -> egui::CursorIcon {
        use egui::CursorIcon::*;
        match (self.left || self.right, self.top || self.bottom) {
            _ if self.inside => Grab,
            (true, true) if (self.left && self.top) || (self.right && self.bottom) => ResizeNwSe,
            (true, true) => ResizeNeSw,
            (true, false) => ResizeHorizontal,
            (false, true) => ResizeVertical,
            _ => Default,
        }
    }
}

/// Smallest crop side, as a fraction of the frame.
pub const MIN_CROP: f32 = 0.02;

/// The crop after dragging `grab` by `(dx, dy)` (fractions of the frame) from
/// `start`. `aspect` is the locked width/height ratio *in normalised units*
/// (pixel ratio × frame height / frame width), or `None` for free.
pub fn drag_crop(start: [f32; 4], grab: CropGrab, dx: f32, dy: f32, aspect: Option<f32>) -> [f32; 4] {
    let [mut x0, mut y0, mut x1, mut y1] = start;
    if grab.inside {
        let (w, h) = (x1 - x0, y1 - y0);
        let nx = (x0 + dx).clamp(0.0, 1.0 - w);
        let ny = (y0 + dy).clamp(0.0, 1.0 - h);
        return [nx, ny, nx + w, ny + h];
    }
    if grab.left {
        x0 = (x0 + dx).clamp(0.0, x1 - MIN_CROP);
    }
    if grab.right {
        x1 = (x1 + dx).clamp(x0 + MIN_CROP, 1.0);
    }
    if grab.top {
        y0 = (y0 + dy).clamp(0.0, y1 - MIN_CROP);
    }
    if grab.bottom {
        y1 = (y1 + dy).clamp(y0 + MIN_CROP, 1.0);
    }
    let Some(a) = aspect else { return [x0, y0, x1, y1] };
    // Keep the ratio: a pure top/bottom drag sets the height and the width
    // follows about the centre; anything else sets the width and the height
    // follows away from the dragged edge.
    let horizontal = grab.left || grab.right;
    let (mut w, mut h) = (x1 - x0, y1 - y0);
    if horizontal { h = w / a } else { w = h * a }
    // Shrink to fit the frame from the anchored side.
    let (ax, ay) = (if grab.left { x1 } else { x0 }, if grab.top { y1 } else { y0 });
    let room_x = if horizontal { if grab.left { ax } else { 1.0 - ax } } else { 1.0 };
    let room_y = if grab.top { ay } else { 1.0 - ay };
    let k = (room_x / w).min(room_y / h).min(1.0);
    (w, h) = (w * k, h * k);
    let (x0, x1) = if horizontal {
        if grab.left { (ax - w, ax) } else { (ax, ax + w) }
    } else {
        let cx = ((start[0] + start[2]) / 2.0).clamp(w / 2.0, 1.0 - w / 2.0);
        (cx - w / 2.0, cx + w / 2.0)
    };
    let (y0, y1) = if grab.top { (ay - h, ay) } else { (ay, ay + h) };
    [x0, y0, x1, y1]
}

/// The largest crop of ratio `aspect` (normalised, as in [`drag_crop`])
/// inside `crop`, centred on it.
pub fn fit_aspect(crop: [f32; 4], aspect: f32) -> [f32; 4] {
    let [x0, y0, x1, y1] = crop;
    let (w, h) = (x1 - x0, y1 - y0);
    let (nw, nh) = if w / h > aspect { (h * aspect, h) } else { (w, w / aspect) };
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    [cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0]
}

/// Overlaid RGB histogram plus a per-channel clipped-percentage line.
pub fn histogram(ui: &mut egui::Ui, h: &crate::histogram::Histogram) {
    use crate::histogram::BINS;
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 80.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, Color32::from_gray(18));
    let colours = [
        Color32::from_rgba_unmultiplied(255, 70, 70, 120),
        Color32::from_rgba_unmultiplied(70, 230, 70, 120),
        Color32::from_rgba_unmultiplied(80, 120, 255, 140),
    ];
    let heights = h.heights();
    let bin_w = rect.width() / BINS as f32;
    // Adjoining rectangles in one mesh: no anti-aliased seams, and no gaps
    // from snapping ~1px strokes to the pixel grid.
    let mut mesh = egui::Mesh::default();
    for (ch, colour) in colours.iter().enumerate() {
        for (k, &v) in heights[ch].iter().enumerate() {
            if v <= 0.0 {
                continue;
            }
            let x0 = rect.left() + k as f32 * bin_w;
            let top = rect.bottom() - v * (rect.height() - 2.0);
            mesh.add_colored_rect(
                egui::Rect::from_min_max(Pos2::new(x0, top), Pos2::new(x0 + bin_w, rect.bottom())),
                *colour,
            );
        }
    }
    painter.add(mesh);

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Clipped").small());
        for (name, frac) in ["R", "G", "B"].iter().zip(h.clipped) {
            let pct = frac * 100.0;
            // Anything over a tenth of a percent is worth noticing.
            let text = egui::RichText::new(format!("{name} {pct:.1}%")).small();
            ui.label(if pct >= 0.1 { text.color(Color32::from_rgb(255, 140, 90)) } else { text.weak() });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run frames of a one-slider UI, feeding `events` into the last one.
    /// Returns the slider's response from the final frame.
    fn run_slider(
        ctx: &egui::Context,
        value: &mut f32,
        events: Vec<egui::Event>,
    ) -> egui::Response {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(Pos2::ZERO, Vec2::new(600.0, 200.0))),
            // Advance the clock so double-click timing works.
            time: Some(ctx.input(|i| i.time) + 0.05),
            events,
            ..Default::default()
        };
        let mut out = None;
        let mut output = ctx.run_ui(input, |ui| {
            out = Some(resettable(ui, value, 0.0, |ui, v| ui.add(egui::Slider::new(v, -1.0..=1.0))));
        });
        // No renderer here; drop the font atlas upload instead of applying it.
        output.textures_delta.clear();
        out.unwrap()
    }

    fn click(at: Pos2) -> Vec<egui::Event> {
        let button = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };
        vec![egui::Event::PointerMoved(at), button(true), button(false)]
    }

    fn approx(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn crop_grab_finds_edges_corners_and_inside() {
        let r = egui::Rect::from_min_max(Pos2::new(100.0, 100.0), Pos2::new(300.0, 200.0));
        let g = CropGrab::at(r, Pos2::new(101.0, 199.0), 8.0);
        assert!(g.left && g.bottom && !g.right && !g.top);
        assert!(CropGrab::at(r, Pos2::new(200.0, 150.0), 8.0).inside);
        assert!(CropGrab::at(r, Pos2::new(200.0, 98.0), 8.0).top);
        assert!(CropGrab::at(r, Pos2::new(500.0, 150.0), 8.0).is_none());
    }

    #[test]
    fn free_drag_moves_only_the_grabbed_edges() {
        let start = [0.1, 0.1, 0.9, 0.9];
        let right = CropGrab { right: true, ..Default::default() };
        assert!(approx(drag_crop(start, right, -0.2, 0.3, None), [0.1, 0.1, 0.7, 0.9]));
        // Clamped to the frame and to the minimum size.
        assert!(approx(drag_crop(start, right, 0.5, 0.0, None), [0.1, 0.1, 1.0, 0.9]));
        assert!(approx(drag_crop(start, right, -2.0, 0.0, None), [0.1, 0.1, 0.1 + MIN_CROP, 0.9]));
    }

    #[test]
    fn moving_keeps_the_size_inside_the_frame() {
        let inside = CropGrab { inside: true, ..Default::default() };
        assert!(approx(drag_crop([0.2, 0.2, 0.6, 0.5], inside, 0.9, -0.9, None), [0.6, 0.0, 1.0, 0.3]));
    }

    #[test]
    fn locked_aspect_holds_through_a_corner_drag() {
        let br = CropGrab { right: true, bottom: true, ..Default::default() };
        let c = drag_crop([0.0, 0.0, 0.5, 0.5], br, 0.2, 0.05, Some(1.0));
        assert!(((c[2] - c[0]) - (c[3] - c[1])).abs() < 1e-5, "{c:?}");
        assert_eq!([c[0], c[1]], [0.0, 0.0], "top-left stays anchored");
        // Too big for the frame: shrinks to fit instead of spilling out.
        let c = drag_crop([0.0, 0.5, 0.4, 0.9], br, 0.5, 0.0, Some(1.0));
        assert!(c.iter().all(|v| (0.0..=1.0 + 1e-5).contains(v)), "{c:?}");
        assert!(((c[2] - c[0]) - (c[3] - c[1])).abs() < 1e-5, "{c:?}");
    }

    #[test]
    fn fit_aspect_is_centred_and_inside() {
        let c = fit_aspect([0.0, 0.0, 1.0, 0.5], 1.0);
        assert!(approx(c, [0.25, 0.0, 0.75, 0.5]), "{c:?}");
    }

    #[test]
    fn reset_button_restores_the_default() {
        let ctx = egui::Context::default();
        let mut v = 0.5;
        let slider = run_slider(&ctx, &mut v, vec![]);
        // The ↺ sits just right of the slider in the same row.
        let at = Pos2::new(slider.rect.right() + 10.0, slider.rect.center().y);
        run_slider(&ctx, &mut v, click(at));
        assert_eq!(v, 0.0);
    }

    #[test]
    fn untouched_value_has_no_visible_reset() {
        let ctx = egui::Context::default();
        let mut v = 0.0;
        let slider = run_slider(&ctx, &mut v, vec![]);
        let at = Pos2::new(slider.rect.right() + 10.0, slider.rect.center().y);
        run_slider(&ctx, &mut v, click(at));
        assert_eq!(v, 0.0);
    }

    #[test]
    fn double_click_on_the_slider_resets_it() {
        let ctx = egui::Context::default();
        let mut v = 0.5;
        let slider = run_slider(&ctx, &mut v, vec![]);
        // Off-centre on the rail: a single click would *set* the value there.
        let at = Pos2::new(slider.rect.left() + 20.0, slider.rect.center().y);
        // One event per frame, as a real pointer delivers them.
        let [mv, down, up] = <[egui::Event; 3]>::try_from(click(at)).unwrap();
        for e in [mv, down.clone(), up.clone(), down, up] {
            run_slider(&ctx, &mut v, vec![e]);
        }
        assert_eq!(v, 0.0);
    }

    #[test]
    fn knob_chroma_roundtrip() {
        for u in [
            Vec2::new(0.0, 0.0),
            Vec2::new(0.7, 0.0),
            Vec2::new(-0.3, 0.6),
            Vec2::new(0.0, -1.0),
        ] {
            let back = knob_from_chroma(chroma_from_knob(u, 0.5), 0.5);
            assert!((back - u).length() < 1e-5, "{u:?} -> {back:?}");
        }
    }

    #[test]
    fn chroma_is_luma_neutral() {
        let c = chroma_from_knob(Vec2::new(0.4, -0.8), 0.5);
        assert!(c.iter().sum::<f32>().abs() < 1e-6, "{c:?}");
    }

    #[test]
    fn split_combine_roundtrip() {
        for (v, neutral) in [([0.1f32, -0.2, 0.05], 0.0), ([1.3, 0.9, 1.0], 1.0)] {
            let (m, ch) = split(&v, neutral);
            let back = combine(m, ch, neutral);
            for i in 0..3 {
                assert!((back[i] - v[i]).abs() < 1e-6, "{v:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn pushing_the_knob_up_pushes_red() {
        // R sits at the top of the wheel.
        let c = chroma_from_knob(Vec2::new(0.0, 1.0), 0.5);
        assert!(c[0] > 0.0 && c[1] < 0.0 && c[2] < 0.0, "{c:?}");
    }

    #[test]
    fn rim_is_red_at_the_top() {
        let c = rim_color(std::f32::consts::FRAC_PI_2);
        assert!(c.r() > c.g() && c.r() > c.b());
    }
}
