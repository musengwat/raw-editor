//! Bespoke egui widgets. Currently just the Resolve-style colour wheel used to
//! drive the lift / gamma / gain params.

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

#[cfg(test)]
mod tests {
    use super::*;

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
