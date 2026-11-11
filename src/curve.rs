//! Point tone curves: one for all three channels and one each for red,
//! green and blue, drawn by dragging points.
//!
//! A curve maps display-encoded values (sRGB transfer, 0..1 — the scale the
//! histogram is drawn on, so the midpoint of the curve is the midpoint of
//! the histogram) to display-encoded values. Between its points it is a
//! monotone cubic (Fritsch–Carlson): it passes through every point and never
//! overshoots between two of them, so a gentle S stays an S. Left of the
//! first point and right of the last it is flat.
//!
//! The shader evaluates the same cubic from the points and tangents packed
//! into its uniform block ([`Curves::uniforms`]) rather than from a lookup
//! texture: there is nothing to quantise, so a 16-bit export gets the exact
//! curve, and the CPU mirror ([`Curves::apply`]) is the same arithmetic.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::pipeline::srgb_encode;

/// Most points one curve can have, ends included. Fixed so `AdjustParams`
/// stays `Copy` and the uniform block has a fixed size.
pub const MAX_POINTS: usize = 8;
/// Closest two points may sit in x.
pub const MIN_GAP: f32 = 0.02;

/// One curve: `n` points sorted by x, the first `n` entries of `pts`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Curve {
    n: u8,
    pts: [[f32; 2]; MAX_POINTS],
}

impl Default for Curve {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Curve {
    /// The straight line from black to white: no change.
    pub const IDENTITY: Curve = {
        let mut pts = [[0.0; 2]; MAX_POINTS];
        pts[1] = [1.0, 1.0];
        Curve { n: 2, pts }
    };

    /// A curve through `points` (x, y in 0..1). They are sorted, clamped,
    /// pulled at least [`MIN_GAP`] apart and cut to [`MAX_POINTS`]; fewer
    /// than two gives the identity.
    pub fn from_points(points: &[[f32; 2]]) -> Self {
        let mut v: Vec<[f32; 2]> = points
            .iter()
            .filter(|p| p[0].is_finite() && p[1].is_finite())
            .map(|p| [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0)])
            .collect();
        v.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let mut kept: Vec<[f32; 2]> = Vec::new();
        for p in v {
            if kept.last().is_none_or(|l| p[0] - l[0] >= MIN_GAP) && kept.len() < MAX_POINTS {
                kept.push(p);
            }
        }
        if kept.len() < 2 {
            return Self::IDENTITY;
        }
        let mut c = Curve { n: kept.len() as u8, pts: [[0.0; 2]; MAX_POINTS] };
        c.pts[..kept.len()].copy_from_slice(&kept);
        c
    }

    pub fn points(&self) -> &[[f32; 2]] {
        &self.pts[..self.n as usize]
    }

    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    pub fn is_full(&self) -> bool {
        self.n as usize == MAX_POINTS
    }

    /// Add a point; returns its index, or `None` when the curve is full or
    /// the spot is within [`MIN_GAP`] of an existing point (grab that one).
    pub fn insert(&mut self, x: f32, y: f32) -> Option<usize> {
        let (x, y) = (x.clamp(0.0, 1.0), y.clamp(0.0, 1.0));
        let n = self.n as usize;
        if n == MAX_POINTS || self.points().iter().any(|p| (p[0] - x).abs() < MIN_GAP) {
            return None;
        }
        let i = self.points().iter().position(|p| p[0] > x).unwrap_or(n);
        self.pts.copy_within(i..n, i + 1);
        self.pts[i] = [x, y];
        self.n += 1;
        Some(i)
    }

    /// Remove a point. A curve keeps at least two; with two left, removing
    /// is refused.
    pub fn remove(&mut self, i: usize) -> bool {
        let n = self.n as usize;
        if n <= 2 || i >= n {
            return false;
        }
        self.pts.copy_within(i + 1..n, i);
        self.pts[n - 1] = [0.0; 2];
        self.n -= 1;
        true
    }

    /// Move point `i`, keeping it between its neighbours (points cannot
    /// cross) and inside the box.
    pub fn move_point(&mut self, i: usize, x: f32, y: f32) {
        let n = self.n as usize;
        if i >= n {
            return;
        }
        let lo = if i == 0 { 0.0 } else { self.pts[i - 1][0] + MIN_GAP };
        let hi = if i + 1 == n { 1.0 } else { self.pts[i + 1][0] - MIN_GAP };
        self.pts[i] = [x.clamp(lo, hi.max(lo)), y.clamp(0.0, 1.0)];
    }

    /// The point within `slop` (in the units of `scale`: the editor's size
    /// in pixels) of a position, the nearest if several are.
    pub fn hit(&self, at: [f32; 2], scale: [f32; 2], slop: f32) -> Option<usize> {
        self.points()
            .iter()
            .enumerate()
            .map(|(i, p)| (i, (((p[0] - at[0]) * scale[0]).powi(2) + ((p[1] - at[1]) * scale[1]).powi(2)).sqrt()))
            .filter(|(_, d)| *d <= slop)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    /// The curve as the shader evaluates it.
    pub fn spline(&self) -> Spline {
        Spline::new(self)
    }
}

impl Serialize for Curve {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(self.points())
    }
}

impl<'de> Deserialize<'de> for Curve {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Curve::from_points(&Vec::<[f32; 2]>::deserialize(d)?))
    }
}

/// A curve with its tangents worked out: what is evaluated per pixel.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Spline {
    n: usize,
    /// `[x, y, tangent]` per point.
    k: [[f32; 3]; MAX_POINTS],
}

impl Spline {
    /// Fritsch–Carlson tangents: the average of the two neighbouring slopes,
    /// zero at a peak or a flat stretch, and scaled back wherever a cubic
    /// through them would overshoot.
    pub fn new(c: &Curve) -> Self {
        let p = c.points();
        let n = p.len();
        let mut d = [0f32; MAX_POINTS];
        for i in 0..n - 1 {
            d[i] = (p[i + 1][1] - p[i][1]) / (p[i + 1][0] - p[i][0]);
        }
        let mut m = [0f32; MAX_POINTS];
        m[0] = d[0];
        m[n - 1] = d[n - 2];
        for i in 1..n - 1 {
            m[i] = if d[i - 1] * d[i] <= 0.0 { 0.0 } else { (d[i - 1] + d[i]) / 2.0 };
        }
        for i in 0..n - 1 {
            if d[i] == 0.0 {
                m[i] = 0.0;
                m[i + 1] = 0.0;
                continue;
            }
            let (a, b) = (m[i] / d[i], m[i + 1] / d[i]);
            let s = a * a + b * b;
            if s > 9.0 {
                let t = 3.0 / s.sqrt();
                m[i] = t * a * d[i];
                m[i + 1] = t * b * d[i];
            }
        }
        let mut k = [[0f32; 3]; MAX_POINTS];
        for i in 0..n {
            k[i] = [p[i][0], p[i][1], m[i]];
        }
        Self { n, k }
    }

    /// The curve at `x`. Mirrors `curve_eval` in the shader.
    pub fn eval(&self, x: f32) -> f32 {
        let k = &self.k;
        if x <= k[0][0] {
            return k[0][1];
        }
        let last = self.n - 1;
        if x >= k[last][0] {
            return k[last][1];
        }
        let mut i = 0;
        for j in 1..last {
            if x >= k[j][0] {
                i = j;
            }
        }
        let h = k[i + 1][0] - k[i][0];
        let t = (x - k[i][0]) / h;
        let (t2, t3) = (t * t, t * t * t);
        let y = (2.0 * t3 - 3.0 * t2 + 1.0) * k[i][1]
            + (t3 - 2.0 * t2 + t) * h * k[i][2]
            + (-2.0 * t3 + 3.0 * t2) * k[i + 1][1]
            + (t3 - t2) * h * k[i + 1][2];
        y.clamp(0.0, 1.0)
    }
}

/// The four curves of a photo.
#[derive(Clone, Copy, PartialEq, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Curves {
    /// Applied to all three channels, first.
    pub rgb: Curve,
    pub r: Curve,
    pub g: Curve,
    pub b: Curve,
}

/// Points and tangents for the shader: `[x, y, tangent, _]`, eight per
/// curve, in the order rgb, r, g, b.
pub type CurveUniform = [[f32; 4]; 4 * MAX_POINTS];

fn srgb_decode(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

impl Curves {
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }

    pub fn channel(&self, i: usize) -> &Curve {
        [&self.rgb, &self.r, &self.g, &self.b][i]
    }

    pub fn channel_mut(&mut self, i: usize) -> &mut Curve {
        [&mut self.rgb, &mut self.r, &mut self.g, &mut self.b][i]
    }

    /// `[points in rgb, r, g, b]` — 0 for a curve that is the identity, so
    /// the shader skips it — and the packed points.
    pub fn uniforms(&self) -> ([f32; 4], CurveUniform) {
        let mut info = [0f32; 4];
        let mut data = [[0f32; 4]; 4 * MAX_POINTS];
        for c in 0..4 {
            let curve = self.channel(c);
            if curve.is_identity() {
                continue;
            }
            let s = curve.spline();
            info[c] = s.n as f32;
            for i in 0..s.n {
                data[c * MAX_POINTS + i] = [s.k[i][0], s.k[i][1], s.k[i][2], 0.0];
            }
        }
        (info, data)
    }

    /// The curves applied to one linear RGB pixel. Values above white keep
    /// their ratio to white (the curve is only defined up to it), so an
    /// unclipped highlight stays brighter than white. Mirrors `curves` in
    /// the shader.
    pub fn apply(&self, c: [f32; 3]) -> [f32; 3] {
        if self.is_identity() {
            return c;
        }
        let master = (!self.rgb.is_identity()).then(|| self.rgb.spline());
        let channels = [&self.r, &self.g, &self.b].map(|k| (!k.is_identity()).then(|| k.spline()));
        std::array::from_fn(|i| {
            let mut v = srgb_encode(c[i].clamp(0.0, 1.0));
            if let Some(s) = &master {
                v = s.eval(v);
            }
            if let Some(s) = &channels[i] {
                v = s.eval(v);
            }
            srgb_decode(v) * c[i].max(1.0)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn the_identity_curve_changes_nothing() {
        let s = Curve::IDENTITY.spline();
        for i in 0..=100 {
            let x = i as f32 / 100.0;
            assert!(near(s.eval(x), x), "{x} -> {}", s.eval(x));
        }
        let c = [0.02, 0.4, 1.7];
        assert_eq!(Curves::default().apply(c), c, "and is skipped outright");
    }

    #[test]
    fn it_passes_through_every_point() {
        let c = Curve::from_points(&[[0.0, 0.05], [0.2, 0.1], [0.5, 0.6], [0.8, 0.85], [1.0, 0.95]]);
        let s = c.spline();
        for p in c.points() {
            assert!(near(s.eval(p[0]), p[1]), "{p:?} -> {}", s.eval(p[0]));
        }
    }

    #[test]
    fn rising_points_give_a_rising_curve_with_no_overshoot() {
        // A step between two flat stretches: a plain cubic spline rings here.
        let c = Curve::from_points(&[[0.0, 0.0], [0.3, 0.02], [0.4, 0.9], [0.7, 0.92], [1.0, 1.0]]);
        let s = c.spline();
        let mut prev = s.eval(0.0);
        for i in 1..=1000 {
            let y = s.eval(i as f32 / 1000.0);
            assert!(y >= prev - 1e-6, "falls at {}: {prev} -> {y}", i as f32 / 1000.0);
            prev = y;
        }
        // Between the two flat-ish points it stays between their values.
        for i in 0..=30 {
            let y = s.eval(0.4 + 0.3 * i as f32 / 30.0);
            assert!((0.9 - 1e-6..=0.92 + 1e-6).contains(&y), "{y}");
        }
    }

    #[test]
    fn it_is_flat_outside_its_end_points() {
        let s = Curve::from_points(&[[0.2, 0.1], [0.8, 0.9]]).spline();
        assert_eq!(s.eval(0.0), 0.1);
        assert_eq!(s.eval(0.2), 0.1);
        assert_eq!(s.eval(0.9), 0.9);
        assert!(near(s.eval(0.5), 0.5));
    }

    #[test]
    fn a_peak_is_allowed_and_stays_inside_the_box() {
        let s = Curve::from_points(&[[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]).spline();
        assert!(near(s.eval(0.5), 1.0));
        for i in 0..=100 {
            let y = s.eval(i as f32 / 100.0);
            assert!((0.0..=1.0).contains(&y));
        }
    }

    #[test]
    fn points_are_inserted_in_order_and_cannot_cross() {
        let mut c = Curve::IDENTITY;
        assert_eq!(c.insert(0.5, 0.6), Some(1));
        assert_eq!(c.insert(0.25, 0.2), Some(1));
        assert_eq!(c.insert(0.75, 0.8), Some(3));
        assert_eq!(c.points().iter().map(|p| p[0]).collect::<Vec<_>>(), [0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(c.insert(0.505, 0.1), None, "on top of an existing point");
        // Dragged past its neighbour, a point stops just short of it.
        c.move_point(2, 0.9, 0.3);
        assert!(near(c.points()[2][0], 0.75 - MIN_GAP) && c.points()[2][1] == 0.3);
        c.move_point(2, -1.0, 2.0);
        assert!(near(c.points()[2][0], 0.25 + MIN_GAP) && c.points()[2][1] == 1.0);
        // The ends move too, within the box.
        c.move_point(0, 0.1, 0.2);
        assert_eq!(c.points()[0], [0.1, 0.2]);
        c.move_point(4, 5.0, 0.7);
        assert_eq!(c.points()[4], [1.0, 0.7]);
    }

    #[test]
    fn a_curve_holds_at_most_eight_points_and_at_least_two() {
        let mut c = Curve::IDENTITY;
        for i in 1..=6 {
            assert!(c.insert(i as f32 / 7.0, 0.5).is_some());
        }
        assert!(c.is_full());
        assert_eq!(c.insert(0.95, 0.5), None);
        for _ in 0..6 {
            assert!(c.remove(1));
        }
        assert!(!c.remove(1), "the last two stay");
        assert!(c.is_identity());
    }

    #[test]
    fn hit_finds_the_nearest_point_in_pixels() {
        let c = Curve::from_points(&[[0.0, 0.0], [0.5, 0.5], [0.55, 0.9], [1.0, 1.0]]);
        let scale = [200.0, 200.0];
        assert_eq!(c.hit([0.51, 0.52], scale, 10.0), Some(1));
        assert_eq!(c.hit([0.54, 0.86], scale, 10.0), Some(2));
        assert_eq!(c.hit([0.3, 0.7], scale, 10.0), None);
    }

    #[test]
    fn sidecars_hold_the_points_and_bad_ones_are_repaired() {
        let curves = Curves { rgb: Curve::from_points(&[[0.0, 0.0], [0.4, 0.5], [1.0, 1.0]]), ..Default::default() };
        let json = serde_json::to_value(curves).unwrap();
        assert_eq!(json["rgb"].as_array().unwrap().len(), 3);
        assert_eq!(json["r"], serde_json::json!([[0.0, 0.0], [1.0, 1.0]]));
        assert_eq!(serde_json::from_value::<Curves>(json).unwrap(), curves);
        // Out of order, out of range, too close together, too many, too few.
        let messy: Curve = serde_json::from_str("[[0.9, 2.0], [0.1, -1.0], [0.105, 0.5]]").unwrap();
        assert_eq!(messy.points(), [[0.1, 0.0], [0.9, 1.0]]);
        let many: Curve = serde_json::from_value(serde_json::json!((0..20).map(|i| [i as f32 / 19.0, 0.5]).collect::<Vec<_>>())).unwrap();
        assert_eq!(many.points().len(), MAX_POINTS);
        let one: Curve = serde_json::from_str("[[0.5, 0.5]]").unwrap();
        assert!(one.is_identity());
        let old: Curves = serde_json::from_str("{}").unwrap();
        assert!(old.is_identity());
    }

    #[test]
    fn the_master_curve_runs_before_the_channel_curves() {
        let lift = Curve::from_points(&[[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]]);
        let crush = Curve::from_points(&[[0.0, 0.0], [0.7, 0.3], [1.0, 1.0]]);
        let curves = Curves { rgb: lift, r: crush, ..Default::default() };
        // A display mid-grey: 0.5 encoded.
        let grey = srgb_decode(0.5);
        let out = curves.apply([grey; 3]);
        assert!(near(srgb_encode(out[1]), 0.7), "green: the master curve only");
        assert!(near(srgb_encode(out[0]), 0.3), "red: 0.5 -> 0.7 -> 0.3, got {}", srgb_encode(out[0]));
    }

    #[test]
    fn above_white_keeps_its_ratio_to_white() {
        let curves = Curves { rgb: Curve::from_points(&[[0.0, 0.0], [0.5, 0.6], [1.0, 0.9]]), ..Default::default() };
        let white = curves.apply([1.0; 3])[0];
        assert!(near(srgb_encode(white), 0.9));
        let over = curves.apply([2.0; 3])[0];
        assert!(near(over, white * 2.0), "continuous at white and proportional above it");
        assert_eq!(curves.apply([-0.1; 3])[0], 0.0, "below black is black");
    }

    #[test]
    fn identity_curves_are_sent_as_empty() {
        let curves = Curves { g: Curve::from_points(&[[0.0, 0.1], [1.0, 1.0]]), ..Default::default() };
        let (info, data) = curves.uniforms();
        assert_eq!(info, [0.0, 0.0, 2.0, 0.0]);
        assert_eq!(data[2 * MAX_POINTS], [0.0, 0.1, 0.9, 0.0]);
        assert_eq!(data[2 * MAX_POINTS + 1], [1.0, 1.0, 0.9, 0.0]);
    }
}
