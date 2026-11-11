// Adjustment pipeline. Mirrored in src/pipeline.rs (`apply`) — keep in sync,
// covered by the parity test in tests/shader_parity.rs.

// One mask slot — packed by `masks::uniforms` in src/masks.rs.
struct MaskU {
    v0: vec4<f32>,  // a.xy, b.xy (linear: the two ends; radial: centre, semi-axes)
    v1: vec4<f32>,  // cos, sin, feather, kind (0 off, 1 linear, 2 radial, 3 range)
    v2: vec4<f32>,  // amount, invert, sharpen, nr_luma
    v3: vec4<f32>,  // exposure, contrast, temp, tint
    v4: vec4<f32>,  // saturation, highlights, shadows, whites
    v5: vec4<f32>,  // blacks
    v6: vec4<f32>,  // brightness range: from, to, feather, on
    v7: vec4<f32>,  // colour range: hue, width, feather (fractions of the circle), on
};

struct Uniforms {
    // vec4s first: WGSL uniform buffers need 16-byte alignment for them.
    wb: vec4<f32>,         // .rgb = white-balance gains
    m0: vec4<f32>,         // rows of the camera -> working matrix (.rgb)
    m1: vec4<f32>,
    m2: vec4<f32>,
    lift: vec4<f32>,       // .rgb = per-channel lift  (neutral 0)
    gamma: vec4<f32>,      // .rgb = per-channel gamma (neutral 1)
    gain: vec4<f32>,       // .rgb = per-channel gain  (neutral 1)
    hotspot: vec4<f32>,    // .rgb = hot-spot strength per channel, .a = radius
    optics: vec4<f32>,     // .xy = hot-spot centre (uv), .z = aspect w/h, .w = flat field on
    hsl: array<vec4<f32>, 8>, // per-colour [hue, sat, lum, _]; hsl[0].w = 1 when active
    geom0: vec4<f32>,      // output uv -> source uv, row 0 (.xyz); see pipeline::Geometry
    geom1: vec4<f32>,      // row 1
    mask_info: vec4<f32>,  // .x = slots to evaluate, .y = overlay slot + 1 (0 = none)
    masks: array<MaskU, 8>,
    curve_info: vec4<f32>, // points in the rgb, r, g, b point curves (0 = off)
    curves: array<vec4<f32>, 32>, // [x, y, tangent, _], eight per curve
    vig0: vec4<f32>,       // vignette: ev, start, end, power (all zero = off)
    vig1: vec4<f32>,       // vignette: kx, ky, norm
    effects: vec4<f32>,    // grain amount, grain size, vibrance
    local0: vec4<f32>,     // .rgb = base luminance weights on camera RGB, .w = bases bound
    local1: vec4<f32>,     // texture, clarity
    grade_s: vec4<f32>,    // colour grading: shadows triple
    grade_m: vec4<f32>,    // mid-tones
    grade_h: vec4<f32>,    // highlights
    grade_info: vec4<f32>, // blending, balance exponent, active
    scale: vec2<f32>,      // NDC half-extent of the drawn quad
    offset: vec2<f32>,     // NDC translation, for panning when zoomed
    texel: vec2<f32>,      // 1 / source texture size
    curve: vec2<f32>,      // .x = tone-curve strength (0 = linear)
    exposure_ev: f32,
    contrast: f32,
    temp: f32,
    tint: f32,
    saturation: f32,
    highlights: f32,
    shadows: f32,
    whites: f32,
    blacks: f32,
    target_is_srgb: f32,   // 1.0 => hardware sRGB target, skip manual encode
    sharpen_amount: f32,
    sharpen_radius: f32,
    sharpen_threshold: f32,
    nr_luma: f32,
    nr_chroma: f32,
    show_clip: f32,        // 1.0 => clipping overlay; also pads to 16 bytes
};

@group(0) @binding(0) var<uniform> u: Uniforms;
// .rgb = demosaiced camera RGB (not white balanced), .a = clip mask.
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
// Lens flat field: camera-RGB gains on a coarse grid (1x1 white when none).
@group(0) @binding(3) var flat_tex: texture_2d<f32>;
// Texture / clarity bases: .r = fine, .g = coarse smoothed luminance.
@group(0) @binding(4) var local_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

const CONTRAST_PIVOT: f32 = 0.18;

// Tone curve — mirrors `pipeline::filmic` (Hill curve fitted to camera JPEGs).
const TONE_N: f32 = 1.6;
const TONE_K: f32 = 0.25;
const TONE_WHITE: f32 = 1.0;

fn filmic(x: vec3<f32>) -> vec3<f32> {
    let xn = pow(x, vec3(TONE_N));
    return (1.0 + pow(TONE_K / TONE_WHITE, TONE_N)) * xn / (xn + pow(TONE_K, TONE_N));
}

// Tonal-range mask shape — must match the constants in src/pipeline.rs.
const MASK_GAMMA: f32 = 2.2;
const HIGH_LO: f32 = 0.40;
const HIGH_HI: f32 = 0.95;
const SHAD_LO: f32 = 0.05;
const SHAD_HI: f32 = 0.60;
const WHITE_LO: f32 = 0.65;
const WHITE_HI: f32 = 1.00;
const BLACK_LO: f32 = 0.00;
const BLACK_HI: f32 = 0.30;
const WHITES_EV: f32 = 0.75;
const BLACKS_LIFT: f32 = 0.05;
const MIN_GAMMA: f32 = 0.01;

const LUMA_709: vec3<f32> = vec3(0.2126, 0.7152, 0.0722);
const CLIP_VIEW_DIM: f32 = 0.35;

// Detail-stage constants — must match src/pipeline.rs.
const DETAIL_R: i32 = 4;
const NR_SIGMA: f32 = 2.0;
const NR_RANGE: f32 = 0.08;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Two-triangle quad, corners in [-1,1].
    var corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    let c = corners[vi];
    var out: VsOut;
    out.pos = vec4(c * u.scale + u.offset, 0.0, 1.0);
    // NDC y is up, texture v is down.
    out.uv = vec2(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    return out;
}

fn srgb_decode(v: vec3<f32>) -> vec3<f32> {
    let lo = v / 12.92;
    let hi = pow((v + 0.055) / 1.055, vec3(2.4));
    return select(hi, lo, v <= vec3(0.04045));
}

// --- Point curves — mirrors src/curve.rs ----------------------------------

// Curve `c` (0 rgb, 1 r, 2 g, 3 b) with `n` points at `x`: a cubic Hermite
// through the points with the tangents worked out on the CPU, flat outside
// the end points. Mirrors `Spline::eval`.
fn curve_eval(c: u32, n: u32, x: f32) -> f32 {
    let base = c * 8u;
    let first = u.curves[base];
    if x <= first.x {
        return first.y;
    }
    let last = u.curves[base + n - 1u];
    if x >= last.x {
        return last.y;
    }
    var i = 0u;
    for (var j = 1u; j < n - 1u; j++) {
        if x >= u.curves[base + j].x {
            i = j;
        }
    }
    let p0 = u.curves[base + i];
    let p1 = u.curves[base + i + 1u];
    let h = p1.x - p0.x;
    let t = (x - p0.x) / h;
    let t2 = t * t;
    let t3 = t2 * t;
    let y = (2.0 * t3 - 3.0 * t2 + 1.0) * p0.y + (t3 - 2.0 * t2 + t) * h * p0.z
        + (-2.0 * t3 + 3.0 * t2) * p1.y + (t3 - t2) * h * p1.z;
    return clamp(y, 0.0, 1.0);
}

// The point curves on a linear pixel. Mirrors `Curves::apply`.
fn curves(c: vec3<f32>) -> vec3<f32> {
    var v = srgb_encode(clamp(c, vec3(0.0), vec3(1.0)));
    let n = vec4<u32>(u.curve_info);
    if n.x > 0u {
        v = vec3(curve_eval(0u, n.x, v.r), curve_eval(0u, n.x, v.g), curve_eval(0u, n.x, v.b));
    }
    if n.y > 0u {
        v.r = curve_eval(1u, n.y, v.r);
    }
    if n.z > 0u {
        v.g = curve_eval(2u, n.z, v.g);
    }
    if n.w > 0u {
        v.b = curve_eval(3u, n.w, v.b);
    }
    return srgb_decode(v) * max(c, vec3(1.0));
}

fn srgb_encode(v: vec3<f32>) -> vec3<f32> {
    let lo = v * 12.92;
    let hi = 1.055 * pow(v, vec3(1.0 / 2.4)) - 0.055;
    return select(hi, lo, v <= vec3(0.0031308));
}

// Flat field and radial hot-spot gain at a texture position. Mirrored in
// src/flatfield.rs::Optics::gain. `uv` is clamped to the outermost texel
// centres first: the detail stage samples past the edge, where the texture
// clamps, and the CPU mirror clamps the pixel — so must the position.
fn optics(uv_in: vec2<f32>) -> vec3<f32> {
    let uv = clamp(uv_in, 0.5 * u.texel, vec2(1.0) - 0.5 * u.texel);
    var k = vec3(1.0);
    if u.optics.w > 0.5 {
        k = textureSampleLevel(flat_tex, samp, uv, 0.0).rgb;
    }
    if any(u.hotspot.rgb != vec3(0.0)) {
        let aspect = u.optics.z;
        let hd = 0.5 * sqrt(aspect * aspect + 1.0);
        let d = (uv - u.optics.xy) * vec2(aspect, 1.0) / hd;
        let g = exp(-dot(d, d) / (2.0 * u.hotspot.a * u.hotspot.a));
        k = k / (vec3(1.0) + u.hotspot.rgb * g);
    }
    return k;
}

// Camera RGB -> working linear RGB: optics, white balance, clipped pixels to
// neutral, matrix. Mirrored in src/pipeline.rs::to_working (plus
// DevelopedImage::camera_rgb for the optics). The clip mask is filtered like
// everything else, so it blends rather than switches.
fn to_working(s: vec4<f32>, uv: vec2<f32>) -> vec3<f32> {
    var c = s.rgb * optics(uv) * u.wb.rgb;
    let m = max(c.r, max(c.g, c.b));
    c = mix(c, vec3(m), s.a);
    return vec3(dot(u.m0.rgb, c), dot(u.m1.rgb, c), dot(u.m2.rgb, c));
}

// Noise reduction and sharpening over one 9x9 neighbourhood. Mirrored in
// src/detail.rs::pixel — keep the two in lockstep.
//
// textureSampleLevel (not textureSample) because this runs under control flow
// that depends on uniforms; an explicit LOD sidesteps the uniformity rules.
fn detail(uv: vec2<f32>, center: vec3<f32>, sharpen_amount: f32, nr_luma: f32) -> vec3<f32> {
    let cl = dot(center, LUMA_709);
    let s_nr = 2.0 * NR_SIGMA * NR_SIGMA;
    let s_sh = 2.0 * pow(max(u.sharpen_radius, 0.3), 2.0);
    let inv_range = 1.0 / NR_RANGE;

    var bil_l = 0.0;
    var bil_w = 0.0;
    var chroma_sum = vec3(0.0);
    var chroma_w = 0.0;
    var sharp_l = 0.0;
    var sharp_w = 0.0;

    for (var dy = -DETAIL_R; dy <= DETAIL_R; dy++) {
        for (var dx = -DETAIL_R; dx <= DETAIL_R; dx++) {
            let suv = uv + vec2(f32(dx), f32(dy)) * u.texel;
            let s = to_working(textureSampleLevel(tex, samp, suv, 0.0), suv);
            let sl = dot(s, LUMA_709);
            let d2 = f32(dx * dx + dy * dy);

            let w_spatial = exp(-d2 / s_nr);
            let dl = (sl - cl) * inv_range;
            let w = w_spatial * exp(-0.5 * dl * dl);
            bil_l += sl * w;
            bil_w += w;

            chroma_sum += (s - sl) * w_spatial;
            chroma_w += w_spatial;

            let w_sharp = exp(-d2 / s_sh);
            sharp_l += sl * w_sharp;
            sharp_w += w_sharp;
        }
    }

    let nr_l = cl + (bil_l / max(bil_w, 1e-6) - cl) * nr_luma;
    let chroma = center - cl;
    let out_chroma = chroma + (chroma_sum / max(chroma_w, 1e-6) - chroma) * u.nr_chroma;

    let blur_l = sharp_l / max(sharp_w, 1e-6);
    var d = cl - blur_l;
    d = sign(d) * max(abs(d) - u.sharpen_threshold, 0.0);

    return max(out_chroma + nr_l + sharpen_amount * d, vec3(0.0));
}

// --- Masks — mirrors src/masks.rs -----------------------------------------

// The slider values in force at one pixel — `pipeline::Locals`.
struct Locals {
    exposure_ev: f32,
    contrast: f32,
    temp: f32,
    tint: f32,
    saturation: f32,
    highlights: f32,
    shadows: f32,
    whites: f32,
    blacks: f32,
    sharpen_amount: f32,
    nr_luma: f32,
};

const MASK_OVERLAY_OPACITY: f32 = 0.55;
const MASK_OVERLAY_COLOUR: vec3<f32> = vec3(1.0, 0.08, 0.08);

// The shape of mask `i` at a source uv: no range, not inverted, not scaled
// by its amount. Mirrors `masks::shape`. Distances are in "aspect space" (u
// times width / height), so circles are round.
fn mask_shape(i: u32, uv: vec2<f32>) -> f32 {
    let m = u.masks[i];
    let asp = vec2(u.optics.z, 1.0);
    let p = uv * asp;
    if m.v1.w == 1.0 {
        let a = m.v0.xy * asp;
        let ab = m.v0.zw * asp - a;
        let t = dot(p - a, ab) / max(dot(ab, ab), 1e-12);
        return 1.0 - smoothstep(0.0, 1.0, t);
    } else if m.v1.w == 2.0 {
        let d = p - m.v0.xy * asp;
        let r = vec2(m.v1.x * d.x + m.v1.y * d.y, -m.v1.y * d.x + m.v1.x * d.y) / m.v0.zw;
        return 1.0 - smoothstep(1.0 - m.v1.z, 1.0, length(r));
    } else if m.v1.w == 3.0 {
        return 1.0;
    }
    return 0.0;
}

// HSV hue (0..1) and saturation; (0, 0) for a grey. Mirrors `masks::hue_sat`.
fn hue_sat(c: vec3<f32>) -> vec2<f32> {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let d = mx - mn;
    if mx <= 1e-6 || d <= 1e-6 {
        return vec2(0.0, 0.0);
    }
    var h6: f32;
    if mx == c.r {
        h6 = (c.g - c.b) / d;
    } else if mx == c.g {
        h6 = (c.b - c.r) / d + 2.0;
    } else {
        h6 = (c.r - c.g) / d + 4.0;
    }
    return vec2(fract(h6 / 6.0), d / mx);
}

// What the ranges are judged on: (display brightness, hue, saturation) of a
// working pixel after the *global* exposure, white balance and tone curve.
// Mirrors `masks::RangeRef::of`.
fn range_ref(c: vec3<f32>) -> vec3<f32> {
    let g = c * exp2(u.exposure_ev) * vec3(1.0 + 0.4 * u.temp, 1.0 - 0.3 * u.tint, 1.0 - 0.4 * u.temp);
    var y = clamp(dot(g, LUMA_709), 0.0, 1.0);
    y = y + u.curve.x * (filmic(vec3(y)).x - y);
    return vec3(pow(y, 1.0 / MASK_GAMMA), hue_sat(g));
}

// What mask `i` selects at a pixel: shape x range, inverted if the mask is,
// not scaled by its amount. Mirrors `masks::selection`.
fn mask_select(i: u32, uv: vec2<f32>, px: vec3<f32>) -> f32 {
    let m = u.masks[i];
    if m.v1.w == 0.0 {
        return 0.0;
    }
    var w = mask_shape(i, uv);
    if m.v6.w > 0.5 {
        w *= smoothstep(m.v6.x - m.v6.z, m.v6.x, px.x) * (1.0 - smoothstep(m.v6.y, m.v6.y + m.v6.z, px.x));
    }
    if m.v7.w > 0.5 {
        let dist = abs(fract(px.y - m.v7.x + 0.5) - 0.5);
        w *= (1.0 - smoothstep(m.v7.y, m.v7.y + m.v7.z, dist)) * clamp(px.z / HSL_FULL_SAT, 0.0, 1.0);
    }
    if m.v2.y > 0.5 {
        w = 1.0 - w;
    }
    return w;
}

// Global sliders plus every mask's offsets at its weight. `c` is the working
// pixel before the detail stage. Mirrors `masks::locals`.
fn locals(uv: vec2<f32>, c: vec3<f32>) -> Locals {
    var l: Locals;
    l.exposure_ev = u.exposure_ev;
    l.contrast = u.contrast;
    l.temp = u.temp;
    l.tint = u.tint;
    l.saturation = u.saturation;
    l.highlights = u.highlights;
    l.shadows = u.shadows;
    l.whites = u.whites;
    l.blacks = u.blacks;
    l.sharpen_amount = u.sharpen_amount;
    l.nr_luma = u.nr_luma;
    let n = u32(u.mask_info.x);
    if n == 0u {
        return l;
    }
    let px = range_ref(c);
    for (var i = 0u; i < n; i++) {
        let m = u.masks[i];
        let w = mask_select(i, uv, px) * m.v2.x;
        l.exposure_ev += w * m.v3.x;
        l.contrast += w * m.v3.y;
        l.temp += w * m.v3.z;
        l.tint += w * m.v3.w;
        l.saturation += w * m.v4.x;
        l.highlights += w * m.v4.y;
        l.shadows += w * m.v4.z;
        l.whites += w * m.v4.w;
        l.blacks += w * m.v5.x;
        l.sharpen_amount += w * m.v2.z;
        l.nr_luma += w * m.v2.w;
    }
    l.contrast = max(l.contrast, -1.0);
    l.saturation = max(l.saturation, -1.0);
    l.sharpen_amount = max(l.sharpen_amount, 0.0);
    l.nr_luma = clamp(l.nr_luma, 0.0, 1.0);
    return l;
}

// --- Colour grading — mirrors src/pipeline.rs::grade -----------------------

const GRADE_TINT: f32 = 1.0;
const GRADE_LUM_EV: f32 = 2.0;

// Shadows / mid-tones / highlights weights at display brightness `d`.
// Mirrors `pipeline::grade_weights`.
fn grade_weights(d_in: f32) -> vec3<f32> {
    let d = pow(clamp(d_in, 0.0, 1.0), u.grade_info.y);
    let b = u.grade_info.x;
    let shadows = 1.0 - smoothstep(0.0, 0.25 + 0.5 * b, d);
    let highlights = smoothstep(0.75 - 0.5 * b, 1.0, d);
    return vec3(shadows, max(1.0 - shadows - highlights, 0.0), highlights);
}

fn grade(c: vec3<f32>) -> vec3<f32> {
    let y = dot(c, LUMA_709);
    let w = grade_weights(pow(clamp(y, 0.0, 1.0), 1.0 / MASK_GAMMA));
    let t = u.grade_s.rgb * w.x + u.grade_m.rgb * w.y + u.grade_h.rgb * w.z;
    let master = (t.r + t.g + t.b) / 3.0;
    var out = c * max(vec3(1.0) + GRADE_TINT * (t - vec3(master)), vec3(0.0));
    let tinted = dot(out, LUMA_709);
    var scale = 1.0;
    if tinted > 1e-6 && y > 0.0 {
        scale = y / tinted;
    }
    return out * (scale * exp2(master * GRADE_LUM_EV));
}

// --- Texture and clarity — mirrors src/local.rs ---------------------------

const TEXTURE_GAIN: f32 = 1.0;
const CLARITY_GAIN: f32 = 0.8;
const LOCAL_MAX_RATIO: f32 = 4.0;

// What to multiply the working pixel by: how far the pixel's luminance is
// from the two smoothed bases, raised to the sliders. `cam` is the camera
// RGB sample as stored. Mirrors `local::gain`.
fn local_gain(cam: vec3<f32>, uv: vec2<f32>) -> f32 {
    let base = textureSampleLevel(local_tex, samp, uv, 0.0).rg;
    let y = max(dot(cam, u.local0.rgb), 1e-6);
    let ratio = clamp(vec2(y) / max(base, vec2(1e-6)), vec2(1.0 / LOCAL_MAX_RATIO), vec2(LOCAL_MAX_RATIO));
    return pow(ratio.x, u.local1.x * TEXTURE_GAIN) * pow(ratio.y, u.local1.y * CLARITY_GAIN);
}

// --- Vignette and grain — mirrors src/effects.rs --------------------------

const GRAIN_MAX: f32 = 0.14;

// Exposure change at an output uv. Mirrors `Vignette::ev_at`.
fn vignette_ev(uv: vec2<f32>) -> f32 {
    let q = abs((uv - vec2(0.5)) * 2.0 * u.vig1.xy);
    let dist = pow(pow(q.x, u.vig0.w) + pow(q.y, u.vig0.w), 1.0 / u.vig0.w) * u.vig1.z;
    return u.vig0.x * smoothstep(u.vig0.y, u.vig0.z, dist);
}

// Mirrors `effects::hash`: integer arithmetic, so the same bits as the CPU.
fn grain_hash(x: i32, y: i32) -> f32 {
    var h = (u32(x) * 0x9E3779B1u) ^ (u32(y) * 0x85EBCA77u);
    h ^= h >> 15u;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12u;
    h *= 0x297A2D39u;
    h ^= h >> 15u;
    return f32(h >> 8u) / 8388608.0 - 1.0;
}

// Mirrors `effects::grain_noise`. `px` is in source pixels.
fn grain_noise(px: vec2<f32>) -> f32 {
    let p = (px - vec2(0.5)) / u.effects.y;
    let i = floor(p);
    let f = p - i;
    let x = i32(i.x);
    let y = i32(i.y);
    let top = mix(grain_hash(x, y), grain_hash(x + 1, y), f.x);
    let bottom = mix(grain_hash(x, y + 1), grain_hash(x + 1, y + 1), f.x);
    return mix(top, bottom, f.y);
}

// Mirrors `effects::grain`. `strength` is the amount, less whatever the
// view's scale averages away.
fn grain(c: vec3<f32>, n: f32, strength: f32) -> vec3<f32> {
    let e = srgb_encode(clamp(c, vec3(0.0), vec3(1.0)));
    let y = dot(e, LUMA_709);
    let d = n * strength * GRAIN_MAX * (0.4 + 2.4 * y * (1.0 - y));
    return srgb_decode(clamp(e + vec3(d), vec3(0.0), vec3(1.0))) + max(c - vec3(1.0), vec3(0.0));
}

// --- Per-colour adjustments — mirrors src/pipeline.rs::hsl_adjust ---------

// Band centres in degrees; must match pipeline::HSL_BANDS.
const HSL_CENTRES = array<f32, 8>(0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0);
const HSL_HUE_MAX: f32 = 0.083333336; // 30 / 360
const HSL_LUM_EV: f32 = 1.0;
const HSL_FULL_SAT: f32 = 0.25;

fn hsl_at(h: f32) -> vec3<f32> {
    let deg = h * 360.0;
    var i = 7u;
    for (var k = 0u; k < 8u; k++) {
        if deg >= HSL_CENTRES[k] {
            i = k;
        }
    }
    let lo = HSL_CENTRES[i];
    let hi = select(HSL_CENTRES[min(i + 1u, 7u)], 360.0, i == 7u);
    let t = (deg - lo) / (hi - lo);
    return mix(u.hsl[i].xyz, u.hsl[(i + 1u) % 8u].xyz, t);
}

fn hsl_adjust(c: vec3<f32>) -> vec3<f32> {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let d = mx - mn;
    if mx <= 1e-6 || d <= 1e-6 {
        return c;
    }
    var h6: f32;
    if mx == c.r {
        h6 = (c.g - c.b) / d;
    } else if mx == c.g {
        h6 = (c.b - c.r) / d + 2.0;
    } else {
        h6 = (c.r - c.g) / d + 4.0;
    }
    var h = fract(h6 / 6.0);
    var s = d / mx;
    let w = clamp(s / HSL_FULL_SAT, 0.0, 1.0);
    let adj = hsl_at(h) * w;
    h = fract(h + adj.x * HSL_HUE_MAX);
    s = s * (1.0 + adj.y);
    let k = (vec3(5.0, 3.0, 1.0) + h * 6.0) % vec3(6.0);
    let rgb = mx - mx * s * clamp(min(k, vec3(4.0) - k), vec3(0.0), vec3(1.0));
    return rgb * exp2(adj.z * HSL_LUM_EV);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Crop / rotate / straighten: where in the source this output pixel is.
    // Everything after works in source uv, so optics and detail are unchanged.
    let src = vec2(dot(u.geom0.xyz, vec3(in.uv, 1.0)), dot(u.geom1.xyz, vec3(in.uv, 1.0)));
    // Sampled before the outside test: textureSample needs uniform control flow.
    let stored = textureSample(tex, samp, src);
    var c = to_working(stored, src);
    let outside = any(src < vec2(0.0)) || any(src > vec2(1.0));
    // Source pixels per screen pixel (1 at 100% and in exports). Derivatives
    // need uniform control flow too, so this is taken here, unconditionally.
    let footprint = max(length(dpdx(src) / u.texel), length(dpdy(src) / u.texel));

    // 0. Detail work happens on the linear source, before tone and colour.
    //    Skipped entirely when the controls are neutral, which is the norm.
    //    Masks first: they decide the slider values this pixel gets.
    let before_detail = c;
    var l = locals(src, c);
    if u.vig0.x != 0.0 {
        l.exposure_ev += vignette_ev(in.uv);
    }
    if l.nr_luma > 0.0 || u.nr_chroma > 0.0 || l.sharpen_amount > 0.0 {
        c = detail(src, c, l.sharpen_amount, l.nr_luma);
    }

    // 0b. Texture and clarity: a local exposure change, before the tone chain.
    if u.local0.w > 0.5 && (u.local1.x != 0.0 || u.local1.y != 0.0) {
        c *= local_gain(stored.rgb, src);
    }

    // 1. Exposure + white balance as combined per-channel gains.
    let gain = exp2(l.exposure_ev);
    c *= gain * vec3(1.0 + 0.4 * l.temp, 1.0 - 0.3 * l.tint, 1.0 - 0.4 * l.temp);

    // 2. Pivot-based contrast in linear space.
    c = CONTRAST_PIVOT * pow(max(c, vec3(0.0)) / CONTRAST_PIVOT, vec3(1.0 + l.contrast));

    // 3. Tonal ranges. Masks live in an approximate display domain so they
    //    line up with what the eye calls "highlights"/"shadows", and they are
    //    driven by luminance only, so hue is preserved.
    let d = pow(clamp(dot(c, LUMA_709), 0.0, 1.0), 1.0 / MASK_GAMMA);
    let m_high = smoothstep(HIGH_LO, HIGH_HI, d);
    let m_shad = 1.0 - smoothstep(SHAD_LO, SHAD_HI, d);
    let m_white = smoothstep(WHITE_LO, WHITE_HI, d);
    let m_black = 1.0 - smoothstep(BLACK_LO, BLACK_HI, d);
    let tone_gain = exp2(l.highlights * m_high + l.shadows * m_shad + l.whites * WHITES_EV * m_white);
    c = c * tone_gain + l.blacks * BLACKS_LIFT * m_black;

    // 3b. Tone curve (filmic shoulder), blended in by strength.
    if u.curve.x != 0.0 {
        let xc = max(c, vec3(0.0));
        c = xc + u.curve.x * (filmic(xc) - xc);
    }

    // 3c. Point curves.
    if any(u.curve_info > vec4(0.0)) {
        c = curves(c);
    }

    // 3d. Colour grading by tonal range.
    if u.grade_info.z > 0.5 {
        c = grade(c);
    }

    // 4. Lift / gamma / gain:  out = gain * (in + lift*(1-in))^(1/gamma)
    let x = max(c, vec3(0.0));
    let lifted = max(x + u.lift.rgb * (vec3(1.0) - x), vec3(0.0));
    c = u.gain.rgb * pow(lifted, vec3(1.0) / max(u.gamma.rgb, vec3(MIN_GAMMA)));

    // 5. Luma-preserving saturation.
    let luma = dot(c, LUMA_709);
    var sat = 1.0 + l.saturation;
    if u.effects.z != 0.0 {
        let mx = max(c.r, max(c.g, c.b));
        let mn = min(c.r, min(c.g, c.b));
        let have = clamp((mx - mn) / max(mx, 1e-6), 0.0, 1.0);
        sat = max(sat + u.effects.z * (1.0 - have), 0.0);
    }
    c = luma + (c - luma) * sat;

    // 6. Per-colour hue / saturation / luminance.
    if u.hsl[0].w > 0.5 {
        c = hsl_adjust(c);
    }

    // 7. Grain, on the source pixel grid. Zoomed out, each screen pixel is
    //    the average of many grains, so the strength drops with the view's
    //    scale — as it would if the export were shown at that size.
    if u.effects.x > 0.0 {
        let strength = u.effects.x * min(1.0, u.effects.y / max(footprint, 1e-6));
        c = grain(c, grain_noise(src / u.texel), strength);
    }

    // Clipping overlay — mirrored in src/pipeline.rs::clip_view.
    if u.show_clip > 0.5 {
        let over = select(vec3(0.0), vec3(1.0), c >= vec3(1.0));
        if any(over > vec3(0.0)) {
            c = over;
        } else {
            c = vec3(dot(clamp(c, vec3(0.0), vec3(1.0)), LUMA_709) * CLIP_VIEW_DIM);
        }
    }

    c = clamp(c, vec3(0.0), vec3(1.0));
    // Mask overlay — mirrored in src/masks.rs::overlay.
    if u.mask_info.y > 0.5 {
        let sel = mask_select(u32(u.mask_info.y) - 1u, src, range_ref(before_detail));
        c = mix(c, MASK_OVERLAY_COLOUR, MASK_OVERLAY_OPACITY * sel);
    }
    // Straightened corners with no source behind them.
    if outside {
        c = vec3(0.0);
    }
    if u.target_is_srgb == 0.0 {
        c = srgb_encode(c);
    }
    return vec4(c, 1.0);
}
