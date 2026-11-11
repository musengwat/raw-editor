// Adjustment pipeline. Mirrored in src/pipeline.rs (`apply`) — keep in sync,
// covered by the parity test in tests/shader_parity.rs.

// One mask slot — packed by `masks::uniforms` in src/masks.rs.
struct MaskU {
    v0: vec4<f32>,  // a.xy, b.xy (linear: the two ends; radial: centre, semi-axes)
    v1: vec4<f32>,  // cos, sin, feather, kind (0 off, 1 linear, 2 radial)
    v2: vec4<f32>,  // amount, invert, sharpen, nr_luma
    v3: vec4<f32>,  // exposure, contrast, temp, tint
    v4: vec4<f32>,  // saturation, highlights, shadows, whites
    v5: vec4<f32>,  // blacks
    v6: vec4<f32>,  // reserved: luminance range
    v7: vec4<f32>,  // reserved: colour range
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

// The shape of mask `i` at a source uv, inverted if the mask is, not scaled
// by its amount. Mirrors `masks::shape`. Distances are in "aspect space" (u
// times width / height), so circles are round.
fn mask_shape(i: u32, uv: vec2<f32>) -> f32 {
    let m = u.masks[i];
    let asp = vec2(u.optics.z, 1.0);
    let p = uv * asp;
    var w = 0.0;
    if m.v1.w == 1.0 {
        let a = m.v0.xy * asp;
        let ab = m.v0.zw * asp - a;
        let t = dot(p - a, ab) / max(dot(ab, ab), 1e-12);
        w = 1.0 - smoothstep(0.0, 1.0, t);
    } else if m.v1.w == 2.0 {
        let d = p - m.v0.xy * asp;
        let r = vec2(m.v1.x * d.x + m.v1.y * d.y, -m.v1.y * d.x + m.v1.x * d.y) / m.v0.zw;
        w = 1.0 - smoothstep(1.0 - m.v1.z, 1.0, length(r));
    } else {
        return 0.0;
    }
    if m.v2.y > 0.5 {
        w = 1.0 - w;
    }
    return w;
}

// Global sliders plus every mask's offsets at its weight. Mirrors
// `masks::locals`.
fn locals(uv: vec2<f32>) -> Locals {
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
    for (var i = 0u; i < n; i++) {
        let m = u.masks[i];
        let w = mask_shape(i, uv) * m.v2.x;
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
    var c = to_working(textureSample(tex, samp, src), src);
    let outside = any(src < vec2(0.0)) || any(src > vec2(1.0));

    // 0. Detail work happens on the linear source, before tone and colour.
    //    Skipped entirely when the controls are neutral, which is the norm.
    //    Masks first: they decide the slider values this pixel gets.
    let l = locals(src);
    if l.nr_luma > 0.0 || u.nr_chroma > 0.0 || l.sharpen_amount > 0.0 {
        c = detail(src, c, l.sharpen_amount, l.nr_luma);
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

    // 4. Lift / gamma / gain:  out = gain * (in + lift*(1-in))^(1/gamma)
    let x = max(c, vec3(0.0));
    let lifted = max(x + u.lift.rgb * (vec3(1.0) - x), vec3(0.0));
    c = u.gain.rgb * pow(lifted, vec3(1.0) / max(u.gamma.rgb, vec3(MIN_GAMMA)));

    // 5. Luma-preserving saturation.
    let luma = dot(c, LUMA_709);
    c = luma + (c - luma) * (1.0 + l.saturation);

    // 6. Per-colour hue / saturation / luminance.
    if u.hsl[0].w > 0.5 {
        c = hsl_adjust(c);
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
        let shape = mask_shape(u32(u.mask_info.y) - 1u, src);
        c = mix(c, MASK_OVERLAY_COLOUR, MASK_OVERLAY_OPACITY * shape);
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
