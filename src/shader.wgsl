// Adjustment pipeline. Mirrored in src/pipeline.rs (`apply`) — keep in sync,
// covered by the parity test in tests/shader_parity.rs.

struct Uniforms {
    // vec4s first: WGSL uniform buffers need 16-byte alignment for them.
    lift: vec4<f32>,       // .rgb = per-channel lift  (neutral 0)
    gamma: vec4<f32>,      // .rgb = per-channel gamma (neutral 1)
    gain: vec4<f32>,       // .rgb = per-channel gain  (neutral 1)
    scale: vec2<f32>,      // NDC half-extent of the aspect-fitted quad
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
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

const CONTRAST_PIVOT: f32 = 0.18;

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

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Two-triangle quad, corners in [-1,1].
    var corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    let c = corners[vi];
    var out: VsOut;
    out.pos = vec4(c * u.scale, 0.0, 1.0);
    // NDC y is up, texture v is down.
    out.uv = vec2(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    return out;
}

fn srgb_encode(v: vec3<f32>) -> vec3<f32> {
    let lo = v * 12.92;
    let hi = 1.055 * pow(v, vec3(1.0 / 2.4)) - 0.055;
    return select(hi, lo, v <= vec3(0.0031308));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var c = textureSample(tex, samp, in.uv).rgb;

    // 1. Exposure + white balance as combined per-channel gains.
    let gain = exp2(u.exposure_ev);
    c *= gain * vec3(1.0 + 0.4 * u.temp, 1.0 - 0.3 * u.tint, 1.0 - 0.4 * u.temp);

    // 2. Pivot-based contrast in linear space.
    c = CONTRAST_PIVOT * pow(max(c, vec3(0.0)) / CONTRAST_PIVOT, vec3(1.0 + u.contrast));

    // 3. Tonal ranges. Masks live in an approximate display domain so they
    //    line up with what the eye calls "highlights"/"shadows", and they are
    //    driven by luminance only, so hue is preserved.
    let d = pow(clamp(dot(c, LUMA_709), 0.0, 1.0), 1.0 / MASK_GAMMA);
    let m_high = smoothstep(HIGH_LO, HIGH_HI, d);
    let m_shad = 1.0 - smoothstep(SHAD_LO, SHAD_HI, d);
    let m_white = smoothstep(WHITE_LO, WHITE_HI, d);
    let m_black = 1.0 - smoothstep(BLACK_LO, BLACK_HI, d);
    let tone_gain = exp2(u.highlights * m_high + u.shadows * m_shad + u.whites * WHITES_EV * m_white);
    c = c * tone_gain + u.blacks * BLACKS_LIFT * m_black;

    // 4. Lift / gamma / gain:  out = gain * (in + lift*(1-in))^(1/gamma)
    let x = max(c, vec3(0.0));
    let lifted = max(x + u.lift.rgb * (vec3(1.0) - x), vec3(0.0));
    c = u.gain.rgb * pow(lifted, vec3(1.0) / max(u.gamma.rgb, vec3(MIN_GAMMA)));

    // 5. Luma-preserving saturation.
    let luma = dot(c, LUMA_709);
    c = luma + (c - luma) * (1.0 + u.saturation);

    c = clamp(c, vec3(0.0), vec3(1.0));
    if u.target_is_srgb == 0.0 {
        c = srgb_encode(c);
    }
    return vec4(c, 1.0);
}
