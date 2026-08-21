// Adjustment pipeline. Mirrored in src/pipeline.rs (`apply`) — keep in sync,
// covered by the parity test in tests/shader_parity.rs.

struct Uniforms {
    scale: vec2<f32>,      // NDC half-extent of the aspect-fitted quad
    exposure_ev: f32,
    contrast: f32,
    temp: f32,
    tint: f32,
    saturation: f32,
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

    // Exposure + white balance as combined per-channel gains.
    let gain = exp2(u.exposure_ev);
    c *= gain * vec3(1.0 + 0.4 * u.temp, 1.0 - 0.3 * u.tint, 1.0 - 0.4 * u.temp);

    // Pivot-based contrast in linear space.
    c = CONTRAST_PIVOT * pow(max(c, vec3(0.0)) / CONTRAST_PIVOT, vec3(1.0 + u.contrast));

    // Luma-preserving saturation.
    let luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = luma + (c - luma) * (1.0 + u.saturation);

    c = clamp(c, vec3(0.0), vec3(1.0));
    if u.target_is_srgb == 0.0 {
        c = srgb_encode(c);
    }
    return vec4(c, 1.0);
}
