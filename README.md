# raw-editor

A basic GPU-accelerated RAW image editor for Panasonic `.rw2` files, with
DaVinci-Resolve-style color adjustments. Rust + wgpu + egui.

## Architecture

- **Decode** (once per file, CPU): [`rawler`](https://lib.rs/crates/rawler)
  extracts CFA sensor data + metadata (black/white levels, as-shot white
  balance, camera color matrices).
- **Develop** (once per file, CPU, `src/develop.rs`): black-level subtract →
  normalize → white-balance gains → demosaic → clipped-highlight handling →
  camera→linear-sRGB matrix. Output: scene-referred linear RGB f32, unclamped,
  uploaded as an `rgba16float` wgpu texture.
- **Adjust** (every slider change, GPU): one WGSL fragment shader applies
  exposure → temp/tint → contrast → highlights/shadows → whites/blacks →
  lift/gamma/gain → saturation → sRGB transfer. Real-time at full resolution.
- **Export**: full-res offscreen render → readback → JPEG / 16-bit TIFF.
- **Persistence**: adjustment params serialized to a `<name>.rw2.edit.json`
  sidecar; the RW2 is never modified.

All shader math is mirrored in pure Rust so `cargo test` can verify it and
export can fall back to CPU.

### Clipped highlights

A saturated photosite only records "at least this much", so its channel ratios
are meaningless — and white-balance gains applied to them turn every blown area
magenta. The develop stage therefore marks clipped photosites, and any pixel
that touches one is forced neutral at the brightest channel's level (that
channel is the least likely to have clipped, so it is the best luminance
estimate left). This is the plain "clip to neutral" behaviour; true highlight
*reconstruction* from neighbouring detail is out of scope.

Finding the clipping point is the subtle part, because it is not always the
reported white level. On Panasonic's extended low ISO (100 on a G85, below its
base of 200) the sensor is overexposed and the result scaled back down, so the
analog ceiling lands around raw 2880 while the file still reports 4095 —
highlights clip without ever approaching the reported level. `saturation_levels`
detects this from the histogram: a real ceiling is a narrow, tall plateau with
almost nothing above it, and because saturation is a property of the sensor it
must show up in at least two colours at nearly the same level. A single colour
peaking alone is just a large flat bright area in the scene, and treating that
as clipping would drain the colour out of it. Set `RAW_EDITOR_DEBUG=1` to see
what it decides per colour.

## Status

- [x] Phase 0 — toolchain + decode proof (`cargo run --bin probe -- file.rw2`)
- [x] Phase 1 — develop stage + display the photo
- [x] Phase 2 — real-time core sliders (exposure, contrast, temp/tint, saturation)
- [x] Phase 3 — highlights/shadows/whites/blacks + lift/gamma/gain wheels
- [x] Phase 4 — JSON sidecar persistence + JPEG/TIFF export
- [ ] Phase 5 — sharpening + noise reduction (compute passes)
- [ ] Phase 6 — Malvar-He-Cutler demosaic (replaces the current bilinear)

Algorithms and coefficients for phases 5 and 6, including the transposed-filter
trap in the MHC paper's figure, are worked out in
[`docs/phase5-6-algorithms.md`](docs/phase5-6-algorithms.md).

## Controls

**Basic**: exposure (EV), contrast (power curve about a 0.18 pivot), temp/tint
(opposing channel gains), saturation (luma-preserving).

**Tonal range**: highlights, shadows, whites, blacks. Each drives a smooth
`smoothstep` mask over an approximate *display-domain* luminance
(`clamp(luma,0,1)^(1/2.2)`), so the ranges land where the eye expects rather
than where scene-linear values sit. Highlights/shadows/whites become one
combined exposure multiplier; blacks is an additive lift of the floor. Sign
convention follows Lightroom's basic panel — **positive brightens that range**,
so negative highlights is the recovery/compression direction. Highlight
*reconstruction* of clipped channels is explicitly out of scope.

**Lift / gamma / gain**: three Resolve-style colour wheels, per channel
`out = gain * (in + lift*(1-in))^(1/gamma)`. Each wheel is a custom egui widget
(`src/widgets.rs`): the circular pad writes a luma-neutral chroma offset (R at
the top, G and B 120° apart, so the three offsets sum to zero) and the value
box under it moves all three channels together. Double-click a wheel to reset it.

**Compare**: hold `B` to render the same frame with neutral params — the
"before" view goes through the identical shader, so only the adjustments differ.

## Persistence and export

Edits are non-destructive: the RW2 is opened read-only and adjustments live in
`<name>.rw2.edit.json` beside it. The sidecar is loaded when a file is opened,
written 600 ms after the last change (and on file switch / app exit), and
*deleted* when the params return to neutral rather than left behind full of
zeroes. It is written via a temp file + rename, and missing fields fall back to
defaults so sidecars from older builds still load.

Export renders the whole image through the same WGSL shader in one offscreen
pass into an `Rgba16Float` target, reads it back (256-byte row padding handled)
and encodes it. JPEG takes a quality slider; TIFF is genuine 16-bit-per-channel
RGB. Output lands next to the raw as `<stem>_edit.jpg` / `<stem>_edit.tif`, so a
camera JPEG sitting alongside is never overwritten. `export::render_cpu` is the
pure-Rust equivalent and is used automatically if the GPU pass fails.

Because the developed image is uploaded as `rgba16float`, half-float is the
pipeline's precision floor; the 16-bit TIFF removes 8-bit banding but is not 16
*significant* bits end to end.

## Reference camera

Panasonic DMC-GF1: 4060×3016 sensor (4000×3000 active), GBRG Bayer, 12-bit
(whitelevel 3986; sensor values can exceed it — clamp on normalize), D65 + StdA
color matrices embedded. Test files live in `testdata/` (gitignored).

## Running

```
cargo run --release -- testdata/f.rw2      # the editor GUI (path optional; drag-and-drop works)
cargo run --release --bin probe -- f.rw2   # dump decode metadata (2nd arg = write a developed PNG)
cargo run --release --bin export -- f.rw2 jpg 92   # headless export using the file's sidecar
cargo test --release                       # develop/pipeline math + headless GPU parity/export tests
```

The GPU tests share a single wgpu device behind a `OnceLock` + `Mutex`
(`tests/common/mod.rs`), so they are safe under the default parallel test
runner — no `--test-threads=1` needed.
