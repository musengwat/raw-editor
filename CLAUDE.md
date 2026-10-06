# raw-editor — project instructions

Read `README.md` first: architecture, phase status, and reference-camera facts
(GF1 quirks: GBRG CFA, whitelevel 3986 but sensor data exceeds it, deprecated
`xyz_to_cam` is zeroed — use `RawImage::color_matrix` HashMap).

## Conventions

- Pipeline math lives twice on purpose: WGSL shader (GPU path) + pure-Rust
  mirror (tests, CPU export fallback). Any change to one must change both, and
  a parity test must cover it.
- Scene-referred linear RGB f32, unclamped, until the final sRGB transfer.
  Never bake clamps into intermediate stages except the whitelevel clamp at
  normalize time.
- Test raws are in `testdata/` (gitignored — never commit raw files):
  Panasonic RW2 at the top and in `ir/`, `fs/`, `portrait/`; Sony A7 IV ARW
  samples (raw.pixls.us) in `sony/`. Tests that need them skip without them.
  Update README phase checkboxes + this file when behavior changes.
- `Uniforms` is a hand-packed `#[repr(C)]` block matched by hand to the WGSL
  `struct Uniforms`. `vec4` members must sit on 16-byte offsets, so they go
  first; scalars pack after. A test asserts the size and the alignments — when
  you add a param, add it to the struct, the shader, `Uniforms::new`, `apply`,
  and update that test.
- Run `cargo test` before every commit; commit + push each completed phase.

## Build notes

- `image` crate is pinned to jpeg/png/tiff features only (default features
  pull in rav1e/AVIF — very slow compiles). Don't re-enable defaults.
- eframe uses the wgpu renderer (`Renderer::Wgpu`), not glow.
- egui 0.36 API: panels are the unified `egui::Panel::{left,right,top,bottom}`
  shown against the `&mut Ui` that `eframe::App::ui()` hands you (the old
  `SidePanel`/`TopBottomPanel` + `update(ctx, ..)` are gone). Show side/edge
  panels before `CentralPanel`; each one shrinks the parent `Ui`'s cursor.
- `tests/common/mod.rs` shares one wgpu device via `OnceLock` and locks a
  `Mutex` per render. Creating an instance per test thread segfaults llvmpipe.
  Keep new GPU tests on that helper rather than reintroducing `--test-threads=1`.
- Export renders into `Rgba16Float` (renderable everywhere; `Rgba16Unorm` needs
  an optional feature). Readback rows must be padded to
  `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`. `src/bin/export.rs` is the headless
  end-to-end check — run it on a real RW2 after touching the export path.
- Sidecars are gitignored (`*.edit.json`). Never write anything into the `.rw2`.
- macOS: `packaging/macos/bundle.sh` builds the universal `.app` (CI runs it).
  Keep `MACOSX_DEPLOYMENT_TARGET=10.13` — the target is a 2012+ Intel MacBook
  Air. File dialogs (rfd) must stay on the UI thread; macOS requires it. There
  is no Mac here: `cargo check --target x86_64-apple-darwin` works from Linux
  only with stub `CC_x86_64_apple_darwin`/`AR_x86_64_apple_darwin` scripts,
  because `ring` compiles C and there's no Apple clang.

## Verifying the GUI visually

There is a live X11 display on `DISPLAY=:10`. Launch detached, grab the window
with `xwd`, and decode it with Python + PIL (`xwdtopnm`/ImageMagick are not
installed). **The XWD pixel data starts at `header_size + ncolors * 12`** — the
colormap follows the header even for TrueColor visuals (this box reports
`ncolors = 256`). Skipping it shifts the whole image sideways by 768 px with
wraparound, which looks exactly like a broken panel layout. Decode:

```python
import struct; from PIL import Image
d = open('win.xwd','rb').read(); h = struct.unpack('>25I', d[:100])
off = h[0] + h[19] * 12                      # header_size + ncolors*12
mode = 'BGRX' if h[7] == 0 else 'XRGB'
Image.frombytes('RGB', (h[4], h[5]), d[off:], 'raw', mode, h[12]).save('win.png')
```

## Highlight clipping and white levels

The reported white level is not always where a file clips. Panasonic's extended
low ISO (100 on the G85) overexposes and scales down, so the analog ceiling sits
near raw 2880 while the file reports 4095. `develop::saturation_levels` detects
that from the histogram; see README for the criteria and the reasoning. Two
rules matter when touching it:

- Clipped pixels are flagged after demosaicing (never per photosite) and forced
  neutral in `pipeline::to_working` / the shader's `to_working`, **after**
  white balance. The flag rides in the texture's alpha channel.
  Green saturates first (its WB gain is 1.0 while red's is 2.3), so a
  per-photosite fix trades a magenta cast for a green one.
- Detection must be corroborated across colours. A single colour peaking alone
  is a scene feature, not a sensor ceiling.

Regression check for this area: develop every file in `testdata/` and confirm
`P1050113.rw2` (ISO 100, overcast sky) renders a neutral sky while every other
file stays byte-identical. `RAW_EDITOR_DEBUG=1` prints the per-colour decision.

## Demosaic (Phase 6)

`develop::demosaic` runs Malvar-He-Cutler over a `Mosaic` (normalized plane,
saturated-photosite map, CFA lookup, balance gains). Bilinear stays in the
code as `Demosaic::Bilinear`: it is the tests' reference and the per-pixel
fallback. Rules:

- The clip flag is always the 3x3 neighbourhood, whatever the method. Pixels
  whose only saturated neighbours are the ±2 axial samples use bilinear (MHC's
  Laplacian would read the ceiling as detail and ring).
- MHC runs on **white-balanced** values (`balance` = as-shot WB) and divides the
  gains back out. In raw units it fringes every edge teal/pink. Keep the
  `mhc_runs_in_white_balanced_units` test.
- Edges mirror (`reflect`), which keeps the Bayer phase; clamping would not.
- Output is clamped at 0 — the Laplacian overshoots below zero at hard edges.
- A transposed G-site Laplacian passes a mixed-scene PSNR test; only the
  per-axis stripe test catches it. See `docs/phase5-6-algorithms.md`.
- Regression method: export `testdata` to TIFF with the old and new binaries and
  compare (mean diff, luma edge energy, high-frequency chroma) plus a 6x zoom of
  the P1050113 branches against the sky — that crop shows fringing at once.

## Camera RGB, white balance and the matrix (IR-1)

The texture holds demosaiced **camera** RGB, not white balanced or matrixed;
`DevelopedImage.colour` (`ImageColour`) carries the as-shot WB and the
camera→sRGB matrix, and `Colour::resolve` picks what one render actually uses.
Both the shader and the CPU path run every sample — including all 81 detail
neighbours — through `to_working` first, so detail/tone code still sees linear
sRGB exactly as before. `DevelopedImage::linear_srgb` builds an identity-colour
image for synthetic tests. Reason: an infrared-converted GF1 needs white
balance far outside the camera's range and often no camera matrix at all.

`AdjustParams.wb_gains` (`None` = as shot) and `colour_mode` are
`#[serde(default)]`, so old sidecars load unchanged; an older binary reading a
new sidecar silently ignores them (serde skips unknown fields) — remember that
when a test export "ignores" a WB edit: rebuild **all** bins (`--bins`), not
just the GUI. Eyedropper = `DevelopedImage::neutral_gains`; screen→pixel =
`View::image_pixel`. Single-key shortcuts go through `shortcut()` in main.rs so
they don't fire while a text field has focus.

Histogram (`src/histogram.rs`) runs `to_working` → `apply` on a point-sampled
`downsample` (long side 512, ~4.5 ms per update on this box) and is recomputed
only when params change. The clipping overlay is `Uniforms.show_clip` +
`pipeline::clip_view` (shader twin in `fs_main`); it is a view setting, not an
`AdjustParams` field, so it is never saved or exported.

Precision note: moving the matrix after the f16 texture means strongly
out-of-gamut colours (one channel pinned, one near zero) can differ from the
old baked path by 1–2 8-bit levels in the near-zero channel — the matrix's big
coefficients cancel there and amplify f16 rounding. Measured on the testdata
set: mean diff ~3/65535, neutral/ordinary colours at rounding level. Accepted.
Regression method: export `testdata` with the previous binary and diff the
16-bit TIFFs.

## Channel mixer and looks (IR-2)

`AdjustParams.mixer` is a row-major 3x3 applied *after* the colour matrix.
`Colour::resolve` multiplies it into the matrix (`mixer · matrix`), so the
shader, histogram, eyedropper and export get it for free — there is no mixer
code in the shader, and none should be added. `parity_channel_mixer_looks`
covers it on the GPU.

`src/looks.rs` holds `Look` (mixer + optional colour mode / WB / saturation;
`None`/`LookWb::Keep` leave the photo's value alone) and the built-ins. Every
built-in row sums to 1 so white-balanced neutrals stay neutral — a test
enforces it. The built-ins are reasoned from the filter physics (behind IR
Chrome the blue photosites see ~pure IR; Aerochrome = IR→R, R→G, G→B), not
tuned on real daylight shots. Retune them when those exist.

## Tuned looks (IR-5)

The *Full spectrum* and *#16 orange* built-ins were tuned on real frames in
`testdata/fs` (P1320275–295; P1320083 is the same view on the stock camera).
The looks tests pin them to region means measured on those frames
(`ORANGE_16` / `UNFILTERED` in `looks.rs`), so retuning must keep those
tests' claims true or change them deliberately. Built-ins were renamed with a
filter prefix; `looks::RENAMED` keeps old names resolving (`find_builtin`) and
`Config::migrated` rewrites stored body-look names on load. Tuning method: dump
1/8-scale camera RGB per file, average hand-picked regions, prototype
WB + mixer in numpy, then confirm with full-res exports through
`target/release/export` using hand-written sidecars (XDG_CONFIG_HOME pointed at
a scratch dir).

## Saved looks and per-body looks (IR-3)

`Config.looks` holds saved looks; `Config.body_looks` maps body serial → look
name (saved looks shadow built-ins of the same name). `sidecar::initial` is
the one place that decides what a photo opens with — sidecar, else the body
look, else neutral — and the GUI, GUI batch export and CLI all call it. Keep
it that way, or exports will stop matching the screen.

The GUI keeps `baseline` (what the photo opens with *without* a sidecar) and
only removes a sidecar when params equal it, not when they are neutral. That
is what lets one photo opt out of its body's look: resetting it writes a
neutral sidecar instead of deleting one.

Serials come from `metadata::camera_ids`: Panasonic stores
`InternalSerialNumber` (0x25) and `LensType` (0x51) only in a maker note
inside the embedded preview JPEG's EXIF, not in standard EXIF, and rawler
does not parse it. `panasonic_ids` is a bounds-checked IFD walk over the first
4 MB (fuzzed by truncating at every length in the tests). GF1 =
`F921004200220`, G85 = `XEL2112060047`. Stock GF1 shots carry the same serial
as IR ones (same body), and the conversion reset the clock, so nothing in the
metadata tells pre- from post-conversion files.

The CLI honours body looks too. To test without touching the real settings,
point `XDG_CONFIG_HOME` at a scratch directory.

## Optics: hot spot and flat field (IR-4)

Both are a position-dependent gain on **camera RGB, before white balance**:
`optics()` at the top of the shader's `to_working(s, uv)`, mirrored by
`flatfield::Optics::gain` and applied on the CPU through
`DevelopedImage::camera_rgb` / `working(&params)` — so the histogram, the
eyedropper and CPU export all see it. `Colour` alone no longer describes a
render; that is why `working` takes `AdjustParams`.

- The flat field is a coarse grid (32 cells on the long side) of per-channel
  gains = frame mean / cell mean, f16-rounded so CPU and GPU agree, uploaded
  as a small rgba16float texture on binding 3 and sampled with the image's
  linear clamp sampler. `FlatField::sample` mirrors that filtering (texel
  centres at `(i + 0.5) / n`). No extra blur on top of cell averaging: one 3x3
  pass left 5.6% of a tight hot spot behind, against 1.9% without.
- Flat fields live in `Config.flat_fields` keyed by lens (maker-note
  `LensType`); `DevelopedImage.flat` carries the one for the open photo (set by
  the caller, since it comes from settings). `sidecar::baseline` turns
  `flat_field` on for a lens that has one.
- The shader clamps the optics `uv` to the outermost texel centres: the detail
  stage samples past the edge, where the CPU mirror clamps the *pixel*.
  `parity_optics_with_detail_at_the_edges` catches a missing clamp (43 levels
  off without it, 1 with) — it needed strong, wide sharpening to do so; with
  mild settings the error hid inside the 3-level tolerance.
- Uniforms grew two vec4s (`hotspot`, `optics`) to 240 bytes.
- `tests/export.rs::flat_field_on_a_real_frame_through_the_gpu_export` runs a
  12 MP real file with a flat field through GPU and CPU export (they agree to
  ~6/65535 mean, 45 worst).

Screenshot note: a window taller than ~800 px came back from `xwd` at 24 bits
per pixel, which the decode recipe above (32-bit) turns into a grey, striped,
narrowed image. Check `h[11]` (bits per pixel) before assuming a broken UI.

## Detail stage (noise reduction + sharpening)

`src/detail.rs` mirrors the `detail()` function in `src/shader.wgsl`; both run
over one 9x9 neighbourhood and share the constants in `pipeline.rs`
(`DETAIL_RADIUS`, `NR_SIGMA`, `NR_RANGE`). Change one, change the other, and
extend `tests/shader_parity.rs` — the detail parity tests render a 32x32 frame
through the real shader and compare it against the CPU path pixel for pixel,
edge clamping included.

- The stage runs **before** the tone chain, on scene-referred linear data.
- The shader early-outs when all three detail controls are zero, so the common
  case costs nothing. Keep that guard.
- `Uniforms` must stay a multiple of 16 bytes (WGSL rounds uniform structs up to
  their largest member alignment); the trailing `show_clip` doubles as that pad. The size is
  asserted in `pipeline::tests`.
- Detail acts on source pixels, so it cannot be judged in the fit-to-window
  view. That is why the 1:1 zoom (Z / View menu) exists — do not remove it
  without replacing it with something equivalent.

## Export, metadata and Immich

- `src/immich.rs` is the *only* module that makes network requests. Do not add
  HTTP anywhere else.
- The API key must never reach a log line, an error message or the status bar.
  `Config`'s `Debug` is hand-written to redact it, errors name only the host,
  and there is a test asserting the key does not leak. Keep all three.
- Uploading is opt-in and happens after the local file is written. An upload
  failure is reported but must never fail the export.
- The Immich contract was taken from the running server's OpenAPI document, not
  the public guides, which are stale (they list `deviceAssetId`/`deviceId` as
  required; 3.1.0 does not). Re-check `/api/spec.json` before changing it.
- `tests/immich_live.rs` only runs when `RAW_EDITOR_IMMICH_TEST_URL` is set, so
  CI never touches the network. Its useful trick: a bogus key must yield **401**
  rather than 400, which proves the request shape without needing a credential.
- TIFF is encoded by `export::write_tiff` through the `tiff` crate, not `image`.
  Metadata goes in during encoding: baseline tags in the image IFD, the rest in
  an EXIF sub-IFD written *first* (the image IFD stores a pointer, so the offset
  has to exist already). Compression is Deflate + horizontal predictor and must
  stay lossless — `compressed_tiff_round_trips_exactly` asserts sample-exact
  equality, so do not swap in anything lossy.

## Reset controls

`widgets::resettable` wraps one control with a ↺ button (space always
reserved via `add_visible`, so rows don't shift) and double-click-to-reset.
egui sliders sense **drag only** — their `Response` never reports clicks — so
the double-click is read off `ui.input` pointer state against the slider's
rect; and a right-click menu is deliberately absent because any button drags a
slider. Sections are `pipeline::Section`; `AdjustParams::reset_section` is the
single list of which field belongs where, and `resetting_every_section_resets_everything`
fails when a new param is added without assigning it to a section. Widget
tests drive a headless `egui::Context` with synthetic pointer events (advance
`RawInput.time` per frame, one event per frame for double-clicks, and
`textures_delta.clear()` or the drop asserts).

## Undo

`src/history.rs` is a generic `History<T>`; the GUI calls `observe(params,
settled)` every frame with `settled = no pointer button down`, so a drag is one
step. Undo mid-drag returns to the pre-drag state. Ctrl+Z is consumed *before*
the single-key shortcuts, and `shortcut()` ignores keys pressed with Ctrl/⌘ —
otherwise Ctrl+Z also toggled the Z zoom. Real-input GUI testing works through
`libXtst` via Python ctypes (`XTestFakeKeyEvent`/`MotionEvent`/`ButtonEvent`
on `:10`) even though `xdotool` is not installed.

## Per-colour HSL and look strength (IR-6)

`pipeline::hsl_adjust` (step 6 of `apply`, after saturation) and `hsl_adjust`
in the shader are twins: RGB→HSV hue, band parameters linearly interpolated
between the two nearest `HSL_BANDS` centres (cyclic, magenta→red wraps), weight
`clamp(s / HSL_FULL_SAT)` so greys are untouched, then the branch-free HSV→RGB
`v − v·s·clamp(min(k, 4−k))`. Keep the tie order of the max-channel branches
identical on both sides. `Uniforms.hsl` is `array<vec4, 8>` (uniform arrays
need 16-byte strides, so vec4) and `hsl[0].w` is the "active" flag the shader
early-outs on; Uniforms was then 368 bytes (400 now — see crop and the tone curve). `parity_hsl_and_look_strength`
covers both. `mixer_strength` is folded into the matrix in `Colour::resolve`
(lerp towards identity), so the shader has no code for it.

## Fit-to-window preview: mipmaps

The image texture carries a full mip chain (`gpu::mip_chain`, 2x2 box
averages built on the CPU at load) and the sampler is trilinear. `fs_main`'s
`textureSample` picks the level from screen derivatives, so the fit view shows
averaged pixels; exports and the GPU parity tests render at 1:1 and stay on
level 0, and `detail()` samples level 0 explicitly. Without the chain the fit
view point-sampled one source pixel per screen pixel and showed
full-resolution noise at a quarter of the size — IR looks that subtract
channels looked far worse on screen than in their exports.
`downscaled_view_averages_instead_of_point_sampling` fails (38 levels off)
if the chain is dropped.

## Aerochrome effects and look thumbnails

`Look` can also carry `hsl`, `contrast` and `blacks` (all `Option`, serde
default), so a built-in can be a finished effect; `Look::finish` fills them.
The #16 Aerochrome effects use `WB_ORANGE_16_CLOUDS` (red gain 0.72) instead
of the asphalt balance — `orange_16_aerochrome_effect_keeps_clouds_white`
fails with 0.63. Looks never set exposure. Tuning was done by exporting
sidecar variants through `target/release/export` and comparing contact sheets
of P1320278 (f/3.9, 0 EV) and P1320294 (f/20, +1 EV).

Thumbnails: `histogram::box_downsample` (box-averaged; the histogram's own
`downsample` point-samples on purpose) → `looks::thumbnail` → `render_cpu`
with detail off. `App::update_thumbs` re-renders only when params changed
and no pointer button is down.

## Crop, rotate, straighten

`pipeline::Geometry::resolve` turns `rotation` (+ the camera's
`ImageColour::turns`), `straighten` and `crop` into one affine map from output
uv to source uv (`Uniforms.geom0/geom1`); the shader samples at that source uv
first thing, so optics and detail stay in source space and the texel stays the
source's. Outside the source renders black (sampled first, then masked —
`textureSample` needs uniform control flow). The crop snaps to whole pixels
of the rotated frame, so without straightening the CPU path (`render_cpu`:
detail on the source grid, then bilinear at the mapped uv) is exact;
`parity_rotation_and_crop` / `parity_straighten` cover both.
Anything sized from a render must use the **render's** size, not the
source's — the look thumbnails crashed egui on the first straighten for that
reason (`thumbnail_size_follows_the_crop`).

Orientation: rawler leaves `RawImage::orientation` at Normal for RW2, so
`metadata::develop_file` (decode + develop + turns from the EXIF block) is the
one entry point the GUI, batch export and CLI use. Exports write Orientation
1 because the pixels are already upright. `testdata/portrait/P1050191.rw2`
(G85, orientation 8) backs `portrait_frames_export_upright`.

Crop UI: `widgets::{CropGrab, drag_crop, fit_aspect}` are pure and tested; the
overlay hit-tests at `pointer.press_origin()` because egui reports a drag only
after a few points of movement, enough to miss a corner.

## Panasonic black level (+15)

rawler takes the RW2 black-level tags (0x1C–0x1E) as is, but they are 15
short: the GF1 reports 0 and its darkest values clip at 16; the G85 reports
128 and its floor is ~143. `develop::black_pedestal` adds 15 for Panasonic
(dcraw/LibRaw do the same). Without it the #16 blue channel — median ~95 on a
starved frame like P1320300 — carried a ~17% pedestal that the Aerochrome mix
turned into a maroon sky and red shadows. Measured by fitting B = k·G + c on
the least-IR pixels (c went from +0.0015…+0.0043 to ≈0). The GF1's "masked"
border columns carry image content, so they cannot be used for black.
Renders of Panasonic files are slightly darker in deep shadow than before.

## Other cameras (S1)

`src/formats.rs` is the one list of what the editor opens (`is_supported`,
`photos_in`, `dialog_extensions`); the open dialog, drag and drop, batch
export and the CLI all go through it — do not add extension checks
elsewhere. `develop` uses `RawImage::crop_area` (falling back to the active
area); on the Panasonic test bodies the two are identical, so their output
did not change. `metadata::develop_file` catches decoder panics and returns
them as errors (`tests/cameras.rs` truncates a real ARW to prove it).

Every wgpu device is created with `gpu::device_descriptor`, which asks for
the adapter's `max_texture_dimension_2d` and `max_buffer_size` instead of the
8192 px / 256 MiB defaults (egui-wgpu's default descriptor also pins 8192).
`gpu::check_fits` refuses an image that still does not fit before it reaches
wgpu, where it would abort the app; the GUI checks on load, export checks in
`render_gpu`. A 33 MP Sony export peaks at ~2 GB of RAM.

## Full-colour input: TIFF and reduced-size raws (S2)

`src/rgb_input.rs` builds a `DevelopedImage` without a demosaic.

- **cpp = 3 raws** (Sony M/S lossless): `develop` hands them to
  `develop_linear_raw`. The reported white level is wrong for these files, so
  `rgb_ceiling` finds the clip plateau from the histogram (most common value
  in the top half, standing ≥8x above values 30–60 away, corroborated by two
  channels within 1%; else the maximum). The first version demanded 0.05% of
  the frame and missed the A7 IV samples' 0.04% patch of blown sky.
  `RAW_EDITOR_DEBUG=1` prints black / ceiling. As-shot WB is unit (baked in);
  the camera matrix still applies.
- **TIFF**: `metadata::develop_file` routes `formats::is_tiff` files to
  `open_tiff` (via `image`, `into_rgb32f`): integer files are sRGB-decoded to
  linear, float files are used as is, colour is `ImageColour::LINEAR_SRGB`.
  `metadata::read` reads TIFF tags with little_exif (`read_tiff`) so
  orientation and camera/lens info survive. ICC profiles are not applied;
  `tiff_profile_note` produces the status-bar warning for non-sRGB ones.
- Clip flag for both: **all three** channels ≥ 0.995. Never "any channel" —
  these pixels are already balanced or finished, and a pure red would turn
  white. `tests/tiff_input.rs` covers it and checks that an unedited 16-bit
  export re-opens and exports to within 2/65535 of itself.
- `.dng` stays a raw (rawler), even though it is a TIFF container.

## Tone curve (P1a)

`AdjustParams.tone_curve` (0..1, default 0 so old sidecars render exactly as
before) blends in `pipeline::filmic`, a Hill curve
`(1 + (K/W)^n) · x^n / (x^n + K^n)` with n = 1.6, K = 0.25, W = 1.0, per
channel, as step 3b of `apply` (after the tonal ranges, before lift/gamma/
gain). The shader twin is `filmic()` in `shader.wgsl`; the strength rides in
`Uniforms.curve.x` (the old `_pad` slot, so the block is still 400 bytes);
`parity_tone_curve` covers it.

The constants are **fitted, not chosen**: neutral exports of the G85 files
and three A7 IV frames were compared with the full-size JPEGs embedded in the
same raws (largest JPEG found by scanning for FFD8FF; Sony embeds full
resolution, the G85 1920×1440), centre 60% of the frame, binned by our linear
luminance → median camera luminance. Both bodies give nearly the same curve
(our 0.106 → 0.24, 0.185 → 0.43, 0.43 → 0.73, 0.73 → 0.90; deep shadows
darkened). A grid search in log space fit n, K, W to ~0.11 EV RMS; W = 1 costs
nothing and puts the clip level on white. An earlier hand-picked Hable curve
was ~0.5 EV too dark in the mid-tones and lifted the shadows the cameras
crush. Saturation: with the curve, the Sony JPEGs match at +0.15 and the
G85's (muted "Standard") at ≤ 0 — Natural uses +0.1.
`tone_curve_is_identity_at_zero_and_camera_like_at_one` pins the measured
points to within 0.2 EV; redo the measurement before changing the constants.
