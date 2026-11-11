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

`src/looks.rs` holds `Look` (mixer + optional colour mode / WB / saturation
and, since P1b, every other tone/colour field; `None`/`LookWb::Keep` leave
the photo's value alone) and the built-ins. Every built-in row sums to 1 so
white-balanced neutrals stay neutral — a test enforces it. The built-ins are reasoned from the filter physics (behind IR
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

## Looks for every camera and the default look (P1b)

- `Look.group` (`LookGroup`, `#[serde(skip)]` → saved looks are always
  `Saved`) orders the picker; `builtin()` must stay grouped in
  `LookGroup::ALL` order (`every_builtin_has_a_note_and_a_group_in_picker_order`).
- Non-IR built-ins are made with `Look::general`, which sets **every** look
  field (unset = neutral, `tone_curve` = 1, `colour_mode` = Camera, WB Keep),
  so switching looks never leaves part of the previous one behind
  (`general_looks_replace_the_previous_look_completely`). "None" is a
  general look with `tone_curve` 0 and `colour_mode` None — it clears every
  look field and keeps WB/colour mode. IR built-ins (`Look::builtin`) still
  set only what they name; do not convert them, or the GF1's body look would
  change.
- `Config.default_look` (`None` = `looks::DEFAULT_LOOK`, "Everyday —
  Natural"; `Some("None")` = neutral). `sidecar::baseline(config, ids, path)`
  decides: TIFF → neutral; body look; default look (`Origin::DefaultLook`);
  else neutral. It needs the path for the TIFF rule — every caller passes it
  (`App::refresh_baseline` after settings changes).
- GUI: *Reset all* and Edit → *Reset to how it opened* go to `baseline`;
  Edit → *Reset to neutral* goes to `AdjustParams::default()` (and so writes a
  neutral sidecar, the per-photo opt-out). The B compare shows `baseline`.
- Tuning: contact sheets of every look on the A7 IV L frame, P1050090 and
  P1050019 via `looks::thumbnail` (a throwaway example binary). Thumbnails
  for all 42 looks take ~20 ms per refresh on this box.
- The grey-ramp ordering test skips the IR group: "#16 orange — Aerochrome"
  dips to −0.004 next to black on a grey ramp (meaningless input for it).

## Everyday conveniences (E1)

- **Loading**: `App::start_load` never runs two decodes at once — while one
  is in flight the request goes to `queued_load` (newest wins) and
  `poll_loader` starts it when the current one lands, *without* showing the
  superseded photo. Holding → on 33 MP files would otherwise run several
  ~1 GB develops at once and could show them out of order. Successful loads
  go to `Config.recent` (`push_recent`, max 10).
- **Folder stepping**: `formats::neighbour` (pure, tested; no wrap; copes with
  the current file having vanished). Arrow keys are ignored while any widget
  has focus (a focused slider takes arrows) and in crop mode.
- **Copy/paste**: egui-winit turns *any* Ctrl+C / Ctrl+V (Shift or not) into
  `Event::Copy` / `Event::Paste(text)` and returns before emitting a key
  event, so `consume_shortcut` can never see them, and `Paste` only arrives
  when the system clipboard holds text. Hence: copy writes
  `sidecar::to_clipboard` JSON (tagged `raw_editor_edits`) with
  `ctx.copy_text` and keeps `App.copied`; paste parses the event text with
  `from_clipboard`, falling back to `copied`. `AdjustParams::pasted` keeps the
  target's crop/rotation/straighten and `flat_field`.
- **Auto**: `src/auto.rs`, exposure only. Median sRGB-encoded output luma →
  0.45 by bisection through the full `apply` (so it respects the look), a
  ≤1% new-clip guard that may pull back ≤1.5 EV (sensor-clipped pixels
  excluded), and a soft knee halving brightening beyond +1 EV (two low-key
  G85 frames asked for +2.0/+2.2 EV). 12k samples × 12 bisection steps: ~35
  ms on the 512 px preview; the first version (40k × 20) took 200–400 ms,
  a visible stall on click.

## HDR from brackets (H1)

`src/bracket.rs` finds sets, `src/hdr.rs` merges and tone maps,
`src/bin/hdr.rs` is the CLI, `App::bracket_box` / `start_hdr_merge` /
`pick_and_merge_folder` the GUI. The merge is CPU-only and produces a file;
it adds nothing to the shader or `AdjustParams`, so no parity tests apply.

- **Detection** needs `ExportMetadata.sub_sec_time_original` (added for
  this). Sony keeps "Exposure Bracketing" in its maker note, which rawler
  does not expose; timing + exposure spread is the camera-agnostic signal.
  Thomas's real sets are in `testdata/hdr` (copied from Immich,
  2026-10-20/22/24); several have repeated exposures (ISO 100 floor) and
  `DSC01350–54` is a 2-frame set 1.36 s before a 3-frame one.
- **Alignment**: MTB at a *shared percentile*, not always the median — a
  +2 EV frame can be >50% clipped and its median bitmap is blank (the
  synthetic test failed exactly that way). Translation-only left visible
  seams in the corners of handheld 33 MP frames, hence the tiled affine fit
  (`align_warp`, outlier tiles dropped, |m − I| ≤ 0.03 else fall back).
  Synthetic test scenes must be non-periodic (value noise): a sine pattern
  gave MTB a wrong-but-consistent answer.
- **Weights**: brighter frames ×(1/scale), faded out over max channel
  0.75–0.95, zero where clipped; non-reference frames feathered over 64 px at
  their own edges (a frame shifted 70 px otherwise leaves a seam where it
  runs out); darkest frame is the fallback.
- **Ghosts**: per 16 px block, |log2 ratio| 0.3–0.6 stops → exclusion, dilated
  3 blocks, blurred, then `smoothstep(0, 0.5)` — the core must reach 1
  because a +2 EV frame outweighs the reference 4×.
- **Tone map** (`local_tone_map`): fast guided filter at ¼ res. Base curve is
  `d / (1 + k·d)` toe/shoulder around the median (slope 1 at mid-tones —
  the first, linear squeeze dimmed white walls and looked worse than the
  single frame). Detail split into clarity (base→fine layer, boosted) and
  texture (fine→pixel, kept at 1 so ISO 12800 grain is not amplified).
  `soft_limit` keeps everything ≤ 1 so the 16-bit sRGB TIFF never clips.
- **Output** is `<first>_hdr.tif`; `formats::is_hdr_merge` makes
  `sidecar::baseline` open it in the default look (other TIFFs stay
  neutral). Pixels are *not* turned upright; the camera's Orientation tag
  is copied so `develop_file` turns them on open.
- Tuning method: `target/release/hdr testdata/hdr --out <scratch>`, then
  `target/release/export <scratch> jpg --out <view>` (XDG_CONFIG_HOME at a
  scratch dir) and side-by-side sheets vs the reference frame's own export,
  plus 100% corner crops for alignment.

## Zoom and pan (Z1)

`pipeline::Zoom` (`Fit` / `Fill` / `Scale(screen px per image px)`) and
`View::placed(out, viewport, zoom, pan, texel)` are the whole model; the pan
is NDC and `placed` clamps it, so read it back from `view.offset`.
`View::zoom_about` keeps the pixel under the pointer in place. All of it is
pure and tested in `pipeline::tests`; `image_panel` is the only caller that
knows the viewport, so keys and the View menu just set `App.zoom_cmd`
(`ZoomCmd`) and `image_panel` carries it out.

- Zooming out snaps to `Zoom::Fit` (so the photo follows the window again);
  an image smaller than the viewport instead stops at 100%.
- `ctx.options_mut(|o| o.zoom_with_keyboard = false)` at startup: egui
  otherwise takes Ctrl + / − / 0 to scale the whole interface.
- Crop mode forces `Fit` for the frame without touching `App.zoom`.
  `App::tool_owns_drag` says when a tool takes plain drags (crop today,
  mask handles next); Space + drag and the middle button always pan.
- Wheel speed is `WHEEL_ZOOM` per point of `smooth_scroll_delta`; one notch
  is ~80 points on this box's X server (17% per notch).
- Before/after: `App.show_before` (the `\` key) latches what holding B
  shows; it is cleared on every load.
- GUI check recipe: XTest buttons 4/5 are the wheel. A synthetic
  double-click needs the two clicks <0.3 s apart — sleeping 0.15 s after
  every event misses egui's double-click window and looks like a bug.

## Masks (M1)

`src/masks.rs` is the whole feature apart from the GUI: `Mask` (kind, shape,
`MaskAdjust` offsets), `shape` / `weight` / `locals`, the uniform packing,
the list helpers, the sidecar form and the pure handle geometry
(`handles`, `outline`, `drag`, `MaskGrab`).

- **One pipeline, not two.** `pipeline::Locals` holds the eleven slider
  values a mask can change. `masks::locals(p, uv, aspect)` = global values +
  Σ weight·offset; `pipeline::apply_local(p, &locals, rgb)` is the chain and
  `apply(p, rgb)` is that with `Locals::of(p)`. The shader does the same in
  `locals(src)` at the top of `fs_main`. Anything that renders pixels with a
  position must use `apply_at` / `apply_local` (CPU export, histogram, Auto
  do); plain `apply` ignores masks.
- **`AdjustParams` stays `Copy`**: `masks: [Mask; MAX_MASKS]` (8), used
  slots packed at the front (`masks::add` / `remove` keep it so — the GUI
  list and `count` rely on it). The sidecar lists only used masks
  (`masks::slots`); extra entries are dropped, not refused.
- **Source uv.** Geometry is stored in the texture's uv, before the camera
  turn / rotation / straighten / crop, so reframing never moves a mask.
  Distances are in "aspect space" (u × width/height; unit = source height) so
  circles are round; the shader gets the aspect from `u.optics.z`. Screen ↔
  source goes `MaskMap` (main.rs) → `View::ndc` / `quad_uv_unbounded` →
  `Geometry::output_uv` / `source_uv`.
- **Detail.** Local sharpen / luma NR make the amounts per pixel:
  `detail()` in the shader takes them as arguments, the guard around it
  tests the *local* values, and `detail::apply` mirrors that guard per pixel
  (`wanted`) — a masked photo's untouched pixels skip the stage on both
  sides, bit for bit. Mask weights are taken at the centre pixel only.
- **Uniforms** grew to 1440 bytes: `mask_info` (count, overlay slot + 1) and
  `masks: array<MaskU, 8>` of eight vec4s (two reserved for M2's ranges),
  placed with the other vec4s. A disabled mask is sent with amount 0, not
  kind 0, so the overlay can still show it.
- **Overlay** is a view setting like the clipping overlay: the GUI sets
  `uniforms.mask_info` itself; `Uniforms::new` always passes `None`, so
  exports can never contain it. CPU twin `masks::overlay`.
- **Clamps** (contrast ≥ −1, saturation ≥ −1, sharpen ≥ 0, NR 0..1) apply
  only when a mask is active, so mask-free renders are byte-identical to
  before (checked by exporting five real files with the old and new binary).
- **GUI.** `mask_interact` runs *before* panning (a drag that starts on a
  handle is the mask's, anything else pans) against the view that is on
  screen; `mask_draw` runs *after* the image callback is added — drawn
  before it, the handles end up underneath the photo. `App::selected_mask`
  re-checks the selection every use, because undo / reset / paste can remove
  the mask it points at. New masks are placed in the *output* frame's centre
  and converted to source uv.
- egui's default font has no `＋` (fullwidth plus) or `⧉`; they render as
  boxes. `✖` and `↺` are fine.
- Tests: `masks::tests` (maths, list, sidecar, handles), `parity_*mask*` in
  `tests/shader_parity.rs` (48×32 non-square frame; each asserts the masks
  really changed the render), `masks_on_a_real_frame_through_the_gpu_export`
  (portrait frame + crop).

## Mask ranges (M2)

`MaskRange` (on every `Mask`) limits the selection by brightness and / or
hue; `MaskKind::Range` is "no shape" (shape = 1), so a range alone is a
whole-frame mask. `masks::selection` = shape × range, **then** invert — the
invert flips the whole selection, which is what makes inverting a pure range
mask meaningful. `masks::shape` is geometry only.

- Ranges are judged on `RangeRef::of(p, rgb)` / shader `range_ref(c)`: the
  working pixel **before the detail stage**, after the *global* exposure,
  temp / tint and tone-curve strength, as display-domain brightness
  (`MASK_GAMMA`) and HSV hue / saturation. Deliberately not after contrast,
  the tonal sliders or any mask offset: the selection must not move while
  the mask's own sliders do. The CPU export therefore passes the pre-detail
  `working` pixel to `masks::locals` and the detailed one to `apply_local`;
  the shader keeps `before_detail` for the overlay.
- `apply_at(p, uv, aspect, rgb)` uses `rgb` for both — only right where no
  detail stage sits in between (histogram, Auto).
- Colour weight includes `clamp(sat / HSL_FULL_SAT)`, like the per-colour
  panel, so greys drop out; `hue_sat` in masks.rs and the shader must keep
  the same max-channel tie order. Hue jumps where two channels tie, so the
  colour-range parity tests allow 4 levels instead of 3.
- Feathers are floored (`MIN_RANGE_FEATHER`) when packed into the uniforms
  and in the CPU maths: WGSL `smoothstep` with equal edges is undefined.
  Hues are stored in degrees in the sidecar and sent as fractions of the
  circle (`v7`); brightness range in `v6`.
- GUI: `App.mask_pick` arms the range eyedropper (`handle_range_pick`,
  reading `DevelopedImage::working_patch`); adding a range mask arms it at
  once. While armed, mask handles, pan cursor and double-click zoom stand
  down, and it cancels the WB eyedropper (and vice versa). A grey click
  keeps the colour picker armed instead of setting a meaningless hue.
- Tests: `masks::tests` (range weights, hue seam, greys, invert, picking),
  `parity_mask_*range*` on `range_source()` (every hue × brightness ramp).

## Point curves (T1)

`src/curve.rs`: `Curve` (2–8 points, `Copy`, serialised as a plain list of
`[x, y]`), `Spline` (points + Fritsch–Carlson tangents), `Curves` (rgb, r,
g, b). `AdjustParams.curves`; step 3c of `apply_local` / `curves()` in the
shader — after the filmic `tone_curve`, before lift/gamma/gain.

- **No lookup texture.** The shader evaluates the cubic Hermite itself from
  `Uniforms.curves` (`array<vec4, 32>`: `[x, y, tangent, _]`, eight per
  curve) and `curve_info` (points per curve; 0 = identity, skipped). The
  tangents are computed once on the CPU, so both sides run the same
  arithmetic on the same numbers and a 16-bit export has no LUT
  quantisation. Uniforms are now 1968 bytes.
- The curve works on **sRGB-encoded** values so its x axis is the
  histogram's. Above white the result is `decode(curve(1)) · c`: continuous
  at white, and an unclipped highlight stays above it.
- `Curve::from_points` is the only way in from outside (sidecar, tests): it
  sorts, clamps, enforces `MIN_GAP` and the point limit, and falls back to
  the identity — a hand-edited sidecar cannot produce a curve the spline
  divides by zero on. Unused slots must stay zeroed: `is_identity` is a
  plain `==` against `Curve::IDENTITY`.
- Curves belong to the photo, like masks: looks do not set them.
- Widget: `widgets::curve_editor` (press on empty space inserts and keeps
  dragging; hit-test at `press_origin`). Its headless tests drive real
  pointer events through `ctx.run_ui`.
- Tests: `curve::tests`, `parity_point_curves` (four curve sets × three
  sources, including the non-monotone eight-point one).

## Vignette, grain, vibrance (T2)

`src/effects.rs` (vignette, grain) and step 5 of `apply_local` (vibrance).

- **Vignette = exposure by position**, in *output* uv. `Vignette::resolve`
  turns the four sliders into the numbers both sides use (`vig0`, `vig1`);
  `pipeline::Frame` (source aspect + `Geometry` + vignette) is built once
  per render and `Frame::locals(p, src_uv, out_uv, rgb)` = masks + vignette.
  `apply_at(p, &frame, uv, rgb)` is the positioned chain for histogram and
  Auto. The shader adds `vignette_ev(in.uv)` to `l.exposure_ev`.
- **Grain** is the last step (after HSL, before the overlays), added in
  sRGB-encoded space, luminance only. The noise is an integer hash of the
  *source* pixel grid (`effects::hash` / `grain_hash` — identical bits on
  both sides), bilinearly blended between grid points `grain_size` apart.
  The shader scales its strength by `min(1, size / footprint)`, with
  `footprint` = source pixels per screen pixel from `dpdx/dpdy(src)` taken
  at the top of `fs_main` (derivatives need uniform control flow); exports
  and parity tests are 1:1, where it is 1. `grain_fades_in_a_downscaled_view`
  guards it. Histogram and Auto ignore grain; look thumbnails switch it off.
- **Vibrance**: `sat_factor += vibrance · (1 − HSV saturation)`, floored at
  0; skipped when 0 so old renders are bit-identical.
- Uniforms are 2016 bytes. New `Section::Effects`.

## Texture and clarity (T3)

`src/local.rs`. `LocalBase` holds two guided-filtered copies of the photo's
luminance (fine for texture, coarse for clarity) at a quarter of its size;
`gain(y, base, texture, clarity)` = (y / fine)^texture · (y / coarse)^clarity
(ratios clamped to 4x) multiplies the working pixel as step 0b — after the
detail stage, before exposure. Shader twin: `local_gain`, binding 4
(`local_tex`), uniforms `local0` (weights + "bases bound") and `local1`.

- **The base is in camera space and so is the pixel.** Luminance =
  `as_shot_wb · [0.25, 0.5, 0.25]` on the *stored* camera RGB (before
  optics, WB, matrix), for the base and for the pixel alike, so the ratio
  depends on no slider. Do not "simplify" this to working luminance: every
  WB or exposure move would then change what counts as detail.
- **`DevelopedImage.local`** is built by `metadata::develop_file` (the one
  entry point) via `with_local`; `develop_file_plain` skips it (HDR merge
  frames). Synthetic images have `None` and the sliders do nothing — on
  both sides (`texture_and_clarity_need_the_bases`). The histogram / look
  thumbnail copies share the parent's `Arc`.
- **`Uniforms::new` cannot know about the base** (it takes `ImageColour`):
  every caller that binds an image must chain `.with_local(img.local
  .as_deref())` — the GUI's `image_panel`, `render_gpu`, the parity
  harness. Forget it and the sliders silently do nothing on the GPU.
- Guided filter, not a blur: with a Gaussian base a 1.5-stop edge overshot
  by ~50% at full clarity; with `COARSE_EPS` 0.1 it is under 10%
  (`clarity_lifts_local_contrast_without_ringing_a_hard_edge`). The filter
  runs in log2 so the edge threshold is in stops.
- Cost: the coarse filter runs on a 1/16-size copy and only its
  coefficients are upsampled ("fast guided filter"); `hdr::box_blur`'s
  transposes were made parallel for this. Measured on this box: +80 ms on a
  33 MP A7 IV file (755 → 835 ms develop), +40 ms on 16 MP. The first
  version (both filters at quarter size, serial transposes) cost +270 ms.
- Base values are f16-rounded on the CPU because they are uploaded as
  rgba16float; `LocalBase::sample` mirrors the sampler like
  `FlatField::sample`.
- Masks' ranges are judged on the pixel *before* this gain (and before
  detail), as in the shader. Texture / clarity are global only for now;
  making them maskable means adding them to `Locals`.
- The gain is a ratio, so it preserves the *median* tone, not the mean:
  on a dark noisy frame the arithmetic mean rose 9% at texture 0.7 /
  clarity 0.8 while the median stayed put. The real-frame test checks the
  median for that reason.
- Uniforms are 2048 bytes.

## Colour grading (T4)

`pipeline::grade` / shader `grade`, step 3d (after the point curves, before
lift/gamma/gain). `AdjustParams.grade_{shadows,midtones,highlights}` are RGB
triples around 0 exactly as `widgets::ColorWheel` writes them (mean =
master, remainder = chroma), plus `grade_blending` and `grade_balance`.

- `grade_weights(d, blending, balance)` on display-domain luminance
  (`MASK_GAMMA`): shadows `1 − smoothstep(0, 0.25 + b/2)`, highlights
  `smoothstep(0.75 − b/2, 1)`, mid-tones the rest; they always sum to 1.
  Balance is an exponent on `d` (`2^−balance`, precomputed into
  `grade_info.y`).
- The blended triple's chroma multiplies the pixel (so black stays black),
  the result is rescaled to the original luminance, then the master applies
  as `2^(master · GRADE_LUM_EV)`. The wheel's chroma sums to zero but is
  *not* luminance-neutral (a blue tint darkens), hence the rescale.
- Uniforms are 2112 bytes (`grade_s`, `grade_m`, `grade_h`, `grade_info`).
- GUI: `Section::Grading`; the lift/gamma/gain wheels (`Section::Wheels`)
  are unchanged in the pipeline and sit in a `CollapsingHeader` with a dot
  when non-neutral — they still apply when collapsed, so the dot matters.
- Tests: `grading_*` in `pipeline::tests`, `parity_colour_grading`.

## Releases

Every push to `master` publishes a `build-YYYYMMDD-<sha7>` GitHub Release
(marked Latest; ten newest kept) and every `v*` tag a named one, both through
`packaging/publish-release.sh`, which both platform workflows call after their
tests pass. The tag is derived from the commit, not the run, so the two
workflows land on the same release. v0.2.1 predated Sony support — a user on
an old download reported ARWs "not editable" — so keep releases flowing.
