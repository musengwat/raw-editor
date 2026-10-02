# raw-editor

A basic GPU-accelerated RAW image editor for Panasonic `.rw2` files, with
DaVinci-Resolve-style color adjustments. Rust + wgpu + egui.

## Architecture

- **Decode** (once per file, CPU): [`rawler`](https://lib.rs/crates/rawler)
  extracts CFA sensor data + metadata (black/white levels, as-shot white
  balance, camera color matrices).
- **Develop** (once per file, CPU, `src/develop.rs`): black-level subtract →
  normalize → Malvar-He-Cutler demosaic → clip mask. Output: scene-referred *camera* RGB f32,
  unclamped and **not** white balanced, uploaded as an `rgba16float` wgpu
  texture with the clip mask in alpha. The file's as-shot white balance and
  camera→sRGB matrix travel alongside as `ImageColour`.
- **Adjust** (every slider change, GPU): one WGSL fragment shader turns every
  texture sample into working linear RGB (white balance → clipped pixels to
  neutral → matrix), then applies
  noise reduction and sharpening → exposure → temp/tint → contrast →
  highlights/shadows → whites/blacks → lift/gamma/gain → saturation → sRGB
  transfer. Real-time at full resolution.
- **Export**: full-res offscreen render → readback → JPEG / 16-bit TIFF.
- **Persistence**: adjustment params serialized to a `<name>.rw2.edit.json`
  sidecar; the RW2 is never modified.

All shader math is mirrored in pure Rust so `cargo test` can verify it and
export can fall back to CPU.

### Clipped highlights

A saturated photosite only records "at least this much", so its channel ratios
are meaningless — and white-balance gains applied to them turn every blown area
magenta. The develop stage therefore marks clipped photosites, and any pixel
that touches one is forced neutral — after white balance, in the shader, so it
holds at any white balance — at the brightest channel's level (that
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
- [x] Phase 5 — sharpening + noise reduction
- [x] Phase 6 — Malvar-He-Cutler demosaic (replaces bilinear): roughly 2x the
  fine detail with ~35% less false colour on the testdata set; develop takes
  ~270 ms instead of ~150 ms for a 16 MP file

Infrared (the GF1 is now a full-spectrum conversion):

- [x] IR-1 — white balance + matrix moved into the shader; eyedropper and
  0.1–10× gains; Camera / Sensor RGB mode; RGB histogram, clip readout and
  clipping overlay
- [x] IR-2 — 3×3 channel mixer + looks (Aerochrome classic / pink, red–blue
  swap, IR mono). The looks are derived from what the sensor sees behind an
  IR Chrome filter, **not yet tuned** against daylight shots through it
- [x] IR-3 — saved looks (in the settings file); a camera body can be set to
  start in a look, keyed by the serial in Panasonic's maker note. It applies
  to photos with no sidecar of their own. The serial identifies the *body*,
  not the conversion, so the GF1's pre-conversion shots also open in its IR
  look — Reset on such a photo saves a neutral sidecar and opts it out
- [x] IR-4 — hot-spot correction: a radial model (amount, radius, centre,
  per-channel colour) plus a per-lens flat field made from one evenly lit,
  defocused frame (*Optics → Make from this photo*). A lens with a flat field
  opens with it on. Not yet tried on a real IR hot spot — none of the test
  shots has one

Algorithms and coefficients for phases 5 and 6, including the transposed-filter
trap in the MHC paper's figure, are worked out in
[`docs/phase5-6-algorithms.md`](docs/phase5-6-algorithms.md).

## Controls

**Resetting**: every slider gets a small ↺ beside it once it has moved off
neutral; click it, or double-click the slider, to put that one control back.
Each section heading (Colour, Basic, Tonal range, Lift / Gamma / Gain, Detail,
Optics) has its own *Reset*, enabled while anything under it has changed, and
*Reset all* at the bottom (also *Edit → Reset adjustments*) clears everything.
All of them reset to neutral, not to the photo's camera look.

**Undo**: **Ctrl+Z** (⌘Z on macOS) undoes, **Ctrl+Shift+Z** or **Ctrl+Y**
redoes; both are also in the *Edit* menu. A whole slider drag is one step.
History is per photo and starts fresh when another one is opened. While a text
field has focus, Ctrl+Z belongs to the field.

**Colour**: white balance and how camera RGB becomes working RGB — built for
the infrared-converted GF1, useful on any file.
*Pick neutral* (or **W**) arms an eyedropper: click something that should be
neutral (foliage in IR, a white wall, a grey card) and it sets the gains that
make that 9×9-pixel spot neutral, ignoring clipped pixels; Esc cancels.
*Red gain* / *Blue gain* set the gains directly on a log scale from 0.1× to 10×
(a camera's own white balance never takes red below 1×, which an IR frame
under warm light needs). *As shot* returns to the camera's white balance.
*Camera* mode applies the file's camera→sRGB matrix; *Sensor RGB* skips it,
because a matrix calibrated for visible light scrambles hues once the sensor
sees infrared. Temp and tint still apply on top as fine trims.

**Histogram**: RGB, overlaid, of the output as it currently looks, with the
percentage of pixels clipping in each channel underneath (orange once it passes
0.1%). It is computed on the CPU from a 1/8-scale copy through the same code
path the shader mirrors, so it tracks every control except noise reduction and
sharpening, which a downscaled copy cannot show. Square-root scaled, and scaled
against the tallest *interior* bin so a big clipped spike doesn't flatten the
rest. *Show clipping* (or **C**) turns the image into a clipping map: clipped
pixels show which channels clipped (red, yellow, white…), the rest go dim grey.
It is a viewing aid only and never reaches an export.

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

**Detail**: noise reduction (luminance and colour) and sharpening, all sharing
one 9×9 neighbourhood pass. Luminance NR is bilateral — neighbours are weighted
by distance *and* by luminance similarity, so flat areas smooth while edges
survive. Colour NR is a plain blur of the chroma difference, which is where the
blotchy high-ISO noise lives and where there is no detail worth keeping.
Sharpening is a thresholded unsharp mask on luminance; the threshold matters,
because at zero it happily amplifies grain along with detail. On a real ISO 800
frame, NR at 0.9 cuts pixel-to-pixel gradient by 56% while leaving local
contrast alone.

These act on *source* pixels, so they are invisible in a fit-to-window view of a
16MP image — that view has already thrown away the frequencies they work on.
Press **Z** (or use the *View* menu) for a 1:1 view and drag to pan; that is the
only honest way to judge them, and it is why the zoom exists at all. Export is
always full resolution, so what 1:1 shows is what you get.

**Menu**: *File → Open…* picks a raw through the native file dialog.
*File → Export as JPEG… / 16-bit TIFF…* and the panel's export button both ask
where to write — every export goes through a save dialog, which also means the
system asks before overwriting anything. *File → Batch export folder…* takes a
folder of raws and a destination and exports every one of them using its own
sidecar, counting progress in the status bar; a file that fails is reported and
the rest carry on. *Edit → Reset adjustments* mirrors the Reset all button.
Drag-and-drop still works.

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
and encodes it. JPEG takes a quality slider. TIFF is genuine 16-bit-per-channel
RGB, encoded through the `tiff` crate (not `image`, which exposes neither
compression nor extra tags) with Deflate plus a horizontal predictor — losslessly,
which a round-trip test pins down. The predictor matters: Deflate alone managed
only 12% on a real frame, and storing differences between neighbouring pixels
instead took it to 25% (68.7 → 51.5 MiB). Output lands next to the raw as `<stem>_edit.jpg` / `<stem>_edit.tif`, so a
camera JPEG sitting alongside is never overwritten. `export::render_cpu` is the
pure-Rust equivalent and is used automatically if the GPU pass fails.

Exported JPEGs carry the camera's own metadata, read straight from the raw:
make, model, lens, date, exposure time, aperture, focal length, ISO,
orientation, artist and copyright, plus `Software = raw-editor` and
`ColorSpace = sRGB` so viewers stop guessing at the encoding. Orientation is
passed through deliberately — the develop stage does not rotate pixels, so the
tag is what makes a portrait frame display upright.

TIFFs carry the same set, written during encoding into a real EXIF sub-IFD
rather than bolted on afterwards.

Because the developed image is uploaded as `rgba16float`, half-float is the
pipeline's precision floor; the 16-bit TIFF removes 8-bit banding but is not 16
*significant* bits end to end.

## Uploading to Immich

*File → Settings…* takes an Immich server URL and API key (Immich → Account
Settings → API Keys) and has a **Test connection** button that reports the
server version. Once both are set, "Upload to Immich after export" sends each
finished export; the CLI does the same with `--upload`, and refuses to run
rather than silently skipping if nothing is configured.

The upload is deliberately opt-in and always happens *after* the local file is
written, so a server that is down or misconfigured never costs you an export —
the failure is reported and the file stays on disk. Immich deduplicates by
checksum, so re-uploading the same export is harmless and reports `duplicate`.

Capture dates are not sent explicitly: `fileCreatedAt` is a filesystem
timestamp, and Immich reads the real capture time from the EXIF written into the
JPEG, so photos file under when they were shot.

The API key is a credential. It lives in `config.json` in your OS config
directory (`~/.config/raw-editor/` on Linux, `%APPDATA%` on Windows,
`~/Library/Application Support/raw-editor/` on macOS), written
`0600` on Unix, masked in the settings dialog, redacted from `Debug`, and kept
out of log lines and error messages — `src/immich.rs` names only the host when a
request fails. That module is the only place in the project that touches the
network; keep it that way.

## Windows build

Every push to `master` builds the executable on a Windows runner
(`.github/workflows/windows.yml`) and attaches `raw-editor-windows-x86_64.zip`
to the run (uploaded with `archive: false`, so it is one zip, not a zip in a
zip). Pushing a version tag (`git tag v0.2.0 && git push origin v0.2.0`) also
publishes the Windows and macOS zips to a **GitHub Release** — the easiest
place to download from. Unzip, double-click. It is portable: no installer, no
runtime to install, and wgpu picks DX12 on Windows. `workflow_dispatch` lets
you trigger a build by hand.

The executable is not code-signed, so Chrome may call the download
"uncommon" or "dangerous" (open the downloads page, ⋮ → *Keep*) and Windows
SmartScreen may say "Windows protected your PC" (*More info* → *Run anyway*).
Only a code-signing certificate removes those warnings.

Cross-compiling from Linux was avoided: rustc has a known crash cross-compiling
egui to `x86_64-pc-windows-gnu`, and a native MSVC build is what we want to ship
regardless. (The GUI stack is pure Rust, but `ring` — pulled in by `ureq`'s TLS
for the Immich upload — compiles C, so any cross build needs a C cross-compiler.)

Nothing strips the binary, so expect roughly 36 MB (about 15 MB zipped). Adding
`strip = true` and `lto = true` under `[profile.release]` would cut that
substantially if it ever matters.

## macOS build

Every push to `master` also builds a universal `raw-editor.app` (Intel + Apple
Silicon) on a macOS runner (`.github/workflows/macos.yml`) and attaches
`raw-editor-macos-universal.zip` to the run. It targets **macOS 10.13 High
Sierra or later**, so it runs on any 2012-or-newer Intel MacBook Air; wgpu
renders through Metal, which is also why nothing older works — pre-2012 Macs
have no Metal GPU, and supporting them would mean an OpenGL path for the whole
shader pipeline.

GitHub has no Intel macOS runners any more, so the Intel slice is cross-built on
Apple Silicon (Apple's toolchain does that natively) and its tests run under
Rosetta when the runner has it. All the packaging lives in
`packaging/macos/bundle.sh`, which also works by hand on a Mac. It refuses to
produce a bundle unless the binary is universal and the Intel slice really
targets 10.13 — without `MACOSX_DEPLOYMENT_TARGET` the binary silently inherits
the build machine's macOS version and won't launch on an old one.

**First launch.** The app is ad-hoc signed, not signed with an Apple developer
ID, so Gatekeeper blocks a plain double-click. Unzip it, move `raw-editor.app`
to Applications, then **right-click → Open → Open** once; after that it opens
normally. (On macOS 15 and later: try to open it, then *System Settings →
Privacy & Security → Open Anyway*.) Alternatively
`xattr -dr com.apple.quarantine /Applications/raw-editor.app`.

**Opening files.** Use *File → Open…* or drag a `.rw2` onto the window.
Double-clicking a raw in Finder, *Open With*, or dropping onto the Dock icon do
**not** work: macOS delivers those as an Apple Event, which winit (under eframe)
doesn't expose, so the app deliberately doesn't claim `.rw2` files at all.

Everything works offline except the Immich upload, whose server URL (in
settings) needs to be reachable from wherever the laptop is.

## Reference camera

Two bodies are exercised by the test files in `testdata/` (gitignored):

- **DMC-GF1**: 4060×3016 sensor (4000×3000 active), GBRG Bayer, black level 0,
  reported whitelevel 3986.
- **DMC-G85**: 4816×3464 sensor (4592×3448 active at offset 8,8), BGGR Bayer,
  black level 128, reported whitelevel 4095 — but see *Clipped highlights*: at
  ISO 100 it really saturates near 2880.

Both embed D65 + StdA colour matrices, and on both the deprecated `xyz_to_cam`
field is zeroed, so `RawImage::color_matrix` is the one to read. Stray samples
above the reported whitelevel are normal (hot pixels) and are not evidence of
where the sensor clips.

## Running

```
cargo run --release -- testdata/f.rw2      # the editor GUI (path optional; drag-and-drop works)
cargo run --release --bin probe -- f.rw2   # dump decode metadata (2nd arg = write a developed PNG)
cargo run --release --bin export -- f.rw2 jpg 92   # headless export using the file's sidecar
cargo run --release --bin export -- photos/ jpg 90 --out out/   # batch a whole folder
cargo test --release                       # develop/pipeline math + headless GPU parity/export tests
```

The GPU tests share a single wgpu device behind a `OnceLock` + `Mutex`
(`tests/common/mod.rs`), so they are safe under the default parallel test
runner — no `--test-threads=1` needed.
