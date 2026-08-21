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
- Test RW2s are in `testdata/` (gitignored — never commit raw files).
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

- Clipped pixels are forced neutral **after** demosaicing, never per photosite.
  Green saturates first (its WB gain is 1.0 while red's is 2.3), so a
  per-photosite fix trades a magenta cast for a green one.
- Detection must be corroborated across colours. A single colour peaking alone
  is a scene feature, not a sensor ceiling.

Regression check for this area: develop every file in `testdata/` and confirm
`P1050113.rw2` (ISO 100, overcast sky) renders a neutral sky while every other
file stays byte-identical. `RAW_EDITOR_DEBUG=1` prints the per-colour decision.

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
  their largest member alignment); `_tail` exists only for that. The size is
  asserted in `pipeline::tests`.
- Detail acts on source pixels, so it cannot be judged in the fit-to-window
  view. That is why the 1:1 zoom (Z / View menu) exists — do not remove it
  without replacing it with something equivalent.
