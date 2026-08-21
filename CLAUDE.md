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
- Run `cargo test` before every commit; commit + push each completed phase.

## Build notes

- `image` crate is pinned to jpeg/png/tiff features only (default features
  pull in rav1e/AVIF — very slow compiles). Don't re-enable defaults.
- eframe uses the wgpu renderer (`Renderer::Wgpu`), not glow.
- egui 0.36 API: panels are the unified `egui::Panel::{left,right,top,bottom}`
  shown against the `&mut Ui` that `eframe::App::ui()` hands you (the old
  `SidePanel`/`TopBottomPanel` + `update(ctx, ..)` are gone). Show side/edge
  panels before `CentralPanel`; each one shrinks the parent `Ui`'s cursor.
- `tests/shader_parity.rs` shares one wgpu device via `OnceLock` and locks a
  `Mutex` per render. Creating an instance per test thread segfaults llvmpipe.
  Keep new GPU tests on that helper rather than reintroducing `--test-threads=1`.

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
