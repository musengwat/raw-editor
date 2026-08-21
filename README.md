# raw-editor

A basic GPU-accelerated RAW image editor for Panasonic `.rw2` files, with
DaVinci-Resolve-style color adjustments. Rust + wgpu + egui.

## Architecture

- **Decode** (once per file, CPU): [`rawler`](https://lib.rs/crates/rawler)
  extracts CFA sensor data + metadata (black/white levels, as-shot white
  balance, camera color matrices).
- **Develop** (once per file, CPU, `src/develop.rs`): black-level subtract →
  normalize → white-balance gains → demosaic → camera→linear-sRGB matrix.
  Output: scene-referred linear RGB f32, unclamped, uploaded as an
  `rgba16float` wgpu texture.
- **Adjust** (every slider change, GPU): one WGSL fragment shader applies
  exposure → temp/tint → contrast → highlights/shadows → whites/blacks →
  lift/gamma/gain → saturation → sRGB transfer. Real-time at full resolution.
- **Export**: full-res offscreen render → readback → JPEG / 16-bit TIFF.
- **Persistence**: adjustment params serialized to a `<name>.rw2.edit.json`
  sidecar; the RW2 is never modified.

All shader math is mirrored in pure Rust so `cargo test` can verify it and
export can fall back to CPU.

## Status

- [x] Phase 0 — toolchain + decode proof (`cargo run --bin probe -- file.rw2`)
- [x] Phase 1 — develop stage + display the photo
- [x] Phase 2 — real-time core sliders (exposure, contrast, temp/tint, saturation)
- [ ] Phase 3 — highlights/shadows/whites/blacks + lift/gamma/gain wheels
- [ ] Phase 4 — JSON sidecar persistence + JPEG/TIFF export
- [ ] Phase 5 — sharpening + noise reduction (compute passes)

## Reference camera

Panasonic DMC-GF1: 4060×3016 sensor (4000×3000 active), GBRG Bayer, 12-bit
(whitelevel 3986; sensor values can exceed it — clamp on normalize), D65 + StdA
color matrices embedded. Test files live in `testdata/` (gitignored).

## Running

```
cargo run --release -- testdata/f.rw2      # the editor GUI (path optional; drag-and-drop works)
cargo run --release --bin probe -- f.rw2   # dump decode metadata (2nd arg = write a developed PNG)
cargo test --release                       # develop/pipeline math + headless GPU parity tests
```

The GPU parity tests share a single wgpu device behind a `OnceLock` + `Mutex`
(`tests/shader_parity.rs`), so they are safe under the default parallel test
runner — no `--test-threads=1` needed.
