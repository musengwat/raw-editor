//! Full-resolution export: run the adjustment pipeline over the whole image
//! and write a JPEG or a 16-bit TIFF.
//!
//! The GPU path renders one offscreen pass into an `Rgba16Float` target and
//! reads it back. That is the same shader the viewport uses, so what you see
//! is what you get. `render_cpu` is the pure-Rust equivalent built on
//! `pipeline::apply` — used when no adapter is available, and by tests.
//!
//! Precision note: the source texture is `rgba16float`, so half-float is the
//! pipeline's precision floor either way. The 16-bit TIFF still buys a lot
//! over 8-bit (no quantisation banding in gradients), it just is not 16
//! *significant* bits end to end.

use std::path::Path;

use anyhow::{Context, Result};
use eframe::wgpu;
use image::ImageEncoder;
use rayon::prelude::*;

use crate::develop::DevelopedImage;
use crate::gpu::ImageRenderer;
use crate::detail;
use crate::metadata::ExportMetadata;
use crate::pipeline::{AdjustParams, Uniforms, View, apply, srgb_encode};

/// wgpu requires texture→buffer copy rows to be a multiple of this.
const COPY_ALIGN: usize = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExportFormat {
    /// 8-bit baseline JPEG, quality 1..=100.
    Jpeg,
    /// 16-bit-per-channel RGB TIFF (uncompressed).
    Tiff16,
}

impl ExportFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Tiff16 => "tif",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Jpeg => "JPEG",
            Self::Tiff16 => "16-bit TIFF",
        }
    }
}

/// Display-referred, sRGB-encoded RGB at 16 bits per channel.
pub struct Rendered {
    pub width: usize,
    pub height: usize,
    /// `width * height * 3` samples, row-major.
    pub data: Vec<u16>,
}

fn to_u16(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

/// Pure-CPU render of the full adjustment chain. Mirrors the shader exactly
/// (both call the same `pipeline::apply`).
pub fn render_cpu(img: &DevelopedImage, params: &AdjustParams) -> Rendered {
    // Neighbourhood work first, then the per-pixel chain — same order as the
    // shader.
    let detailed = detail::apply(img.width, img.height, &img.data, params);
    let mut data = vec![0u16; img.width * img.height * 3];
    data.par_chunks_mut(3).zip(detailed.par_chunks(3)).for_each(|(dst, src)| {
        let c = apply(params, [src[0], src[1], src[2]]);
        for i in 0..3 {
            dst[i] = to_u16(srgb_encode(c[i]));
        }
    });
    Rendered { width: img.width, height: img.height, data }
}

/// Render the whole image through the real WGSL pipeline in one offscreen
/// pass and read it back.
pub fn render_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img: &DevelopedImage,
    params: &AdjustParams,
) -> Result<Rendered> {
    let (w, h) = (img.width, img.height);
    let limit = device.limits().max_texture_dimension_2d as usize;
    if w > limit || h > limit {
        anyhow::bail!("image {w}x{h} exceeds the device's {limit}px texture limit");
    }

    // Rgba16Float is renderable everywhere and matches the source texture's
    // precision; the shader writes sRGB-encoded values into it.
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    let mut renderer = ImageRenderer::from_device(device, FORMAT);
    renderer.set_image(device, queue, img);
    // One output pixel per source pixel, so the detail filters step by exactly
    // one source texel and the export matches the 1:1 preview.
    let view = View::one_to_one([1.0 / w as f32, 1.0 / h as f32]);
    renderer.write_uniforms(queue, &Uniforms::new(params, view, false));

    let size = wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 };
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("export target"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());

    let unpadded_row = w * 8; // rgba16float = 8 bytes/px
    let padded_row = unpadded_row.div_ceil(COPY_ALIGN) * COPY_ALIGN;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("export readback"),
        size: (padded_row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("export"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if !renderer.draw(&mut pass) {
            anyhow::bail!("no image bound to the export renderer");
        }
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row as u32),
                rows_per_image: Some(h as u32),
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| {
        if let Err(e) = r {
            eprintln!("export readback map failed: {e}");
        }
    });
    device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .context("waiting for the export render to finish")?;
    let mapped = slice.get_mapped_range().context("mapping the export readback buffer")?;

    let mut data = vec![0u16; w * h * 3];
    data.par_chunks_mut(w * 3).enumerate().for_each(|(row, out)| {
        let src: &[half::f16] =
            bytemuck::cast_slice(&mapped[row * padded_row..row * padded_row + unpadded_row]);
        for col in 0..w {
            for ch in 0..3 {
                out[col * 3 + ch] = to_u16(src[col * 4 + ch].to_f32());
            }
        }
    });
    drop(mapped);
    readback.unmap();

    Ok(Rendered { width: w, height: h, data })
}

/// Encode and write `rendered` to `path`. `quality` is only used for JPEG.
/// Write the rendered image, then stamp the camera's metadata into it.
///
/// Metadata failure is reported but never discards the image: the file on disk
/// is already complete and correct at that point, and losing a full-resolution
/// export because a tag would not encode would be a bad trade.
pub fn write(
    path: &Path,
    rendered: &Rendered,
    format: ExportFormat,
    quality: u8,
    meta: Option<&ExportMetadata>,
) -> Result<()> {
    write_image(path, rendered, format, quality, meta)?;
    // JPEG takes its EXIF as an APP1 segment bolted onto the finished file;
    // TIFF already had its tags written during encoding.
    if let Some(meta) = meta.filter(|_| format == ExportFormat::Jpeg) {
        if let Err(e) = meta.write_into(path) {
            eprintln!("warning: {} written, but metadata failed: {e:#}", path.display());
        }
    }
    Ok(())
}

fn write_image(
    path: &Path,
    rendered: &Rendered,
    format: ExportFormat,
    quality: u8,
    meta: Option<&ExportMetadata>,
) -> Result<()> {
    let (w, h) = (rendered.width as u32, rendered.height as u32);
    let file = std::fs::File::create(path)
        .with_context(|| format!("creating {}", path.display()))?;
    let writer = std::io::BufWriter::new(file);

    match format {
        ExportFormat::Jpeg => {
            // 65535/257 == 255 exactly, so this is a clean 16→8 bit reduction.
            let bytes: Vec<u8> = rendered.data.par_iter().map(|&v| (v / 257) as u8).collect();
            image::codecs::jpeg::JpegEncoder::new_with_quality(writer, quality.clamp(1, 100))
                .write_image(&bytes, w, h, image::ExtendedColorType::Rgb8)
        }
        ExportFormat::Tiff16 => {
            // Encoded through the `tiff` crate rather than `image`, which
            // exposes neither compression nor extra tags. Metadata is written
            // during encoding here, not bolted on afterwards.
            return write_tiff(path, rendered, meta);
        }
    }
    .with_context(|| format!("encoding {}", path.display()))?;
    Ok(())
}

/// Deflate: on 16-bit photographic data it beats LZW comfortably and every
/// TIFF reader worth the name handles it. Balanced level — Best buys a couple
/// of percent for a lot of time on a 70 MB image.
const TIFF_COMPRESSION: tiff::encoder::Compression =
    tiff::encoder::Compression::Deflate(tiff::encoder::DeflateLevel::Balanced);

/// Horizontal differencing before compression. Continuous-tone images have
/// small differences between neighbouring pixels, so storing those differences
/// gives the compressor far more to work with — worth roughly another 20% on
/// top of plain Deflate here, and the reason a compressed TIFF is worth having
/// at all.
const TIFF_PREDICTOR: tiff::tags::Predictor = tiff::tags::Predictor::Horizontal;

/// Write a compressed 16-bit TIFF, with the camera metadata in a proper EXIF
/// sub-IFD.
///
/// The EXIF directory has to be written *before* the image directory, because
/// the image IFD stores a pointer to it and we can only know the offset once
/// the sub-directory is on disk.
fn write_tiff(path: &Path, rendered: &Rendered, meta: Option<&ExportMetadata>) -> Result<()> {
    use tiff::encoder::{Rational, TiffEncoder, colortype};
    use tiff::tags::Tag;

    let (w, h) = (rendered.width as u32, rendered.height as u32);
    let file = std::fs::File::create(path)
        .with_context(|| format!("creating {}", path.display()))?;
    let mut enc = TiffEncoder::new(std::io::BufWriter::new(file))
        .with_context(|| format!("starting {}", path.display()))?
        .with_compression(TIFF_COMPRESSION)
        .with_predictor(TIFF_PREDICTOR);

    let rat = |(n, d): (u32, u32)| Rational { n, d };

    // 1. The EXIF sub-directory, if we have anything to put in it.
    let mut exif_offset = None;
    if let Some(m) = meta {
        let mut dir = enc.extra_directory().context("starting the EXIF directory")?;
        if let Some(v) = m.exposure_time {
            dir.write_tag(Tag::Unknown(0x829a), rat(v))?;
        }
        if let Some(v) = m.fnumber {
            dir.write_tag(Tag::Unknown(0x829d), rat(v))?;
        }
        if let Some(v) = m.iso {
            dir.write_tag(Tag::Unknown(0x8827), v)?;
        }
        if let Some(v) = &m.date_time_original {
            dir.write_tag(Tag::Unknown(0x9003), v.as_str())?;
        }
        if let Some(v) = m.focal_length {
            dir.write_tag(Tag::Unknown(0x920a), rat(v))?;
        }
        if let Some(v) = &m.lens_make {
            dir.write_tag(Tag::Unknown(0xa433), v.as_str())?;
        }
        if let Some(v) = &m.lens_model {
            dir.write_tag(Tag::Unknown(0xa434), v.as_str())?;
        }
        // sRGB, so viewers stop guessing.
        dir.write_tag(Tag::Unknown(0xa001), 1u16)?;
        exif_offset = Some(dir.finish_with_offsets().context("writing the EXIF directory")?);
    }

    // 2. The image itself, with the baseline tags and a pointer to the above.
    let mut image = enc
        .new_image::<colortype::RGB16>(w, h)
        .with_context(|| format!("encoding {}", path.display()))?;
    {
        let dir = image.encoder();
        if let Some(m) = meta {
            if let Some(v) = &m.make {
                dir.write_tag(Tag::Make, v.as_str())?;
            }
            if let Some(v) = &m.model {
                dir.write_tag(Tag::Model, v.as_str())?;
            }
            if let Some(v) = m.orientation {
                dir.write_tag(Tag::Orientation, v)?;
            }
            if let Some(v) = &m.date_time_original {
                dir.write_tag(Tag::Unknown(0x0132), v.as_str())?;
            }
            if let Some(v) = &m.artist {
                dir.write_tag(Tag::Artist, v.as_str())?;
            }
            if let Some(v) = &m.copyright {
                dir.write_tag(Tag::Copyright, v.as_str())?;
            }
        }
        dir.write_tag(Tag::Software, "raw-editor")?;
        if let Some(off) = &exif_offset {
            dir.write_tag(Tag::ExifDirectory, off.offset)?;
        }
    }
    image
        .write_data(&rendered.data)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Default output path: `<stem>_edit.<ext>` next to the raw, so a camera JPEG
/// sitting beside the RW2 is never overwritten.
pub fn output_path(raw: &Path, format: ExportFormat) -> std::path::PathBuf {
    let stem = raw.file_stem().unwrap_or_default().to_string_lossy().into_owned();
    raw.with_file_name(format!("{stem}_edit.{}", format.extension()))
}

/// Every raw file directly inside `dir`, sorted, matched case-insensitively.
///
/// Subdirectories are not walked: batch export is a "do this folder" action,
/// and quietly recursing into a photo library would be a nasty surprise.
pub fn raw_files_in(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let is_raw = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("rw2"));
        if is_raw {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

/// Where a batch export writes `raw`'s result, given an optional output
/// directory. Without one the file lands beside its raw, as before.
pub fn batch_output_path(
    raw: &Path,
    out_dir: Option<&Path>,
    format: ExportFormat,
) -> std::path::PathBuf {
    let named = output_path(raw, format);
    match out_dir {
        Some(dir) => dir.join(named.file_name().unwrap_or_default()),
        None => named,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_output_path_redirects_into_a_folder() {
        let raw = Path::new("/photos/P1320071.rw2");
        assert_eq!(
            batch_output_path(raw, None, ExportFormat::Jpeg),
            Path::new("/photos/P1320071_edit.jpg")
        );
        assert_eq!(
            batch_output_path(raw, Some(Path::new("/out")), ExportFormat::Jpeg),
            Path::new("/out/P1320071_edit.jpg")
        );
    }

    #[test]
    fn raw_discovery_is_case_insensitive_and_skips_other_files() {
        let dir = std::env::temp_dir().join(format!("raw-editor-discover-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["b.rw2", "a.RW2", "note.txt", "old_edit.jpg"] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        std::fs::create_dir_all(dir.join("nested")).unwrap();
        std::fs::write(dir.join("nested/deep.rw2"), b"x").unwrap();

        let found = raw_files_in(&dir).unwrap();
        let names: Vec<String> =
            found.iter().map(|p| p.file_name().unwrap().to_string_lossy().into()).collect();
        assert_eq!(names, vec!["a.RW2", "b.rw2"], "sorted, raws only, no recursion");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn output_path_is_suffixed_next_to_the_raw() {
        let p = output_path(Path::new("/photos/P1320071.rw2"), ExportFormat::Jpeg);
        assert_eq!(p, Path::new("/photos/P1320071_edit.jpg"));
        let p = output_path(Path::new("/photos/P1320071.rw2"), ExportFormat::Tiff16);
        assert_eq!(p, Path::new("/photos/P1320071_edit.tif"));
    }

    #[test]
    fn cpu_render_encodes_srgb() {
        let img = DevelopedImage {
            width: 2,
            height: 1,
            data: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        };
        let r = render_cpu(&img, &AdjustParams::default());
        assert_eq!(r.data[0..3], [0, 0, 0]);
        assert_eq!(r.data[3..6], [65535, 65535, 65535]);
    }

    #[test]
    fn cpu_render_applies_params() {
        let img = DevelopedImage { width: 1, height: 1, data: vec![0.18, 0.18, 0.18] };
        let dark = render_cpu(&img, &AdjustParams { exposure_ev: -2.0, ..Default::default() });
        let bright = render_cpu(&img, &AdjustParams { exposure_ev: 2.0, ..Default::default() });
        assert!(dark.data[0] < 20000 && bright.data[0] > 50000, "{:?} {:?}", dark.data, bright.data);
    }
}
