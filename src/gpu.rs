//! wgpu render path: the developed linear image lives in an rgba16float
//! texture; every frame the adjustment shader draws it aspect-fitted inside
//! the egui image panel via an egui_wgpu paint callback.

use eframe::egui_wgpu::{CallbackResources, CallbackTrait, RenderState, ScreenDescriptor};
use eframe::wgpu;
use rayon::prelude::*;

use crate::develop::DevelopedImage;
use crate::pipeline::Uniforms;

pub struct ImageRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniforms: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    pub image_size: Option<[usize; 2]>,
    pub target_is_srgb: bool,
}

impl ImageRenderer {
    pub fn new(rs: &RenderState) -> Self {
        Self::from_device(&rs.device, rs.target_format)
    }

    /// egui-independent constructor, also used by the headless parity test.
    pub fn from_device(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adjust"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("adjust"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The lens flat field: a small grid of camera-RGB gains,
                // sampled with the same linear sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("adjust"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("adjust"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(target_format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("adjust"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            // Trilinear: the fit-to-window view samples an averaged mip level
            // instead of point-picking one source pixel per screen pixel,
            // which showed full-resolution noise at a fraction of the size.
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adjust"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            uniforms,
            bind_group: None,
            image_size: None,
            target_is_srgb: target_format.is_srgb(),
        }
    }

    /// Upload a developed image as the active rgba16float texture.
    pub fn set_image(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, img: &DevelopedImage) {
        let (w, h) = (img.width, img.height);
        // RGB f32 → RGBA f16 (rgba16float; there is no 3-channel float format).
        // Alpha carries the clip mask; see `to_working` in the shader.
        let base: Vec<[f32; 4]> = (0..w * h)
            .into_par_iter()
            .map(|i| [img.data[i * 3], img.data[i * 3 + 1], img.data[i * 3 + 2], img.clip_at(i)])
            .collect();
        let levels = mip_chain(base, w, h);

        let size = wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("linear image"),
            size,
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (level, (lw, lh, px)) in levels.iter().enumerate() {
            let pixels: Vec<half::f16> = px.par_iter().flat_map_iter(|p| p.map(half::f16::from_f32)).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&pixels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some((lw * 4 * 2) as u32),
                    rows_per_image: Some(*lh as u32),
                },
                wgpu::Extent3d { width: *lw as u32, height: *lh as u32, depth_or_array_layers: 1 },
            );
        }

        let view = texture.create_view(&Default::default());
        let flat_view = match &img.flat {
            Some(f) => upload_rgba16f(device, queue, "flat field", f.width, f.height, &f.gains),
            // Unit gain everywhere; the shader only samples it when the flat
            // field is switched on, but the binding must always exist.
            None => upload_rgba16f(device, queue, "no flat field", 1, 1, &[[1.0; 3]]),
        };
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adjust"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&flat_view),
                },
            ],
        }));
        self.image_size = Some([w, h]);
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, uniforms: &Uniforms) {
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(uniforms));
    }

    /// Issue the draw into an active render pass. Returns false when no image
    /// is loaded yet.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) -> bool {
        let Some(bg) = &self.bind_group else { return false };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..6, 0..1);
        true
    }
}

/// Upload RGB values as an rgba16float texture (alpha 1) and return its view.
/// Full mip chain by 2x2 box averaging, down to 1x1. Level 0 is `base`.
/// Odd sizes drop their last row/column at each step, as GPUs size mips.
/// Averaging camera RGB before white balance and the matrix is exact for
/// those linear stages; only the tone curve after them sees averaged input.
pub fn mip_chain(base: Vec<[f32; 4]>, w: usize, h: usize) -> Vec<(usize, usize, Vec<[f32; 4]>)> {
    let mut levels = vec![(w, h, base)];
    while let Some(&(pw, ph, ref prev)) = levels.last() {
        if pw == 1 && ph == 1 {
            break;
        }
        let (nw, nh) = ((pw / 2).max(1), (ph / 2).max(1));
        let next: Vec<[f32; 4]> = (0..nw * nh)
            .into_par_iter()
            .map(|i| {
                let (x, y) = (i % nw, i / nw);
                let (x0, y0) = ((x * 2).min(pw - 1), (y * 2).min(ph - 1));
                let (x1, y1) = ((x0 + 1).min(pw - 1), (y0 + 1).min(ph - 1));
                let at = |x: usize, y: usize| prev[y * pw + x];
                let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
                std::array::from_fn(|k| (a[k] + b[k] + c[k] + d[k]) * 0.25)
            })
            .collect();
        levels.push((nw, nh, next));
    }
    levels
}

fn upload_rgba16f(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    w: usize,
    h: usize,
    rgb: &[[f32; 3]],
) -> wgpu::TextureView {
    let pixels: Vec<half::f16> = rgb
        .iter()
        .flat_map(|c| [c[0], c[1], c[2], 1.0].map(half::f16::from_f32))
        .collect();
    let size = wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&pixels),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some((w * 8) as u32), rows_per_image: Some(h as u32) },
        size,
    );
    texture.create_view(&Default::default())
}

/// Per-frame paint callback carrying the current uniform values.
pub struct ImageCallback {
    pub uniforms: Uniforms,
}

impl CallbackTrait for ImageCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(r) = resources.get::<ImageRenderer>() {
            r.write_uniforms(queue, &self.uniforms);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let Some(r) = resources.get::<ImageRenderer>() else { return };
        r.draw(pass);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_chain_averages_down_to_one_pixel() {
        let (w, h) = (6, 3);
        let base: Vec<[f32; 4]> = (0..w * h).map(|i| [i as f32, 1.0, 0.0, 0.0]).collect();
        let levels = mip_chain(base, w, h);
        let sizes: Vec<_> = levels.iter().map(|(w, h, _)| (*w, *h)).collect();
        assert_eq!(sizes, [(6, 3), (3, 1), (1, 1)]);
        // Level 1, pixel 0: mean of (0,1,6,7).
        assert_eq!(levels[1].2[0], [3.5, 1.0, 0.0, 0.0]);
        // Constant channels stay constant all the way down.
        assert!(levels.iter().all(|(_, _, px)| px.iter().all(|p| p[1] == 1.0)));
    }
}
