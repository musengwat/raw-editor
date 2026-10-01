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
        let mut pixels = vec![half::f16::ZERO; w * h * 4];
        pixels
            .par_chunks_mut(4)
            .zip(img.data.par_chunks(3))
            .enumerate()
            .for_each(|(i, (dst, src))| {
                dst[0] = half::f16::from_f32(src[0]);
                dst[1] = half::f16::from_f32(src[1]);
                dst[2] = half::f16::from_f32(src[2]);
                // Alpha carries the clip mask; see `to_working` in the shader.
                dst[3] = half::f16::from_f32(img.clip_at(i));
            });

        let size = wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("linear image"),
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
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((w * 4 * 2) as u32),
                rows_per_image: Some(h as u32),
            },
            size,
        );

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
