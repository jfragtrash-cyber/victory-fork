//! Two presented planes as one image: the console's display puts the second
//! (the interface) over the first (the scene), each scaled to the screen.
//! Titles present so on Series S. (After codec-d/cool_fox 1ad6e67.)

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

use crate::host::Host;

const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=2), visibility=SHADER_VISIBILITY_PIXEL), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP)"
Texture2D<float4> scene : register(t0);
Texture2D<float4> over : register(t1);
SamplerState smooth_ : register(s0);
struct V { float4 at : SV_Position; float2 uv : TEXCOORD0; };
[RootSignature(RS)]
V vs(uint id : SV_VertexID) {
    V v;
    v.uv = float2((id << 1) & 2, id & 2);
    v.at = float4(v.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return v;
}
[RootSignature(RS)]
float4 ps(V v) : SV_Target {
    float4 s = scene.SampleLevel(smooth_, v.uv, 0);
    float4 o = over.SampleLevel(smooth_, v.uv, 0);
    return float4(saturate(o.rgb + s.rgb * (1 - saturate(o.a))), 1);
}
"#;

/// A format a view can read `format` as: a typeless plane (the scene is
/// one) is read as its family's display type; a view of a typeless format
/// reads nothing.
fn typed(format: DXGI_FORMAT) -> DXGI_FORMAT {
    match format {
        DXGI_FORMAT_R32G32B32A32_TYPELESS => DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R10G10B10A2_TYPELESS => DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R8G8B8A8_TYPELESS => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_TYPELESS => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_B8G8R8X8_TYPELESS => DXGI_FORMAT_B8G8R8X8_UNORM,
        other => other,
    }
}

/// The pass that draws the two planes into a back buffer of one format.
pub(crate) struct Composer {
    format: DXGI_FORMAT,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// Two views a list slot: a slot's are written again only once the
    /// list that last read them has run.
    views: Vec<ID3D12DescriptorHeap>,
    step: u32,
}

fn compile(entry: PCSTR, target: PCSTR) -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; the out pointer is
    // ours.
    unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("compose"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            None,
        )?;
    }
    code.ok_or_else(windows::core::Error::empty)
}

impl Composer {
    pub(crate) fn new(host: &Host, format: DXGI_FORMAT) -> windows::core::Result<Self> {
        let (vs, ps) = (
            compile(s!("vs"), s!("vs_5_1"))?,
            compile(s!("ps"), s!("ps_5_1"))?,
        );
        // SAFETY: the blobs' own bytes, alive through the calls.
        unsafe {
            let bytes = |b: &ID3DBlob| {
                std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())
            };
            let device = &host.device;
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes(&vs))?;
            let code = |b: &ID3DBlob| D3D12_SHADER_BYTECODE {
                pShaderBytecode: b.GetBufferPointer(),
                BytecodeLength: b.GetBufferSize(),
            };
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs),
                PS: code(&ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            desc.RTVFormats[0] = format;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
                D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let pso = device.CreateGraphicsPipelineState(&desc)?;
            let step =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            Ok(Composer {
                format,
                root,
                pso,
                views: Vec::new(),
                step,
            })
        }
    }

    pub(crate) fn format(&self) -> DXGI_FORMAT {
        self.format
    }

    /// Records into `list` the draw of `planes` (the scene, then the
    /// interface; both readable by pixel shaders) over all of `back`, a
    /// render target of `width` by `height`. `slot` is the list's own.
    pub(crate) fn record(
        &mut self,
        host: &Host,
        slot: usize,
        list: &ID3D12GraphicsCommandList,
        planes: [&ID3D12Resource; 2],
        back: &ID3D12Resource,
        (width, height): (u32, u32),
    ) -> windows::core::Result<()> {
        while self.views.len() <= slot {
            // SAFETY: a plain heap description.
            let heap = unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: 2,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })?
            };
            self.views.push(heap);
        }
        let views = &self.views[slot];
        // SAFETY: an open list; views no list in the air reads; live
        // textures.
        unsafe {
            let (cpu, gpu) = (
                views.GetCPUDescriptorHandleForHeapStart(),
                views.GetGPUDescriptorHandleForHeapStart(),
            );
            for (i, plane) in planes.iter().enumerate() {
                let at = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu.ptr + i * self.step as usize,
                };
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: typed(plane.GetDesc().Format),
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV {
                            MipLevels: 1,
                            ..Default::default()
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(Some(*plane), Some(&desc), at);
            }
            let rtv = host.rtv(Some(back), None);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.SetDescriptorHeaps(&[Some(views.clone())]);
            list.SetGraphicsRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetGraphicsRootDescriptorTable(0, gpu);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: width as f32,
                Height: height as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            }]);
            list.DrawInstanced(3, 1, 0, 0);
        }
        Ok(())
    }
}
