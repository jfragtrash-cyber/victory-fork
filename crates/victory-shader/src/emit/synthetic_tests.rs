//! Synthetic GFX10 programs of what the 2020 compiler does not produce,
//! lowered to DXIL, validated by `dxil.dll` when present, and run on the
//! host GPU and WARP: lane permutes (`ds_bpermute_b32`), and a pixel
//! program's ordered section (POPS) on rasterizer-ordered views. (Upstream
//! 0b35b7f's tests of its own gathers at a level are left out: this tree
//! reads the level's texels in `Lower::image` its own way.)

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

fn build(words: &[u32]) -> Result<HostShader, crate::ShaderError> {
    let range = |kind, table_offset_dwords, dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 8, 0),
            range(ResourceType::Srv, 8, 4, 16),
            range(ResourceType::Uav, 16, 4, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let lowered = lowered(&native)?;
    Ok(HostShader {
        loop_report: false,
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| super::StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    })
}

fn lowered(native: &NativeProgram) -> Result<lower::Lowered, crate::ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        loop_report: None,
    })
}

#[test]
fn permutes_read_the_lane_named_and_zero_from_lanes_left_out() {
    // Lanes below 24 run the permute; each reads lane (13 × lane + 5) mod 32
    // (its address in dwords, offset 20 bytes) of 1000 + lane.
    let shader = build(&[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x1602_00B4, // v_mul_u32_u24 v1, 52, v0
        0x4A04_00FF,
        0x0000_03E8, // v_add_nc_u32 v2, 1000, v0
        0x7E06_02FF,
        0x0000_DEAD, // v_mov_b32 v3, 0xdead
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0xDACC_0014,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2 offset:20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ])
    .expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let want: Vec<u32> = (0..32)
        .map(|l| match (13 * l + 5) % 32 {
            _ if l >= 24 => 0xDEAD,
            src if src < 24 => 1000 + src,
            _ => 0,
        })
        .collect();
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(32),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        // The program declares the console's wave of 32, which WARP refuses.
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("{label}: no pipeline for a wave of 32; skipped");
            continue;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "no device ran the permute");
}

#[test]
fn ordered_pixel_sections_use_rasterizer_ordered_views() {
    // A pixel program's ordered section (POPS) as the title's compiler
    // writes it: the collision id after the primitive mask (s10, user data
    // ending at s9), a wait loop on the exiting wave id, a read-modify-write
    // of a UAV, the end of the section.
    let code: Vec<u32> = vec![
        0xBE84_0308, // s_mov_b32 s4, s8
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0x9000_9F0A, // s_lshr_b32 s0, s10, 31
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0x8580_807E, // s_cselect_b64 s[0:1], exec, 0
        0x8102_81EF, // s_add_i32 s2, src_pops_exiting_wave_id, 1
        0xBF08_8502, // s_cmp_gt_u32 s2, 5
        0x8582_807E, // s_cselect_b64 s[2:3], exec, 0
        0x8880_0200, // s_or_b64 s[0:1], s[0:1], s[2:3]
        0x87EA_007E, // s_and_b64 vcc, exec, s[0:1]
        0xBF87_0004, // s_cbranch_vccnz +4
        0xBF8E_0001, // s_sleep 1
        0x8100_81EF, // s_add_i32 s0, src_pops_exiting_wave_id, 1
        0xBF0A_8500, // s_cmp_lt_u32 s0, 5
        0xBF84_FFFC, // s_cbranch_scc0 -4
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF000_3F88,
        0x0003_0200, // image_load v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm dlc glc
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x0604_04F2, // v_add_f32 v2, 1.0, v2
        0xF020_1F08,
        0x0003_0200, // image_store v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xF800_1890,
        0x0000_0000, // exp null off, off, off, off done vm
        0xBF81_0000, // s_endpgm
    ];
    let native = |db_shader_control| NativeProgram {
        stage: HwStage::Ps,
        wave64: true,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 8,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 8,
            slot: 0,
        }],
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control,
            patches: Vec::new(),
        },
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let l = lowered(&native(1 << 16)).expect("lower");
    assert!(l.rovs);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
    // Without the ordered mode the collision id is no value the host has.
    let Err(crate::ShaderError::Unsupported { what, .. }) = lowered(&native(0)) else {
        panic!("translated a collision id outside an ordered section");
    };
    assert_eq!(what, "a hardware SGPR input");
}
