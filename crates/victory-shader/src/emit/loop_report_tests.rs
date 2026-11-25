//! End-to-end tests of the loop report: synthetic GFX10 loops that never
//! leave on their own, lowered with the report, validated by `dxil.dll`
//! when present and run on the host GPU and WARP; the iteration limit stops
//! the lane and the report names the loop.

use super::host::{Dispatch, Host, Root};
use super::{HostShader, lower, validator};
use crate::native::{ComputeIo, HwStage, NativeProgram, StageIo};
use crate::{io, ir, resource, structure};

const TAG: u32 = 0x1234_5678;

/// Lowers a one-thread wave32 compute program with the loop report.
fn lowered(words: &[u32]) -> lower::Lowered {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 0,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        loop_report: Some(TAG),
    })
    .expect("lower")
}

/// [`lowered`], checked to write the report, validated.
fn build(words: &[u32]) -> HostShader {
    let lowered = lowered(words);
    assert!(lowered.loop_report, "the loop writes no report");
    let shader = HostShader {
        loop_report: lowered.loop_report,
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Runs `words` on the host GPU and WARP: the loop report of each, by label.
fn reports(words: &[u32]) -> Vec<(&'static str, Vec<u32>)> {
    let shader = build(words);
    let mut all = Vec::new();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud)), (1, Root::LoopReport)],
                    views: &[],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        all.push((label, out.last().expect("the report").clone()));
    }
    all
}

/// Runs `words` and checks that the report holds one stopped lane, in the
/// slot of the loop whose header is at console byte `header`.
fn check(words: &[u32], header: u32) {
    for (label, report) in reports(words) {
        assert_eq!(report[..2], [1, 0], "{label}: stopped, lost");
        let slot = 4 + 4 * super::report_slot(TAG, header + 1) as usize;
        assert_eq!(report[slot..slot + 3], [TAG, header + 1, 1], "{label}: the loop's slot");
    }
}

#[test]
fn a_loop_without_a_way_out_is_reported_at_its_back_jump() {
    let words = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBF06_8100, // loop: s_cmp_eq_u32 s0, 1
        0xBF85_0001, // s_cbranch_scc1 exit
        0xBF82_FFFD, // s_branch loop
        0xBF81_0000, // exit: s_endpgm
    ];
    check(&words, 4);
}

#[test]
fn a_loop_without_a_way_out_is_reported_at_its_exit_edge() {
    let words = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBF06_8100, // loop: s_cmp_eq_u32 s0, 1
        0xBF84_FFFE, // s_cbranch_scc0 loop
        0xBF81_0000, // s_endpgm
    ];
    check(&words, 4);
}

#[test]
fn a_loop_a_counter_ends_has_no_limit() {
    // 300 iterations: past every stage's limit, but always over
    let words = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0x8100_8100, // loop: s_add_i32 s0, s0, 1
        0xB180_012C, // s_cmpk_eq_i32 s0, 300
        0xBF84_FFFD, // s_cbranch_scc0 loop
        0xBF81_0000, // s_endpgm
    ];
    assert!(!lowered(&words).loop_report, "the loop is counted");
}

#[test]
fn a_counter_that_never_meets_its_end_keeps_the_limit() {
    // 0, 2, 4 ... never 301
    let words = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0x8100_8200, // loop: s_add_i32 s0, s0, 2
        0xB180_012D, // s_cmpk_eq_i32 s0, 301
        0xBF84_FFFD, // s_cbranch_scc0 loop
        0xBF81_0000, // s_endpgm
    ];
    check(&words, 4);
}

#[test]
fn a_lane_outside_exec_leaves_a_loop_it_could_not_leave_at_once() {
    // The light loops' shape (`parked_stay`): a lane done takes the
    // `execz`-not side and leaves once no lane is still at it; a lane
    // outside EXEC only ever goes round again. It leaves the program at
    // the back edge, not at the iteration limit.
    let words = [
        0x7DA4_0081, // v_cmpx_eq_u32 exec, 1, v0 (the one lane is outside)
        0xBE81_037E, // s_mov_b32 s1, exec_lo (lanes still at it)
        0xBE82_037E, // loop: s_mov_b32 s2, exec_lo
        0x4A02_0281, // v_add_nc_u32 v1, 1, v1
        0x7DA2_0283, // v_cmpx_lt_u32 exec, 3, v1 (lanes done)
        0xBF88_0002, // s_cbranch_execz back
        0x8A01_7E01, // s_andn2_b32 s1, s1, exec_lo
        0xBF84_0003, // s_cbranch_scc0 out
        0x8A7E_7E02, // back: s_andn2_b32 exec_lo, s2, exec_lo
        0x877E_017E, // s_and_b32 exec_lo, exec_lo, s1
        0xBF82_FFF7, // s_branch loop
        0xBF81_0000, // out: s_endpgm
    ];
    for (label, report) in reports(&words) {
        assert_eq!(report[..2], [0, 0], "{label}: stopped, lost");
    }
}
