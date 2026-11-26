//! Lane semantics of a console wave64 on a host of 32-lane waves, against a CPU reference
//! (the handover's §12 test list). Ported with [`crate::dualbank`] from the dual-bank fork
//! of fxceless/victory-fork, whose `CLAUDE.md` and `docs/XBOX_SERIES_GPU_TRANSLATION.md`
//! these comments cite.
//!
//! [`model`] is a deliberately small model of one 64-lane wave, written from the public
//! RDNA 2 ISA and sharing nothing with the translator: vector registers of 64 slots, EXEC
//! as 64 bits, every instruction reading its sources before writing (source snapshots).
//! Where the ISA leaves a result undefined, or this model does not claim to know it, a lane
//! is `None` and is not compared.
//!
//! The GPU tests build synthetic GFX10 programs of one console wave (64 threads), lower
//! them for a host of 32-lane waves (each console wave becomes two host waves; cross-lane
//! code goes through the paired-wave exchange), validate the DXIL with `dxil.dll` when
//! present, run them on the hardware device and compare every lane with the model. Without
//! a hardware device that takes the program they print "not run" and pass: report them as
//! not run, never as passed.

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::wave64_tests::{lowered, lowered_on, native_of};
use super::{HostShader, lower, validator};
use crate::ir;
use crate::native::RootType;

/// The reference: one console wave of 64 lanes.
pub(super) mod model {
    /// The handover's input: lane g holds `0x1000 + g`.
    pub fn input() -> [u32; 64] {
        std::array::from_fn(|g| 0x1000 + g as u32)
    }

    pub fn active(exec: u64, lane: usize) -> bool {
        exec >> lane & 1 != 0
    }

    /// `ds_bpermute_b32 dst, addr, data` for each active lane: the lane `addr[l] / 4 mod 64`
    /// is read. The ISA's rule for a source lane that is not active is not relied on here:
    /// such lanes are `None`. Lanes not active keep `old`.
    pub fn bpermute(
        old: &[u32; 64],
        addr: &[u32; 64],
        data: &[u32; 64],
        exec: u64,
    ) -> [Option<u32>; 64] {
        std::array::from_fn(|l| {
            if !active(exec, l) {
                return Some(old[l]);
            }
            let src = (addr[l] >> 2 & 63) as usize;
            active(exec, src).then(|| data[src])
        })
    }

    /// `v_readlane_b32`: lane `lane mod 64` of `v`, whatever EXEC is.
    pub fn readlane(v: &[u32; 64], lane: u32) -> u32 {
        v[(lane & 63) as usize]
    }

    /// `v_writelane_b32`: `value` into lane `lane mod 64`, whatever EXEC is.
    pub fn writelane(v: &[u32; 64], value: u32, lane: u32) -> [u32; 64] {
        let mut out = *v;
        out[(lane & 63) as usize] = value;
        out
    }

    /// `v_readfirstlane_b32`: the lowest active lane's value; `None` with EXEC empty.
    pub fn readfirstlane(v: &[u32; 64], exec: u64) -> Option<u32> {
        (exec != 0).then(|| v[exec.trailing_zeros() as usize])
    }

    /// A `v_cmp` into VCC: the predicate's bit for each active lane, 0 for the others.
    pub fn ballot(exec: u64, predicate: impl Fn(usize) -> bool) -> u64 {
        (0..64)
            .filter(|&l| active(exec, l) && predicate(l))
            .fold(0, |m, l| m | 1 << l)
    }

    /// `s_ff1_i32_b64`: the lowest set bit, -1 for none.
    pub fn ff1(mask: u64) -> i32 {
        if mask == 0 {
            -1
        } else {
            mask.trailing_zeros() as i32
        }
    }

    /// `v_mbcnt_lo_u32_b32` then `v_mbcnt_hi_u32_b32` over `mask`: for lane g the set bits of
    /// `mask` below g (the exclusive prefix count).
    pub fn mbcnt(mask: u64) -> [u32; 64] {
        std::array::from_fn(|g| (mask & ((1u64 << g) - 1)).count_ones())
    }

    /// Lanes active by `exec` get `v`, the others keep `old`.
    pub fn masked(old: &[u32; 64], v: &[u32; 64], exec: u64) -> [u32; 64] {
        std::array::from_fn(|l| if active(exec, l) { v[l] } else { old[l] })
    }

    /// The address each lane of the shuffle programs gives: `((g ^ x) + y) mod 64`, in bytes.
    pub fn shuffle_addr(x: u32, y: u32) -> [u32; 64] {
        std::array::from_fn(|g| ((g as u32 ^ x).wrapping_add(y) & 63) << 2)
    }
}

/// The handover's deterministic expectations, on the model alone (always runs).
#[test]
fn the_model_gives_the_handovers_expected_values() {
    use model::*;
    let d = input();
    let all = u64::MAX;
    let shuffle = |x, y| bpermute(&[0; 64], &shuffle_addr(x, y), &d, all);
    assert_eq!(shuffle(0, 0)[17], Some(0x1011), "identity");
    assert_eq!(shuffle(0, 1)[31], Some(0x1020), "lane 31 reads lane 32");
    assert_eq!(shuffle(0, 63)[32], Some(0x101F), "lane 32 reads lane 31");
    for g in 0..64 {
        assert_eq!(shuffle(32, 0)[g], Some(0x1000 + (g as u32 ^ 32)), "XOR-32");
        assert_eq!(shuffle(1, 0)[g], Some(0x1000 + (g as u32 ^ 1)), "XOR-1");
    }
    let set = |lanes: &[usize]| lanes.iter().fold(0u64, |m, &l| m | 1 << l);
    let b = ballot(set(&[0, 31, 32, 63]), |_| true);
    assert_eq!(
        (b as u32, (b >> 32) as u32),
        (0x8000_0001, 0x8000_0001),
        "ballot"
    );
    assert_eq!(ff1(1 << 37), 37, "first active lane, high half only");
    assert_eq!(readfirstlane(&d, 1 << 37), Some(0x1025));
    assert_eq!(
        d.iter().map(|v| v - 0x1000).sum::<u32>(),
        2016,
        "sum of lane ids"
    );
    assert_eq!(
        mbcnt(u64::MAX),
        std::array::from_fn(|g| g as u32),
        "exclusive prefix of ones"
    );
    // a source lane that is not active is not claimed
    assert_eq!(
        bpermute(&[7; 64], &shuffle_addr(0, 1), &d, !(1 << 32))[31],
        None
    );
    assert_eq!(
        bpermute(&[7; 64], &shuffle_addr(0, 1), &d, !(1 << 31))[31],
        Some(7)
    );
}

// ---- GFX10 encodings (the decoder's tables name each one: `gfx10/opcodes.rs`) ----

/// Operand codes: SGPRs are their number, VGPRs 256 up, inline integers 128 + n.
const fn v(n: u32) -> u32 {
    256 + n
}
const fn int(n: u32) -> u32 {
    128 + n
}
const LITERAL: u32 = 255;
const VCC_LO: u32 = 106;
const VCC_HI: u32 = 107;
const EXEC: u32 = 126;

const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}
const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}
const fn vop3(op: u32, vdst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD400_0000 | op << 16 | vdst, src1 << 9 | src0]
}
const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | vsrc1 << 9 | src0
}
const fn sop1(op: u32, sdst: u32, ssrc0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | ssrc0
}
const fn ds(op: u32, vdst: u32, addr: u32, data0: u32) -> [u32; 2] {
    [0x36 << 26 | op << 18, vdst << 24 | data0 << 8 | addr]
}

const V_MOV_B32: u32 = 0x01;
const V_READFIRSTLANE_B32: u32 = 0x02;
const V_AND_B32: u32 = 0x1B;
const V_XOR_B32: u32 = 0x1D;
const V_LSHLREV_B32: u32 = 0x1A;
const V_ADD_NC_U32: u32 = 0x25;
const V_CMP_LE_U32: u32 = 0xC3;
const V_CMPX_GT_U32: u32 = 0xD4;
const V_READLANE_B32: u32 = 0x360;
const V_WRITELANE_B32: u32 = 0x361;
const V_MBCNT_LO: u32 = 0x365;
const V_MBCNT_HI: u32 = 0x366;
const S_MOV_B64: u32 = 0x04;
const S_BCNT1_I32_B64: u32 = 0x10;
const S_FF1_I32_B64: u32 = 0x14;
const DS_BPERMUTE_B32: u32 = 0xB3;
const S_WAITCNT_LGKM0: u32 = 0xBF8C_C07F;
/// Stores: `buffer_store_dword v12, v14, s[8:11], 0 offen` (v14 = lane * 4), and
/// `buffer_store_dword vN, off, s[8:11], 0 offset:o`.
const STORE_V12: [u32; 2] = [0xE070_1000, 0x8002_0C0E];
const fn store_at(vdata: u32, offset: u32) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | offset,
        0x80 << 24 | 2 << 16 | vdata << 8,
    ]
}

/// What every program starts with: v1 = 0x1000 + lane, v14 = lane * 4, v12 = 0xDEAD0000
/// (what lanes left out of a write keep), EXEC saved in s[22:23].
fn prologue() -> Vec<u32> {
    vec![
        vop2(V_ADD_NC_U32, 1, LITERAL, 0),
        0x1000,
        vop2(V_LSHLREV_B32, 14, int(2), 0),
        vop1(V_MOV_B32, 12, LITERAL),
        0xDEAD_0000,
        sop1(S_MOV_B64, 22, EXEC),
    ]
}

/// EXEC back from s[22:23] and v12 stored at v14.
fn epilogue(words: &mut Vec<u32>) {
    words.push(sop1(S_MOV_B64, EXEC, 22));
    words.extend(STORE_V12);
}

const DEAD: u32 = 0xDEAD_0000;

/// A program lowered for a host of 32-lane waves (validated), and for 64-lane waves (validated
/// only: the RTX 3060 runs at most 32), on the hardware device.
struct Card {
    host: Host,
    root: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
    pso: windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    shader: HostShader,
    threads: u32,
}

fn validate(shader: &HostShader) {
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
}

/// `body` (after [`prologue`], before the program's end) as one console wave; None (and a
/// line) without a hardware device of 32-lane waves that takes it.
fn card(what: &str, body: &[u32]) -> Option<Card> {
    card_of(what, body, 64)
}

/// [`card`] for a group of `threads` threads (console waves of 64, the last one partial).
fn card_of(what: &str, body: &[u32], threads: u32) -> Option<Card> {
    card_with(what, body, |n| {
        if let crate::native::StageIo::Cs(cs) = &mut n.io {
            cs.threads = [threads, 1, 1];
        }
    })
}

/// [`card`] for a program as `adjust` changes it (group shape, thread-id VGPRs).
fn card_with(
    what: &str,
    body: &[u32],
    adjust: impl FnOnce(&mut crate::native::NativeProgram),
) -> Option<Card> {
    let mut console = native_of(body, true, 64);
    adjust(&mut console);
    let threads = match &console.io {
        crate::native::StageIo::Cs(cs) => cs.threads.iter().product::<u32>(),
        _ => 64,
    };
    let dual = DUAL.get();
    let what = &if dual {
        format!("{what} [dual-bank]")
    } else {
        what.to_owned()
    };
    let (native, shader) = if dual {
        // one host wave of 32 per console wave, two console lanes per host lane
        let (native, banks) = crate::dualbank::transform(&console)
            .unwrap_or_else(|e| panic!("{what}: not rewritten: {e}"));
        let p = ir::build_banked(&native.code, native.wave64, native.vgprs, Some(&banks))
            .unwrap_or_else(|e| panic!("{what}: ir::build: {e}"));
        let shader = lower::with_dual_bank(|| lowered(&native, &p, &HashSet::new()))
            .unwrap_or_else(|e| panic!("{what}: not lowered: {e}"));
        (native, shader)
    } else {
        let p = ir::build(&console.code, console.wave64, console.vgprs).expect("ir::build");
        let wide =
            lowered_on(&console, &p, &HashSet::new(), 64).expect("lowered for 64-lane waves");
        validate(&wide);
        let shader = lowered(&console, &p, &HashSet::new())
            .unwrap_or_else(|e| panic!("{what}: not lowered for 32-lane waves: {e}"));
        (console, shader)
    };
    let _ = &native;
    let form = lower::last_form();
    validate(&shader);
    let host = match Host::new(false) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("{what}: no SM 6.6 hardware device; NOT RUN");
            return None;
        }
    };
    if host.wave_lanes().1 > 32 {
        eprintln!(
            "{what}: the hardware device runs waves over 32 lanes; the 32-lane path is NOT RUN here"
        );
        return None;
    }
    let (root, pso) = host
        .compute_pipeline(&shader)
        .unwrap_or_else(|e| panic!("{what}: the hardware device refuses the 32-lane form: {e}"));
    eprintln!(
        "{what}: lowered for 32-lane waves (exchange {}, local {}), running on the hardware device",
        form & lower::form::EXCHANGE != 0,
        form & lower::form::LOCAL != 0
    );
    Some(Card {
        host,
        root,
        pso,
        shader,
        threads,
    })
}

impl Card {
    /// Runs the program with root constants s0..s3; the UAV's first 64 dwords.
    fn run(&self, s: [u32; 4]) -> Vec<u32> {
        let mut ud = vec![0u32; self.shader.user_data_dwords().max(1) as usize];
        for u in &self.shader.user_data {
            if let (RootType::Constants, 0, d @ 0..=3) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = s[d as usize];
            }
        }
        let mut out = self
            .host
            .dispatch(
                &self.root,
                &self.pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(self.threads.max(64))],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        out.swap_remove(0)
    }

    /// GPU milliseconds of a dispatch of `groups` thread groups (timestamps around the
    /// dispatch, `host::GPU_MS`): the median of `reps` after one warm-up.
    fn time(&self, s: [u32; 4], groups: u32, reps: usize) -> f64 {
        let mut ud = vec![0u32; self.shader.user_data_dwords().max(1) as usize];
        for u in &self.shader.user_data {
            if let (RootType::Constants, 0, d @ 0..=3) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = s[d as usize];
            }
        }
        let once = || {
            self.host
                .dispatch(
                    &self.root,
                    &self.pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud.clone()))],
                        views: &[View::RawUav(self.threads.max(64))],
                        samplers: &[],
                        groups: [groups, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
            super::host::GPU_MS.get()
        };
        once();
        let mut ms: Vec<f64> = (0..reps).map(|_| once()).collect();
        ms.sort_by(f64::total_cmp);
        ms[reps / 2]
    }
}

/// Paired waves against dual-bank on the hardware device (the handover's §12 benchmarks;
/// CLAUDE.md: measure both before choosing defaults). Each workload is one console wave's
/// straight-line program, dispatched as 65,535 groups (4 million console threads); the time
/// of an empty program (the store alone) is subtracted, so what is left is the workload's
/// GPU time plus noise. Ignored by default: `cargo test -p victory-shader --lib
/// bench_paired_against_dual_bank -- --ignored --nocapture --test-threads=1`.
#[test]
#[ignore = "a benchmark: run it on purpose"]
fn bench_paired_against_dual_bank() {
    const GROUPS: u32 = 65_535;
    const REPS: usize = 5;
    const N: usize = 128;
    let body = |inner: &dyn Fn(&mut Vec<u32>)| {
        let mut w = prologue();
        w.extend([
            vop2(V_XOR_B32, 2, 0, 0),
            vop2(V_ADD_NC_U32, 2, 1, 2),
            vop2(V_AND_B32, 2, int(63), 2),
            vop2(V_LSHLREV_B32, 2, int(2), 2),
        ]);
        inner(&mut w);
        w.push(vop1(V_MOV_B32, 12, v(1)));
        epilogue(&mut w);
        w
    };
    let workloads: Vec<(&str, Vec<u32>)> = vec![
        ("empty (the store alone)", body(&|_| {})),
        (
            "128 dependent shuffles across the halves (ds_bpermute)",
            body(&|w| {
                for _ in 0..N {
                    w.extend(ds(DS_BPERMUTE_B32, 1, 2, 1));
                    w.push(S_WAITCNT_LGKM0);
                }
            }),
        ),
        (
            "128 lane reads by a scalar lane (v_readlane) and adds",
            body(&|w| {
                for _ in 0..N {
                    w.extend(vop3(V_READLANE_B32, 20, v(1), 0));
                    w.push(vop2(V_ADD_NC_U32, 1, 20, 1));
                }
            }),
        ),
        (
            "128 compares read back as data (vcc_hi into a VGPR)",
            body(&|w| {
                for _ in 0..N {
                    w.push(vopc(V_CMP_LE_U32, 0, 1));
                    w.push(vop2(V_ADD_NC_U32, 1, VCC_HI, 1));
                }
            }),
        ),
        (
            "1024 adds and xors, no cross-lane code",
            body(&|w| {
                for _ in 0..512 {
                    w.push(vop2(V_ADD_NC_U32, 1, int(3), 1));
                    w.push(vop2(V_XOR_B32, 1, 0, 1));
                }
            }),
        ),
        (
            "8 whole-wave sums (the compiler's DPP/permlane form)",
            body(&|w| {
                // (the sum's own scratch is v1 and v2: the running value is kept in v9)
                w.push(vop1(V_MOV_B32, 9, v(1)));
                for _ in 0..8 {
                    w.push(vop1(V_MOV_B32, 7, v(9)));
                    w.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&x| {
                        if x >> 25 == 0x1C {
                            x & 0x01FF_FFFF | V_ADD_NC_U32 << 25
                        } else {
                            x
                        }
                    }));
                    w.push(vop2(V_ADD_NC_U32, 9, 7, 9));
                }
                w.push(vop1(V_MOV_B32, 1, v(9)));
            }),
        ),
    ];
    let _one = one_at_a_time();
    let mut base = [0.0f64; 2];
    let mut rows = Vec::new();
    for (n, (name, words)) in workloads.iter().enumerate() {
        let mut ms = [f64::NAN; 2];
        for (m, dual) in [false, true].into_iter().enumerate() {
            DUAL.set(dual);
            let card = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| card(name, words)));
            DUAL.set(false);
            match card {
                Ok(Some(c)) => ms[m] = c.time([0, 1, u32::MAX, u32::MAX], GROUPS, REPS),
                Ok(None) => return,
                Err(_) => eprintln!(
                    "{name}: the {} form does not lower",
                    if dual { "dual-bank" } else { "paired" }
                ),
            }
        }
        if n == 0 {
            base = ms;
        }
        rows.push((name, ms));
    }
    eprintln!("\nms per dispatch of {GROUPS} groups (median of {REPS}), less the empty program's");
    eprintln!(
        "{:<58} {:>10} {:>10} {:>8}",
        "workload", "paired", "dual-bank", "ratio"
    );
    for (name, ms) in rows {
        let (p, d) = (ms[0] - base[0], ms[1] - base[1]);
        eprintln!("{name:<58} {p:>10.2} {d:>10.2} {:>8.2}", d / p);
    }
    eprintln!(
        "empty program itself: paired {:.2} ms, dual-bank {:.2} ms",
        base[0], base[1]
    );
}

/// Every lane the model claims, compared; the first few differences named.
fn compare(what: &str, got: &[u32], want: &[Option<u32>; 64]) {
    let wrong: Vec<String> = (0..64)
        .filter_map(|l| match want[l] {
            Some(w) if got[l] != w => Some(format!("lane {l}: got {:#x}, want {w:#x}", got[l])),
            _ => None,
        })
        .take(6)
        .collect();
    assert!(wrong.is_empty(), "{what}: {}", wrong.join("; "));
}

fn some(v: [u32; 64]) -> [Option<u32>; 64] {
    v.map(Some)
}

/// xorshift64*: the same "random" cases every run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// `ds_bpermute_b32` with lane g reading lane `((g ^ s0) + s1) mod 64`, under EXEC = s[2:3];
/// `in_place`: the destination is the data register itself (old values must be read first).
fn shuffle_program(in_place: bool) -> Vec<u32> {
    let mut w = prologue();
    w.extend([
        vop2(V_XOR_B32, 2, 0, 0),       // v2 = s0 ^ v0
        vop2(V_ADD_NC_U32, 2, 1, 2),    // v2 = s1 + v2
        vop2(V_AND_B32, 2, int(63), 2), // v2 &= 63
        vop2(V_LSHLREV_B32, 2, int(2), 2),
        sop1(S_MOV_B64, EXEC, 2), // EXEC = s[2:3]
    ]);
    if in_place {
        w.extend(ds(DS_BPERMUTE_B32, 1, 2, 1));
        w.push(S_WAITCNT_LGKM0);
        w.push(vop1(V_MOV_B32, 12, v(1)));
    } else {
        w.extend(ds(DS_BPERMUTE_B32, 12, 2, 1));
        w.push(S_WAITCNT_LGKM0);
    }
    epilogue(&mut w);
    w
}

fn shuffle_want(x: u32, y: u32, exec: u64) -> [Option<u32>; 64] {
    model::bpermute(
        &[DEAD; 64],
        &model::shuffle_addr(x, y),
        &model::input(),
        exec,
    )
}

fn split(m: u64) -> (u32, u32) {
    (m as u32, (m >> 32) as u32)
}

/// Identity, the 31/32 boundary both ways, XOR-1, XOR-32, and every (source, destination)
/// pair of the wave (4,096: 64 rotations), all lanes active; then the same in place.
#[test]
fn a_shuffle_reads_every_lane_of_the_console_wave() {
    let _one = one_at_a_time();
    for in_place in [false, true] {
        let what = if in_place {
            "ds_bpermute in place"
        } else {
            "ds_bpermute"
        };
        let Some(card) = card(what, &shuffle_program(in_place)) else {
            return;
        };
        let all = u64::MAX;
        for (x, y, name) in [
            (0, 0, "identity"),
            (0, 1, "31 reads 32"),
            (0, 63, "32 reads 31"),
            (1, 0, "XOR-1"),
            (32, 0, "XOR-32"),
            (63, 0, "XOR-63"),
        ] {
            let got = card.run([x, y, u32::MAX, u32::MAX]);
            compare(&format!("{what}, {name}"), &got, &shuffle_want(x, y, all));
        }
        for y in 0..64 {
            let got = card.run([0, y, u32::MAX, u32::MAX]);
            compare(
                &format!("{what}, rotation {y}"),
                &got,
                &shuffle_want(0, y, all),
            );
        }
        eprintln!("{what}: identity, 31<->32, XOR-1/32/63 and all 4,096 pairs as the model");
    }
}

/// The same shuffles under random EXEC masks (inactive lanes keep their value; a lane whose
/// source is inactive is not compared), with random XOR and rotation.
#[test]
fn a_shuffle_under_random_masks_keeps_the_inactive_lanes() {
    let _one = one_at_a_time();
    for in_place in [false, true] {
        let what = if in_place {
            "ds_bpermute in place, random masks"
        } else {
            "ds_bpermute, random masks"
        };
        let Some(card) = card(what, &shuffle_program(in_place)) else {
            return;
        };
        let mut rng = Rng(0x5EED_0000_0000_0001 + u64::from(in_place));
        for case in 0..64 {
            let exec = match case {
                0 => 0,
                1 => 0xFFFF_FFFF,
                2 => 0xFFFF_FFFF_0000_0000,
                3 => 1 << 31 | 1 << 32,
                _ => rng.next() & rng.next() | rng.next() & 1 << (rng.next() % 64),
            };
            let (x, y) = ((rng.next() % 64) as u32, (rng.next() % 64) as u32);
            let (lo, hi) = split(exec);
            let got = card.run([x, y, lo, hi]);
            compare(
                &format!("{what}, exec {exec:#018x}, x {x}, y {y}"),
                &got,
                &shuffle_want(x, y, exec),
            );
        }
        eprintln!("{what}: 64 cases as the model");
    }
}

/// `v_readlane_b32` from lanes 0, 31, 32, 63 and random ones, under random EXEC (the lanes
/// active take the value), including lanes read while inactive.
#[test]
fn a_lane_read_reaches_both_halves() {
    let mut w = prologue();
    w.push(sop1(S_MOV_B64, EXEC, 2));
    w.extend(vop3(V_READLANE_B32, 20, v(1), 0));
    w.push(vop1(V_MOV_B32, 12, 20));
    epilogue(&mut w);
    let _one = one_at_a_time();
    let Some(card) = card("v_readlane", &w) else {
        return;
    };
    let mut rng = Rng(0x5EED_0000_0000_0002);
    for case in 0..40u32 {
        let lane = if case < 4 {
            [0, 31, 32, 63][case as usize]
        } else {
            (rng.next() % 64) as u32
        };
        let exec = if case < 8 { u64::MAX } else { rng.next() | 1 };
        let (lo, hi) = split(exec);
        let got = card.run([lane, 0, lo, hi]);
        let value = model::readlane(&model::input(), lane);
        compare(
            &format!("v_readlane lane {lane}, exec {exec:#x}"),
            &got,
            &some(model::masked(&[DEAD; 64], &[value; 64], exec)),
        );
    }
    eprintln!("v_readlane: 40 cases as the model");
}

/// `v_writelane_b32` into lanes 31, 32 and random ones, under random EXEC (it ignores EXEC).
#[test]
fn a_lane_write_reaches_both_halves() {
    let mut w = prologue();
    w.push(vop1(V_MOV_B32, 12, v(1)));
    w.push(sop1(S_MOV_B64, EXEC, 2));
    w.extend(vop3(V_WRITELANE_B32, 12, 1, 0));
    epilogue(&mut w);
    let _one = one_at_a_time();
    let Some(card) = card("v_writelane", &w) else {
        return;
    };
    let mut rng = Rng(0x5EED_0000_0000_0003);
    for case in 0..40u32 {
        let lane = if case < 4 {
            [31, 32, 0, 63][case as usize]
        } else {
            (rng.next() % 64) as u32
        };
        let exec = if case < 4 { u64::MAX } else { rng.next() };
        let (lo, hi) = split(exec);
        let value = 0xC0DE_0000 | case;
        let got = card.run([lane, value, lo, hi]);
        compare(
            &format!("v_writelane lane {lane}, exec {exec:#x}"),
            &got,
            &some(model::writelane(&model::input(), value, lane)),
        );
    }
    eprintln!("v_writelane: 40 cases as the model");
}

/// `v_readfirstlane_b32` under masks whose first lane is in either half, among them the
/// handover's high-half-only mask (lane 37).
#[test]
fn the_first_active_lane_is_found_in_either_half() {
    let mut w = prologue();
    w.push(sop1(S_MOV_B64, EXEC, 2));
    w.push(vop1(V_READFIRSTLANE_B32, 20, v(1)));
    w.push(sop1(S_MOV_B64, EXEC, 22));
    w.push(vop1(V_MOV_B32, 12, 20));
    w.extend(STORE_V12);
    let _one = one_at_a_time();
    let Some(card) = card("v_readfirstlane", &w) else {
        return;
    };
    let mut rng = Rng(0x5EED_0000_0000_0004);
    for case in 0..40u32 {
        let exec = match case {
            0 => 1 << 37,
            1 => 1 << 32,
            2 => 1 << 31 | 1 << 63,
            3 => u64::MAX,
            _ => rng.next() >> (rng.next() % 64) | 1 << 63,
        };
        let (lo, hi) = split(exec);
        let got = card.run([0, 0, lo, hi]);
        let first = model::readfirstlane(&model::input(), exec);
        compare(
            &format!("v_readfirstlane exec {exec:#x}"),
            &got,
            &[first; 64],
        );
    }
    eprintln!("v_readfirstlane: 40 cases as the model");
}

/// A ballot through `v_cmp` (lanes g >= s0 among the active ones), read as data: its two
/// halves, `s_ff1_i32_b64` and `s_bcnt1_i32_b64` of it.
#[test]
fn a_ballot_holds_both_halves() {
    let mut w = prologue();
    w.push(sop1(S_MOV_B64, EXEC, 2));
    w.push(vopc(V_CMP_LE_U32, 0, 0)); // vcc = s0 <= v0
    w.push(sop1(S_MOV_B64, EXEC, 22));
    w.push(sop1(S_FF1_I32_B64, 24, VCC_LO));
    w.push(sop1(S_BCNT1_I32_B64, 25, VCC_LO));
    w.push(vop1(V_MOV_B32, 3, VCC_LO));
    w.push(vop1(V_MOV_B32, 4, VCC_HI));
    w.push(vop1(V_MOV_B32, 5, 24));
    w.push(vop1(V_MOV_B32, 6, 25));
    for (i, r) in [3, 4, 5, 6].into_iter().enumerate() {
        w.extend(store_at(r, 4 * i as u32));
    }
    let _one = one_at_a_time();
    let Some(card) = card("ballot", &w) else {
        return;
    };
    let set = |lanes: &[u32]| lanes.iter().fold(0u64, |m, &l| m | 1 << l);
    let mut rng = Rng(0x5EED_0000_0000_0005);
    for case in 0..48u32 {
        let (exec, from) = match case {
            0 => (set(&[0, 31, 32, 63]), 0),
            1 => (1 << 37, 0),
            2 => (u64::MAX, 0),
            3 => (u64::MAX, 32),
            4 => (0, 0),
            _ => (rng.next(), (rng.next() % 65) as u32),
        };
        let (lo, hi) = split(exec);
        let got = card.run([from, 0, lo, hi]);
        let b = model::ballot(exec, |l| from as usize <= l);
        let want = [
            b as u32,
            (b >> 32) as u32,
            model::ff1(b) as u32,
            b.count_ones(),
        ];
        assert_eq!(
            &got[..4],
            &want,
            "ballot of g >= {from} over exec {exec:#x}: [lo, hi, ff1, bcnt]"
        );
    }
    eprintln!(
        "ballot: 48 cases as the model (the handover's {{0,31,32,63}} gives 0x80000001 twice)"
    );
}

/// `v_mbcnt_lo` then `v_mbcnt_hi` over a mask in s[0:1]: each lane's exclusive prefix count,
/// all ones first (lane g gets g).
#[test]
fn a_prefix_count_runs_across_the_halves() {
    let mut w = prologue();
    w.extend(vop3(V_MBCNT_LO, 15, 0, int(0)));
    w.extend(vop3(V_MBCNT_HI, 15, 1, v(15)));
    w.push(vop1(V_MOV_B32, 12, v(15)));
    epilogue(&mut w);
    let _one = one_at_a_time();
    let Some(card) = card("v_mbcnt", &w) else {
        return;
    };
    let mut rng = Rng(0x5EED_0000_0000_0006);
    for case in 0..40 {
        let mask = match case {
            0 => u64::MAX,
            1 => 0,
            2 => 1 << 31 | 1 << 32,
            _ => rng.next(),
        };
        let (lo, hi) = split(mask);
        let got = card.run([lo, hi, u32::MAX, u32::MAX]);
        compare(
            &format!("v_mbcnt over {mask:#x}"),
            &got,
            &some(model::mbcnt(mask)),
        );
    }
    eprintln!("v_mbcnt: 40 cases as the model (all ones: lane g gets g)");
}

/// The console compiler's whole-wave sum (identity in inactive lanes, DPP, `v_permlanex16`,
/// `v_readlane` 32 and 0) of the lane ids: 2016 in every lane.
#[test]
fn a_whole_wave_sum_of_lane_ids_is_2016() {
    let mut w = prologue();
    w.push(vop1(V_MOV_B32, 7, v(0)));
    // (the compiler's wave OR with v_add_nc_u32 for v_or_b32)
    w.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&x| {
        if x >> 25 == 0x1C {
            x & 0x01FF_FFFF | V_ADD_NC_U32 << 25
        } else {
            x
        }
    }));
    w.push(vop1(V_MOV_B32, 12, 7));
    epilogue(&mut w);
    let _one = one_at_a_time();
    let Some(card) = card("wave sum", &w) else {
        return;
    };
    let got = card.run([0; 4]);
    compare("wave sum of lane ids", &got, &[Some(2016); 64]);
    eprintln!("wave sum: 2016 in every lane");
}

/// Divergence: EXEC = the lanes below s0 (`v_cmpx`), a write, EXEC restored. The lanes left
/// out keep their old value, on both sides of the 31/32 boundary.
#[test]
fn lanes_left_out_by_exec_keep_their_values() {
    let mut w = prologue();
    w.push(vopc(V_CMPX_GT_U32, 0, 0)); // exec = s0 > v0
    w.push(vop1(V_MOV_B32, 12, v(1)));
    epilogue(&mut w);
    let _one = one_at_a_time();
    let Some(card) = card("v_cmpx divergence", &w) else {
        return;
    };
    for below in [0u32, 1, 31, 32, 33, 63, 64] {
        let got = card.run([below, 0, 0, 0]);
        let exec = if below >= 64 {
            u64::MAX
        } else {
            (1u64 << below) - 1
        };
        compare(
            &format!("lanes below {below}"),
            &got,
            &some(model::masked(&[DEAD; 64], &model::input(), exec)),
        );
    }
    eprintln!("v_cmpx: lanes 0, 1, 31, 32, 33, 63, 64 active as the model");
}

/// Partial groups and several console waves per group: 48 threads (one console wave whose
/// upper host wave has 16 threads) and 96 (two console waves, the second with 32). Each
/// console wave shuffles among its own lanes; a lane that reads a lane the group does not
/// have is not compared (not there on the console: outside EXEC). The barriers must not
/// hang with a host wave short.
#[test]
fn a_shuffle_in_partial_groups_stays_in_its_console_wave() {
    let _one = one_at_a_time();
    for threads in [48u32, 96] {
        let what = format!("ds_bpermute, {threads} threads");
        let Some(card) = card_of(&what, &shuffle_program(false), threads) else {
            return;
        };
        let mut rng = Rng(0x5EED_0000_0000_0007 + u64::from(threads));
        for case in 0..32 {
            let (x, y) = match case {
                0 => (0, 1),
                1 => (0, 63),
                2 => (32, 0),
                _ => ((rng.next() % 64) as u32, (rng.next() % 64) as u32),
            };
            let got = card.run([x, y, u32::MAX, u32::MAX]);
            for wave in 0..threads.div_ceil(64) {
                let first = wave * 64;
                let lanes = (threads - first).min(64);
                let exec = if lanes == 64 {
                    u64::MAX
                } else {
                    (1u64 << lanes) - 1
                };
                let data: [u32; 64] = std::array::from_fn(|l| 0x1000 + first + l as u32);
                let mut want =
                    model::bpermute(&[DEAD; 64], &model::shuffle_addr(x, y), &data, exec);
                for (l, w) in want.iter_mut().enumerate() {
                    if l as u32 >= lanes {
                        *w = None;
                    }
                }
                let got_wave: Vec<u32> = (0..64)
                    .map(|l| got.get((first + l) as usize).copied().unwrap_or(0))
                    .collect();
                compare(
                    &format!("{what}, console wave {wave}, x {x}, y {y}"),
                    &got_wave,
                    &want,
                );
            }
        }
        eprintln!("{what}: 32 cases as the model in every console wave");
    }
}

thread_local! {
    /// The cards of this thread's tests take the dual-bank form ([`crate::dualbank`]).
    static DUAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Every lane test once more with the console wave in one host wave of 32, two console lanes
/// per host lane (dual-bank), against the same model.
mod dual_bank {
    fn dual(test: fn()) {
        super::DUAL.set(true);
        test();
        super::DUAL.set(false);
    }
    #[test]
    fn a_shuffle_reads_every_lane_of_the_console_wave() {
        dual(super::a_shuffle_reads_every_lane_of_the_console_wave);
    }
    #[test]
    fn a_shuffle_under_random_masks_keeps_the_inactive_lanes() {
        dual(super::a_shuffle_under_random_masks_keeps_the_inactive_lanes);
    }
    #[test]
    fn a_lane_read_reaches_both_halves() {
        dual(super::a_lane_read_reaches_both_halves);
    }
    #[test]
    fn a_lane_write_reaches_both_halves() {
        dual(super::a_lane_write_reaches_both_halves);
    }
    #[test]
    fn the_first_active_lane_is_found_in_either_half() {
        dual(super::the_first_active_lane_is_found_in_either_half);
    }
    #[test]
    fn a_ballot_holds_both_halves() {
        dual(super::a_ballot_holds_both_halves);
    }
    #[test]
    fn a_prefix_count_runs_across_the_halves() {
        dual(super::a_prefix_count_runs_across_the_halves);
    }
    #[test]
    fn a_whole_wave_sum_of_lane_ids_is_2016() {
        dual(super::a_whole_wave_sum_of_lane_ids_is_2016);
    }
    #[test]
    fn lanes_left_out_by_exec_keep_their_values() {
        dual(super::lanes_left_out_by_exec_keep_their_values);
    }
    #[test]
    fn a_shuffle_in_partial_groups_stays_in_its_console_wave() {
        dual(super::a_shuffle_in_partial_groups_stays_in_its_console_wave);
    }
}

/// A per-lane if/else as the console compiler writes it: `v_cmp`, `s_and_saveexec_b64`,
/// `s_cbranch_execz` past the `then` lanes, `s_andn2_b64` to the `else` lanes and
/// `s_cbranch_execz` past them, EXEC restored. Lanes below s0 take v1, the others 7, on
/// both sides of the 31/32 boundary, including when one side has no lane (both branches
/// taken).
#[test]
fn an_if_else_on_lane_masks_runs_each_side_in_its_lanes() {
    use crate::gfx10::{Encoding, opcode};
    let sop1 = |name: &str, sdst: u32, ssrc0: u32| {
        0xBE80_0000 | sdst << 16 | u32::from(opcode(Encoding::Sop1, name).expect(name)) << 8 | ssrc0
    };
    let sop2 = |name: &str, sdst: u32, ssrc0: u32, ssrc1: u32| {
        0x8000_0000
            | u32::from(opcode(Encoding::Sop2, name).expect(name)) << 23
            | sdst << 16
            | ssrc1 << 8
            | ssrc0
    };
    let execz =
        0xBF80_0000 | u32::from(opcode(Encoding::Sopp, "s_cbranch_execz").expect("execz")) << 16;
    let mut w = prologue();
    w.push(vopc(0xC4, 0, 0)); // v_cmp_gt_u32 vcc, s0, v0
    w.push(sop1("s_and_saveexec_b64", 20, VCC_LO)); // s[20:21] = exec; exec &= vcc
    let skip_then = w.len();
    w.push(execz);
    w.push(vop1(V_MOV_B32, 12, v(1)));
    let else_at = w.len();
    w.push(sop2("s_andn2_b64", EXEC, 20, EXEC)); // exec = saved & ~exec
    let skip_else = w.len();
    w.push(execz);
    w.push(vop1(V_MOV_B32, 12, int(7)));
    let end = w.len();
    epilogue(&mut w);
    let rel = |from: usize, to: usize| (to as i32 - (from as i32 + 1)) as u16 as u32;
    w[skip_then] |= rel(skip_then, else_at);
    w[skip_else] |= rel(skip_else, end);
    let _one = one_at_a_time();
    let Some(card) = card("if/else", &w) else {
        return;
    };
    for below in [0u32, 1, 31, 32, 33, 63, 64] {
        let got = card.run([below, 0, 0, 0]);
        let want: [u32; 64] = std::array::from_fn(|l| {
            if (l as u32) < below {
                0x1000 + l as u32
            } else {
                7
            }
        });
        compare(
            &format!("if/else, lanes below {below} take v1"),
            &got,
            &some(want),
        );
    }
    eprintln!("if/else: lanes 0, 1, 31, 32, 33, 63, 64 as the model");
}

#[test]
fn dual_bank_an_if_else_on_lane_masks_runs_each_side_in_its_lanes() {
    DUAL.set(true);
    an_if_else_on_lane_masks_runs_each_side_in_its_lanes();
    DUAL.set(false);
}

/// 3D thread groups: lanes follow the linear thread id `x + X * (y + Y * z)` (x fastest), so
/// each thread computes its linear id from v0, v1, v2 and shuffles by it. Groups of 4x4x4
/// (one console wave) and 8x4x4 (two). Every thread must have its own (x, y, z), and each
/// console wave's shuffle stays among its own lanes.
fn three_d(threads: [u32; 3]) {
    let [x, y, _] = threads;
    let (lx, ly) = (x.trailing_zeros(), y.trailing_zeros());
    let mut w = vec![
        vop2(V_LSHLREV_B32, 3, int(lx + ly), 2), // v3 = z << (lx + ly)
        vop2(V_LSHLREV_B32, 4, int(lx), 1),      // v4 = y << lx
        vop2(V_ADD_NC_U32, 3, v(4), 3),
        vop2(V_ADD_NC_U32, 3, v(0), 3), // v3 = the linear id
        vop2(V_ADD_NC_U32, 1, LITERAL, 3),
        0x1000,
        vop2(V_LSHLREV_B32, 14, int(2), 3),
        vop1(V_MOV_B32, 12, LITERAL),
        DEAD,
        sop1(S_MOV_B64, 22, EXEC),
        vop2(V_XOR_B32, 2, 0, 3), // v2 = s0 ^ lin
        vop2(V_ADD_NC_U32, 2, 1, 2),
        vop2(V_AND_B32, 2, int(63), 2),
        vop2(V_LSHLREV_B32, 2, int(2), 2),
        sop1(S_MOV_B64, EXEC, 2),
    ];
    w.extend(ds(DS_BPERMUTE_B32, 12, 2, 1));
    w.push(S_WAITCNT_LGKM0);
    epilogue(&mut w);
    let group: u32 = threads.iter().product();
    let what = format!("3D group {threads:?}");
    let Some(card) = card_with(&what, &w, move |n| {
        if let crate::native::StageIo::Cs(cs) = &mut n.io {
            cs.threads = threads;
            cs.rsrc2 |= 2 << 11; // v0, v1, v2: x, y, z
        }
    }) else {
        return;
    };
    let mut rng = Rng(0x5EED_0000_0000_0008 + u64::from(group));
    for case in 0..24 {
        let (sx, sy) = match case {
            0 => (0, 0),
            1 => (0, 1),
            2 => (32, 0),
            _ => ((rng.next() % 64) as u32, (rng.next() % 64) as u32),
        };
        let got = card.run([sx, sy, u32::MAX, u32::MAX]);
        for wave in 0..group / 64 {
            let first = wave * 64;
            let data: [u32; 64] = std::array::from_fn(|l| 0x1000 + first + l as u32);
            let want = model::bpermute(&[DEAD; 64], &model::shuffle_addr(sx, sy), &data, u64::MAX);
            let got_wave: Vec<u32> = (0..64).map(|l| got[(first + l) as usize]).collect();
            compare(
                &format!("{what}, console wave {wave}, x {sx}, y {sy}"),
                &got_wave,
                &want,
            );
        }
    }
    eprintln!("{what}: 24 cases as the model in every console wave");
}

#[test]
fn three_dimensional_groups_number_lanes_x_first() {
    let _one = one_at_a_time();
    three_d([4, 4, 4]);
    three_d([8, 4, 4]);
}

#[test]
fn dual_bank_three_dimensional_groups_number_lanes_x_first() {
    DUAL.set(true);
    three_dimensional_groups_number_lanes_x_first();
    DUAL.set(false);
}

/// A loop whose trip count is per lane, `(lane & 7) + (lane >> 5)` (so the halves differ),
/// leaving it lane by lane (`s_and_b64 exec, exec, vcc`, `s_cbranch_execz`), with an if
/// inside it on odd iterations (`s_and_saveexec_b64`, `s_cbranch_execz`, `s_or_b64` back).
/// Each lane adds 1 per iteration and 100 more on odd ones.
#[test]
fn a_loop_with_an_if_inside_runs_each_lane_its_own_count() {
    use crate::gfx10::{Encoding, opcode};
    let sop1n = |name: &str, sdst: u32, ssrc0: u32| {
        0xBE80_0000 | sdst << 16 | u32::from(opcode(Encoding::Sop1, name).expect(name)) << 8 | ssrc0
    };
    let sop2n = |name: &str, sdst: u32, ssrc0: u32, ssrc1: u32| {
        0x8000_0000
            | u32::from(opcode(Encoding::Sop2, name).expect(name)) << 23
            | sdst << 16
            | ssrc1 << 8
            | ssrc0
    };
    let sopp =
        |name: &str| 0xBF80_0000 | u32::from(opcode(Encoding::Sopp, name).expect(name)) << 16;
    let mut w = prologue();
    w.extend([
        vop1(V_MOV_B32, 12, int(0)),    // acc
        vop1(V_MOV_B32, 6, int(0)),     // i
        vop2(0x16, 8, int(5), 0),       // v_lshrrev_b32 v8, 5, v0
        vop2(V_AND_B32, 5, int(7), 0),  // v5 = lane & 7
        vop2(V_ADD_NC_U32, 5, v(8), 5), // t = (lane & 7) + (lane >> 5)
    ]);
    let top = w.len();
    w.push(vopc(0xC1, v(6), 5)); // v_cmp_lt_u32 vcc, v6, v5: i < t
    w.push(sop2n("s_and_b64", EXEC, EXEC, VCC_LO));
    let leave = w.len();
    w.push(sopp("s_cbranch_execz"));
    w.push(vop2(V_AND_B32, 7, int(1), 6));
    w.push(vopc(0xC5, int(0), 7)); // v_cmp_ne_u32 vcc, 0, v7: i odd
    w.push(sop1n("s_and_saveexec_b64", 24, VCC_LO));
    let skip = w.len();
    w.push(sopp("s_cbranch_execz"));
    w.push(vop2(V_ADD_NC_U32, 12, int(64), 12));
    w.push(vop2(V_ADD_NC_U32, 12, int(36), 12)); // + 100
    let join = w.len();
    w.push(sop2n("s_or_b64", EXEC, EXEC, 24));
    w.push(vop2(V_ADD_NC_U32, 12, int(1), 12));
    w.push(vop2(V_ADD_NC_U32, 6, int(1), 6));
    let back = w.len();
    w.push(sopp("s_branch"));
    let end = w.len();
    epilogue(&mut w);
    let rel = |from: usize, to: usize| (to as i32 - (from as i32 + 1)) as u16 as u32;
    w[leave] |= rel(leave, end);
    w[skip] |= rel(skip, join);
    w[back] |= rel(back, top);
    let _one = one_at_a_time();
    let Some(card) = card("loop with an if", &w) else {
        return;
    };
    let got = card.run([0; 4]);
    let want: [u32; 64] = std::array::from_fn(|l| {
        let t = (l as u32 & 7) + (l as u32 >> 5);
        (0..t).map(|i| 1 + if i & 1 == 1 { 100 } else { 0 }).sum()
    });
    compare("loop with an if inside", &got, &some(want));
    eprintln!("loop with an if: every lane its own count, as the model");
}

#[test]
fn dual_bank_a_loop_with_an_if_inside_runs_each_lane_its_own_count() {
    DUAL.set(true);
    a_loop_with_an_if_inside_runs_each_lane_its_own_count();
    DUAL.set(false);
}

/// A far jump (`s_getpc_b64`, `s_add_u32` of a literal, `s_addc_u32`, `s_setpc_b64`), the form
/// the console's compiler gives a branch past a 16-bit offset's reach (the early-out of the
/// cascaded shadows' compute program, 169 KB of code). With s0 0 the jump goes to code that
/// gives every lane 9; else lanes below s0 take 7 from SCC (`s_cselect_b32` after the branch
/// on it: SCC must come through), the others keep their value.
#[test]
fn a_far_jump_goes_where_it_points() {
    use crate::gfx10::{Encoding, opcode};
    let sop1n = |name: &str, sdst: u32, ssrc0: u32| {
        0xBE80_0000 | sdst << 16 | u32::from(opcode(Encoding::Sop1, name).expect(name)) << 8 | ssrc0
    };
    let sop2n = |name: &str, sdst: u32, ssrc0: u32, ssrc1: u32| {
        0x8000_0000
            | u32::from(opcode(Encoding::Sop2, name).expect(name)) << 23
            | sdst << 16
            | ssrc1 << 8
            | ssrc0
    };
    let sopp =
        |name: &str| 0xBF80_0000 | u32::from(opcode(Encoding::Sopp, name).expect(name)) << 16;
    let cmp = u32::from(opcode(Encoding::Sopc, "s_cmp_lg_u32").expect("s_cmp_lg_u32"));
    let mut w = prologue();
    w.push(0xBF00_0000 | cmp << 16 | int(0) << 8); // s_cmp_lg_u32 s0, 0
    let over = w.len();
    w.push(sopp("s_cbranch_scc1"));
    let pc = w.len();
    w.push(sop1n("s_getpc_b64", 30, 0));
    w.push(sop2n("s_add_u32", 30, 30, LITERAL));
    let lit = w.len();
    w.push(0);
    w.push(sop2n("s_addc_u32", 31, 31, int(0)));
    w.push(sop1n("s_setpc_b64", 0, 30));
    let then = w.len();
    w.push(sop2n("s_cselect_b32", 40, int(7), int(8)));
    w.push(vopc(0xC4, 0, 0)); // v_cmp_gt_u32 vcc, s0, v0
    w.push(sop1n("s_and_saveexec_b64", 20, VCC_LO));
    w.push(vop1(V_MOV_B32, 12, 40));
    let to_end = w.len();
    w.push(sopp("s_branch"));
    let none = w.len();
    w.push(vop1(V_MOV_B32, 12, int(9)));
    let end = w.len();
    epilogue(&mut w);
    let rel = |from: usize, to: usize| (to as i32 - (from as i32 + 1)) as u16 as u32;
    w[over] |= rel(over, then);
    w[to_end] |= rel(to_end, end);
    w[lit] = 4 * (none - (pc + 1)) as u32;
    let _one = one_at_a_time();
    let Some(card) = card("far jump", &w) else {
        return;
    };
    for below in [0u32, 1, 32, 33, 64] {
        let got = card.run([below, 0, 0, 0]);
        let want: [u32; 64] = std::array::from_fn(|l| match below {
            0 => 9,
            _ if (l as u32) < below => 7,
            _ => DEAD,
        });
        compare(&format!("far jump, lanes below {below}"), &got, &some(want));
    }
    eprintln!("far jump: lanes below 0, 1, 32, 33, 64 as the model");
}

#[test]
fn dual_bank_a_far_jump_goes_where_it_points() {
    DUAL.set(true);
    a_far_jump_goes_where_it_points();
    DUAL.set(false);
}

/// Dual-bank with every branch written as a far jump ([`crate::dualbank`]: what a branch the
/// rewrite has put out of its offset's reach becomes), forward and backward, on EXEC and
/// unconditional, against the same model.
#[test]
fn dual_bank_branches_as_far_jumps_run_as_the_short_ones() {
    crate::dualbank::ALL_FAR.set(true);
    DUAL.set(true);
    a_far_jump_goes_where_it_points();
    a_loop_with_an_if_inside_runs_each_lane_its_own_count();
    DUAL.set(false);
    crate::dualbank::ALL_FAR.set(false);
}

/// The model of a DPP16 `row_shr:n` compare `src0 > vsrc1` (unsigned) over a 64-lane wave:
/// lane l reads src0 of lane l - n in its row of 16; a lane whose source is past the row's
/// start takes 0 with bound_ctrl, else is not written (its bit 0); rows off in `rows` and
/// lanes outside EXEC are not written either. Returns the bits, and the lanes not claimed:
/// a source lane in the row but outside EXEC (with FI clear the ISA treats it as out of
/// range; the translator reads it as it is; not settled, so not compared).
fn dpp_shr_compare(
    src0: &[u32; 64],
    vsrc1: &[u32; 64],
    n: usize,
    bound: bool,
    rows: u32,
    exec: u64,
) -> (u64, u64) {
    let (mut bits, mut unknown) = (0u64, 0u64);
    for l in 0..64 {
        if !model::active(exec, l) || rows >> (l / 16) & 1 == 0 {
            continue;
        }
        let from = (l % 16).checked_sub(n).map(|k| l / 16 * 16 + k);
        let value = match from {
            Some(s) if model::active(exec, s) => Some(src0[s]),
            Some(_) => {
                unknown |= 1 << l;
                continue;
            }
            None if bound => Some(0),
            None => None,
        };
        if value.is_some_and(|v| v > vsrc1[l]) {
            bits |= 1 << l;
        }
    }
    (bits, unknown)
}

/// `v_cmp_gt_u32` (or `v_cmpx_gt_u32`) with DPP16 `row_shr:1` on src0 = the lane number,
/// vsrc1 = s0, under EXEC = s[2:3]. The compare: VCC's two halves at dwords 0 and 1. The
/// `v_cmpx`: v12 = 1 in the lanes the new EXEC keeps.
fn dpp_compare_program(cmpx: bool, bound: bool, rows: u32) -> Vec<u32> {
    let ext = 2 | 0x111 << 8 | u32::from(bound) << 19 | 0xF << 24 | rows << 28; // src0 v2, row_shr:1
    let mut w = prologue();
    w.push(vop1(V_MOV_B32, 2, v(0)));
    w.push(vop1(V_MOV_B32, 4, 0)); // v4 = s0
    w.push(sop1(S_MOV_B64, EXEC, 2));
    let op = if cmpx { V_CMPX_GT_U32 } else { 0xC4 };
    w.push(0x7C00_0000 | op << 17 | 4 << 9 | 0xFA);
    w.push(ext);
    if cmpx {
        w.push(vop1(V_MOV_B32, 12, int(1)));
        epilogue(&mut w);
    } else {
        w.push(sop1(S_MOV_B64, EXEC, 22));
        w.push(vop1(V_MOV_B32, 3, VCC_LO));
        w.push(vop1(V_MOV_B32, 5, VCC_HI));
        w.extend(store_at(3, 0));
        w.extend(store_at(5, 4));
    }
    w
}

#[test]
fn a_dpp_compare_reads_its_row_neighbour() {
    let _one = one_at_a_time();
    let lanes: [u32; 64] = std::array::from_fn(|l| l as u32);
    for cmpx in [false, true] {
        for (bound, rows) in [(true, 0xF), (false, 0xF), (true, 0b0101), (false, 0b1010)] {
            let what = format!(
                "DPP row_shr:1 {} bound_ctrl {bound} rows {rows:#06b}",
                if cmpx { "v_cmpx" } else { "v_cmp" }
            );
            let Some(card) = card(&what, &dpp_compare_program(cmpx, bound, rows)) else {
                return;
            };
            let mut rng = Rng(0x5EED_0000_0000_0009
                + u64::from(rows)
                + u64::from(bound) * 16
                + u64::from(cmpx) * 32);
            for case in 0..24 {
                let exec = if case < 2 { u64::MAX } else { rng.next() };
                let threshold = [0u32, 7, 20, 40][case % 4];
                let (lo, hi) = split(exec);
                let got = card.run([threshold, 0, lo, hi]);
                let (want, unknown) =
                    dpp_shr_compare(&lanes, &[threshold; 64], 1, bound, rows, exec);
                if cmpx {
                    let expect: [Option<u32>; 64] = std::array::from_fn(|l| {
                        (unknown >> l & 1 == 0).then_some(if want >> l & 1 == 1 { 1 } else { DEAD })
                    });
                    compare(
                        &format!("{what}, exec {exec:#x}, s0 {threshold}"),
                        &got,
                        &expect,
                    );
                } else {
                    let got64 = u64::from(got[0]) | u64::from(got[1]) << 32;
                    assert_eq!(
                        got64 & !unknown,
                        want,
                        "{what}, exec {exec:#x}, s0 {threshold}: VCC (lanes {unknown:#x} not compared)"
                    );
                }
            }
            eprintln!("{what}: 24 cases as the model");
        }
    }
}

#[test]
fn dual_bank_a_dpp_compare_reads_its_row_neighbour() {
    DUAL.set(true);
    a_dpp_compare_reads_its_row_neighbour();
    DUAL.set(false);
}

/// A program of 156 VGPRs, as the volumetric fog program with local lights has: through
/// dual-bank its bank 1 lies past the machine word's 256 ([`crate::dualbank::Banks`]). The
/// shuffle's data goes through v155, and lane 37's value of it is added to every lane.
#[test]
fn a_program_of_156_vgprs_runs_as_the_model() {
    let mut w = prologue();
    w.extend([
        vop1(V_MOV_B32, 155, v(1)),
        vop2(V_XOR_B32, 2, 0, 0),       // v2 = s0 ^ v0
        vop2(V_ADD_NC_U32, 2, 1, 2),    // v2 = s1 + v2
        vop2(V_AND_B32, 2, int(63), 2), // v2 &= 63
        vop2(V_LSHLREV_B32, 2, int(2), 2),
    ]);
    w.extend(ds(DS_BPERMUTE_B32, 150, 2, 155));
    w.push(S_WAITCNT_LGKM0);
    w.extend(vop3(V_READLANE_B32, 40, v(155), int(37))); // s40 = lane 37 of v155
    w.push(vop2(V_ADD_NC_U32, 12, 40, 150)); // v12 = s40 + v150
    epilogue(&mut w);
    let _one = one_at_a_time();
    let Some(card) = card_with("156 VGPRs", &w, |n| n.vgprs = 156) else {
        return;
    };
    for (x, y) in [(0, 0), (0, 1), (32, 0), (5, 17)] {
        let got = card.run([x, y, 0, 0]);
        let want = shuffle_want(x, y, u64::MAX).map(|l| l.map(|d| d + 0x1000 + 37));
        compare(&format!("156 VGPRs, x {x}, y {y}"), &got, &want);
    }
    eprintln!("156 VGPRs: shuffles and a lane read as the model");
}

#[test]
fn dual_bank_a_program_of_156_vgprs_runs_as_the_model() {
    DUAL.set(true);
    a_program_of_156_vgprs_runs_as_the_model();
    DUAL.set(false);
}

/// A lane pair of a VGPR the compiler parks scalars in holding a lane mask, then one of its
/// lanes a scalar alone (as the underwater volumetric fog program reuses lanes 18 and 19 of
/// v128): the save and the restore that come together move as a pair, the lone lane as a
/// scalar (`ir::unspill`'s pairs, for dual-bank programs where some saves and restores of the
/// two lanes come together), so the mask comes back as a mask. Restored where EXEC holds fewer
/// lanes than it; lanes below s0 and s2 take v1, every lane adds s1 restored from lane 4 alone.
#[test]
fn dual_bank_a_parked_mask_beside_a_parked_scalar_comes_back_as_a_mask() {
    use crate::gfx10::{Encoding, opcode};
    let vop3n = |name: &str, vdst: u32, src0: u32, src1: u32| {
        let o = u32::from(opcode(Encoding::Vop3, name).expect(name));
        [0xD400_0000 | o << 16 | vdst, src1 << 9 | src0]
    };
    let mut w = prologue();
    w.push(vopc(0xC4, 0, 0)); // v_cmp_gt_u32 vcc, s0, v0
    w.push(sop1(S_MOV_B64, 26, VCC_LO));
    w.extend(vop3n("v_writelane_b32", 15, 26, int(4)));
    w.extend(vop3n("v_writelane_b32", 15, 27, int(5)));
    w.push(vopc(0xC4, 2, 0)); // v_cmp_gt_u32 vcc, s2, v0
    let saveexec = u32::from(opcode(Encoding::Sop1, "s_and_saveexec_b64").expect("saveexec"));
    w.push(sop1(saveexec, 20, VCC_LO)); // EXEC = the lanes below s2
    w.extend(vop3n("v_readlane_b32", 24, v(15), int(4)));
    w.extend(vop3n("v_readlane_b32", 25, v(15), int(5)));
    let and = u32::from(opcode(Encoding::Sop2, "s_and_b64").expect("s_and_b64"));
    w.push(0x8000_0000 | and << 23 | EXEC << 16 | 24 << 8 | EXEC); // and below s0
    w.push(vop1(V_MOV_B32, 12, v(1)));
    w.push(sop1(S_MOV_B64, EXEC, 22));
    w.extend(vop3n("v_writelane_b32", 15, 1, int(4))); // lane 4 alone: s1
    w.extend(vop3n("v_readlane_b32", 31, v(15), int(4)));
    w.push(vop2(V_ADD_NC_U32, 12, 31, 12)); // v12 += s31
    epilogue(&mut w);
    // the pair's save (from s[26:27]) and restore (to s[24:25]) as two `s_mov_b64`
    let (dual, banks) = crate::dualbank::transform(&native_of(&w, true, 64)).expect("dual-bank");
    let p = ir::build_banked(&dual.code, false, dual.vgprs, Some(&banks)).expect("ir::build");
    let at = |o: Option<&crate::gfx10::Operand>, c: u8| matches!(o, Some(&crate::gfx10::Operand::Scalar { code, dwords: 2 }) if code == c);
    let pairs = p.blocks.iter().flat_map(|b| &b.insts).filter(|i| {
        matches!(&i.fields, crate::gfx10::Fields::Sop { dst, src, .. }
            if i.name() == "s_mov_b64" && (at(src.first(), 26) || at(dst.as_ref(), 24)))
    });
    assert_eq!(pairs.count(), 2, "the parked pair's save and restore");
    DUAL.set(true);
    let _one = one_at_a_time();
    let card = card("parked mask and scalar", &w);
    DUAL.set(false);
    let Some(card) = card else {
        return;
    };
    for (below, and) in [
        (0u32, 64u32),
        (1, 64),
        (31, 40),
        (40, 33),
        (64, 32),
        (64, 64),
    ] {
        let got = card.run([below, 5, and, 0]);
        let want: [u32; 64] = std::array::from_fn(|l| {
            let l = l as u32;
            let x = if l < below && l < and {
                0x1000 + l
            } else {
                DEAD
            };
            x.wrapping_add(5)
        });
        compare(
            &format!("parked mask, lanes below {below} and {and}"),
            &got,
            &some(want),
        );
    }
    eprintln!("parked mask and scalar: six masks as the model");
}

/// Every dual-bank lane test once more numbered as a program of more than 124 VGPRs is
/// (`crate::dualbank::Banks`: bank 1 by bank 0's numbers in the words, the scratch VGPRs at
/// the top, put in place by the IR), against the same model.
#[test]
fn dual_bank_numbered_past_124_vgprs_runs_every_lane_test() {
    crate::dualbank::ALL_WIDE.set(true);
    DUAL.set(true);
    let tests: [fn(); 15] = [
        a_shuffle_reads_every_lane_of_the_console_wave,
        a_shuffle_under_random_masks_keeps_the_inactive_lanes,
        a_lane_read_reaches_both_halves,
        a_lane_write_reaches_both_halves,
        the_first_active_lane_is_found_in_either_half,
        a_ballot_holds_both_halves,
        a_prefix_count_runs_across_the_halves,
        a_whole_wave_sum_of_lane_ids_is_2016,
        lanes_left_out_by_exec_keep_their_values,
        a_shuffle_in_partial_groups_stays_in_its_console_wave,
        an_if_else_on_lane_masks_runs_each_side_in_its_lanes,
        three_dimensional_groups_number_lanes_x_first,
        a_loop_with_an_if_inside_runs_each_lane_its_own_count,
        a_far_jump_goes_where_it_points,
        a_dpp_compare_reads_its_row_neighbour,
    ];
    for test in tests {
        test();
    }
    DUAL.set(false);
    crate::dualbank::ALL_WIDE.set(false);
}
