//! Dual-bank Wave64, ported from the dual-bank fork of fxceless/victory-fork (its
//! `docs/XBOX_SERIES_GPU_TRANSLATION.md` §6 and `docs/wave-audit.md`): a console wave64 compute
//! program rewritten as a console wave32 program in which each lane carries two console
//! lanes, p (bank 0) and p + 32 (bank 1), so that one host wave of 32 runs a whole console
//! wave of 64. Nothing has to meet across host waves: cross-lane code, lane masks read as
//! data and branches the console wave takes together are all inside one host wave, with no
//! barrier.
//!
//! The rewrite works on machine code; the result goes through the translator as any wave32
//! program does ([`crate::emit::emit`] tries it before the approximate half-wave forms).
//!
//! - Bank 0 keeps the program's VGPRs; bank 1's are `K` higher (`K` = the program's VGPR
//!   count). A few scratch VGPRs follow at `2K`.
//! - The console's 64-bit EXEC lives in a pair of spare SGPRs `E`, `E+1` (lanes 0-31, 32-63):
//!   every scalar instruction that names EXEC names that pair instead. Control flow stays
//!   scalar, as the console's is.
//! - Each vector instruction is written twice, bank 0 then bank 1, each after
//!   `s_mov_b32 exec_lo, E+b`: real EXEC masks that bank's lanes (inactive lanes keep their
//!   values, side effects happen only in active lanes, as on the console).
//! - A 64-bit lane mask in `s[x:x+1]` (VCC, compare results, carries, the EXEC pair) splits
//!   the same way: bank b's 32 bits are `s[x+b]`. Bank 1's compares and carries that VOPC
//!   and VOP2 write implicitly to VCC are written in VOP3 form to `vcc_hi`.
//! - Lane-identity instructions are rewritten: `v_readlane`/`v_writelane`/`v_readfirstlane`
//!   pick the bank by bit 5 of the lane, `v_mbcnt_lo`/`_hi` count the other bank's bits,
//!   `ds_bpermute` reads both banks and picks by bit 5 of the lane it names, and branches on
//!   EXEC or VCC being zero test both halves of the pair. The added code leaves SCC and VCC
//!   as they were (a far jump it writes for a branch the longer code puts out of reach saves
//!   SCC and gives it back).
//! - A program of more than 124 VGPRs has no room for bank 1 in the machine word's 256: its
//!   bank 1 instructions name bank 0's numbers and say so in [`Banks`], which the IR applies
//!   (bank 1 is then `K` up in the IR's wider VGPR numbers, the scratch VGPRs at `2K`).
//!
//! Refused (with a reason), for this first version: thread groups whose dimensions are not
//! powers of two, more than 249 VGPRs, no spare SGPRs, program-counter instructions other
//! than far jumps (`s_getpc_b64` of a table: inserted code moves the program), scratch
//! memory and LDS loads from buffers (per-lane addresses the two banks would share),
//! `ds_permute_b32`, `ds_append`/`ds_consume`/ordered counts/GWS, `*_addtid`, and SDWA or
//! DPP forms of the instructions bank 1 has to rewrite (VOPC, VOP2 with an implicit VCC).
//! shortcut: host waves are taken to hold 32 consecutive threads of a one-dimensional
//! group (the host group here is `[n, 1, 1]`), as every GPU this ran on does.

use crate::ShaderError;
use crate::gfx10::{self, Encoding, Instruction};
use crate::native::{ComputeIo, HwStage, NativeProgram, StageIo};

/// Why a program is not rewritten.
type Why = &'static str;

#[cfg(test)]
thread_local! {
    /// Tests: every branch written as a far jump ([`branch`]), as one the rewrite has put out of
    /// its offset's reach is.
    pub(crate) static ALL_FAR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: any program numbered as one of more than 124 VGPRs is ([`Banks`]).
    pub(crate) static ALL_WIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Every branch written as a far jump ([`ALL_FAR`]).
fn all_far() -> bool {
    #[cfg(test)]
    return ALL_FAR.get();
    #[cfg(not(test))]
    false
}

/// Every program numbered as one of more than 124 VGPRs ([`ALL_WIDE`]).
fn all_wide() -> bool {
    #[cfg(test)]
    return ALL_WIDE.get();
    #[cfg(not(test))]
    false
}

const EXEC_LO: u32 = 126;
const EXEC_HI: u32 = 127;
const VCC_LO: u32 = 106;
const VCC_HI: u32 = 107;
const NULL: u32 = 125;
const LITERAL: u32 = 255;
const MINUS_ONE: u32 = 193;
const fn int(n: u32) -> u32 {
    128 + n
}

/// The scratch VGPRs (from `2K`) and SGPRs (from `E+2`) the rewrite uses.
#[derive(Clone, Copy)]
struct Regs {
    /// Bank 1's VGPR offset.
    k: u32,
    /// The console EXEC pair.
    e: u32,
    /// Scratch VGPRs.
    vx: u32,
    vy: u32,
    vp: u32,
    vq: u32,
    vz: u32,
    vr0: u32,
    vr1: u32,
    /// Scratch SGPRs: a pair for copies, a lane mask, two values.
    st: u32,
    sz: u32,
    sa: u32,
    sb: u32,
}

const SCRATCH_VGPRS: u32 = 7;

/// The machine word's VGPR field of the first scratch VGPR in a program of more than 124
/// ([`Banks`]): the fields from it up name the scratch VGPRs in every instruction.
const SCRATCH_FIELD: u32 = 256 - SCRATCH_VGPRS;

/// What the IR ([`crate::ir::build_banked`]) must know of a rewritten program: that it is one
/// (its parked lane masks are pairs of bank dwords), and the VGPR numbering of one of more
/// than 124 VGPRs (`wide`), whose bank 1 does not fit the machine word's 256 (the volumetric
/// fog programs: 129 and 156). Those words name bank 1's registers by bank 0's numbers and
/// the scratch VGPRs from [`SCRATCH_FIELD`]; the IR puts each instruction's VGPRs in place:
/// bank 1's `K` up, the scratch ones at `2K`. DXIL has no register limit of its own: the
/// host's compiler spills where it must.
#[derive(Debug, Clone)]
pub struct Banks {
    /// Numbered past the machine word's VGPRs.
    wide: bool,
    /// The program's VGPRs (bank 1's offset).
    k: u16,
    /// Byte offsets of the instructions that name bank 1's registers, in order.
    bank1: Vec<usize>,
}

impl Banks {
    /// The VGPR numbers of the instruction at `offset` put in place.
    pub(crate) fn remap(&self, offset: usize, fields: &mut gfx10::Fields) {
        if !self.wide {
            return;
        }
        let base = if self.bank1.binary_search(&offset).is_ok() {
            self.k
        } else {
            0
        };
        let (k, scratch) = (self.k, SCRATCH_FIELD as u16);
        vgprs_mut(fields, |reg| {
            *reg = if *reg >= scratch {
                2 * k + (*reg - scratch)
            } else {
                *reg + base
            };
        });
    }
}

/// `f` on every VGPR number `fields` names.
fn vgprs_mut(fields: &mut gfx10::Fields, f: impl Fn(&mut u16)) {
    use gfx10::{Fields, Operand};
    let mut each = |o: &mut Operand| {
        if let Operand::Vector { reg, .. } = o {
            f(reg);
        }
    };
    match fields {
        Fields::Sop { dst, src, .. } => dst.iter_mut().chain(src).for_each(&mut each),
        Fields::Sopp { .. } | Fields::Smem { .. } => {}
        Fields::Valu(v) => {
            v.dst
                .iter_mut()
                .chain(&mut v.carry)
                .chain(&mut v.src)
                .for_each(&mut each);
        }
        Fields::Vintrp { dst, src, attr } => [dst, src, attr].into_iter().for_each(&mut each),
        Fields::Ds {
            dst, addr, data, ..
        } => {
            dst.iter_mut().chain(addr).chain(data).for_each(&mut each);
        }
        Fields::Buffer { data, addr, .. } => data.iter_mut().chain([addr]).for_each(&mut each),
        Fields::Image { data, addr, .. } => [data].into_iter().chain(addr).for_each(&mut each),
        Fields::Flat {
            dst, addr, data, ..
        } => dst.iter_mut().chain(addr).chain(data).for_each(&mut each),
        Fields::Exp { src, .. } => src.iter_mut().flatten().for_each(&mut each),
    }
}

fn op(encoding: Encoding, name: &str) -> Result<u32, Why> {
    gfx10::opcode(encoding, name)
        .map(u32::from)
        .ok_or("an opcode the rewrite writes is not in the tables")
}

/// A source operand for a constant: an inline code, or the literal code and its dword.
fn constant(v: i32) -> (u32, Option<u32>) {
    match v {
        0..=64 => (128 + v as u32, None),
        -16..=-1 => (192 + v.unsigned_abs(), None),
        _ => (LITERAL, Some(v as u32)),
    }
}

struct Out {
    words: Vec<u32>,
    /// Bank whose EXEC real EXEC holds now (`None`: unknown).
    real: Option<u32>,
    /// Bank whose registers the instructions pushed now name (for [`Banks`]).
    bank: u32,
    /// Words where an instruction of bank 1 starts.
    bank1: Vec<usize>,
}

impl Out {
    fn push(&mut self, w: &[u32]) {
        if self.bank == 1 {
            self.bank1.push(self.words.len());
        }
        self.words.extend_from_slice(w);
    }
    fn sop1(&mut self, name: &str, sdst: u32, ssrc0: u32) -> Result<(), Why> {
        let o = op(Encoding::Sop1, name)?;
        self.push(&[0xBE80_0000 | sdst << 16 | o << 8 | ssrc0]);
        Ok(())
    }
    fn sop2(&mut self, name: &str, sdst: u32, ssrc0: u32, ssrc1: u32) -> Result<(), Why> {
        let o = op(Encoding::Sop2, name)?;
        self.push(&[0x8000_0000 | o << 23 | sdst << 16 | ssrc1 << 8 | ssrc0]);
        Ok(())
    }
    /// VOP3 (or VOP3B with `sdst`), with an optional literal.
    fn vop3(
        &mut self,
        name: &str,
        vdst: u32,
        src: [u32; 3],
        sdst: Option<u32>,
        literal: Option<u32>,
    ) -> Result<(), Why> {
        let o = op(Encoding::Vop3, name)?;
        self.push(&[
            0xD400_0000 | o << 16 | sdst.unwrap_or(0) << 8 | vdst,
            src[0] | src[1] << 9 | src[2] << 18,
        ]);
        if let Some(l) = literal {
            self.push(&[l]);
        }
        Ok(())
    }
    fn vop1(&mut self, name: &str, vdst: u32, src0: u32) -> Result<(), Why> {
        let o = op(Encoding::Vop1, name)?;
        self.push(&[0x7E00_0000 | vdst << 17 | o << 9 | src0]);
        Ok(())
    }
    fn vopc(&mut self, name: &str, src0: u32, vsrc1: u32) -> Result<(), Why> {
        let o = op(Encoding::Vopc, name)?;
        self.push(&[0x7C00_0000 | o << 17 | vsrc1 << 9 | src0]);
        Ok(())
    }
    /// Real EXEC to `exec` (a scalar code); the bank it belongs to, if any, for [`Out::real`].
    fn exec(&mut self, exec: u32, bank: Option<u32>) -> Result<(), Why> {
        if let Some(b) = bank {
            self.bank = b;
        }
        if bank.is_some() && self.real == bank {
            return Ok(());
        }
        self.sop1("s_mov_b32", EXEC_LO, exec)?;
        self.real = bank;
        Ok(())
    }
    /// Every lane on (for the rewrite's own vector code).
    fn all_lanes(&mut self) -> Result<(), Why> {
        self.exec(MINUS_ONE, None)
    }
}

/// The program rewritten for dual-bank execution, and what the IR must know of it ([`Banks`];
/// build it with [`crate::spill::build_banked`]); or why not.
pub fn transform(native: &NativeProgram) -> Result<(NativeProgram, Banks), ShaderError> {
    rewrite(native).map_err(|what| ShaderError::Unsupported { offset: 0, what })
}

/// [`rewrite_with`] until every branch reaches: the rewrite about doubles the code, and a
/// branch whose 16-bit offset no longer reaches its target is written as a far jump in the next
/// round (the cascaded shadows' 169 KB compute program has such branches).
fn rewrite(native: &NativeProgram) -> Result<(NativeProgram, Banks), Why> {
    let mut long = std::collections::HashSet::new();
    loop {
        let mut too_far = Vec::new();
        match rewrite_with(native, &long, &mut too_far) {
            Err(_) if !too_far.is_empty() => long.extend(too_far),
            r => return r,
        }
    }
}

/// The rewrite, with the branches of the instructions in `long` written as far jumps; the
/// others that do not reach are added to `too_far`.
fn rewrite_with(
    native: &NativeProgram,
    long: &std::collections::HashSet<usize>,
    too_far: &mut Vec<usize>,
) -> Result<(NativeProgram, Banks), Why> {
    if native.stage != HwStage::Cs || !native.wave64 {
        return Err("dual-bank: not a console wave64 compute program");
    }
    let StageIo::Cs(cs) = native.io else {
        return Err("dual-bank: not a console wave64 compute program");
    };
    let [x, y, z] = cs.threads;
    // (one dimension: the linear id is x; more: x, y, z by shifts and masks)
    let one_dimension = y == 1 && z == 1;
    if !one_dimension && !(x.is_power_of_two() && y.is_power_of_two() && z.is_power_of_two()) {
        return Err("dual-bank: a thread group whose dimensions are not powers of two");
    }
    let group = x * y * z;
    let k = u32::from(native.vgprs).max(3);
    // past 124 VGPRs, bank 1 by bank 0's numbers and the scratch VGPRs at the top ([`Banks`])
    let wide = 2 * k + SCRATCH_VGPRS > 256 || all_wide();
    if wide && k > SCRATCH_FIELD {
        return Err("dual-bank: more than 249 VGPRs");
    }
    let (field_k, scratch) = if wide { (0, SCRATCH_FIELD) } else { (k, 2 * k) };
    let insts = gfx10::decode(&native.code).map_err(|_| "dual-bank: code that does not decode")?;
    // the spare SGPRs: past every SGPR the program names (and the ones the hardware sets)
    // (the trap temporaries ttmp0-6 first: compute code never names them, and the
    // translator takes them as registers like any other; `ir.rs` parks spilled scalars in
    // the ones still free. Else past the highest SGPR the program names.)
    let mut top = u32::from(cs.initialized_sgprs());
    let mut ttmp_named = false;
    for i in &insts {
        for (code, dwords) in scalars(i) {
            if code < VCC_LO {
                top = top.max(code + dwords);
            }
            if (108..=123).contains(&code)
                || (code..code + dwords).any(|c| (108..=114).contains(&c))
            {
                ttmp_named = true;
            }
        }
    }
    let e = if !ttmp_named {
        108
    } else {
        let e = top.next_multiple_of(2);
        if e + 7 > VCC_LO {
            return Err("dual-bank: no spare SGPRs");
        }
        e
    };
    let r = Regs {
        k: field_k,
        e,
        vx: scratch,
        vy: scratch + 1,
        vp: scratch + 2,
        vq: scratch + 3,
        vz: scratch + 4,
        vr0: scratch + 5,
        vr1: scratch + 6,
        st: e + 2,
        sz: e + 4,
        sa: e + 5,
        sb: e + 6,
    };
    let mut out = Out {
        words: Vec::new(),
        real: None,
        bank: 0,
        bank1: Vec::new(),
    };
    prologue(&mut out, &r, cs, group)?;
    // instruction index by byte offset, for branch targets
    let index_of: std::collections::HashMap<usize, usize> = insts
        .iter()
        .enumerate()
        .map(|(n, i)| (i.offset, n))
        .collect();
    let far = far_jumps(&insts, native.code.len(), &index_of)?;
    let targets: std::collections::HashSet<usize> = insts
        .iter()
        .filter_map(branch_target)
        .chain(far.iter().map(|j| insts[j.target].offset))
        .collect();
    let mut starts = Vec::with_capacity(insts.len());
    // (word of a branch in `out`, the instruction index it goes to, its own index)
    let mut fixups: Vec<(usize, usize, usize)> = Vec::new();
    // branches written as far jumps: (word of the `s_getpc_b64`, of the literal, the target)
    let mut jumps: Vec<(usize, usize, usize)> = Vec::new();
    // far jumps: (word of the `s_getpc_b64`, word of the addition's literal) by far jump
    let mut far_words: Vec<(Option<usize>, Option<usize>)> = vec![(None, None); far.len()];
    for (n, i) in insts.iter().enumerate() {
        if let Some(j) = far.iter().position(|j| j.getpc == n) {
            far_words[j].0 = Some(out.words.len());
        }
        if let Some(j) = far.iter().position(|j| j.add == n) {
            far_words[j].1 = Some(out.words.len() + 1);
        }
        if targets.contains(&i.offset) {
            out.real = None;
        }
        starts.push(out.words.len());
        let name = i.name().unwrap_or("");
        match i.encoding {
            Encoding::Sop1 | Encoding::Sop2 | Encoding::Sopc | Encoding::Sopk => {
                scalar(&mut out, &r, i, name)?
            }
            Encoding::Smem => out.push(&i.words),
            Encoding::Sopp => {
                let target = match branch_target(i) {
                    Some(t) => Some(
                        *index_of
                            .get(&t)
                            .ok_or("dual-bank: a branch into an instruction")?,
                    ),
                    None => None,
                };
                match name {
                    // A skip: a forward `s_cbranch_execz` over straight-line code (no branch
                    // inside, the target the region's end). The console's compiler inserts it to
                    // save time; the code it skips is correct to run with no lane active, and
                    // here every vector instruction of it runs under its bank's EXEC, so with
                    // none active it does nothing. Left out: its scalar code then always runs,
                    // as code after the region may read (a sampler built inside the region and
                    // read after it, seen in one compute program). `VICTORY_DUAL_BANK_SKIPS=1`
                    // keeps the branches.
                    "s_cbranch_execz"
                        if target.is_some_and(|t| t > n && straight(&insts[n + 1..t]))
                            && !keep_skips() => {}
                    "s_cbranch_execz" | "s_cbranch_execnz" | "s_cbranch_vccz"
                    | "s_cbranch_vccnz" => {
                        let pair = if name.contains("exec") { r.e } else { VCC_LO };
                        any_lane(&mut out, &r, pair)?;
                        let (_, t) = (0, target.ok_or("dual-bank: a branch without a target")?);
                        let to = if name.ends_with("nz") {
                            "s_cbranch_execnz"
                        } else {
                            "s_cbranch_execz"
                        };
                        let o = op(Encoding::Sopp, to)?;
                        let word = 0xBF80_0000 | o << 16;
                        let how = (long.contains(&n) || all_far(), &mut fixups, &mut jumps);
                        branch(&mut out, &r, to, word, (n, t), how)?;
                    }
                    n if n.starts_with("s_cbranch_cdbg") || n == "s_setkill" => {
                        return Err("dual-bank: a debug or kill branch");
                    }
                    _ => match target {
                        Some(t) => {
                            let how = (long.contains(&n) || all_far(), &mut fixups, &mut jumps);
                            branch(&mut out, &r, name, i.words[0], (n, t), how)?;
                        }
                        None => out.push(&i.words),
                    },
                }
                if target.is_some() {
                    out.real = None;
                }
            }
            Encoding::Vop1 | Encoding::Vop2 | Encoding::Vopc | Encoding::Vop3 | Encoding::Vop3p => {
                vector(&mut out, &r, i, name)?;
            }
            Encoding::Ds => ds(&mut out, &r, i, name)?,
            Encoding::Mubuf | Encoding::Mtbuf => {
                if i.encoding == Encoding::Mubuf && i.words[0] >> 16 & 1 != 0 {
                    return Err("dual-bank: a buffer load into LDS");
                }
                memory(&mut out, &r, i, &[(1, 0), (1, 8)])?;
            }
            Encoding::Mimg => {
                let fields = i.fields(true);
                let nsa = match &fields {
                    gfx10::Fields::Image { addr, .. } if i.words[0] >> 1 & 3 != 0 => {
                        addr.len().saturating_sub(1)
                    }
                    _ => 0,
                };
                let mut at = vec![(1usize, 0u32), (1, 8)];
                for n in 0..nsa {
                    at.push((2 + n / 4, 8 * (n % 4) as u32));
                }
                memory(&mut out, &r, i, &at)?;
            }
            Encoding::Flat | Encoding::Global => {
                let fields = i.fields(true);
                let mut at = Vec::new();
                if let gfx10::Fields::Flat {
                    dst, addr, data, ..
                } = &fields
                {
                    if addr.is_some() {
                        at.push((1, 0));
                    }
                    if data.is_some() {
                        at.push((1, 8));
                    }
                    if dst.is_some() {
                        at.push((1, 24));
                    }
                }
                memory(&mut out, &r, i, &at)?;
            }
            Encoding::Scratch => return Err("dual-bank: scratch memory"),
            Encoding::Vintrp | Encoding::Exp => return Err("dual-bank: not a compute instruction"),
        }
        let far_jump = far.iter().any(|j| j.getpc == n || j.setpc == n);
        if (name == "s_getpc_b64" || name == "s_setpc_b64") && !far_jump
            || name == "s_swappc_b64"
            || name.starts_with("s_call")
        {
            return Err("dual-bank: program-counter instructions");
        }
        if name == "s_setpc_b64" {
            out.real = None;
        }
    }
    // far jumps: the literal is the distance from the `s_getpc_b64`'s next word to the target
    let far_words = far_words
        .iter()
        .zip(&far)
        .map(|(&(pc, lit), j)| match (pc, lit) {
            (Some(pc), Some(lit)) => Ok((pc, lit, j.target, j.subtract)),
            _ => Err("dual-bank: a far jump out of order"),
        });
    let jumps = jumps.iter().map(|&(pc, lit, t)| Ok((pc, lit, t, false)));
    for jump in far_words.chain(jumps) {
        let (pc, lit, t, subtract) = jump?;
        let to = 4 * (starts[t] as i64 - (pc as i64 + 1));
        let to = if subtract { -to } else { to };
        out.words[lit] = i32::try_from(to).map_err(|_| "dual-bank: a far jump too far")? as u32;
    }
    // branch offsets in dwords from the word after the branch
    for (at, t, n) in fixups {
        let to = starts[t] as i64 - (at as i64 + 1);
        match i16::try_from(to) {
            Ok(imm) => out.words[at] = out.words[at] & 0xFFFF_0000 | u32::from(imm as u16),
            Err(_) => too_far.push(n),
        }
    }
    if !too_far.is_empty() {
        return Err("dual-bank: a branch too far after the rewrite");
    }
    let host = group.div_ceil(64) * 32;
    let banks = Banks {
        wide,
        k: k as u16,
        bank1: out.bank1.iter().map(|w| 4 * w).collect(),
    };
    let program = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: (2 * k + SCRATCH_VGPRS) as u16,
        layout: native.layout.clone(),
        used: native.used.clone(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: cs.rsrc2,
            threads: [host, 1, 1],
        }),
        code: out.words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    Ok((program, banks))
}

/// A far jump: `s_getpc_b64 s[n:n+1]`, `s_add_u32` (or `s_sub_u32`) `sn, sn, literal`, a carry
/// into `s(n+1)`, `s_setpc_b64 s[n:n+1]`, the form the console's compiler gives a branch past
/// the reach of a 16-bit offset (seen in one compute program of the cascaded shadows, 169 KB
/// of code, as the early-out to its end). By instruction index.
struct FarJump {
    getpc: usize,
    add: usize,
    setpc: usize,
    target: usize,
    subtract: bool,
}

/// The far jumps among `insts` (as [`gfx10::control`] resolves them). The rewrite keeps the
/// four instructions in order and in one piece, and gives the addition the literal that
/// reaches the target's rewritten code.
fn far_jumps(
    insts: &[Instruction],
    len: usize,
    index_of: &std::collections::HashMap<usize, usize>,
) -> Result<Vec<FarJump>, Why> {
    let mut out = Vec::new();
    for (setpc, i) in insts.iter().enumerate() {
        if i.name() != Some("s_setpc_b64") {
            continue;
        }
        let before = |at: usize| insts[..insts.partition_point(|i| i.offset < at)].last();
        let Ok(gfx10::Flow::Jump(t)) = gfx10::control(i, len, before) else {
            continue;
        };
        let target = *index_of
            .get(&t)
            .ok_or("dual-bank: a far jump into an instruction")?;
        let n = i.words[0] & 0xFF;
        let (mut getpc, mut add) = (None, None);
        for k in (setpc.saturating_sub(4)..setpc).rev() {
            let w = insts[k].words[0];
            match insts[k].name() {
                Some(op @ ("s_add_u32" | "s_sub_u32"))
                    if w >> 16 & 0x7F == n && w & 0xFF == n && w >> 8 & 0xFF == LITERAL =>
                {
                    add = add.or(Some((k, op == "s_sub_u32")));
                }
                Some("s_getpc_b64") if w >> 16 & 0x7F == n => {
                    getpc = Some(k);
                    break;
                }
                _ => {}
            }
        }
        let (Some(getpc), Some((add, subtract))) = (getpc, add) else {
            return Err("dual-bank: a far jump of another form");
        };
        if add < getpc {
            return Err("dual-bank: a far jump of another form");
        }
        out.push(FarJump {
            getpc,
            add,
            setpc,
            target,
            subtract,
        });
    }
    Ok(out)
}

/// A branch (SOPP `word`, of instruction `at.0`, to instruction `at.1`): as it is, its offset
/// filled in once the code is laid out; or with `far.0` as a far jump through the scratch pair,
/// SCC kept through it (a scratch SGPR holds it), a conditional one stepping over it on the
/// reverse condition.
#[allow(clippy::type_complexity)]
fn branch(
    out: &mut Out,
    r: &Regs,
    name: &str,
    word: u32,
    at: (usize, usize),
    far: (
        bool,
        &mut Vec<(usize, usize, usize)>,
        &mut Vec<(usize, usize, usize)>,
    ),
) -> Result<(), Why> {
    let ((n, t), (long, fixups, jumps)) = (at, far);
    if !long {
        fixups.push((out.words.len(), t, n));
        out.push(&[word]);
        return Ok(());
    }
    let reverse = match name {
        "s_branch" => None,
        "s_cbranch_scc0" => Some("s_cbranch_scc1"),
        "s_cbranch_scc1" => Some("s_cbranch_scc0"),
        "s_cbranch_execz" => Some("s_cbranch_execnz"),
        "s_cbranch_execnz" => Some("s_cbranch_execz"),
        _ => return Err("dual-bank: a branch too far after the rewrite"),
    };
    if let Some(reverse) = reverse {
        // over the 7 words below
        out.push(&[0xBF80_0000 | op(Encoding::Sopp, reverse)? << 16 | 7]);
    }
    out.sop2("s_cselect_b32", r.sa, int(1), int(0))?;
    let pc = out.words.len();
    out.sop1("s_getpc_b64", r.st, 0)?;
    out.sop2("s_add_u32", r.st, r.st, LITERAL)?;
    jumps.push((pc, out.words.len(), t));
    out.push(&[0]);
    let high = if t > n { int(0) } else { MINUS_ONE };
    out.sop2("s_addc_u32", r.st + 1, r.st + 1, high)?;
    let cmp = op(Encoding::Sopc, "s_cmp_lg_u32")?;
    out.push(&[0xBF00_0000 | cmp << 16 | int(0) << 8 | r.sa]);
    out.sop1("s_setpc_b64", 0, r.st)?;
    Ok(())
}

/// The byte offset a SOPP branch goes to.
fn branch_target(i: &Instruction) -> Option<usize> {
    let name = i.name()?;
    if i.encoding != Encoding::Sopp || !(name == "s_branch" || name.starts_with("s_cbranch")) {
        return None;
    }
    let imm = i.words[0] as u16 as i16;
    Some((i.offset as i64 + 4 + 4 * i64::from(imm)) as usize)
}

/// Every SGPR range an instruction names, as (first code, dwords).
fn scalars(i: &Instruction) -> Vec<(u32, u32)> {
    use gfx10::{Fields, Operand};
    let mut out = Vec::new();
    let mut add = |o: &Operand| {
        if let Operand::Scalar { code, dwords } = *o {
            out.push((u32::from(code), u32::from(dwords)));
        }
    };
    match i.fields(true) {
        Fields::Sop { dst, src, .. } => {
            dst.iter().chain(&src).for_each(&mut add);
        }
        Fields::Smem {
            data,
            base,
            soffset,
            ..
        } => {
            data.iter().chain([&base, &soffset]).for_each(&mut add);
        }
        Fields::Valu(v) => {
            v.dst
                .iter()
                .chain(&v.carry)
                .chain(&v.src)
                .for_each(&mut add);
        }
        Fields::Buffer { rsrc, soffset, .. } => {
            [&rsrc, &soffset].into_iter().for_each(&mut add);
        }
        Fields::Image { rsrc, sampler, .. } => {
            std::iter::once(&rsrc).chain(&sampler).for_each(&mut add);
        }
        Fields::Flat { saddr, .. } => saddr.iter().for_each(&mut add),
        _ => {}
    }
    out
}

/// Console thread ids for both banks, and the console EXEC of the lanes the group has.
fn prologue(out: &mut Out, r: &Regs, cs: ComputeIo, group: u32) -> Result<(), Why> {
    out.all_lanes()?;
    out.bank = 0;
    // host thread h (v0) is console thread (h & ~31) * 2 + (h & 31) in bank 0, + 32 in bank 1
    out.push(&[0x1B << 25 | r.vz << 17 | LITERAL, 0xFFFF_FFE0]); // v_and_b32 vz, ~31, v0 (vsrc1 v0)
    out.push(&[0x25 << 25 | r.vr0 << 17 | (256 + r.vz)]); // v_add_nc_u32 vr0, vz, v0 (vsrc1 v0)
    out.push(&[0x25 << 25 | r.vr1 << 17 | r.vr0 << 9 | int(32)]); // v_add_nc_u32 vr1, 32, vr0
    // the console's initial EXEC: the lanes whose thread the group has
    let (n, lit) = constant(group as i32);
    out.vop3("v_cmp_gt_u32", r.e, [n, 256 + r.vr0, 0], None, lit)?;
    out.vop3("v_cmp_gt_u32", r.e + 1, [n, 256 + r.vr1, 0], None, lit)?;
    // thread id components (dimensions are powers of two): x, y, z of the linear id
    let [x, y, _] = cs.threads;
    let (lx, ly) = (x.trailing_zeros(), y.trailing_zeros());
    let ids = cs.thread_id_vgprs();
    for (b, lin) in [(0, r.vr0), (1, r.vr1)] {
        out.bank = b;
        let base = b * r.k;
        if ids >= 2 {
            // v1 = (lin >> lx) & (y - 1)
            out.push(&[0x16 << 25 | (base + 1) << 17 | lin << 9 | int(lx)]); // v_lshrrev_b32
            let (m, l) = constant((y - 1) as i32);
            out.push(&[0x1B << 25 | (base + 1) << 17 | (base + 1) << 9 | m]);
            out.push(&l.map_or(vec![], |l| vec![l]));
        }
        if ids >= 3 {
            out.push(&[0x16 << 25 | (base + 2) << 17 | lin << 9 | int(lx + ly)]);
        }
        if cs.threads[1] == 1 && cs.threads[2] == 1 {
            // one dimension: x is the linear id (any group size)
            out.vop1("v_mov_b32", base, 256 + lin)?;
        } else {
            // v0 = lin & (x - 1)
            let (m, l) = constant((x - 1) as i32);
            out.push(&[0x1B << 25 | base << 17 | lin << 9 | m]);
            out.push(&l.map_or(vec![], |l| vec![l]));
        }
    }
    out.real = None;
    Ok(())
}

/// A scalar register code with the console EXEC renamed to its pair.
fn data(r: &Regs, code: u32) -> u32 {
    match code {
        EXEC_LO => r.e,
        EXEC_HI => r.e + 1,
        c => c,
    }
}

/// Bank `b`'s 32 bits of the 64-bit lane mask whose first register is `code`.
fn mask(r: &Regs, code: u32, b: u32) -> u32 {
    match code {
        EXEC_LO => r.e + b,
        NULL => NULL,
        c if c < 128 => c + b,
        c => c,
    }
}

/// SALU and SOPK: EXEC renamed; the saveexec and wrexec forms written out.
fn scalar(out: &mut Out, r: &Regs, i: &Instruction, name: &str) -> Result<(), Why> {
    let w = i.words[0];
    if let Some(stem) = name.strip_suffix("_saveexec_b64") {
        // D = EXEC; EXEC = f(S, EXEC); SCC = EXEC != 0
        let (d, s) = (w >> 16 & 0x7F, w & 0xFF);
        let s = data(r, s);
        let lit = i.words.get(1).copied();
        // S into the scratch pair first if D would overwrite it
        let s = if s == LITERAL || s < 128 && (s == d || s == d + 1 || s + 1 == d) {
            let mut word = 0xBE80_0000 | r.st << 16 | op(Encoding::Sop1, "s_mov_b64")? << 8 | s;
            if s == LITERAL {
                word |= 0;
                out.push(&[word, lit.unwrap_or(0)]);
            } else {
                out.push(&[word]);
            }
            r.st
        } else {
            s
        };
        out.sop1("s_mov_b64", data(r, d), r.e)?;
        let (o, a, b) = match stem {
            "s_and" => ("s_and_b64", s, r.e),
            "s_or" => ("s_or_b64", s, r.e),
            "s_xor" => ("s_xor_b64", s, r.e),
            "s_nand" => ("s_nand_b64", s, r.e),
            "s_nor" => ("s_nor_b64", s, r.e),
            "s_xnor" => ("s_xnor_b64", s, r.e),
            "s_andn1" => ("s_andn2_b64", r.e, s),
            "s_andn2" => ("s_andn2_b64", s, r.e),
            "s_orn1" => ("s_orn2_b64", r.e, s),
            "s_orn2" => ("s_orn2_b64", s, r.e),
            _ => return Err("dual-bank: a saveexec form the rewrite does not know"),
        };
        out.sop2(o, r.e, a, b)?;
        out.real = None;
        return Ok(());
    }
    if let Some(stem) = name.strip_suffix("_wrexec_b64") {
        // EXEC = D = f(S, EXEC); SCC = EXEC != 0
        let (d, s) = (w >> 16 & 0x7F, data(r, w & 0xFF));
        if s == LITERAL {
            return Err("dual-bank: a wrexec with a literal");
        }
        match stem {
            "s_andn1" => out.sop2("s_andn2_b64", r.e, r.e, s)?,
            "s_andn2" => out.sop2("s_andn2_b64", r.e, s, r.e)?,
            _ => return Err("dual-bank: a wrexec form the rewrite does not know"),
        }
        out.sop1("s_mov_b64", data(r, d), r.e)?;
        out.real = None;
        return Ok(());
    }
    let mut words = i.words.clone();
    let rename = |w: u32, shift: u32, bits: u32| -> u32 {
        let field = w >> shift & ((1 << bits) - 1);
        let new = data(r, field);
        w & !(((1 << bits) - 1) << shift) | new << shift
    };
    words[0] = match i.encoding {
        Encoding::Sop2 => rename(rename(rename(w, 16, 7), 8, 8), 0, 8),
        Encoding::Sop1 => rename(rename(w, 16, 7), 0, 8),
        Encoding::Sopc => rename(rename(w, 8, 8), 0, 8),
        Encoding::Sopk => rename(w, 16, 7),
        _ => w,
    };
    // a write to the console EXEC pair: real EXEC no longer known to match
    let sdst = words[0] >> 16 & 0x7F;
    if matches!(i.encoding, Encoding::Sop1 | Encoding::Sop2 | Encoding::Sopk)
        && (sdst == r.e || sdst == r.e + 1)
    {
        out.real = None;
    }
    out.push(&words);
    Ok(())
}

/// Branch helper: real EXEC = the host lanes where bank 0 or bank 1 of the lane mask pair
/// at `pair` has its bit, so that `s_cbranch_exec[n]z` tests the console's 64 bits.
fn any_lane(out: &mut Out, r: &Regs, pair: u32) -> Result<(), Why> {
    out.all_lanes()?;
    out.vop3("v_cndmask_b32", r.vz, [int(0), int(1), pair], None, None)?;
    out.vop3(
        "v_cndmask_b32",
        r.vz,
        [256 + r.vz, int(1), pair + 1],
        None,
        None,
    )?;
    out.vopc("v_cmpx_ne_u32", int(0), r.vz)?;
    out.real = None;
    Ok(())
}

/// The lane `lane` (a scalar code or constant) of a console register held in `v` (bank 0) and
/// `v + K` (bank 1), into SGPR `dst`. Real EXEC all lanes.
fn read_lane(out: &mut Out, r: &Regs, v: u32, lane: u32, dst: u32) -> Result<(), Why> {
    if (128..=192).contains(&lane) {
        let l = lane - 128;
        let bank = l >> 5;
        out.bank = bank;
        out.vop3(
            "v_readlane_b32",
            dst,
            [256 + v + bank * r.k, int(l & 31), 0],
            None,
            None,
        )?;
        return Ok(());
    }
    out.all_lanes()?;
    out.bank = 0;
    out.vop3("v_readlane_b32", r.sa, [256 + v, lane, 0], None, None)?;
    out.bank = 1;
    out.vop3("v_readlane_b32", r.sb, [256 + v + r.k, lane, 0], None, None)?;
    bit5(out, r, lane)?;
    out.vop1("v_mov_b32", r.vx, r.sa)?;
    out.vop3("v_cndmask_b32", r.vx, [256 + r.vx, r.sb, r.sz], None, None)?;
    out.vop1("v_readfirstlane_b32", dst, 256 + r.vx)?;
    Ok(())
}

/// `sz` = every lane where bit 5 of `lane` (a scalar code) is set: the bank it names.
fn bit5(out: &mut Out, r: &Regs, lane: u32) -> Result<(), Why> {
    out.vop3("v_and_b32", r.vz, [int(32), lane, 0], None, None)?;
    out.vop3("v_cmp_ne_u32", r.sz, [int(0), 256 + r.vz, 0], None, None)?;
    Ok(())
}

/// A VGPR field of `words[word]` at `shift` (8 bits), moved to bank `b`.
fn bank_byte(words: &mut [u32], word: usize, shift: u32, add: u32) {
    let v = words[word] >> shift & 0xFF;
    words[word] = words[word] & !(0xFF << shift) | (v + add) << shift;
}

/// A 9-bit VALU source field moved to bank `b` (VGPRs) or renamed (EXEC as data).
fn source_field(r: &Regs, field: u32, add: u32) -> u32 {
    match field {
        256.. => field + add,
        c if c < 128 => data(r, c),
        c => c,
    }
}

/// Vector memory: every listed VGPR byte (word, shift) moved to the bank, once per bank.
fn memory(
    out: &mut Out,
    r: &Regs,
    i: &Instruction,
    vgpr_bytes: &[(usize, u32)],
) -> Result<(), Why> {
    for b in 0..2 {
        out.exec(r.e + b, Some(b))?;
        let mut words = i.words.clone();
        for &(word, shift) in vgpr_bytes {
            if b == 1 {
                bank_byte(&mut words, word, shift, r.k);
            }
        }
        out.push(&words);
    }
    Ok(())
}

fn ds(out: &mut Out, r: &Regs, i: &Instruction, name: &str) -> Result<(), Why> {
    if name == "ds_permute_b32" {
        return Err("dual-bank: ds_permute_b32");
    }
    if name.starts_with("ds_gws")
        || name.contains("addtid")
        || matches!(name, "ds_append" | "ds_consume" | "ds_ordered_count")
    {
        return Err("dual-bank: an LDS instruction that counts the wave's lanes");
    }
    let gfx10::Fields::Ds {
        dst, addr, data: d, ..
    } = i.fields(true)
    else {
        unreachable!()
    };
    if name == "ds_bpermute_b32" {
        return bpermute(out, r, i);
    }
    let mut at = Vec::new();
    if addr.is_some() {
        at.push((1, 0));
    }
    for n in 0..d.len() {
        at.push((1, 8 + 8 * n as u32));
    }
    if dst.is_some() {
        at.push((1, 24));
    }
    memory(out, r, i, &at)
}

/// `ds_bpermute_b32 dst, addr, data offset`: each bank's lanes read lane
/// `(addr + offset) / 4 mod 64` of the console register, 0 from a lane outside EXEC.
fn bpermute(out: &mut Out, r: &Regs, i: &Instruction) -> Result<(), Why> {
    let (w0, w1) = (i.words[0], i.words[1]);
    let offset = w0 & 0xFFFF;
    let (addr, data_reg, dst) = (w1 & 0xFF, w1 >> 8 & 0xFF, w1 >> 24);
    let ds_op = op(Encoding::Ds, "ds_bpermute_b32")?;
    let bperm = |out: &mut Out, vdst: u32, a: u32, d: u32| {
        out.push(&[0xD800_0000 | ds_op << 18 | offset, vdst << 24 | d << 8 | a]);
    };
    out.all_lanes()?;
    // each bank's data, 0 where its lane is outside the console EXEC
    out.bank = 0;
    out.vop3(
        "v_cndmask_b32",
        r.vx,
        [int(0), 256 + data_reg, r.e],
        None,
        None,
    )?;
    out.bank = 1;
    out.vop3(
        "v_cndmask_b32",
        r.vy,
        [int(0), 256 + data_reg + r.k, r.e + 1],
        None,
        None,
    )?;
    for (b, result) in [(0u32, r.vr0), (1, r.vr1)] {
        out.bank = b;
        let a = addr + b * r.k;
        bperm(out, r.vp, a, r.vx);
        bperm(out, r.vq, a, r.vy);
        out.push(&[0xBF8C_C07F]); // s_waitcnt lgkmcnt(0)
        // bit 7 of the byte address (+ offset): the bank of the lane it names
        let (o, lit) = constant(offset as i32);
        out.vop3("v_add_nc_u32", r.vz, [o, 256 + a, 0], None, lit)?;
        out.vop3(
            "v_and_b32",
            r.vz,
            [LITERAL, 256 + r.vz, 0],
            None,
            Some(0x80),
        )?;
        out.vop3("v_cmp_ne_u32", r.sz, [int(0), 256 + r.vz, 0], None, None)?;
        out.vop3(
            "v_cndmask_b32",
            result,
            [256 + r.vp, 256 + r.vq, r.sz],
            None,
            None,
        )?;
    }
    for (b, result) in [(0u32, r.vr0), (1, r.vr1)] {
        out.exec(r.e + b, Some(b))?;
        out.vop1("v_mov_b32", dst + b * r.k, 256 + result)?;
    }
    Ok(())
}

/// Every VALU instruction: lane-identity ones rewritten, the others once per bank.
fn vector(out: &mut Out, r: &Regs, i: &Instruction, name: &str) -> Result<(), Why> {
    let w0 = i.words[0];
    let vop3 = matches!(i.encoding, Encoding::Vop3 | Encoding::Vop3p);
    let w1 = i.words.get(1).copied().unwrap_or(0);
    let field = |n: u32| w1 >> (9 * n) & 0x1FF;
    match name {
        "v_readlane_b32" => {
            // VOP3: sdst in vdst, src0 the VGPR, src1 the lane
            let (dst, v, lane) = (w0 & 0xFF, field(0), field(1));
            let lane = data(r, lane);
            if v < 256 {
                return Err("dual-bank: a lane read of no VGPR");
            }
            read_lane(out, r, v - 256, lane, data(r, dst))?;
            out.real = None;
            return Ok(());
        }
        "v_readfirstlane_b32" => {
            let (dst, src) = if vop3 {
                (w0 & 0xFF, field(0))
            } else {
                (w0 >> 17 & 0xFF, w0 & 0x1FF)
            };
            if src < 256 {
                // a scalar or constant: the same in every lane
                out.all_lanes()?;
                out.vop1("v_readfirstlane_b32", data(r, dst), data(r, src))?;
                out.real = None;
                return Ok(());
            }
            out.sop1("s_ff1_i32_b64", r.st, r.e)?;
            read_lane(out, r, src - 256, r.st, data(r, dst))?;
            out.real = None;
            return Ok(());
        }
        "v_writelane_b32" => {
            let (vd, value, lane) = (w0 & 0xFF, data(r, field(0)), data(r, field(1)));
            let lit = i.words.get(2).copied();
            if (128..=192).contains(&lane) {
                let l = lane - 128;
                out.all_lanes()?;
                out.bank = l >> 5;
                out.vop3(
                    "v_writelane_b32",
                    vd + (l >> 5) * r.k,
                    [value, int(l & 31), 0],
                    None,
                    lit,
                )?;
                out.real = None;
                return Ok(());
            }
            out.all_lanes()?;
            out.bank = 0;
            out.vop1("v_mov_b32", r.vx, 256 + vd)?;
            out.bank = 1;
            out.vop1("v_mov_b32", r.vy, 256 + vd + r.k)?;
            out.vop3("v_writelane_b32", r.vx, [value, lane, 0], None, lit)?;
            out.vop3("v_writelane_b32", r.vy, [value, lane, 0], None, lit)?;
            bit5(out, r, lane)?;
            // bank 0 takes the write where the lane is in bank 0 (bit 5 clear)
            out.bank = 0;
            out.vop3(
                "v_cndmask_b32",
                vd,
                [256 + r.vx, 256 + vd, r.sz],
                None,
                None,
            )?;
            out.bank = 1;
            out.vop3(
                "v_cndmask_b32",
                vd + r.k,
                [256 + vd + r.k, 256 + r.vy, r.sz],
                None,
                None,
            )?;
            out.real = None;
            return Ok(());
        }
        "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
            let (vd, m, acc) = (w0 & 0xFF, data(r, field(0)), field(1));
            let lit = i.words.get(2).copied();
            let acc_of = |b: u32| source_field(r, acc, b * r.k);
            let lo = name == "v_mbcnt_lo_u32_b32";
            for b in 0..2u32 {
                out.exec(r.e + b, Some(b))?;
                let vdst = vd + b * r.k;
                match (lo, b) {
                    // lanes 0-31: the mask's bits below the lane; lanes 32-63: all of them
                    (true, 0) => {
                        out.vop3("v_mbcnt_lo_u32_b32", vdst, [m, acc_of(0), 0], None, lit)?
                    }
                    (true, _) => out.vop3("v_bcnt_u32_b32", vdst, [m, acc_of(1), 0], None, lit)?,
                    // the high half: none below lanes 0-31; below lane 32 + p, its bits below p
                    (false, 0) => {
                        out.vop3("v_add_nc_u32", vdst, [acc_of(0), int(0), 0], None, lit)?
                    }
                    (false, _) => {
                        out.vop3("v_mbcnt_lo_u32_b32", vdst, [m, acc_of(1), 0], None, lit)?
                    }
                }
            }
            return Ok(());
        }
        _ => {}
    }
    if i.encoding == Encoding::Vop3 && (name == "v_div_fmas_f32" || name == "v_div_fmas_f64") {
        // reads VCC implicitly: bank 1 runs with VCC's high half in vcc_lo
        for b in 0..2u32 {
            out.exec(r.e + b, Some(b))?;
            let words = bank_vop3(r, i, name, b)?;
            if b == 1 {
                out.sop1("s_mov_b32", r.sa, VCC_LO)?;
                out.sop1("s_mov_b32", VCC_LO, VCC_HI)?;
                out.push(&words);
                out.sop1("s_mov_b32", VCC_LO, r.sa)?;
            } else {
                out.push(&words);
            }
        }
        return Ok(());
    }
    let cmpx = name.starts_with("v_cmpx");
    for b in 0..2u32 {
        out.exec(r.e + b, Some(b))?;
        if i.encoding == Encoding::Vopc && w0 & 0x1FF == 0xFA && (cmpx || b == 1) {
            dpp_compare(out, r, i, name, b, cmpx)?;
            out.real = None;
            continue;
        }
        if i.encoding == Encoding::Vopc && w0 & 0x1FF == 0xF9 && cmpx {
            sdwa_compare(out, r, i, name, b)?;
            out.real = None;
            continue;
        }
        let words = match i.encoding {
            Encoding::Vop3 | Encoding::Vop3p => bank_vop3(r, i, name, b)?,
            Encoding::Vopc => {
                let sdwa_dpp = matches!(w0 & 0x1FF, 0xF9 | 0xFA | 0xE9 | 0xEA);
                if w0 & 0x1FF == 0xF9 && !cmpx {
                    // SDWA: the compare names its scalar destination in the extra dword
                    // (`sd`, bits 14:8; without `sd` it is VCC): bank b's 32 bits of it
                    let mut w = bank_short(r, i, b);
                    let ext = w[1];
                    let to = if ext >> 15 & 1 == 1 {
                        mask(r, ext >> 8 & 0x7F, b)
                    } else {
                        VCC_LO + b
                    };
                    w[1] = ext & !(0xFF << 8) | 1 << 15 | to << 8;
                    w
                } else if cmpx || b == 1 {
                    if sdwa_dpp {
                        return Err("dual-bank: an SDWA or DPP compare");
                    }
                    // VOP3 form, the result to its 32 bits of VCC or of the console EXEC
                    let to = if cmpx { r.e + b } else { VCC_LO + b };
                    let cmp = if cmpx {
                        name.replacen("v_cmpx_", "v_cmp_", 1)
                    } else {
                        name.to_owned()
                    };
                    let o = op(Encoding::Vop3, &cmp)?;
                    let src0 = source_field(r, w0 & 0x1FF, b * r.k);
                    let src1 = 256 + (w0 >> 9 & 0xFF) + b * r.k;
                    let mut w = vec![0xD400_0000 | o << 16 | to, src0 | src1 << 9];
                    w.extend(i.words.get(1).copied().filter(|_| w0 & 0x1FF == LITERAL));
                    w
                } else {
                    bank_short(r, i, b)
                }
            }
            Encoding::Vop2 if b == 1 && implicit_vcc(name) => {
                if matches!(w0 & 0x1FF, 0xF9 | 0xFA | 0xE9 | 0xEA) {
                    return Err("dual-bank: an SDWA or DPP form with an implicit VCC");
                }
                let o = op(Encoding::Vop3, name)?;
                let carry = name.ends_with("_co_ci_u32");
                let src0 = source_field(r, w0 & 0x1FF, r.k);
                let src1 = 256 + (w0 >> 9 & 0xFF) + r.k;
                let vdst = (w0 >> 17 & 0xFF) + r.k;
                let mut w = vec![
                    0xD400_0000 | o << 16 | if carry { VCC_HI << 8 } else { 0 } | vdst,
                    src0 | src1 << 9 | VCC_HI << 18,
                ];
                w.extend(i.words.get(1).copied().filter(|_| w0 & 0x1FF == LITERAL));
                w
            }
            _ => bank_short(r, i, b),
        };
        out.push(&words);
        if cmpx {
            // the console EXEC changed under real EXEC
            out.real = None;
        }
    }
    if cmpx {
        out.real = None;
    }
    Ok(())
}

/// A DPP16 control word (bits 31:28 the row mask, one bit per 16-lane row of the console's
/// 64-lane wave) for bank `b` of a 32-lane host wave: bank 0 is the console's rows 0-1,
/// bank 1 its rows 2-3, which are the host's rows 0-1. Rows 2-3 of a 32-lane wave do not
/// exist; their bits are left clear.
fn dpp_rows(ext: u32, b: u32) -> u32 {
    let rows = ext >> 28 >> (2 * b) & 0x3;
    ext & 0x0FFF_FFFF | rows << 28
}

/// A DPP16 compare (`v_cmp*_dpp`) for bank `b` where it cannot stay a VOPC (bank 1 writes
/// `vcc_hi`, a `v_cmpx` writes its bank of the console EXEC): GFX10 has no VOP3 form with
/// DPP, so the moved operand is made first and compared after.
/// - `v_mov_b32` with the instruction's DPP (pattern, row and bank masks, bound_ctrl) puts
///   src0 of the lane each lane reads into `vp`; the abs/neg modifiers move to the compare.
/// - A lane whose DPP source is invalid (a masked row or bank, or an inactive source without
///   bound_ctrl) is not written by the console's instruction: its result bit stays 0. A
///   second DPP move of 1 into a zeroed `vq` marks the valid lanes, and a `v_cmpx` takes the
///   others out of real EXEC for the compare (VALU only: SCC and VCC are left alone).
/// - The compare in VOP3 form writes the bank's 32 bits of VCC (or of the console EXEC for
///   `v_cmpx`); lanes outside real EXEC get 0, as on the console.
fn dpp_compare(
    out: &mut Out,
    r: &Regs,
    i: &Instruction,
    name: &str,
    b: u32,
    cmpx: bool,
) -> Result<(), Why> {
    let (w0, ext) = (i.words[0], i.words[1]);
    let add = b * r.k;
    let src0 = (ext & 0xFF) + add;
    let vsrc1 = (w0 >> 9 & 0xFF) + add;
    let (row_mask, bank_mask, bound_ctrl) = (ext >> 28, ext >> 24 & 0xF, ext >> 19 & 1 == 1);
    let mov = op(Encoding::Vop1, "v_mov_b32")?;
    // the DPP word without src0 and without abs/neg (bits 20-23): those go to the compare
    let dpp = dpp_rows(ext, b) & !0xFF & !(0xF << 20);
    out.push(&[0x7E00_0000 | r.vp << 17 | mov << 9 | 0xFA, dpp | src0]);
    let checks_lanes = !bound_ctrl || row_mask != 0xF || bank_mask != 0xF;
    if checks_lanes {
        // vq = 1 where the lane's DPP source is valid, 0 elsewhere (the move writes nothing
        // there): the same DPP, from vz = 1, without bound_ctrl
        out.vop1("v_mov_b32", r.vq, int(0))?;
        out.vop1("v_mov_b32", r.vz, int(1))?;
        out.push(&[
            0x7E00_0000 | r.vq << 17 | mov << 9 | 0xFA,
            dpp & !(1 << 19) | r.vz,
        ]);
        out.vopc("v_cmpx_ne_u32", int(0), r.vq)?;
    }
    let to = if cmpx { r.e + b } else { VCC_LO + b };
    let cmp = if cmpx {
        name.replacen("v_cmpx_", "v_cmp_", 1)
    } else {
        name.to_owned()
    };
    let o = op(Encoding::Vop3, &cmp)?;
    // DPP's modifiers: neg src0 bit 20, abs src0 21, neg src1 22, abs src1 23
    let neg = (ext >> 20 & 1) | (ext >> 22 & 1) << 1;
    let abs = (ext >> 21 & 1) | (ext >> 23 & 1) << 1;
    out.push(&[
        0xD400_0000 | o << 16 | abs << 8 | to,
        (256 + r.vp) | (256 + vsrc1) << 9 | neg << 29,
    ]);
    Ok(())
}

/// An SDWA `v_cmpx` for bank `b`: it writes the console EXEC, which here is the pair `E`, not
/// real EXEC, and GFX10's SDWA compares have no form that writes an SGPR as a `v_cmpx`. So
/// each operand's selected byte or word (with its sign extension) is first moved into a
/// scratch VGPR by `v_mov_b32` SDWA (`vp` for src0, `vq` for src1; an operand may be an
/// SGPR), and a VOP3 compare of the two writes the bank's 32 bits of the console EXEC, with
/// SDWA's abs/neg modifiers. Lanes outside real EXEC get 0, as `v_cmpx` gives them.
fn sdwa_compare(out: &mut Out, r: &Regs, i: &Instruction, name: &str, b: u32) -> Result<(), Why> {
    let (w0, ext) = (i.words[0], i.words[1]);
    let add = b * r.k;
    let mov = op(Encoding::Vop1, "v_mov_b32")?;
    // src0: bits 7:0, sel 18:16, sext 19, an SGPR when bit 23 is set
    let (s0_sgpr, s1_sgpr) = (ext >> 23 & 1 == 1, ext >> 31 == 1);
    let src0 = if s0_sgpr {
        data(r, ext & 0xFF)
    } else {
        (ext & 0xFF) + add
    };
    let sel0 = ext >> 16 & 0xF; // sel and sext
    // src1: the VOPC's vsrc1 field, sel 26:24, sext 27, an SGPR when bit 31 is set
    let v1 = w0 >> 9 & 0xFF;
    let src1 = if s1_sgpr { data(r, v1) } else { v1 + add };
    let sel1 = ext >> 24 & 0xF;
    // v_mov_b32 SDWA: dst_sel dword (6), dst_unused pad (0), src0_sel and sext, s0
    let sdwa_mov = |dst: u32, reg: u32, sel: u32, sgpr: bool| {
        [
            0x7E00_0000 | dst << 17 | mov << 9 | 0xF9,
            reg | 6 << 8 | sel << 16 | u32::from(sgpr) << 23,
        ]
    };
    out.push(&sdwa_mov(r.vp, src0, sel0, s0_sgpr));
    out.push(&sdwa_mov(r.vq, src1, sel1, s1_sgpr));
    let o = op(Encoding::Vop3, &name.replacen("v_cmpx_", "v_cmp_", 1))?;
    // SDWA's modifiers: neg src0 bit 20, abs src0 21, neg src1 28, abs src1 29
    let neg = (ext >> 20 & 1) | (ext >> 28 & 1) << 1;
    let abs = (ext >> 21 & 1) | (ext >> 29 & 1) << 1;
    out.push(&[
        0xD400_0000 | o << 16 | abs << 8 | (r.e + b),
        (256 + r.vp) | (256 + r.vq) << 9 | neg << 29,
    ]);
    Ok(())
}

/// VOP2 instructions that read (and the `_co_ci` ones write) VCC without naming it.
fn implicit_vcc(name: &str) -> bool {
    name == "v_cndmask_b32" || name.ends_with("_co_ci_u32")
}

/// VOP1, VOP2 and VOPC (bank 0's compare) moved to bank `b`, with SDWA and DPP sources.
fn bank_short(r: &Regs, i: &Instruction, b: u32) -> Vec<u32> {
    let mut words = i.words.clone();
    let add = b * r.k;
    let w0 = words[0];
    let src0 = w0 & 0x1FF;
    let vop1 = i.encoding == Encoding::Vop1;
    let vopc = i.encoding == Encoding::Vopc;
    let mut w = w0;
    match src0 {
        0xF9 => {
            // SDWA: the s0 bit says whether src0 is an SGPR; s1 for vsrc1
            let ext = words[1];
            let s0 = ext >> 23 & 1 == 1;
            let s1 = ext >> 31 == 1;
            let reg = ext & 0xFF;
            words[1] = ext & !0xFF | if s0 { data(r, reg) } else { reg + add };
            if !vop1 && !s1 {
                let v1 = w >> 9 & 0xFF;
                w = w & !(0xFF << 9) | (v1 + add) << 9;
            } else if !vop1 {
                let v1 = data(r, w >> 9 & 0xFF);
                w = w & !(0xFF << 9) | v1 << 9;
            }
        }
        0xFA | 0xE9 | 0xEA => {
            bank_byte(&mut words, 1, 0, add);
            if src0 == 0xFA {
                words[1] = dpp_rows(words[1], b);
            }
            if !vop1 {
                let v1 = w >> 9 & 0xFF;
                w = w & !(0xFF << 9) | (v1 + add) << 9;
            }
        }
        _ => {
            w = w & !0x1FF | source_field(r, src0, add);
            if !vop1 {
                let v1 = w >> 9 & 0xFF;
                w = w & !(0xFF << 9) | (v1 + add) << 9;
            }
        }
    }
    if !vopc {
        // the destination: a VGPR (v_readfirstlane is rewritten before this)
        let d = w >> 17 & 0xFF;
        w = w & !(0xFF << 17) | (d + add) << 17;
    }
    words[0] = w;
    words
}

/// VOP3 and VOP3P moved to bank `b`: VGPRs moved, lane masks split, EXEC as data renamed.
fn bank_vop3(r: &Regs, i: &Instruction, name: &str, b: u32) -> Result<Vec<u32>, Why> {
    let mut words = i.words.clone();
    let add = b * r.k;
    let w0 = words[0];
    let compare = i.encoding == Encoding::Vop3 && i.opcode < 0x100;
    let vop3b = matches!(
        name,
        "v_add_co_u32"
            | "v_sub_co_u32"
            | "v_subrev_co_u32"
            | "v_add_co_ci_u32"
            | "v_sub_co_ci_u32"
            | "v_subrev_co_ci_u32"
            | "v_div_scale_f32"
            | "v_div_scale_f64"
            | "v_mad_u64_u32"
            | "v_mad_i64_i32"
    );
    let carry_in = implicit_vcc(name);
    let vd = w0 & 0xFF;
    let mut new0 = if compare {
        // the result is a lane mask: bank b's 32 bits (v_cmpx: of the console EXEC)
        let to = if name.starts_with("v_cmpx") {
            r.e + b
        } else {
            mask(r, vd, b)
        };
        let o = if name.starts_with("v_cmpx") {
            op(Encoding::Vop3, &name.replacen("v_cmpx_", "v_cmp_", 1))?
        } else {
            u32::from(i.opcode)
        };
        w0 & !(0x3FF << 16) & !0xFF | o << 16 | to
    } else {
        w0 & !0xFF | (vd + add)
    };
    if vop3b {
        let sdst = w0 >> 8 & 0x7F;
        new0 = new0 & !(0x7F << 8) | mask(r, sdst, b) << 8;
    }
    words[0] = new0;
    let w1 = words[1];
    let mut new1 = w1 & !0x7FF_FFFF;
    for n in 0..3 {
        let f = w1 >> (9 * n) & 0x1FF;
        let g = if n == 2 && carry_in && i.encoding == Encoding::Vop3 {
            mask(r, f, b)
        } else {
            source_field(r, f, add)
        };
        new1 |= g << (9 * n);
    }
    words[1] = new1;
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_take_inline_codes_where_they_can() {
        assert_eq!(constant(0), (128, None));
        assert_eq!(constant(64), (192, None));
        assert_eq!(constant(-1), (193, None));
        assert_eq!(constant(-16), (208, None));
        assert_eq!(constant(65), (255, Some(65)));
    }

    #[test]
    fn masks_split_by_bank() {
        let r = Regs {
            k: 16,
            e: 20,
            vx: 32,
            vy: 33,
            vp: 34,
            vq: 35,
            vz: 36,
            vr0: 37,
            vr1: 38,
            st: 22,
            sz: 24,
            sa: 25,
            sb: 26,
        };
        assert_eq!((mask(&r, VCC_LO, 0), mask(&r, VCC_LO, 1)), (VCC_LO, VCC_HI));
        assert_eq!((mask(&r, EXEC_LO, 0), mask(&r, EXEC_LO, 1)), (20, 21));
        assert_eq!((mask(&r, 4, 0), mask(&r, 4, 1)), (4, 5));
        assert_eq!(mask(&r, NULL, 1), NULL);
        assert_eq!((data(&r, EXEC_LO), data(&r, EXEC_HI)), (20, 21));
    }
}

/// Debugging aid: `VICTORY_DUAL_DUMP=<container.cdxil>,<byte offset>` prints the rewritten
/// program's instructions around that offset (the compute program of the container).
#[cfg(test)]
#[test]
#[ignore = "debugging aid"]
fn dump_rewrite_around() {
    let Ok(spec) = std::env::var("VICTORY_DUAL_DUMP") else {
        return;
    };
    let (path, at) = spec.split_once(',').expect("<file>,<offset>");
    let at = usize::from_str_radix(at.trim_start_matches("0x"), 16).expect("hex offset");
    let bytes = std::fs::read(path).expect("container");
    let c = crate::Container::parse(&bytes).expect("container");
    let part = c.part(crate::FourCC::DXIL).expect("DXIL part");
    let native = crate::native_programs(part.data)
        .expect("programs")
        .into_iter()
        .find(|p| p.stage == HwStage::Cs)
        .expect("compute");
    let (dual, _) = rewrite(&native).expect("rewrite");
    for i in gfx10::decode(&dual.code).expect("decode") {
        if i.offset + 0x80 >= at && i.offset <= at + 0x40 {
            println!(
                "{:#06x}{} {}",
                i.offset,
                if i.offset == at { " >>" } else { "   " },
                i.text(false)
            );
        }
    }
}

/// Debugging aid: `VICTORY_DUAL_DUMP_ORIG=<container.cdxil>,<name>` prints the original
/// program's instructions before the first instruction of that name.
#[cfg(test)]
#[test]
#[ignore = "debugging aid"]
fn dump_original_before() {
    let Ok(spec) = std::env::var("VICTORY_DUAL_DUMP_ORIG") else {
        return;
    };
    let (path, name) = spec.split_once(',').expect("<file>,<name>");
    let bytes = std::fs::read(path).expect("container");
    let c = crate::Container::parse(&bytes).expect("container");
    let part = c.part(crate::FourCC::DXIL).expect("DXIL part");
    let native = crate::native_programs(part.data)
        .expect("programs")
        .into_iter()
        .find(|p| p.stage == HwStage::Cs)
        .expect("compute");
    let insts = gfx10::decode(&native.code).expect("decode");
    let at = insts
        .iter()
        .position(|i| i.name() == Some(name))
        .expect("instruction");
    for i in &insts[at.saturating_sub(22)..=at + 1] {
        println!("{:#06x}  {}", i.offset, i.text(true));
    }
}

/// Whether `region` has no branch, no end of program and no scalar memory store or message
/// (code a skip may jump over and run all the same).
fn straight(region: &[Instruction]) -> bool {
    region.iter().all(|i| {
        let name = i.name().unwrap_or("");
        branch_target(i).is_none()
            && !matches!(name, "s_endpgm" | "s_setpc_b64" | "s_swappc_b64")
            && !name.starts_with("s_sendmsg")
            && !(i.encoding == Encoding::Smem
                && (name.contains("store") || name.contains("atomic")))
    })
}

/// `VICTORY_DUAL_BANK_SKIPS=1`: skips are kept as branches ([`rewrite`]).
fn keep_skips() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DUAL_BANK_SKIPS").is_some_and(|v| v == "1"))
}
