//! LOCAL (upstream-port, 77e13a9; fxceless/victory-fork PR 927, tippythetaidum):
//! typed buffer loads that move 16-bit halves without the d16 form: a
//! format load of whole dwords whose result the program reads only one
//! half of, as a half float, moves 16-bit integers, like a d16 access
//! ([`super::super::HostShader::half_views`]).

use std::collections::HashSet;

use super::{Fields, Inst, Lower, Operand, Reg, Role, RootType, Source, Value};

impl<'a> Lower<'a> {
    /// Whether every use of `value`, a typed buffer load's result, reads
    /// one 16-bit half of it as a half float: a `v_fma_mix*` source with
    /// its `op_sel_hi` bit set, or `v_cvt_f32_f16`. A program that takes a
    /// format load apart like that reads a 16-bit integer view on the
    /// console (the load gives the zero-extended integer, whose bits are a
    /// half; a float view's value it would not take apart). A use is a
    /// read as an operand ([`crate::ir::operand_reads`]): a later write of
    /// the register reads the value only to keep it in the lanes EXEC
    /// leaves alone, and a phi carries it on, its own uses counting. False
    /// without a use.
    pub(super) fn read_only_as_halves(&self, value: Value) -> bool {
        let p = self.x.p;
        let mut used = false;
        let mut seen = HashSet::from([value]);
        let mut todo = vec![value];
        while let Some(value) = todo.pop() {
            for (bi, b) in p.blocks.iter().enumerate() {
                for phi in b.phis.iter().filter(|phi| phi.args.contains(&value)) {
                    if seen.insert(phi.value) {
                        todo.push(phi.value);
                    }
                }
                for (ii, i) in b.insts.iter().enumerate() {
                    if !i.reads.iter().any(|r| r.1 == value) {
                        continue;
                    }
                    let operands = crate::ir::operand_reads(p, i);
                    let regs: Vec<Reg> = i
                        .reads
                        .iter()
                        .filter(|r| r.1 == value && operands.contains(&r.0))
                        .map(|r| r.0)
                        .collect();
                    // LOCAL (port-fix): a write that only keeps the old value
                    // in some lanes or bits (a guarded one, a half write,
                    // an SDWA one) makes a value of its own that still holds
                    // those bits: every use of it counts as a use of this
                    // one, or a whole-word read of the overwritten register
                    // would pass for halves only.
                    for r in i.reads.iter().filter(|r| r.1 == value && !operands.contains(&r.0)) {
                        if self.keeps_old_bits(i, bi, ii)
                            && let Some(new) = i.write(r.0)
                            && seen.insert(new)
                        {
                            todo.push(new);
                        }
                    }
                    if regs.is_empty() {
                        continue;
                    }
                    used = true;
                    let Fields::Valu(v) = &i.fields else {
                        return false;
                    };
                    match i.name() {
                        "v_cvt_f32_f16" => {}
                        "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                            let whole = v.src.iter().enumerate().any(|(k, op)| {
                                matches!(*op, Operand::Vector { reg, .. } if regs.contains(&Reg::V(reg)))
                                    && v.opsel_hi >> k & 1 == 0
                            });
                            if whole {
                                return false;
                            }
                        }
                        _ => return false,
                    }
                }
            }
        }
        used
    }

    /// LOCAL (port-fix): whether the write of `inst` (block `b`, place `i`)
    /// may keep bits of the register's old value: it is guarded by EXEC
    /// or a branch, or is not plainly a whole-dword write (SDWA, a 16-bit
    /// destination, `mixlo`/`mixhi`, d16). Conservative: only a plain
    /// unguarded 32-bit VALU write is known to replace every bit.
    fn keeps_old_bits(&self, inst: &Inst, b: usize, i: usize) -> bool {
        if self.x.st.guards[b][i].is_some() {
            return true;
        }
        let Fields::Valu(v) = &inst.fields else {
            return true;
        };
        let name = inst.name();
        let whole = v.sdwa.is_none()
            && v.dpp.is_none()
            && ["_b32", "_u32", "_i32", "_f32"].iter().any(|s| name.ends_with(s))
            && !name.contains("d16")
            && !name.contains("mixlo")
            && !name.contains("mixhi");
        !whole
    }

    /// Whether `inst` is a format load of whole dwords (not d16) whose
    /// every channel is read only as halves ([`Self::read_only_as_halves`]).
    pub(super) fn format_load_read_as_halves(&self, inst: &Inst) -> bool {
        // (LOCAL: not one with its own format in the instruction, which
        // goes through `format_load` and no view.)
        let Fields::Buffer {
            data: Some(Operand::Vector { reg, .. }),
            format: None,
            ..
        } = inst.fields
        else {
            return false;
        };
        let name = inst.name();
        let count = match name
            .strip_prefix("buffer_load_format_")
            .or(name.strip_prefix("tbuffer_load_format_"))
        {
            Some("x") => 1,
            Some("xy") => 2,
            Some("xyz") => 3,
            Some("xyzw") => 4,
            _ => return false,
        };
        (0..count).all(|k| {
            inst.write(Reg::V(reg + k))
                .is_some_and(|v| self.read_only_as_halves(v))
        })
    }

    /// The table places ([`Self::table_place`]) every access through which
    /// moves halves: a d16 access, or a format load read only as halves
    /// ([`Self::format_load_read_as_halves`]). A place the program reaches
    /// otherwise too is left out (a format load of numbers beside one of
    /// halves, a raw or scalar load, a descriptor chosen by control flow),
    /// and so is every place of a table the program also picks from at run
    /// time: what the program makes of such a view is not halves alone.
    ///
    /// LOCAL: [`super::Options::half_views`] decides which of these places
    /// count. `HALVES_ALL` is upstream's rule: every such place, d16-only ones
    /// too. `HALVES_READS` (our default) keeps only the places a format load
    /// read as halves reaches: the d16-only places stay with the cool_fox
    /// path (`integer_view`, a run-time choice by the view's real format),
    /// which already moves them, and a place with both kinds of access moves
    /// them all as halves, so the host's one view serves every access.
    pub(super) fn places_read_as_halves(&self) -> std::collections::BTreeSet<(u8, u32)> {
        if self.opts.half_views == crate::emit::HALVES_OFF {
            return Default::default();
        }
        let mut halves = std::collections::BTreeSet::new();
        let mut reads = std::collections::BTreeSet::new();
        let mut other = std::collections::BTreeSet::new();
        let mut picked = HashSet::new();
        for d in &self.x.res.descriptors {
            if d.role == Role::Sampler {
                continue;
            }
            let inst = self
                .x
                .p
                .blocks
                .get(d.site.0)
                .and_then(|b| b.insts.get(d.site.1));
            let half_read = matches!(d.source, Source::Loaded { .. })
                && inst.is_some_and(|i| self.format_load_read_as_halves(i));
            let half = half_read
                || matches!(d.source, Source::Loaded { .. })
                    && inst.is_some_and(|i| {
                        i.name().starts_with("buffer_") && i.name().contains("_format_d16_")
                    });
            for leaf in d.source.leaves() {
                let Source::Loaded { load, .. } = leaf else {
                    continue;
                };
                let Some(a) = self.scalar.get(load) else {
                    continue;
                };
                if a.dynamic.is_some() {
                    if let Some(rp) = self.x.native.layout.get(a.param)
                        && rp.root == RootType::Table
                    {
                        picked.insert(rp.slot);
                    }
                } else if let Some(place) = self.table_place(leaf) {
                    if half {
                        halves.insert(place);
                        if half_read {
                            reads.insert(place);
                        }
                    } else {
                        other.insert(place);
                    }
                }
            }
        }
        halves.retain(|p| !other.contains(p) && !picked.contains(&p.0));
        if self.opts.half_views != crate::emit::HALVES_ALL {
            halves.retain(|p| reads.contains(p));
        }
        halves
    }
}
