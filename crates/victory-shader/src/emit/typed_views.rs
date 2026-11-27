//! LOCAL (port of DexNom/VIctory b8ff22e): typed SRVs a program reads as
//! unsigned integers. A few stripped programs keep masks and indices in
//! `R32_UINT`-like views and read them with `image_load`, `image_gather4_lz`
//! or `buffer_load_format_x`; our float declarations of such views leave
//! what they return to the driver (AMD hands back the bits, others convert).
//! Those views are declared and loaded as `i32` by the exact root-table
//! field ([`super::Options::uint_srvs`], chosen by container digest in
//! `super::super::tiled_lighting_uint_srvs`).
use super::super::bitcode::Ty;
use super::Lower;
use crate::native::RootType;
use crate::resource::Source;

impl Lower<'_> {
    /// Whether the descriptor `src`, of DXIL resource kind `kind`, is one of
    /// the program's integer SRVs: loaded straight from a root table at a
    /// constant offset that names one of them.
    pub(super) fn scalar_uint_srv(&self, src: &Source, kind: u32) -> bool {
        if self.opts.uint_srvs.is_empty() {
            return false;
        }
        let Source::Loaded { load, first } = src else {
            return false;
        };
        let Some(address) = self.scalar.get(load) else {
            return false;
        };
        let Some(root) = self.x.native.layout.get(address.param) else {
            return false;
        };
        if root.root != RootType::Table || address.dynamic.is_some() || !address.offset.is_multiple_of(4) {
            return false;
        }
        self.opts
            .uint_srvs
            .contains(&(root.slot, address.offset / 4 + u32::from(*first), kind))
    }

    /// The result type of a resource load of `f32` channels, or of `i32`
    /// channels where `uint`.
    pub(super) fn resource_return_type(&mut self, uint: bool) -> Ty {
        if uint {
            let i32t = self.s.t.i32;
            self.s.m.named_struct("dx.types.ResRet.i32", vec![i32t; 5])
        } else {
            self.s.t.res_ret
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Options;
    use super::super::tests::{S_ENDPGM, lower_compute, texture_tables};
    use crate::emit::tiled_lighting_uint_srvs;

    /// (DexNom/VIctory b8ff22e) The tiled-lighting vertex program's integer view is the
    /// Texture2D at root table 1, 48 dwords in: only a load of that field, of that dimension,
    /// is declared as integers.
    #[test]
    fn fixed_uint_load_matches_only_the_named_root_field_and_dimension() {
        let vs = 0x8386ec5b_47a3b2de_49309239_7987b35du128.to_be_bytes();
        let views = tiled_lighting_uint_srvs(vs);
        assert!(!views.is_empty());
        let compile = |slot, offset: u32, dim: u32, views| {
            let words = [
                0xBE84_0300,
                0xBE85_0380, // s[4:5] = first root table
                0xF40C_0302,
                0xFA00_0000 | offset, // s_load_dwordx8 s[12:19]
                0xF000_1100 | dim,
                0x0003_0C00, // image_load v12, v[0:1], s[12:19]
                0xD834_0000,
                0x0000_0C00, // ds_write_b32 v0, v12
                S_ENDPGM,
            ];
            let (mut layout, mut used) = texture_tables();
            layout.truncate(1);
            used.truncate(1);
            layout[0].slot = slot;
            used[0].slot = slot;
            used[0].table_offset_dwords = offset / 4;
            let opts = Options {
                uint_srvs: views,
                ..Options::DEFAULT
            };
            lower_compute(&words, layout, used, false, true, opts).unwrap().dxil
        };
        // Texture2D, root 1 + 48 dwords. The native immediate is in bytes.
        let matched = compile(1, 192, 8, views);
        assert_ne!(matched, compile(1, 192, 8, &[]));
        for (slot, offset, dim) in [(2, 192, 8), (1, 196, 8), (1, 192, 16)] {
            assert_eq!(compile(slot, offset, dim, views), compile(slot, offset, dim, &[]));
        }
        #[cfg(windows)]
        if let Ok(mut v) =
            crate::emit::validator::Validator::load(crate::emit::validator::Validator::default_path().as_deref())
        {
            v.validate(&mut matched.clone()).unwrap();
        }
    }

    fn validate(dxil: &[u8]) {
        #[cfg(windows)]
        if let Ok(mut v) =
            crate::emit::validator::Validator::load(crate::emit::validator::Validator::default_path().as_deref())
        {
            v.validate(&mut dxil.to_vec()).unwrap();
        }
        let _ = dxil;
    }

    /// LOCAL: a one-channel gather at level 0 of a listed Texture2D (root table 1, 48 dwords
    /// in, its sampler from table s1) gathers integers; the same gather of an unlisted field
    /// is the float one; a gather of another channel stays float too.
    #[test]
    fn a_listed_view_gathers_integers() {
        let compile = |offset: u32, dmask: u32, views: &'static [(u8, u32, u32)]| {
            let words = [
                0xBE84_0300,
                0xBE85_0380, // s[4:5] = texture table
                0xF40C_0302,
                0xFA00_0000 | offset, // s_load_dwordx8 s[12:19]
                0xBE86_0301,
                0xBE87_0380, // s[6:7] = sampler table
                0xF408_0503,
                0xFA00_0000, // s_load_dwordx4 s[20:23]
                0xF11C_0008 | dmask << 8,
                0x00A3_0C00, // image_gather4_lz v[12:15], v[0:1], s[12:19], s[20:23]
                0xD834_0000,
                0x0000_0C00, // ds_write_b32 v0, v12
                S_ENDPGM,
            ];
            let (mut layout, mut used) = texture_tables();
            layout[0].slot = 1;
            used[0].slot = 1;
            used[0].table_offset_dwords = offset / 4;
            layout[1].slot = 2;
            used[1].slot = 2;
            let opts = Options {
                uint_srvs: views,
                ..Options::DEFAULT
            };
            lower_compute(&words, layout, used, false, true, opts).unwrap().dxil
        };
        const VIEWS: &[(u8, u32, u32)] = &[(1, 48, 2)];
        let matched = compile(192, 1, VIEWS);
        assert_ne!(matched, compile(192, 1, &[]));
        assert_eq!(compile(196, 1, VIEWS), compile(196, 1, &[]));
        assert_eq!(compile(192, 2, VIEWS), compile(192, 2, &[]));
        validate(&matched);
    }

    /// LOCAL: `buffer_load_format_x` of a listed typed buffer (root table 0, 32 dwords in)
    /// loads an integer; of another field, the float load.
    #[test]
    fn a_listed_typed_buffer_loads_an_integer() {
        let compile = |offset: u32, views: &'static [(u8, u32, u32)]| {
            let words = [
                0xBE84_0300,
                0xBE85_0380, // s[4:5] = table
                0xF408_0202,
                0xFA00_0000 | offset, // s_load_dwordx4 s[8:11]
                0xE000_2000,
                0x8002_0C00, // buffer_load_format_x v12, v0, s[8:11], 0 idxen
                0xD834_0000,
                0x0000_0C00, // ds_write_b32 v0, v12
                S_ENDPGM,
            ];
            let (mut layout, mut used) = texture_tables();
            layout.truncate(1);
            used.truncate(1);
            used[0].dwords = 4;
            used[0].table_offset_dwords = offset / 4;
            let opts = Options {
                uint_srvs: views,
                ..Options::DEFAULT
            };
            lower_compute(&words, layout, used, false, true, opts).unwrap().dxil
        };
        const VIEWS: &[(u8, u32, u32)] = &[(0, 32, 10)];
        let matched = compile(128, VIEWS);
        assert_ne!(matched, compile(128, &[]));
        assert_eq!(compile(132, VIEWS), compile(132, &[]));
        validate(&matched);
    }
}
