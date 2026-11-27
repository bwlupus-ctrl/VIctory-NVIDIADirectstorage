//! LOCAL (per-slot copy-back, after DexNom/VIctory bc3cd4b): which root UAVs a lowered
//! program only reads (`Lowered::read_only_root_uavs`, `HostShader`'s, which `victory-gpu`'s
//! `draw.rs` leaves out of the copy back). Root views at s[0:1] (slot 0) and s[2:3] (slot 1),
//! their buffer descriptors built in code as the console compiler builds them.

use super::Options;
use super::tests::{S_ENDPGM, lower_compute, root};
use crate::native::{ResourceType, RootType, UsedResource};

/// s[8:11] from root view s[0:1], s[12:15] from root view s[2:3].
const DESCRIPTORS: [u32; 12] = [
    0xBE88_0300, // s_mov_b32 s8, s0
    0x8809_FF01,
    0x0010_0000, // s_or_b32 s9, s1, 0x100000
    0xBE8A_03C1, // s_mov_b32 s10, -1
    0xBE8B_03FF,
    0x1101_4FAC, // s_mov_b32 s11, 0x11014fac
    0xBE8C_0302, // s_mov_b32 s12, s2
    0x880D_FF03,
    0x0010_0000, // s_or_b32 s13, s3, 0x100000
    0xBE8E_03C1, // s_mov_b32 s14, -1
    0xBE8F_03FF,
    0x1101_4FAC, // s_mov_b32 s15, 0x11014fac
];
/// buffer_load_dword v1, v0, s[8:11], 0 offen
const LOAD_SLOT0: [u32; 2] = [0xE030_1000, 0x8002_0100];
/// buffer_load_dword v4, v0, s[12:15], 0 offen
const LOAD_SLOT1: [u32; 2] = [0xE030_1000, 0x8003_0400];
/// buffer_store_dword v1, v0, s[12:15], 0 offen
const STORE_SLOT1: [u32; 2] = [0xE070_1000, 0x8003_0100];
/// v2 = v0, v3 = 0: an address for the global accesses.
const ADDRESS: [u32; 2] = [0x7E04_0300, 0x7E06_0280];
/// global_store_dword v[2:3], v1, off
const GLOBAL_STORE: [u32; 2] = [0xDC70_8000, 0x007D_0102];
/// global_atomic_add v[2:3], v1, off
const GLOBAL_ATOMIC: [u32; 2] = [0xDCC8_8000, 0x007D_0102];
/// ds_write_b32 v0, v4: keeps a load live.
const KEEP: [u32; 2] = [0xD834_0000, 0x0000_0400];

fn read_only(body: &[[u32; 2]]) -> Vec<u8> {
    let mut words = DESCRIPTORS.to_vec();
    for w in body {
        words.extend(w);
    }
    words.push(S_ENDPGM);
    let view = |user_data, slot| UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data,
        slot,
    };
    let layout = vec![root(RootType::Descriptor, 0, 0), root(RootType::Descriptor, 2, 1)];
    let l = lower_compute(&words, layout, vec![view(0, 0), view(2, 1)], false, true, Options::DEFAULT)
        .expect("lowers");
    assert!(l.read_only_root_uavs.iter().all(|s| l.root_uavs.contains(s)));
    l.read_only_root_uavs
}

/// (c) A root view only loaded from is read only; one stored to is not; both loaded only:
/// both.
#[test]
fn root_views_only_loaded_from_are_read_only() {
    assert_eq!(read_only(&[LOAD_SLOT0, STORE_SLOT1]), [0]);
    assert_eq!(read_only(&[LOAD_SLOT0, LOAD_SLOT1, KEEP]), [0, 1]);
}

/// (a) A store by address may land in a root view's buffer: no root view is read only.
#[test]
fn a_store_by_address_makes_every_root_view_written() {
    assert_eq!(read_only(&[LOAD_SLOT0, LOAD_SLOT1, KEEP]), [0, 1]);
    assert_eq!(read_only(&[LOAD_SLOT0, LOAD_SLOT1, KEEP, ADDRESS, GLOBAL_STORE]), [] as [u8; 0]);
}

/// (b) So does an atomic by address.
#[test]
fn an_atomic_by_address_makes_every_root_view_written() {
    assert_eq!(read_only(&[LOAD_SLOT0, LOAD_SLOT1, KEEP, ADDRESS, GLOBAL_ATOMIC]), [] as [u8; 0]);
}
