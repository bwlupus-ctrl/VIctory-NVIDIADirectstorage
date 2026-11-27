//! LOCAL (`VICTORY_SANITIZE_HASHES`, a test switch): a program told to sanitize
//! (`super::Options::sanitize`) stores 0 for a float result that is a NaN or an
//! infinity, and leaves values that are not float results alone. Lowered,
//! validated by `dxil.dll` when present, and run on the host GPU when there is one.

use super::Options;
use super::tests::{S_ENDPGM, lower_compute, root};
use crate::native::{ResourceType, RootType, UsedResource};

/// `buffer_store_dword v<data>, v0, s[16:19], 0 offen offset:<offset>`
const fn store(data: u32, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x8004_0000 | data << 8]
}

/// One thread: v1 = s1 * s1 and v3 = v1 - v1 (floats), v2 = s1 as it is; stored in that
/// order (v1, v3, v2) to the first three dwords of the table's raw UAV.
fn program() -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0402,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[4:5], 0
        0x7E04_0201, // v_mov_b32 v2, s1
        0x1002_0401, // v_mul_f32 v1, s1, v2
        0x0806_0301, // v_sub_f32 v3, v1, v1
    ];
    w.extend(store(1, 0));
    w.extend(store(3, 4));
    w.extend(store(2, 8));
    w.push(S_ENDPGM);
    w
}

fn lowered(sanitize: bool) -> super::Lowered {
    let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    let output = UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    let opts = Options {
        sanitize,
        ..Options::DEFAULT
    };
    lower_compute(&program(), layout, vec![output], false, false, opts).expect("lowers")
}

#[test]
fn sanitized_stores_differ_and_validate() {
    let (plain, finite) = (lowered(false), lowered(true));
    assert_ne!(plain.dxil, finite.dxil);
    #[cfg(windows)]
    if let Ok(mut v) =
        crate::emit::validator::Validator::load(crate::emit::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut finite.dxil.clone()).unwrap();
    }
}

/// On the card: an infinity and a NaN come out as 0 with the switch, as they are without it;
/// finite results and the stored constant are the same either way.
#[cfg(windows)]
#[test]
fn sanitized_stores_replace_nan_and_infinity_with_zero_on_the_card() {
    use crate::emit::HostShader;
    use crate::emit::host::{Dispatch, Host, Root, View};
    use crate::native::HwStage;
    let _one = crate::emit::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let run = |sanitize: bool, s1: u32| -> Vec<u32> {
        let l = lowered(sanitize);
        let shader = HostShader {
            stage: HwStage::Cs,
            dxil: l.dxil,
            user_data: l.user_data,
            samplers: l.samplers,
            static_samplers: Vec::new(),
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            read_only_root_uavs: Vec::new(),
            root_indexed_views: Vec::new(),
            half_views: Vec::new(),
        };
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        let mut data = vec![0; shader.user_data_dwords().max(1) as usize];
        for param in &shader.user_data {
            if param.root == RootType::Constants {
                data[param.index as usize] = s1;
            }
        }
        let mut out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(data))],
                    views: &[View::RawUav(3)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        out.swap_remove(0)
    };
    let huge = 0x7F00_0000; // 1.7e38: its square is an infinity
    let plain = run(false, huge);
    assert_eq!(plain[0], 0x7F80_0000, "the infinity without the switch");
    assert!(f32::from_bits(plain[1]).is_nan(), "the NaN without the switch");
    assert_eq!(run(true, huge), [0, 0, huge]);
    let two = 2.0f32.to_bits();
    assert_eq!(run(true, two), [4.0f32.to_bits(), 0, two]);
    assert_eq!(run(false, two), run(true, two));
    eprintln!("ran on the hardware device: NaN and infinity stored as 0");
}
