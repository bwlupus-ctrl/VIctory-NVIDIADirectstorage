//! Which processors our own threads run on, where a processor has two kinds
//! of cores. `cpu_sets` and `keep_on_fast_cores` are fxceless/victory-fork's
//! (`framelog.rs`, commit 0b0074e0, GPL-3.0-or-later, as this).

/// The system's processor sets of group 0: each one's id and its efficiency
/// class (higher is faster).
fn cpu_sets() -> Vec<(u32, u8)> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sets = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push((set.Id, set.EfficiencyClass));
        }
        at += info.Size as usize;
    }
    sets
}

/// The sets of the fastest class, if there are slower ones too.
fn fast_of(sets: &[(u32, u8)]) -> Vec<u32> {
    let best = sets.iter().map(|c| c.1).max().unwrap_or(0);
    let fast: Vec<u32> = sets.iter().filter(|c| c.1 == best).map(|c| c.0).collect();
    if fast.len() == sets.len() { Vec::new() } else { fast }
}

/// `VICTORY_FAST_CORES`: 0: no thread asks; 1: the threads a frame waits for
/// one by one; 2 (the default): the threads they share their work with too;
/// 3: those that work beside the rest as well (lists translated, descriptors
/// made): more threads by then than the fast cores run at a time, while the
/// slow ones stood idle.
fn stage() -> u32 {
    static STAGE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *STAGE.get_or_init(|| std::env::var("VICTORY_FAST_CORES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2))
}

/// Asks the system to run the calling thread, which a frame waits for, on
/// the fast cores only, where the processor has two kinds. That fork's submit
/// thread was found on a slow one in 1,529 looks of 24,021. (A selection of
/// processors is a preference the scheduler keeps unless nothing of it is
/// free.) The title's threads are not asked for.
pub(crate) fn keep_on_fast_cores(what: &str) {
    ask(what, 1);
}

/// [`keep_on_fast_cores`] for one of the threads a submission's preparing or
/// a landing shares its work with.
pub(crate) fn keep_helper_on_fast_cores(what: &str) {
    ask(what, 2);
}

/// [`keep_on_fast_cores`] for one of many threads whose work is ahead of what
/// a frame waits for.
pub(crate) fn keep_many_on_fast_cores(what: &str) {
    ask(what, 3);
}

/// LOCAL (vcache): `VICTORY_VCACHE_CORES`: 0 (default): as before; 1: where all
/// cores are of one class but the last-level caches differ in size (Ryzen X3D:
/// one die with 3D V-Cache), the threads asked for by [`keep_on_fast_cores`] and
/// its kin go to the cores of the largest cache; 2: as 1, and the whole process
/// (the title's threads too) defaults to those cores.
fn vcache_mode() -> u32 {
    static MODE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| std::env::var("VICTORY_VCACHE_CORES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// The logical processors (group 0, as a mask) sharing the largest L3, if
/// there are L3s of different sizes; with each L3's size in MB for the log.
fn largest_cache_mask() -> Option<(usize, Vec<(u32, usize)>)> {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationCache, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    let mut bytes = 0u32;
    // SAFETY: a size query.
    unsafe { GetLogicalProcessorInformationEx(RelationCache, std::ptr::null_mut(), &mut bytes) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetLogicalProcessorInformationEx(RelationCache, buffer.as_mut_ptr().cast(), &mut bytes) };
    if ok == 0 {
        return None;
    }
    let mut l3: Vec<(u32, usize)> = Vec::new();
    let mut at = 0usize;
    while at + 8 <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()) };
        if info.Size == 0 {
            break;
        }
        if info.Relationship == RelationCache {
            // SAFETY: a cache record, as asked for.
            let cache = unsafe { &info.Anonymous.Cache };
            // SAFETY: the first group mask is always there.
            let mask = unsafe { cache.Anonymous.GroupMask };
            if cache.Level == 3 && mask.Group == 0 {
                l3.push((cache.CacheSize, mask.Mask));
            }
        }
        at += info.Size as usize;
    }
    let sizes: Vec<(u32, usize)> = l3.iter().map(|&(size, mask)| (size >> 20, mask)).collect();
    let largest = l3.iter().map(|c| c.0).max()?;
    if l3.iter().all(|c| c.0 == largest) {
        return None;
    }
    let mask = l3.iter().filter(|c| c.0 == largest).fold(0usize, |m, c| m | c.1);
    Some((mask, sizes))
}

/// The CPU set ids of the largest-cache cores (empty when not applicable).
fn vcache_sets() -> &'static Vec<u32> {
    static SETS: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    SETS.get_or_init(|| {
        if vcache_mode() == 0 {
            return Vec::new();
        }
        let Some((mask, sizes)) = largest_cache_mask() else {
            tracing::info!("VICTORY_VCACHE_CORES: the last-level caches are alike (or unknown): nothing asked");
            return Vec::new();
        };
        let ids = cpu_set_ids_in(mask);
        tracing::info!(
            l3_mb_and_masks = ?sizes.iter().map(|(mb, m)| format!("{mb} MB {m:#x}")).collect::<Vec<_>>(),
            chosen_mask = format_args!("{mask:#x}"),
            sets = ids.len(),
            "VICTORY_VCACHE_CORES: the cores of the largest last-level cache"
        );
        ids
    })
}

/// The CPU set ids (group 0) whose logical processor is in `mask`.
fn cpu_set_ids_in(mask: usize) -> Vec<u32> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: as in `cpu_sets`.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as in `cpu_sets`.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut ids = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet.
        let set = unsafe { &info.Anonymous.CpuSet };
        let lp = set.LogicalProcessorIndex as u32;
        if set.Group == 0 && lp < usize::BITS && mask >> lp & 1 != 0 {
            ids.push(set.Id);
        }
        at += info.Size as usize;
    }
    ids
}

/// `VICTORY_VCACHE_CORES=2`: the process's threads default to the largest-cache
/// cores (threads that select their own sets keep them).
pub(crate) fn process_default() {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetProcessDefaultCpuSets};
    if vcache_mode() < 2 {
        return;
    }
    let sets = vcache_sets();
    if sets.is_empty() {
        return;
    }
    // SAFETY: our own process and a list of the system's own set ids.
    let ok = unsafe { SetProcessDefaultCpuSets(GetCurrentProcess(), sets.as_ptr(), sets.len() as u32) };
    if ok == 0 {
        tracing::warn!(error = %std::io::Error::last_os_error(), "VICTORY_VCACHE_CORES=2: the process default was not set");
    } else {
        tracing::info!(sets = sets.len(), "VICTORY_VCACHE_CORES=2: the process runs on the largest-cache cores");
    }
}

fn ask(what: &str, from: u32) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};
    if stage() < from {
        return;
    }
    static FAST: std::sync::OnceLock<(Vec<u32>, usize)> = std::sync::OnceLock::new();
    let (fast, of) = FAST.get_or_init(|| {
        let sets = cpu_sets();
        let fast = fast_of(&sets);
        // LOCAL (vcache): one class of core everywhere: the X3D die instead.
        let fast = if fast.is_empty() { vcache_sets().clone() } else { fast };
        (fast, sets.len())
    });
    // All alike (or nothing known): nothing to ask for.
    if fast.is_empty() {
        return;
    }
    // SAFETY: our own thread and a list of the system's own set ids.
    let ok = unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), fast.as_ptr(), fast.len() as u32) };
    if ok == 0 {
        tracing::warn!(what, error = %std::io::Error::last_os_error(), "the thread could not be kept on the fast cores");
    } else {
        tracing::info!(what, fast = fast.len(), of, "the thread is kept on the fast cores (VICTORY_FAST_CORES=0: not)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fast_cores_are_those_of_the_best_class_if_there_is_a_worse_one() {
        assert_eq!(fast_of(&[(256, 1), (257, 1), (258, 0)]), [256, 257]);
        assert!(fast_of(&[(256, 0), (257, 0)]).is_empty());
        assert!(fast_of(&[]).is_empty());
        assert!(!cpu_sets().is_empty());
        std::thread::spawn(|| keep_on_fast_cores("test")).join().unwrap();
    }
}

#[cfg(test)]
mod vcache_tests {
    use super::*;

    /// Prints what this machine reports (the X3D die on a 9950X3D: 96 MB vs 32 MB).
    #[test]
    fn the_largest_cache_is_found_or_all_alike() {
        match largest_cache_mask() {
            Some((mask, sizes)) => {
                eprintln!("L3s: {sizes:?}, largest mask {mask:#x}, sets {:?}", cpu_set_ids_in(mask));
                assert_ne!(mask, 0);
                assert!(!cpu_set_ids_in(mask).is_empty());
            }
            None => eprintln!("all L3s alike"),
        }
    }
}
