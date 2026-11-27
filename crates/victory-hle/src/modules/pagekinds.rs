//! LOCAL (page-kinds): build-specific, measurement-only reader. Never hooks or writes title memory.
//! VirtualQuery validates entire spans; ReadProcessMemory also catches unmap/protection races.
//! Volatile loads are from private copied words, avoiding an unprotected title-pointer dereference.

use std::sync::OnceLock;
use victory_gpu::pagekinds::{Region, Snapshot};
use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_IMAGE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
    VirtualQuery,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const GLOBALS: [usize; 4] = [0x09e1_dac0, 0x09e1_dac8, 0x09e1_dad0, 0x071b_fb08];
// LOCAL (page-kinds): same original-file BLAKE3 build fingerprint as victory-loader/src/hooks.rs.
const BUILD: &str = "b99f876b74d4a77f";

#[derive(Clone, Copy)]
struct Image {
    base: usize,
    bytes: usize,
}
static IMAGE: OnceLock<Result<Image, &'static str>> = OnceLock::new();

fn identity(base: usize, bytes: usize, hash: &str) -> Result<Image, &'static str> {
    if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) || !hash.starts_with(BUILD)
    {
        return Err("unsupported title build (original BLAKE3)");
    }
    if base < 65536
        || !base.is_multiple_of(65536)
        || bytes < GLOBALS[2] + 8
        || !base
            .checked_add(bytes)
            .is_some_and(|end| end <= 1usize << 47)
    {
        return Err("invalid title image extent");
    }
    Ok(Image { base, bytes })
}

/// LOCAL (page-kinds): called with the loader's original file hash and mapped image extent.
/// Registration does not read title memory and is a no-op unless the exact opt-in is set.
pub fn install(base: usize, bytes: usize, original_hash: &str) {
    if !victory_gpu::pagekinds::on() {
        return;
    }
    let _ = IMAGE.set(identity(base, bytes, original_hash));
    victory_gpu::pagekinds::set_reader(sample);
}

trait Memory {
    fn readable(&self, at: usize, bytes: usize, image: Option<usize>) -> bool;
    fn copy_words(&self, at: usize, out: &mut [u64]) -> bool;
}

struct Process;

fn readable_protection(protect: u32) -> bool {
    protect & PAGE_GUARD == 0
        && matches!(
            protect & 0xff,
            PAGE_READONLY
                | PAGE_READWRITE
                | PAGE_WRITECOPY
                | PAGE_EXECUTE_READ
                | PAGE_EXECUTE_READWRITE
                | PAGE_EXECUTE_WRITECOPY
        )
}

impl Memory for Process {
    fn readable(&self, at: usize, bytes: usize, image: Option<usize>) -> bool {
        let Some(end) = at
            .checked_add(bytes)
            .filter(|&e| at >= 65536 && e <= 1usize << 47 && bytes != 0)
        else {
            return false;
        };
        let mut here = at;
        while here < end {
            let mut info = MEMORY_BASIC_INFORMATION::default();
            // LOCAL (page-kinds): SAFETY: query metadata into a valid local structure, no title dereference.
            let got = unsafe {
                VirtualQuery(
                    here as *const _,
                    &mut info,
                    size_of::<MEMORY_BASIC_INFORMATION>(),
                )
            };
            if got != size_of::<MEMORY_BASIC_INFORMATION>()
                || info.State != MEM_COMMIT
                || !readable_protection(info.Protect)
                || image.is_some_and(|base| {
                    info.AllocationBase as usize != base || info.Type != MEM_IMAGE
                })
                || info.BaseAddress as usize > here
            {
                return false;
            }
            let Some(next) = (info.BaseAddress as usize)
                .checked_add(info.RegionSize)
                .filter(|&next| next > here)
            else {
                return false;
            };
            here = next.min(end);
        }
        true
    }

    fn copy_words(&self, at: usize, out: &mut [u64]) -> bool {
        let mut copied = 0;
        let bytes = std::mem::size_of_val(out);
        // LOCAL (page-kinds): SAFETY: kernel probes the source; destination is our valid owned slice.
        // No safe-copy VEH is used: it could open an emulator guard as a side effect.
        unsafe {
            ReadProcessMemory(
                GetCurrentProcess(),
                at as *const _,
                out.as_mut_ptr().cast(),
                bytes,
                &mut copied,
            ) != 0
                && copied == bytes
        }
    }
}

fn read_globals(memory: &impl Memory, image: Image) -> Result<[u64; 4], &'static str> {
    let mut values = [0; 4];
    for (value, rva) in values.iter_mut().zip(GLOBALS) {
        if rva + 8 > image.bytes {
            return Err("global outside image");
        }
        let at = image
            .base
            .checked_add(rva)
            .ok_or("global address overflow")?;
        if !memory.readable(at, 8, Some(image.base)) {
            return Err("unreadable global");
        }
        let mut word = [0];
        if !memory.copy_words(at, &mut word) {
            return Err("global read failed");
        }
        // LOCAL (page-kinds): SAFETY: local aligned word just copied by the kernel.
        *value = unsafe { word.as_ptr().read_volatile() };
    }
    Ok(values)
}

fn read_snapshot(memory: &impl Memory, image: Image) -> Result<Snapshot, &'static str> {
    let globals = read_globals(memory, image)?;
    let [cpu, bytes, gpu, table] = globals;
    let region = Region {
        cpu: cpu as usize,
        bytes: bytes as usize,
        gpu: gpu as usize,
        table: table as usize,
    }
    .validate()?;
    if !memory.readable(region.table, region.table_bytes(), None) {
        return Err("unreadable table");
    }
    let mut words = vec![0; region.bytes / victory_gpu::pagekinds::PAGE];
    if !memory.copy_words(region.table, &mut words) {
        return Err("table read failed");
    }
    if read_globals(memory, image)? != globals {
        return Err("region changed during sample");
    }
    // LOCAL (page-kinds): the copy is an observation, not an atomic title-wide snapshot; writers may run.
    // Recheck protection after copying too, and do not retain pointers to any title pages.
    if !memory.readable(region.table, region.table_bytes(), None) {
        return Err("table became unreadable");
    }
    for word in &mut words {
        // LOCAL (page-kinds): SAFETY: volatile observation of our private copied, aligned u64.
        *word = unsafe { (word as *const u64).read_volatile() };
    }
    Snapshot::new(region, &words)
}

fn sample() -> Result<Snapshot, &'static str> {
    if !victory_gpu::pagekinds::on() {
        return Err("disabled");
    }
    let image = *IMAGE
        .get()
        .ok_or("image not registered")?
        .as_ref()
        .map_err(|&e| e)?;
    read_snapshot(&Process, image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use victory_gpu::pagekinds::PAGE;

    struct Fake {
        globals: [u64; 4],
        deny: Option<usize>,
        fail_copy: bool,
        fail_table_copy: bool,
        change_globals: bool,
        copies: Cell<usize>,
        table_checked: Cell<usize>,
    }

    fn image() -> Image {
        Image {
            base: 0x140000000,
            bytes: 0xa000000,
        }
    }
    fn fake() -> Fake {
        Fake {
            globals: [0x100000, (4 * PAGE) as u64, 0x200000, 0x300000],
            deny: None,
            fail_copy: false,
            fail_table_copy: false,
            change_globals: false,
            copies: Cell::new(0),
            table_checked: Cell::new(0),
        }
    }

    impl Memory for Fake {
        fn readable(&self, at: usize, bytes: usize, _: Option<usize>) -> bool {
            if at == self.globals[3] as usize {
                self.table_checked.set(bytes);
            }
            !self
                .deny
                .is_some_and(|deny| at <= deny && deny < at + bytes)
        }
        fn copy_words(&self, at: usize, out: &mut [u64]) -> bool {
            if self.fail_copy || (self.fail_table_copy && at == self.globals[3] as usize) {
                return false;
            }
            self.copies.set(self.copies.get() + 1);
            if let Some(i) = GLOBALS.iter().position(|&rva| image().base + rva == at) {
                out[0] = self.globals[i];
                if self.change_globals && self.copies.get() > 5 {
                    out[0] ^= 1;
                }
            } else {
                out.copy_from_slice(&[2 << 62, 3 << 62, 1 << 62, 0]);
            }
            true
        }
    }

    #[test]
    fn pagekinds_validated_reader_and_whole_table() {
        let m = fake();
        let s = read_snapshot(&m, image()).unwrap();
        assert_eq!(m.table_checked.get(), 32);
        assert_eq!(
            s.query(0x100000, 4 * PAGE).unwrap().counts.pages,
            [1, 1, 1, 1, 0]
        );
    }

    #[test]
    fn pagekinds_validation_failures_are_unavailable() {
        let mut m = fake();
        m.deny = Some(0x300000 + 31);
        assert_eq!(read_snapshot(&m, image()).unwrap_err(), "unreadable table");
        assert_eq!(m.copies.get(), 4);
        let mut m = fake();
        m.deny = Some(image().base + GLOBALS[3]);
        assert_eq!(read_snapshot(&m, image()).unwrap_err(), "unreadable global");
        let mut m = fake();
        m.globals[0] += 1;
        assert!(read_snapshot(&m, image()).is_err());
        for size in [0, (PAGE + 1) as u64, (16u64 << 30) + PAGE as u64] {
            let mut m = fake();
            m.globals[1] = size;
            assert_eq!(
                read_snapshot(&m, image()).unwrap_err(),
                "invalid region size"
            );
        }
        let mut m = fake();
        m.fail_copy = true;
        assert_eq!(
            read_snapshot(&m, image()).unwrap_err(),
            "global read failed"
        );
        let mut m = fake();
        m.change_globals = true;
        assert_eq!(
            read_snapshot(&m, image()).unwrap_err(),
            "region changed during sample"
        );
        let mut m = fake();
        m.fail_table_copy = true;
        assert_eq!(read_snapshot(&m, image()).unwrap_err(), "table read failed");
    }

    #[test]
    fn pagekinds_refuses_unknown_identity() {
        let known = format!("{BUILD}{}", "0".repeat(48));
        assert!(identity(image().base, image().bytes, &known).is_ok());
        assert!(identity(image().base, image().bytes, &"0".repeat(64)).is_err());
        assert!(identity(image().base, image().bytes, BUILD).is_err());
        assert!(identity(image().base, 0x1000, &known).is_err());
    }

    #[test]
    fn pagekinds_rejects_guard_noaccess_and_execute_only() {
        use windows_sys::Win32::System::Memory::{PAGE_EXECUTE, PAGE_NOACCESS};
        for protect in [0, PAGE_NOACCESS, PAGE_EXECUTE, PAGE_READWRITE | PAGE_GUARD] {
            assert!(!readable_protection(protect));
        }
        assert!(readable_protection(PAGE_READWRITE));
        assert!(!Process.readable(0, 8, None));
        assert!(!Process.readable(usize::MAX - 3, 8, None));
    }

    #[test]
    fn pagekinds_real_memory_spans_and_fault_safe_copy() {
        use windows_sys::Win32::System::Memory::{
            MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc, VirtualFree, VirtualProtect,
        };
        struct Allocation(*mut std::ffi::c_void);
        impl Drop for Allocation {
            fn drop(&mut self) {
                // LOCAL (page-kinds): SAFETY: frees only this test's private allocation.
                unsafe {
                    VirtualFree(self.0, 0, MEM_RELEASE);
                }
            }
        }
        // LOCAL (page-kinds): SAFETY: test-owned anonymous memory, no title loaded or executed.
        let allocation = Allocation(unsafe {
            VirtualAlloc(
                std::ptr::null(),
                8192,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        });
        assert!(!allocation.0.is_null());
        let base = allocation.0 as usize;
        assert!(Process.readable(base, 8192, None));
        assert!(!Process.readable(base, 8, Some(base)));
        let mut copied = [0u64; 2];
        assert!(Process.copy_words(base + 4096 - 8, &mut copied));
        let mut old = 0;
        // LOCAL (page-kinds): SAFETY: protects only the second page owned by this test.
        assert_ne!(
            unsafe { VirtualProtect((base + 4096) as *const _, 4096, PAGE_NOACCESS, &mut old) },
            0
        );
        assert!(Process.readable(base, 4096, None));
        assert!(!Process.readable(base + 4096 - 8, 16, None));
        assert!(!Process.copy_words(base + 4096 - 8, &mut copied));
        // LOCAL (page-kinds): SAFETY: same test-owned page; the meter must leave its guard intact.
        assert_ne!(
            unsafe {
                VirtualProtect(
                    (base + 4096) as *const _,
                    4096,
                    PAGE_READWRITE | PAGE_GUARD,
                    &mut old,
                )
            },
            0
        );
        assert!(!Process.readable(base + 4096, 8, None));
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // LOCAL (page-kinds): SAFETY: metadata query into our local structure.
        unsafe {
            VirtualQuery(
                (base + 4096) as *const _,
                &mut info,
                size_of::<MEMORY_BASIC_INFORMATION>(),
            );
        }
        assert_ne!(info.Protect & PAGE_GUARD, 0);
    }
}
