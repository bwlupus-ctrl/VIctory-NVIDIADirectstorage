//! LOCAL (heap-probe): Stage 0 of moving the title's hot 174 MB range (0xae00000 bytes of pool
//! memory, views of physical pages below 4 GiB) into card memory: only measurements, and only
//! with `VICTORY_HEAP_PROBE=1` (off: every hook is one load of a `OnceLock` and a branch, and
//! nothing else in the process changes).
//!
//! What it answers:
//! 1. Who reads the range, the title's own code or ours (a CPU read of card memory is some 14
//!    MB/s)? Every `VICTORY_HEAP_PROBE_EVERY` frames (30) up to `VICTORY_HEAP_PROBE_PAGES` (64)
//!    pages of 4 KiB, spread over the range, are made `PAGE_NOACCESS` for at most one frame (the
//!    next [`tick`] gives them back). The first access to each faults into `watch::on_write`,
//!    which asks [`on_fault`] first: it notes read/write/execute and whether the instruction is in
//!    a title image, gives the page its protection back and goes on (the instruction runs again).
//! 2. How often its pages are remapped, and how many are shown at more than one address:
//!    [`note_call`] and friends count the runtime's mapping calls that touch the range, `watch`
//!    tells of each view mapped and unmapped, `inplace` knows the multiply shown pages.
//! 3. What writes it: DirectStorage ([`dstorage`]), the GPU's flights ([`flight_fetched`]: bytes
//!    fetched back from card copies) and write faults of the title in the guards ([`guard_fault`]).
//!
//! `VICTORY_HEAP_PROBE_SIZE=<hex>` picks another size of range; `VICTORY_HEAP_PROBE_AT=<hex>,<hex
//! length>` names a range and the registry is not asked.
//!
//! # How it keeps out of the guards' way (`watch.rs`)
//! The guards (`Guards`, under `GUARDS`) give 64 KiB blocks of the same views read-only and
//! writable again, and a `VirtualProtect` of a block that holds a `PAGE_NOACCESS` page of ours
//! would take it for "not ours" or overwrite it. So:
//! - A page is armed only under `GUARDS` (try-locked: a busy lock skips the window) and the
//!   mapping lock (`title_memory_reading_now`: no change of mapping is under way and none begins
//!   while we arm), and not in a block the kernel writes or a change has left alone.
//! - Every place that changes protections or mappings tells us first, under the same lock
//!   ([`release_range`]): `protect_here`, `protect_alias`, `unguard`, `title_memory_changed`
//!   (every remap and unmap), `title_memory_freed` and `kernel_writing`. A released page gets
//!   the protection it had back at once, so the guard code sees what it always saw, and is
//!   dropped from the sample: nothing is restored over what the guards did later.
//! - The restore of a page that is not released is the handler's, or [`end_window`]'s: they claim
//!   the page's slot first (`ARMED` to `CLAIMED`, one compare-exchange), and a release that
//!   finds a claimed slot waits for the claim to end (a `VirtualProtect`, no lock), so two
//!   restores never cross a guard's change. A restore only touches a page that still reads
//!   `PAGE_NOACCESS` (the handler knows it by the fault).
//! - The emulator's own readers that ask Windows what is readable (`texture::readable`,
//!   `host::read_sparse`) would take a page of ours for unmapped: they hold [`emulator_reading`]
//!   around the question, which gives the pages back first and keeps new ones from being armed
//!   until it is dropped (counted as reads of ours that were announced).
//! - The handler takes no lock, allocates nothing, and only claims a page it has armed; any other
//!   fault goes on as before.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
    VirtualProtect, VirtualQuery,
};

const KP: usize = 4096;
/// The most pages armed at once.
const SLOTS: usize = 512;
/// The most ranges followed at once.
const TARGETS: usize = 8;
const BLOCK64: usize = 64 << 10;
const DEFAULT_SIZE: usize = 0xae0_0000;
const FOUR_GIB: usize = 1 << 32;

// ---------------------------------------------------------------- configuration

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Config {
    pub on: bool,
    /// The size of range to follow.
    pub size: usize,
    /// A range named by `VICTORY_HEAP_PROBE_AT`: (start, length).
    pub at: Option<(usize, usize)>,
    /// Frames between windows.
    pub every: u64,
    /// Pages per window.
    pub pages: usize,
    /// `VICTORY_HEAP_PROBE=1` was asked for but `VICTORY_PROFILE` (the stack sampler, which reads
    /// threads' contexts and stacks at any moment) is on: not probing.
    pub blocked_by_profile: bool,
}

fn parse_hex(s: &str) -> Option<usize> {
    let s = s.trim();
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    usize::from_str_radix(s, 16).ok()
}

fn parse_at(s: &str) -> Option<(usize, usize)> {
    let (a, l) = s.split_once(',')?;
    let (a, l) = (parse_hex(a)?, parse_hex(l)?);
    (a != 0 && l != 0 && a.checked_add(l).is_some()).then_some((a, l))
}

pub(crate) fn config_from(get: &dyn Fn(&str) -> Option<String>) -> Config {
    let asked = get("VICTORY_HEAP_PROBE").is_some_and(|v| v != "0");
    let blocked_by_profile = asked && get("VICTORY_PROFILE").is_some();
    let on = asked && !blocked_by_profile;
    let size = get("VICTORY_HEAP_PROBE_SIZE").and_then(|v| parse_hex(&v)).filter(|&s| s != 0).unwrap_or(DEFAULT_SIZE);
    let at = get("VICTORY_HEAP_PROBE_AT").and_then(|v| parse_at(&v));
    let every = get("VICTORY_HEAP_PROBE_EVERY").and_then(|v| v.trim().parse::<u64>().ok()).filter(|&n| n != 0).unwrap_or(30).max(2);
    let pages = get("VICTORY_HEAP_PROBE_PAGES")
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n != 0)
        .unwrap_or(64)
        .min(SLOTS);
    Config { on, size, at, every, pages, blocked_by_profile }
}

fn config() -> &'static Config {
    static CONFIG: OnceLock<Config> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let c = config_from(&|k| std::env::var(k).ok());
        if c.blocked_by_profile {
            tracing::warn!("heap probe: off, because VICTORY_PROFILE (the stack sampler) is on");
        }
        if c.on {
            if let Some((a, l)) = c.at {
                add_target(a, a + l);
            }
            tracing::info!(
                "heap probe on: ranges of {:#x} bytes{}; {} pages every {} frames",
                c.size,
                c.at.map_or(String::new(), |(a, l)| format!(" (named: {a:#x}+{l:#x})")),
                c.pages,
                c.every
            );
        }
        c
    })
}

/// `VICTORY_HEAP_PROBE=1`.
#[inline]
pub fn on() -> bool {
    config().on
}

// ---------------------------------------------------------------- the ranges followed

static TARGET: [(AtomicUsize, AtomicUsize); TARGETS] = [const { (AtomicUsize::new(0), AtomicUsize::new(0)) }; TARGETS];
static TARGET_EDIT: Mutex<()> = Mutex::new(());

fn targets() -> [(usize, usize); TARGETS] {
    let mut out = [(0, 0); TARGETS];
    for (o, t) in out.iter_mut().zip(&TARGET) {
        *o = (t.0.load(Acquire), t.1.load(Acquire));
    }
    out
}

/// LOCAL (page-kinds): the probe's current targets as (start, len), for the
/// page-kind cross-tab (empty unless `VICTORY_HEAP_PROBE=1` found one).
pub(crate) fn target_ranges() -> Vec<(usize, usize)> {
    targets().into_iter().filter(|&(s, e)| e > s).map(|(s, e)| (s, e - s)).collect()
}

fn add_target(start: usize, end: usize) {
    let _edit = TARGET_EDIT.lock().unwrap_or_else(|e| e.into_inner());
    let mut free = None;
    for (i, t) in TARGET.iter().enumerate() {
        let (s, e) = (t.0.load(Relaxed), t.1.load(Relaxed));
        if e == 0 {
            free.get_or_insert(i);
        } else if s < end && start < e {
            // The same range again, or what replaces it.
            if (s, e) == (start, end) {
                return;
            }
            t.1.store(0, Release);
            t.0.store(0, Release);
            free.get_or_insert(i);
        }
    }
    if let Some(i) = free {
        let t = &TARGET[i];
        t.1.store(0, Release);
        t.0.store(start, Release);
        t.1.store(end, Release);
        tracing::info!("heap probe: following {start:#x}+{:#x} ({} MB)", end - start, (end - start) >> 20);
    }
}

fn remove_target(start: usize) {
    let _edit = TARGET_EDIT.lock().unwrap_or_else(|e| e.into_inner());
    for t in &TARGET {
        if t.0.load(Relaxed) == start && t.1.load(Relaxed) != 0 {
            t.1.store(0, Release);
            t.0.store(0, Release);
            tracing::info!("heap probe: {start:#x} released");
        }
    }
}

/// Whether an allocation or view `[start, start + len)` is one to follow.
pub(crate) fn wants(size: usize, start: usize, len: usize) -> bool {
    len == size && start != 0 && start.checked_add(len).is_some_and(|e| e <= FOUR_GIB)
}

/// The bytes of `[at, at + len)` that lie in `ranges` (empty ones are `(0, 0)`).
pub(crate) fn overlap_bytes(ranges: &[(usize, usize)], at: usize, len: usize) -> usize {
    let end = at.saturating_add(len);
    ranges.iter().filter(|r| r.1 > r.0).map(|&(s, e)| e.min(end).saturating_sub(s.max(at))).sum()
}

/// The bytes of `[at, at + len)` in the ranges followed.
pub fn overlap(at: usize, len: usize) -> usize {
    if !on() {
        return 0;
    }
    overlap_bytes(&targets(), at, len)
}

/// `memclass::allocated`: the title reserved `[at, at + size)`; `pool`: for physical pages.
pub fn allocated(at: usize, size: usize, pool: bool) {
    if !on() {
        return;
    }
    let c = config();
    if c.at.is_none() && pool && wants(c.size, at, size) {
        add_target(at, at + size);
    }
}

/// `memclass::pool_mapped`: a view of pool pages at `[at, at + len)`.
pub fn pool_mapped(at: usize, len: usize) {
    if !on() {
        return;
    }
    let c = config();
    if c.at.is_none() && wants(c.size, at, len) {
        add_target(at, at + len);
    }
}

/// `memclass::released`: the title released the allocation at `at`.
pub fn released(at: usize) {
    if on() {
        remove_target(at);
    }
}

// ---------------------------------------------------------------- the title's images

const IMAGES: usize = 16;
static IMAGE: [(AtomicUsize, AtomicUsize); IMAGES] = [const { (AtomicUsize::new(0), AtomicUsize::new(0)) }; IMAGES];

fn query(at: usize) -> Option<MEMORY_BASIC_INFORMATION> {
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    (n != 0).then_some(info)
}

/// The size of the image mapped at `base`: its header's, if that reads, else the regions that
/// share its allocation.
fn image_size(base: usize) -> usize {
    let walked = {
        let mut end = base;
        for _ in 0..4096 {
            match query(end) {
                Some(i) if i.AllocationBase as usize == base && i.State == MEM_COMMIT => {
                    end = (i.BaseAddress as usize).saturating_add(i.RegionSize);
                }
                _ => break,
            }
        }
        end - base
    };
    let header = (|| {
        let i = query(base)?;
        if i.State != MEM_COMMIT || i.RegionSize < 0x200 {
            return None;
        }
        // SAFETY: committed memory of at least 0x200 bytes at `base` (read through unaligned loads).
        unsafe {
            if (base as *const u16).read_unaligned() != 0x5a4d {
                return None;
            }
            let nt = (base as *const u8).add(0x3c).cast::<u32>().read_unaligned() as usize;
            if nt > 0x100 || (base as *const u8).add(nt).cast::<u32>().read_unaligned() != 0x4550 {
                return None;
            }
            Some((base as *const u8).add(nt + 0x18 + 0x38).cast::<u32>().read_unaligned() as usize)
        }
    })();
    header.unwrap_or(0).max(walked)
}

/// A title image is mapped at `base` (`main`: the title's own executable, whose offsets the log
/// gives).
pub fn add_title_image(base: usize, main: bool) {
    if !on() || base == 0 {
        return;
    }
    let size = image_size(base);
    let end = base + size.max(KP);
    if main {
        IMAGE[0].1.store(0, Release);
        IMAGE[0].0.store(base, Release);
        IMAGE[0].1.store(end, Release);
    } else if base != IMAGE[0].0.load(Acquire) {
        for t in IMAGE.iter().skip(1) {
            let b = t.0.load(Acquire);
            if b == base {
                return;
            }
            if b == 0 {
                t.1.store(0, Release);
                t.0.store(base, Release);
                t.1.store(end, Release);
                break;
            }
        }
    }
    tracing::info!("heap probe: title image {base:#x}+{size:#x}{}", if main { " (main)" } else { "" });
}

/// The index and base of the title image holding `rip`.
fn title_image_of(rip: usize) -> Option<(usize, usize)> {
    for (i, t) in IMAGE.iter().enumerate() {
        let (b, e) = (t.0.load(Acquire), t.1.load(Acquire));
        if b != 0 && rip >= b && rip < e {
            return Some((i, b));
        }
    }
    None
}

// ---------------------------------------------------------------- counters

#[derive(Clone, Copy)]
#[repr(usize)]
enum C {
    Sampled,
    Windows,
    ReadTitle,
    ReadOurs,
    WriteTitle,
    WriteOurs,
    ExecTitle,
    ExecOurs,
    OtherFaults,
    Untouched,
    DroppedWatch,
    DroppedEmu,
    Skipped,
    ArmFailed,
    ArmBusy,
    Conflicts,
    ReadHelper,
    MapCalls,
    MapBytes,
    UnmapCalls,
    UnmapBytes,
    ScatterCalls,
    ScatterBytes,
    AliasCalls,
    AliasBytes,
    CopyCalls,
    CopyBytes,
    ViewMapped,
    ViewMappedBytes,
    ViewUnmapped,
    ViewUnmappedBytes,
    Dstorage,
    Flights,
    GuardFaults,
    RingLost,
}
const COUNTERS: usize = C::RingLost as usize + 1;
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

#[inline]
fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

/// The kinds of mapping call counted.
#[derive(Clone, Copy, Debug)]
pub enum Call {
    /// `XMemMapPhysicalPages` with pages.
    Map,
    /// `XMemMapPhysicalPages` without.
    Unmap,
    /// `XMemMapPhysicalPagesScatter`.
    Scatter,
    /// `map_alias` (the queue's page mappings as the runtime does them).
    Alias,
    /// `CopyPageMappingsBatchX`.
    CopyMap,
}

/// One call of `kind` of which `bytes` lie in the range followed (none: not counted).
pub fn note_bytes(kind: Call, bytes: usize) {
    if bytes == 0 {
        return;
    }
    let (calls, total) = match kind {
        Call::Map => (C::MapCalls, C::MapBytes),
        Call::Unmap => (C::UnmapCalls, C::UnmapBytes),
        Call::Scatter => (C::ScatterCalls, C::ScatterBytes),
        Call::Alias => (C::AliasCalls, C::AliasBytes),
        Call::CopyMap => (C::CopyCalls, C::CopyBytes),
    };
    add(calls, 1);
    add(total, bytes as u64);
}

/// A call of `kind` over `[at, at + len)`.
pub fn note_call(kind: Call, at: usize, len: usize) {
    if on() {
        note_bytes(kind, overlap(at, len));
    }
}

/// A call of `kind` over several ranges (one call).
pub fn note_calls(kind: Call, ranges: impl Iterator<Item = (usize, usize)>) {
    if !on() {
        return;
    }
    let t = targets();
    note_bytes(kind, ranges.map(|(at, len)| overlap_bytes(&t, at, len)).sum());
}

/// `watch`: a view was mapped (`mapped`) or is about to be unmapped at `[at, at + len)`.
pub fn view(mapped: bool, at: usize, len: usize) {
    if !on() {
        return;
    }
    let n = overlap(at, len);
    if n == 0 {
        return;
    }
    if mapped {
        add(C::ViewMapped, 1);
        add(C::ViewMappedBytes, n as u64);
    } else {
        add(C::ViewUnmapped, 1);
        add(C::ViewUnmappedBytes, n as u64);
    }
}

/// DirectStorage wrote `[at, at + len)` of title memory.
pub fn dstorage(at: usize, len: usize) {
    if on() {
        add(C::Dstorage, overlap(at, len) as u64);
    }
}

/// The GPU's flights wrote card copies that were fetched back to `[at, at + len)`.
pub fn flight_fetched(at: usize, len: usize) {
    if on() {
        add(C::Flights, overlap(at, len) as u64);
    }
}

/// A write fault of the title in a guarded block at `at`.
pub(crate) fn guard_fault(at: usize) {
    if on() && overlap(at, 1) != 0 {
        add(C::GuardFaults, 1);
    }
}

// ---------------------------------------------------------------- the sampled pages
//
// Ownership: while a slot is ARMING, ARMED or CLAIMED the page's `PAGE_NOACCESS` is the slot's,
// and only whoever moved the slot from ARMED to CLAIMED (one compare-exchange) may give the page
// its protection back, once, through `VirtualProtect`, which reports what was there: anything
// but `PAGE_NOACCESS` means someone else changed the page since, and that is put back at once.
// A slot in HIT, DROPPED or UNTOUCHED owns nothing and is never restored from: nothing is
// "healed" afterwards, so a later legitimate `PAGE_NOACCESS` is never flipped.

const FREE: u32 = 0;
/// Being armed: the protection is about to be changed.
const ARMING: u32 = 6;
const ARMED: u32 = 1;
/// A restore of the page is under way by whoever claimed it (the handler, a release, the end of
/// the window).
const CLAIMED: u32 = 2;
/// Touched: the handler gave the page back.
const HIT: u32 = 3;
/// Given back because the guards or the runtime were about to change it, or the emulator reads.
const DROPPED: u32 = 4;
/// Given back at the end of the window, untouched.
const UNTOUCHED: u32 = 5;

struct Slot {
    addr: AtomicUsize,
    /// The page's own protection (what `VirtualProtect` said was there when it was armed); read
    /// after the claim.
    prot: AtomicU32,
    /// One word: the state (bits 0-7), a generation bumped each time the slot is armed or taken
    /// over (8-31), and the thread that owns the slot, who armed or claimed it (32-63). A claim,
    /// an owner's every transition and a takeover are compare-exchanges of the whole word, so
    /// whoever acts on what it saw before the slot changed hands fails and does nothing more.
    state: AtomicU64,
}

impl Slot {
    const fn new() -> Slot {
        Slot { addr: AtomicUsize::new(0), prot: AtomicU32::new(0), state: AtomicU64::new(FREE as u64) }
    }

    fn st(&self) -> u32 {
        (self.state.load(SeqCst) & 0xff) as u32
    }

    /// ARMED to CLAIMED, by this thread: the owner's token (the word as it stands now), or none.
    fn claim(&self) -> Option<u64> {
        let v = self.state.load(SeqCst);
        if v & 0xff != u64::from(ARMED) {
            return None;
        }
        let new = u64::from(thread_id()) << 32 | v & 0xffff_ff00 | u64::from(CLAIMED);
        self.state.compare_exchange(v, new, SeqCst, SeqCst).is_ok().then_some(new)
    }

    /// The owner's move to `to`; false if the slot is no longer the owner's (taken over).
    fn finish(&self, token: u64, to: u32) -> bool {
        self.state.compare_exchange(token, token & !0xff | u64::from(to), SeqCst, SeqCst).is_ok()
    }

    /// A slot nobody owns any more (HIT, DROPPED, UNTOUCHED) is free to be armed again.
    fn free_if_done(&self) {
        let v = self.state.load(SeqCst);
        if matches!((v & 0xff) as u32, HIT | DROPPED | UNTOUCHED) {
            let _ = self.state.compare_exchange(v, v & !0xff | u64::from(FREE), SeqCst, SeqCst);
        }
    }

    /// A free slot becomes ARMING, owned by this thread, under a generation of its own: the token.
    fn arming(&self) -> u64 {
        let v = self.state.load(SeqCst);
        let generation = (((v >> 8) & 0xff_ffff) + 1) & 0xff_ffff;
        let token = u64::from(thread_id()) << 32 | generation << 8 | u64::from(ARMING);
        self.state.store(token, SeqCst);
        token
    }
}

static SLOT: [Slot; SLOTS] = [const { Slot::new() }; SLOTS];
/// Slots `ARMING`, `ARMED` or `CLAIMED`.
static LIVE: AtomicUsize = AtomicUsize::new(0);
/// The addresses armed pages lie in, `[LO, HI)`.
static LO: AtomicUsize = AtomicUsize::new(usize::MAX);
static HI: AtomicUsize = AtomicUsize::new(0);
/// The addresses pages were ever armed in (only widened): where a late fault is looked at.
static ELO: AtomicUsize = AtomicUsize::new(usize::MAX);
static EHI: AtomicUsize = AtomicUsize::new(0);
/// The pages ever armed, a bit for each 4 KiB page below 4 GiB (hashed above): set before the
/// page is first made inaccessible and never cleared, so a thread that faulted on a page that
/// is given back by the time the handler runs is always told from one with a fault of its own
/// (a bit set by a hash collision only matters when the page can be accessed that way, when no
/// fault could have come).
const SEEN_BITS: usize = 1 << 20;
static SEEN: [AtomicU64; SEEN_BITS / 64] = [const { AtomicU64::new(0) }; SEEN_BITS / 64];

fn seen_bit(page: usize) -> usize {
    let n = page >> 12;
    if page < FOUR_GIB { n } else { (n as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) as usize >> 44 }
}

fn seen_note(page: usize) {
    let b = seen_bit(page);
    SEEN[b / 64].fetch_or(1 << (b % 64), SeqCst);
}

fn seen_has(page: usize) -> bool {
    let b = seen_bit(page);
    SEEN[b / 64].load(SeqCst) & (1 << (b % 64)) != 0
}

/// `watch::writing`: the title's or the emulator's write into `[at, at + len)` follows, perhaps
/// by the kernel: its pages are given back. No page is armed while the count of readers and
/// writers ([`BUSY`]) is not 0, and the release is made inside it, so a page armed before is
/// found and none comes after; a writer whose write goes on after this returns holds
/// [`emulator_reading`] over it (the host's copies do; `kernel_writing` keeps the blocks of its
/// write out of the sample).
pub(crate) fn writing_range(at: usize, len: usize) {
    if !on() || len == 0 {
        return;
    }
    BUSY.fetch_add(1, SeqCst);
    release_range(at, at.saturating_add(len), Why::Watch);
    BUSY.fetch_sub(1, SeqCst);
}

/// Readers and writers of the emulator (and calls of the title) that must not meet an armed page
/// ([`emulator_reading`], [`writing_range`]): while it is not 0 no page is armed. (Armed pages
/// are counted by `LIVE` first, then this is looked at: a reader that raised it first sees the
/// page, or the arming sees the reader.)
static BUSY: AtomicUsize = AtomicUsize::new(0);

/// Whether pages have been armed (their addresses are looked at in faults).
#[inline]
pub(crate) fn listening() -> bool {
    EHI.load(Relaxed) != 0
}

/// Whether any page is armed (or being armed).
#[inline]
pub(crate) fn live() -> bool {
    LIVE.load(SeqCst) != 0
}

fn thread_id() -> u32 {
    // SAFETY: a plain query of the calling thread.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

/// Whether the thread `id` has ended (killed, as the exit ends the title's threads).
fn thread_ended(id: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{OpenThread, WaitForSingleObject};
    const SYNCHRONIZE: u32 = 0x10_0000;
    const ERROR_INVALID_PARAMETER: u32 = 87;
    if id == 0 {
        return false;
    }
    // SAFETY: a handle to a thread of ours, closed at once.
    unsafe {
        let h = OpenThread(SYNCHRONIZE, 0, id);
        if h.is_null() {
            return GetLastError() == ERROR_INVALID_PARAMETER;
        }
        let ended = WaitForSingleObject(h, 0) == WAIT_OBJECT_0;
        CloseHandle(h);
        ended
    }
}

fn pause(i: u32) {
    if i < 64 {
        std::hint::spin_loop();
    } else if i < 2000 {
        // SAFETY: gives the processor to another thread.
        unsafe { windows_sys::Win32::System::Threading::SwitchToThread() };
    } else {
        // SAFETY: sleeps this thread.
        unsafe { windows_sys::Win32::System::Threading::Sleep(1) };
    }
}

enum Take {
    /// The slot was armed and is the caller's to restore now (the owner's token).
    Claimed(u64),
    /// (Handler only.) Another thread's restore was under way and is over, or is still going on:
    /// the faulting instruction runs again; nothing was restored.
    Retry,
    /// Not armed (any more): nothing is owned.
    Gone,
}

/// The handler's claim: no long wait, no lock. A slot being armed or restored by another thread
/// is looked at a few thousand times; then the instruction runs again (and faults again if the
/// page is still ours), which is a wait of the exception path's own.
fn take_quick(s: &Slot) -> Take {
    let mut waited = false;
    for i in 0..2000u32 {
        match s.st() {
            ARMED => {
                if let Some(t) = s.claim() {
                    return Take::Claimed(t);
                }
            }
            ARMING | CLAIMED => {
                waited = true;
                if i % 64 == 63 {
                    // SAFETY: gives the processor to another thread.
                    unsafe { windows_sys::Win32::System::Threading::SwitchToThread() };
                } else {
                    std::hint::spin_loop();
                }
            }
            // The page was NOACCESS when this thread faulted and has been given back since.
            HIT if waited => return Take::Retry,
            _ => return Take::Gone,
        }
    }
    Take::Retry
}

/// A release's claim (guards' lock held, or the end of a window): waits, as long as it takes,
/// for an arming or a restore in progress. Whoever owns a page is in the middle of one
/// `VirtualProtect` (no lock, nothing that blocks), so the wait ends; the owner is never
/// overtaken, so no protection is written by one that lost its page: a protection or a mapping
/// is not changed with the ownership of a page in it unresolved. (A thread suspended inside the
/// handler holds this up until it is resumed; said once after five seconds.)
///
/// What was claimed is whatever the slot holds now: the caller reads its address after the claim
/// and gives the slot back ([`Slot::set`] to ARMED) if it is not the page it came for.
fn take_patient(s: &Slot) -> Take {
    let mut i = 0u32;
    let started = std::time::Instant::now();
    let mut said = false;
    loop {
        match s.st() {
            ARMED => {
                if let Some(t) = s.claim() {
                    return Take::Claimed(t);
                }
                continue;
            }
            ARMING | CLAIMED => {}
            _ => return Take::Gone,
        }
        i = i.saturating_add(1);
        pause(i);
        // The owner was killed (the exit ends the title's threads): it can write nothing more, so
        // the slot is taken over, under a generation of its own.
        if i % 32 == 0 && started.elapsed().as_millis() >= 50 {
            // (From exactly the word seen with that thread's id in it: an owner that came back
            // finds its own transitions refused.)
            let v = s.state.load(SeqCst);
            if matches!(v & 0xff, x if x == u64::from(CLAIMED) || x == u64::from(ARMING)) && thread_ended((v >> 32) as u32) {
                let generation = (((v >> 8) & 0xff_ffff) + 1) & 0xff_ffff;
                let new = u64::from(thread_id()) << 32 | generation << 8 | u64::from(CLAIMED);
                if s.state.compare_exchange(v, new, SeqCst, SeqCst).is_ok() {
                    return Take::Claimed(new);
                }
            }
        }
        if !said && i % 256 == 0 && started.elapsed().as_secs() >= 5 {
            said = true;
            tracing::warn!("heap probe: waiting for a restore of a sampled page for 5 s (a thread suspended in a fault?)");
        }
    }
}

/// Gives a page its protection back; the claimer's. False if the call failed (the page is gone).
/// If the page did not read `PAGE_NOACCESS` it was changed by someone since: that is put back.
fn restore_owned(page: usize, prot: u32) -> bool {
    // A slot whose arming thread was killed between making the page inaccessible and noting its
    // protection (only at the exit, when the sampling thread is ended) has none: the page is
    // given the protection of its allocation if that is plain read-only or read-write, else
    // read-write.
    let prot = if prot != 0 {
        prot
    } else {
        match query(page) {
            Some(i) if i.State == MEM_COMMIT && i.Protect & 0xff == PAGE_NOACCESS => {
                if i.AllocationProtect & 0xff == PAGE_READONLY { PAGE_READONLY } else { PAGE_READWRITE }
            }
            _ => return false,
        }
    };
    let mut old = 0;
    // SAFETY: a 4 KiB page of title memory; only its protection changes.
    let ok = unsafe { VirtualProtect(page as *const _, KP, prot, &mut old) != 0 };
    if ok && old & 0xff != PAGE_NOACCESS {
        let mut ignored = 0;
        // SAFETY: as above.
        unsafe { VirtualProtect(page as *const _, KP, old, &mut ignored) };
        add(C::Conflicts, 1);
    }
    ok
}

/// Arms `page` (4 KiB aligned) with the protection `prot` it is known to have.
pub(crate) fn arm_page(page: usize, prot: u32) -> bool {
    arm_inner(page, Some(prot))
}

/// Arms `page`: `PAGE_NOACCESS` until the handler, a release or [`end_window`] gives it back;
/// its protection is what `VirtualProtect` says was there (`known`: only a test's hint). Pages
/// that are not committed, or not plain read-only or read-write, are left as they are. Under the
/// guards' lock (`watch::heap_probe_try_locked`).
fn arm_inner(page: usize, known: Option<u32>) -> bool {
    let Some(s) = SLOT.iter().find(|s| s.st() == FREE) else {
        add(C::ArmFailed, 1);
        return false;
    };
    s.prot.store(known.unwrap_or(0), Relaxed);
    s.addr.store(page, Relaxed);
    let token = s.arming();
    LO.fetch_min(page, SeqCst);
    HI.fetch_max(page + KP, SeqCst);
    ELO.fetch_min(page, SeqCst);
    EHI.fetch_max(page + KP, SeqCst);
    // For good, before the page is first made inaccessible: a thread that faults on it can always
    // be told from one with a fault of its own.
    seen_note(page);
    LIVE.fetch_add(1, SeqCst);
    // (A refused slot keeps its page: a fault that came before the protection was put back finds
    // it, `on_fault`.) Moved only if it is still ours; if it was taken over, the one who did
    // has done the rest.
    let free = |s: &Slot| {
        if s.finish(token, FREE) {
            LIVE.fetch_sub(1, SeqCst);
        }
    };
    // A reader that began before this is not to meet the page. (It either sees `LIVE` and
    // releases us, or we see it here, before anything is changed.)
    if BUSY.load(SeqCst) != 0 {
        free(s);
        add(C::ArmBusy, 1);
        return false;
    }
    let mut old = 0;
    // SAFETY: a 4 KiB page of title memory; only its protection changes.
    if unsafe { VirtualProtect(page as *const _, KP, PAGE_NOACCESS, &mut old) } == 0 {
        free(s);
        add(C::ArmFailed, 1);
        return false;
    }
    if known.is_none() && (old & PAGE_GUARD != 0 || !matches!(old & 0xff, PAGE_READONLY | PAGE_READWRITE)) {
        // Not ours to look at: as it was.
        let mut ignored = 0;
        // SAFETY: as above.
        unsafe { VirtualProtect(page as *const _, KP, old, &mut ignored) };
        free(s);
        add(C::ArmFailed, 1);
        return false;
    }
    s.prot.store(old, Relaxed);
    s.finish(token, ARMED)
}

/// Why a page was given back before the window's end.
#[derive(Clone, Copy)]
pub(crate) enum Why {
    /// The guards or the runtime change protections or mappings there.
    Watch,
    /// The emulator (or the title's call) is about to read or change protections there.
    Emulator,
}

/// Gives back every armed page that overlaps `[at, end)` and drops it from the sample. Called
/// before a protection or mapping there changes (under the guards' lock where there is one: no
/// page can be armed meanwhile), or before the emulator reads.
pub(crate) fn release_range(at: usize, end: usize, why: Why) {
    if LIVE.load(SeqCst) == 0 || end <= at || end <= LO.load(SeqCst) || at >= HI.load(SeqCst) {
        return;
    }
    for s in &SLOT {
        let a = s.addr.load(Relaxed);
        if a == 0 || a + KP <= at || a >= end {
            continue;
        }
        if let Take::Claimed(token) = take_patient(s) {
            // What the claim got is what the slot holds now (it may have been freed and armed for
            // another page since it was looked at): that is restored, or, if it is not in the
            // range, left to its owner.
            let (a, prot) = (s.addr.load(Relaxed), s.prot.load(Relaxed));
            if a == 0 || a + KP <= at || a >= end {
                s.finish(token, ARMED);
                continue;
            }
            restore_owned(a, prot);
            if s.finish(token, DROPPED) {
                LIVE.fetch_sub(1, SeqCst);
            }
            add(
                match why {
                    Why::Watch => C::DroppedWatch,
                    Why::Emulator => C::DroppedEmu,
                },
                1,
            );
        }
    }
}

/// [`release_range`] for blocks (64 KiB) of the guards.
pub(crate) fn release_blocks(blocks: &[usize]) {
    if live() {
        for &b in blocks {
            release_range(b * BLOCK64, (b + 1) * BLOCK64, Why::Watch);
        }
    }
}

/// (Tests) everything as at the start (views are mapped at the same addresses again and again,
/// and a test that failed may have left pages armed).
#[cfg(test)]
pub(crate) fn reset_for_test() {
    end_window();
    for w in &SEEN {
        w.store(0, SeqCst);
    }
    for s in &SLOT {
        s.addr.store(0, SeqCst);
        s.state.store(FREE as u64, SeqCst);
    }
    BUSY.store(0, SeqCst);
    LIVE.store(0, SeqCst);
    LO.store(usize::MAX, SeqCst);
    HI.store(0, SeqCst);
    ELO.store(usize::MAX, SeqCst);
    EHI.store(0, SeqCst);
}

/// (Tests) the slots that are not free.
#[cfg(test)]
pub(crate) fn debug_slots() -> String {
    SLOT.iter()
        .filter(|s| s.st() != FREE)
        .map(|s| format!("{:#x}:{}", s.addr.load(SeqCst), s.st()))
        .collect::<Vec<_>>()
        .join(",")
        + &format!(" live {}", LIVE.load(SeqCst))
}

/// What [`emulator_reading`] returns.
pub struct EmuRead(bool);

impl Drop for EmuRead {
    fn drop(&mut self) {
        if self.0 {
            BUSY.fetch_sub(1, SeqCst);
        }
    }
}

/// Held by code (ours, or the title's call that we forward) that reads `[at, at + len)` through
/// the kernel, asks Windows what of it can be read, or changes its protection: the pages are
/// given back, and none is armed until it is dropped. A page given back counts as read by us
/// (announced).
#[inline]
pub fn emulator_reading(at: usize, len: usize) -> EmuRead {
    if !on() {
        return EmuRead(false);
    }
    BUSY.fetch_add(1, SeqCst);
    release_range(at, at.saturating_add(len), Why::Emulator);
    EmuRead(true)
}

/// A fault of access kind `kind` (`ExceptionInformation[0]`: 0 read, 1 write, 8 execute) at
/// `at` by the instruction at `rip` with the stack pointer `rsp`, first thing in
/// `watch::on_write`: true if the page was armed by us and has its protection back (or is about
/// to: another thread's restore), so the instruction can run again. No lock, no allocation, no
/// long wait.
pub(crate) fn on_fault(kind: usize, at: usize, rip: usize, rsp: usize) -> bool {
    if at < ELO.load(SeqCst) || at >= EHI.load(SeqCst) {
        return false;
    }
    let page = at & !(KP - 1);
    let mut stale = false;
    for s in &SLOT {
        if s.addr.load(Relaxed) != page {
            continue;
        }
        if !matches!(s.st(), ARMING | ARMED | CLAIMED) {
            // Given back already (a slot keeps its page until it is armed again).
            stale = true;
            continue;
        }
        match take_quick(s) {
            Take::Claimed(token) => {
                // (The slot may have been freed and armed for another page since it was looked at.)
                if s.addr.load(Relaxed) != page {
                    s.finish(token, ARMED);
                    stale = true;
                    continue;
                }
                let prot = s.prot.load(Relaxed);
                note_fault(kind, rip, rsp);
                let ok = restore_owned(page, prot);
                if s.finish(token, if ok { HIT } else { DROPPED }) {
                    LIVE.fetch_sub(1, SeqCst);
                }
                return ok;
            }
            Take::Retry => return true,
            Take::Gone => stale = true,
        }
    }
    // The page was ours when this thread faulted and was given back before the handler came (by
    // a release, the end of the window, another thread's fault): the instruction runs again if
    // the page can be accessed that way now; if not, the fault is for the guards (a write to a
    // page they hold read-only) or the title's.
    (stale || seen_has(page)) && accessible_for(kind, page)
}

/// Whether an access of kind `kind` (0 read, 1 write, 8 execute) to `page` can succeed now.
fn accessible_for(kind: usize, page: usize) -> bool {
    query(page).is_some_and(|i| {
        i.State == MEM_COMMIT
            && i.Protect & PAGE_GUARD == 0
            && match kind {
                0 => matches!(i.Protect & 0xff, 0x02 | 0x04 | 0x08 | 0x20 | 0x40 | 0x80),
                1 => matches!(i.Protect & 0xff, 0x04 | 0x08 | 0x40 | 0x80),
                8 => matches!(i.Protect & 0xff, 0x10 | 0x20 | 0x40 | 0x80),
                _ => false,
            }
    })
}

// ---------------------------------------------------------------- who faulted

const HELPERS: usize = 16;
/// Code that runs for the title's `memcpy`, `memmove` and `memset` (ours that the title is bound
/// to, and the host's they call): `[begin, end)`.
static HELPER: [(AtomicUsize, AtomicUsize); HELPERS] = [const { (AtomicUsize::new(0), AtomicUsize::new(0)) }; HELPERS];

/// The function at `addr` is code the title's copies run (`titlewrite::import`).
pub fn add_helper(addr: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::RtlLookupFunctionEntry;
    if !on() || addr == 0 {
        return;
    }
    let mut image = 0u64;
    // SAFETY: looks up the function table entry covering `addr`; nothing keeps it.
    let entry = unsafe { RtlLookupFunctionEntry(addr as u64, &mut image, std::ptr::null_mut()) };
    let (b, e) = if entry.is_null() {
        (addr, addr + 0x100)
    } else {
        // SAFETY: the entry the system returned.
        unsafe { (image as usize + (*entry).BeginAddress as usize, image as usize + (*entry).EndAddress as usize) }
    };
    for t in &HELPER {
        if t.0.load(Acquire) == b {
            return;
        }
        if t.1.load(Acquire) == 0 {
            t.0.store(b, Release);
            t.1.store(e, Release);
            tracing::info!("heap probe: title copy helper {b:#x}+{:#x}", e - b);
            return;
        }
    }
}

fn in_helper(q: usize) -> bool {
    HELPER.iter().any(|t| {
        let (b, e) = (t.0.load(Acquire), t.1.load(Acquire));
        e != 0 && q >= b && q < e
    })
}

/// Whether a fault at `rip`, with `stack` the words from the stack pointer up, is the title's
/// read: its own code, or code of the copies it is bound to (`memcpy` and its kin, which run
/// ours, which call the host's): then the first title address on the stack, the return into its
/// code, is the site. Returns (title's, the site, through a copy helper).
pub(crate) fn classify_with(rip: usize, stack: &[usize]) -> (bool, usize, bool) {
    if title_image_of(rip).is_some() {
        return (true, rip, false);
    }
    let Some(i) = stack.iter().position(|&q| title_image_of(q).is_some()) else {
        return (false, rip, false);
    };
    if in_helper(rip) || stack[..i].iter().any(|&q| in_helper(q)) {
        (true, stack[i], true)
    } else {
        (false, rip, false)
    }
}

/// Up to 64 words of this thread's stack from `rsp` up.
fn stack_words(rsp: usize) -> &'static [usize] {
    use windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits;
    let (mut lo, mut hi) = (0usize, 0usize);
    // SAFETY: plain out parameters.
    unsafe { GetCurrentThreadStackLimits(&mut lo, &mut hi) };
    if rsp < lo || rsp >= hi || rsp % 8 != 0 {
        return &[];
    }
    // SAFETY: committed stack of this thread between `rsp` and its top, read at once.
    unsafe { std::slice::from_raw_parts(rsp as *const usize, ((hi - rsp) / 8).min(64)) }
}

// The rings the handler fills with the sites of reads. An entry is written before anyone can
// read it: the consumer walks from its tail while entries are there and stops at one not yet
// written; a producer that finds an entry no one has taken (the ring has wrapped) counts a loss.
const RING: usize = 4096;

struct Ring {
    buf: [AtomicUsize; RING],
    reserved: AtomicUsize,
}

impl Ring {
    const fn new() -> Self {
        Ring { buf: [const { AtomicUsize::new(0) }; RING], reserved: AtomicUsize::new(0) }
    }

    fn push(&self, v: usize) {
        let i = self.reserved.fetch_add(1, AcqRel);
        if self.buf[i % RING].swap(v, AcqRel) != 0 {
            add(C::RingLost, 1);
        }
    }

    fn drain(&self, tail: &mut usize, mut f: impl FnMut(usize)) {
        let head = self.reserved.load(Acquire);
        if head.wrapping_sub(*tail) > RING {
            // (Overwritten: the producers counted them.)
            *tail = head - RING;
        }
        while *tail != head {
            let v = self.buf[*tail % RING].swap(0, AcqRel);
            if v == 0 {
                // Reserved, not yet written: next time.
                break;
            }
            f(v);
            *tail += 1;
        }
    }
}

static TITLE_RING: Ring = Ring::new();
static OURS_RING: Ring = Ring::new();

fn note_fault(kind: usize, rip: usize, rsp: usize) {
    let (title, site, helper) = classify_with(rip, stack_words(rsp));
    add(
        match (kind, title) {
            (0, true) => C::ReadTitle,
            (0, false) => C::ReadOurs,
            (1, true) => C::WriteTitle,
            (1, false) => C::WriteOurs,
            (8, true) => C::ExecTitle,
            (8, false) => C::ExecOurs,
            _ => C::OtherFaults,
        },
        1,
    );
    if kind == 0 {
        if helper {
            add(C::ReadHelper, 1);
        }
        if title {
            TITLE_RING.push(site);
        } else {
            OURS_RING.push(rip);
        }
    }
}

/// Up to `k` distinct pages (4 KiB aligned) of `[start, end)`, spread over it, different in each
/// `window`.
pub(crate) fn pick_pages(start: usize, end: usize, k: usize, window: u64) -> Vec<usize> {
    let first = start.div_ceil(KP);
    let n = (end / KP).saturating_sub(first);
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    let step = n / k;
    let shift = (window.wrapping_mul(0x9e37_79b1) % step as u64) as usize;
    (0..k).map(|j| (first + j * step + shift) * KP).collect()
}

/// Arms up to `k` pages of each range of `ranges` (a page that is not committed, or not plain
/// read-only or read-write, is left: `VirtualProtect` says so, with no walk of the region as a
/// query makes). `left_alone(block)`: the guards' blocks not to touch. Under the guards' lock,
/// which is held for 2 ms at the most.
fn sample(ranges: &[(usize, usize)], k: usize, window: u64, left_alone: &dyn Fn(usize) -> bool) -> usize {
    let live: Vec<&(usize, usize)> = ranges.iter().filter(|r| r.1 > r.0).collect();
    if live.is_empty() {
        return 0;
    }
    let started = std::time::Instant::now();
    let each = (k / live.len()).max(1);
    let mut armed = 0;
    'ranges: for &&(start, end) in &live {
        for page in pick_pages(start, end, each, window) {
            if started.elapsed().as_micros() > 2000 {
                break 'ranges;
            }
            if left_alone(page / BLOCK64) {
                continue;
            }
            if arm_inner(page, None) {
                armed += 1;
            }
        }
    }
    armed
}

/// Gives back every page still armed and empties the table. No lock: a claim keeps the guards'
/// changes (which wait for it) and this apart.
pub(crate) fn end_window() -> u64 {
    let mut untouched = 0;
    for s in &SLOT {
        if let Take::Claimed(token) = take_patient(s) {
            let a = s.addr.load(Relaxed);
            let prot = s.prot.load(Relaxed);
            restore_owned(a, prot);
            if s.finish(token, UNTOUCHED) {
                LIVE.fetch_sub(1, SeqCst);
                untouched += 1;
            }
        }
        // Nothing is owned in these: free for the next window.
        // (The slot keeps its page until it is armed again: a thread that faulted on it and comes
        // late to the handler finds it, `on_fault`.)
        s.free_if_done();
    }
    untouched
}

// ---------------------------------------------------------------- the top table

const TOP: usize = 64;

/// The instruction addresses most often seen reading (the space-saving counts: a full table
/// replaces its smallest).
pub(crate) struct Top {
    keys: [u64; TOP],
    counts: [u64; TOP],
}

impl Top {
    pub(crate) const fn new() -> Self {
        Top { keys: [0; TOP], counts: [0; TOP] }
    }

    pub(crate) fn bump(&mut self, key: u64) {
        let mut least = 0;
        for i in 0..TOP {
            if self.counts[i] != 0 && self.keys[i] == key {
                self.counts[i] += 1;
                return;
            }
            if self.counts[i] < self.counts[least] {
                least = i;
            }
        }
        self.keys[least] = key;
        self.counts[least] += 1;
    }

    pub(crate) fn most(&self, n: usize) -> Vec<(u64, u64)> {
        let mut all: Vec<(u64, u64)> = (0..TOP).filter(|&i| self.counts[i] != 0).map(|i| (self.keys[i], self.counts[i])).collect();
        all.sort_by_key(|&(k, c)| (std::cmp::Reverse(c), k));
        all.truncate(n);
        all
    }

    fn clear(&mut self) {
        *self = Top::new();
    }
}

// ---------------------------------------------------------------- per frame

struct State {
    frames: u64,
    window: u64,
    armed: bool,
    period_frames: u64,
    started: Option<std::time::Instant>,
    logged: Option<std::time::Instant>,
    said_none: u64,
    prev: [u64; COUNTERS],
    top: Top,
    top_ours: Top,
    tail: usize,
    tail_ours: usize,
}

static STATE: Mutex<State> = Mutex::new(State {
    frames: 0,
    window: 0,
    armed: false,
    period_frames: 0,
    started: None,
    logged: None,
    said_none: 0,
    prev: [0; COUNTERS],
    top: Top::new(),
    top_ours: Top::new(),
    tail: 0,
    tail_ours: 0,
});

/// A frame was presented: the window of the frame before ends, a new one begins every
/// `VICTORY_HEAP_PROBE_EVERY` frames, and every ~5 s the lines.
pub fn tick() {
    if !on() {
        return;
    }
    let c = config();
    let Ok(mut st) = STATE.try_lock() else {
        return;
    };
    st.frames += 1;
    st.period_frames += 1;
    let now = std::time::Instant::now();
    let started = *st.started.get_or_insert(now);
    if st.armed || live() {
        add(C::Untouched, end_window());
        st.armed = false;
    }
    if st.frames % c.every == 0 {
        begin_window(&mut st, c);
    }
    drain(&mut st);
    let logged = *st.logged.get_or_insert(now);
    if now.duration_since(logged).as_millis() >= 5000 {
        st.logged = Some(now);
        let secs = now.duration_since(logged).as_secs_f64();
        log(&mut st, secs, now.duration_since(started).as_secs());
    }
}

fn begin_window(st: &mut State, c: &Config) {
    let t = targets();
    if t.iter().all(|r| r.1 == 0) {
        return;
    }
    // Other pages in the next window whatever came of this one (one that cannot be armed, a
    // range not mapped yet, must not hold the sample in place).
    let window = st.window;
    st.window += 1;
    // Not while the runtime changes a mapping (or waits to), and not while the guards are busy:
    // the window waits for its turn.
    let Some(_reading) = crate::watch::title_memory_reading_now() else {
        add(C::Skipped, 1);
        return;
    };
    if !live() {
        // (The range of addresses faults are looked at for starts again.)
        LO.store(usize::MAX, SeqCst);
        HI.store(0, SeqCst);
    }
    let armed = crate::watch::heap_probe_try_locked(|left_alone| sample(&t, c.pages, window, left_alone));
    match armed {
        Some(n) if n > 0 => {
            st.armed = true;
            add(C::Windows, 1);
            add(C::Sampled, n as u64);
        }
        Some(_) => {}
        None => add(C::Skipped, 1),
    }
}

fn drain(st: &mut State) {
    let mut tail = st.tail;
    TITLE_RING.drain(&mut tail, |v| {
        if let Some((i, base)) = title_image_of(v) {
            st.top.bump((i as u64) << 48 | (v - base) as u64);
        }
    });
    st.tail = tail;
    let mut tail = st.tail_ours;
    OURS_RING.drain(&mut tail, |v| st.top_ours.bump(v as u64));
    st.tail_ours = tail;
}

/// "module+0xoffset" of the code at `rip`.
fn describe(rip: usize) -> String {
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW,
        GetModuleHandleExW,
    };
    let mut module = std::ptr::null_mut();
    // SAFETY: looks up the module holding the address; the reference count is left alone.
    let found = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            rip as *const u16,
            &mut module,
        )
    };
    if found == 0 || module.is_null() {
        return format!("{rip:#x}");
    }
    let mut name = [0u16; 260];
    // SAFETY: a buffer of the length given.
    let n = unsafe { GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) } as usize;
    let name = String::from_utf16_lossy(&name[..n.min(260)]);
    let name = name.rsplit(['\\', '/']).next().unwrap_or("?");
    format!("{name}+{:#x}", rip - module as usize)
}

/// (blocks of 64 KiB shown, of those shown at more than one address) in `ranges`.
fn multi_mapped(ranges: &[(usize, usize)]) -> (usize, usize) {
    let (mut shown, mut multi) = (0, 0);
    for &(start, end) in ranges.iter().filter(|r| r.1 > r.0) {
        for block in start.div_ceil(BLOCK64)..end / BLOCK64 {
            let at = crate::inplace::pages_of(block * BLOCK64);
            shown += usize::from(!at.is_empty());
            multi += usize::from(at.len() > 1);
        }
    }
    (shown, multi)
}

fn log(st: &mut State, secs: f64, since: u64) {
    let now: Vec<u64> = COUNT.iter().map(|c| c.load(Relaxed)).collect();
    let d = |c: C| now[c as usize].wrapping_sub(st.prev[c as usize]);
    let frames = st.period_frames.max(1) as f64;
    let t = targets();
    let ranges: Vec<String> = t.iter().filter(|r| r.1 > r.0).map(|&(s, e)| format!("{s:#x}+{:#x}", e - s)).collect();
    if ranges.is_empty() {
        if since / 30 >= st.said_none {
            st.said_none = since / 30 + 1;
            tracing::info!("heap probe: no range of {:#x} bytes of pool memory below 4 GiB yet", config().size);
        }
        st.prev.copy_from_slice(&now);
        st.period_frames = 0;
        return;
    }
    let (shown, multi) = multi_mapped(&t);
    let pg = |c: C| d(c) >> 16;
    let reads = d(C::ReadTitle) + d(C::ReadOurs);
    let writes = d(C::WriteTitle) + d(C::WriteOurs);
    let execs = d(C::ExecTitle) + d(C::ExecOurs);
    tracing::info!(
        "heap probe {} [first access per sampled page only: a lower bound, no title hits do not prove no reads]: \
         sampled {} pages in {} windows (ARM REFUSED while emulator readers were busy: {}; other arm failures: {}); faults {reads} reads ({} title, {} of those through memcpy helpers / {} ours, {} of ours announced), \
         {writes} writes ({} title / {} ours), {execs} exec ({} / {}); untouched {}; dropped {} (changes {}, emulator reads {}); \
         skipped {} windows, {} protection conflicts; \
         remaps: map {} ({} pages), unmap {} ({}), scatter {} ({}), alias {} ({}), copy-mappings {} ({}), views +{} ({} pages) -{} ({}); \
         multi-mapped pages {multi} of {shown}; \
         writers: dstorage {:.1} MB/s, GPU flights {:.2} MB/frame, title write faults {:.2}/frame ({} frames, {secs:.1} s)",
        ranges.join(","),
        d(C::Sampled),
        d(C::Windows),
        d(C::ArmBusy),
        d(C::ArmFailed),
        d(C::ReadTitle),
        d(C::ReadHelper),
        d(C::ReadOurs),
        d(C::DroppedEmu),
        d(C::WriteTitle),
        d(C::WriteOurs),
        d(C::ExecTitle),
        d(C::ExecOurs),
        d(C::Untouched),
        d(C::DroppedWatch) + d(C::DroppedEmu),
        d(C::DroppedWatch),
        d(C::DroppedEmu),
        d(C::Skipped),
        d(C::Conflicts),
        d(C::MapCalls),
        pg(C::MapBytes),
        d(C::UnmapCalls),
        pg(C::UnmapBytes),
        d(C::ScatterCalls),
        pg(C::ScatterBytes),
        d(C::AliasCalls),
        pg(C::AliasBytes),
        d(C::CopyCalls),
        pg(C::CopyBytes),
        d(C::ViewMapped),
        pg(C::ViewMappedBytes),
        d(C::ViewUnmapped),
        pg(C::ViewUnmappedBytes),
        d(C::Dstorage) as f64 / 1e6 / secs.max(0.001),
        d(C::Flights) as f64 / 1e6 / frames,
        d(C::GuardFaults) as f64 / frames,
        st.period_frames,
    );
    let top = st.top.most(8);
    if !top.is_empty() {
        tracing::info!(
            "heap probe title read sites (offset from the image base: reads): {}{}",
            top.iter()
                .map(|&(k, c)| if k >> 48 == 0 {
                    format!("+{:#x} x{c}", k & 0xffff_ffff_ffff)
                } else {
                    format!("image{}+{:#x} x{c}", k >> 48, k & 0xffff_ffff_ffff)
                })
                .collect::<Vec<_>>()
                .join(", "),
            if d(C::RingLost) != 0 { format!(" ({} lost)", d(C::RingLost)) } else { String::new() }
        );
    }
    let ours = st.top_ours.most(8);
    if !ours.is_empty() {
        tracing::info!(
            "heap probe our code's read sites: {}",
            ours.iter().map(|&(k, c)| format!("{} x{c}", describe(k as usize))).collect::<Vec<_>>().join(", ")
        );
    }
    st.top.clear();
    st.top_ours.clear();
    st.prev.copy_from_slice(&now);
    st.period_frames = 0;
}

/// The slot table is one for the process: tests that arm pages take turns.
#[cfg(test)]
pub(crate) static TEST_TURN: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_RELEASE, MEM_RESERVE, MapViewOfFile, FILE_MAP_ALL_ACCESS, CreateFileMappingW, UnmapViewOfFile, VirtualAlloc,
        VirtualFree,
    };


    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| pairs.iter().find(|p| p.0 == k).map(|p| p.1.to_owned())
    }

    #[test]
    fn the_switches() {
        let off = config_from(&env(&[]));
        assert!(!off.on);
        assert_eq!((off.size, off.every, off.pages, off.at), (0xae0_0000, 30, 64, None));
        assert!(!config_from(&env(&[("VICTORY_HEAP_PROBE", "0")])).on);
        let c = config_from(&env(&[
            ("VICTORY_HEAP_PROBE", "1"),
            ("VICTORY_HEAP_PROBE_SIZE", "0x1000000"),
            ("VICTORY_HEAP_PROBE_AT", "0x9ff0000, 0xae00000"),
            ("VICTORY_HEAP_PROBE_EVERY", "5"),
            ("VICTORY_HEAP_PROBE_PAGES", "100000"),
        ]));
        assert_eq!(
            c,
            Config { on: true, size: 0x100_0000, at: Some((0x9ff_0000, 0xae0_0000)), every: 5, pages: SLOTS, blocked_by_profile: false }
        );
        // Nonsense falls back.
        let c = config_from(&env(&[
            ("VICTORY_HEAP_PROBE", "1"),
            ("VICTORY_HEAP_PROBE_SIZE", "zz"),
            ("VICTORY_HEAP_PROBE_AT", "0x1000"),
            ("VICTORY_HEAP_PROBE_EVERY", "0"),
        ]));
        assert_eq!((c.size, c.at, c.every), (0xae0_0000, None, 30));
        // Windows every frame would meet their own slots again: at least 2.
        let c = config_from(&env(&[("VICTORY_HEAP_PROBE", "1"), ("VICTORY_HEAP_PROBE_EVERY", "1")]));
        assert_eq!(c.every, 2);
        // The stack sampler reads threads' contexts at any time: no probing with it.
        let c = config_from(&env(&[("VICTORY_HEAP_PROBE", "1"), ("VICTORY_PROFILE", "1,2,3,x")]));
        assert!(!c.on && c.blocked_by_profile);
    }

    #[test]
    fn the_range_is_picked_by_its_size_below_4_gib() {
        let size = 0xae0_0000;
        assert!(wants(size, 0x2b6a_0000, size));
        assert!(wants(size, 0x405a_0000, size));
        assert!(wants(size, 0x9ff_0000, size));
        assert!(!wants(size, 0x9ff_0000, size - 0x1_0000), "another size");
        assert!(!wants(size, 0x9ff_0000, size + 1));
        assert!(!wants(size, 0, size));
        assert!(!wants(size, 0xffff_0000, size), "reaches past 4 GiB");
        assert!(!wants(size, 0x2_0000_0000, size));
        assert!(wants(0x100_0000, 0x4000_0000, 0x100_0000), "another size asked for");
    }

    #[test]
    fn what_of_a_call_lies_in_the_range() {
        let r = [(0x1000_0000, 0x1100_0000), (0, 0), (0x3000_0000, 0x3000_8000)];
        assert_eq!(overlap_bytes(&r, 0x1000_0000, 0x1_0000), 0x1_0000);
        assert_eq!(overlap_bytes(&r, 0x0fff_0000, 0x2_0000), 0x1_0000, "straddles the start");
        assert_eq!(overlap_bytes(&r, 0x10ff_0000, 0x2_0000), 0x1_0000, "straddles the end");
        assert_eq!(overlap_bytes(&r, 0x2000_0000, 0x1_0000), 0);
        assert_eq!(overlap_bytes(&r, 0x2fff_8000, 0x1_0000), 0x8000 - 0x0);
        assert_eq!(overlap_bytes(&r, 0, usize::MAX), 0x100_0000 + 0x8000, "saturates");
        assert_eq!(overlap_bytes(&[], 0, 100), 0);
    }

    #[test]
    fn pages_are_spread_distinct_and_move_from_window_to_window() {
        let (start, end) = (0x9ff_0000, 0x9ff_0000 + 0xae0_0000);
        let a = pick_pages(start, end, 64, 0);
        assert_eq!(a.len(), 64);
        assert!(a.iter().all(|p| p % KP == 0 && *p >= start && *p + KP <= end));
        assert!(a.windows(2).all(|w| w[0] < w[1]), "ascending, distinct");
        assert!(a[63] - a[0] > (end - start) / 2, "spread over the range");
        let b = pick_pages(start, end, 64, 1);
        assert_ne!(a, b, "another window, other pages");
        assert!(b.iter().all(|p| *p >= start && *p + KP <= end));
        // Every window stays inside, whatever the number.
        for w in 0..1000 {
            let p = pick_pages(start, end, 64, w);
            assert!(p.iter().all(|p| *p >= start && *p + KP <= end));
        }
        // Small ranges give what they have; unaligned ends do not run over.
        assert_eq!(pick_pages(0x1000, 0x3000, 64, 7), vec![0x1000, 0x2000]);
        assert_eq!(pick_pages(0x1800, 0x4800, 64, 0), vec![0x2000, 0x3000]);
        assert!(pick_pages(0x1000, 0x1800, 4, 0).is_empty());
        assert!(pick_pages(0x1000, 0x3000, 0, 0).is_empty());
    }

    #[test]
    fn the_top_table_keeps_the_most_seen() {
        let mut t = Top::new();
        for i in 0..1000u64 {
            t.bump(0x100 + i % 7); // seven sites, 143 or so each
        }
        for _ in 0..500 {
            t.bump(0x9999);
        }
        // Noise beyond the table's size does not push the real ones out.
        for i in 0..200u64 {
            t.bump(0x10_0000 + i);
        }
        let most = t.most(8);
        assert_eq!(most[0], (0x9999, 500), "{most:?}");
        assert_eq!(most[1], (0x100, 143), "{most:?}");
        assert_eq!(most.len(), 8);
        assert!((0..7).all(|i| most[1..].iter().any(|&(k, _)| k == 0x100 + i)), "{most:?}");
        assert!(most.windows(2).all(|w| w[0].1 >= w[1].1));
    }

    #[test]
    fn a_range_followed_is_dropped_when_released_and_replaced_when_overlapped() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        // (Far from anything the title does: the process is a test's.)
        let (a, b) = (0x7e00_0000, 0x7e00_0000 + 0xae0_0000);
        add_target(a, b);
        assert!(targets().contains(&(a, b)));
        add_target(a, b);
        assert_eq!(targets().iter().filter(|t| **t == (a, b)).count(), 1);
        add_target(a + 0x10_0000, b + 0x10_0000);
        assert!(!targets().contains(&(a, b)), "overlapping: the new one stands");
        remove_target(a + 0x10_0000);
        assert!(targets().iter().all(|t| t.1 == 0));
    }

    /// A page-file backed view (title pool memory is a view), `n` pages of 4 KiB.
    struct View(usize);

    impl View {
        fn new(n: usize) -> View {
            // SAFETY: a fresh anonymous section and a view of it, undone in `drop`.
            unsafe {
                let h = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (n * KP) as u32, std::ptr::null());
                assert!(!h.is_null());
                let at = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, 0).Value as usize;
                assert_ne!(at, 0);
                windows_sys::Win32::Foundation::CloseHandle(h);
                View(at)
            }
        }
        fn protection(&self, page: usize) -> u32 {
            query(self.0 + page * KP).unwrap().Protect
        }
    }

    impl Drop for View {
        fn drop(&mut self) {
            // (A test that failed leaves no page armed in what is unmapped.)
            end_window();
            // SAFETY: the view made in `new`.
            unsafe {
                UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: self.0 as *mut _ })
            };
        }
    }

    fn read(at: usize) -> u8 {
        // SAFETY: a page of a view of the test's.
        unsafe { (at as *const u8).read_volatile() }
    }

    fn clean() {
        // (Tests that fault need the handler: whichever runs first installs it.)
        crate::watch::install();
        reset_for_test();
        assert_eq!(LIVE.load(SeqCst), 0);
    }

    #[test]
    fn a_sampled_page_read_is_recorded_and_then_succeeds() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        crate::watch::install();
        clean();
        let v = View::new(8);
        // SAFETY: the view's own bytes.
        unsafe { ((v.0 + 3 * KP + 5) as *mut u8).write(0x5a) };
        let before = (COUNT[C::ReadOurs as usize].load(Relaxed), COUNT[C::WriteOurs as usize].load(Relaxed));
        assert!(arm_page(v.0 + 3 * KP, PAGE_READWRITE));
        assert!(arm_page(v.0 + 5 * KP, PAGE_READWRITE));
        assert_eq!(v.protection(3), PAGE_NOACCESS);
        assert_eq!(v.protection(4), PAGE_READWRITE, "its neighbours are as they were");
        // A read of the page by this thread, which is not a title image: ours.
        assert_eq!(read(v.0 + 3 * KP + 5), 0x5a);
        assert_eq!(v.protection(3), PAGE_READWRITE, "given back");
        let slot = SLOT.iter().find(|s| s.addr.load(Relaxed) == v.0 + 3 * KP).unwrap();
        assert_eq!(slot.st(), HIT);
        // A write to the other.
        // SAFETY: the view's own bytes.
        unsafe { ((v.0 + 5 * KP) as *mut u8).write_volatile(7) };
        assert_eq!(v.protection(5), PAGE_READWRITE);
        assert_eq!(read(v.0 + 5 * KP), 7);
        let after = (COUNT[C::ReadOurs as usize].load(Relaxed), COUNT[C::WriteOurs as usize].load(Relaxed));
        assert!(after.0 > before.0 && after.1 > before.1, "{before:?} {after:?}");
        assert_eq!(LIVE.load(SeqCst), 0);
        clean();
    }

    #[test]
    fn a_read_by_a_title_image_is_told_from_ours_and_its_site_noted() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        crate::watch::install();
        clean();
        let v = View::new(2);
        // This test's own function stands for the title's code.
        let here = a_read_by_a_title_image_is_told_from_ours_and_its_site_noted as *const () as usize;
        let saved = (IMAGE[0].0.load(SeqCst), IMAGE[0].1.load(SeqCst));
        IMAGE[0].1.store(0, SeqCst);
        IMAGE[0].0.store(here & !0xfff_ffff, SeqCst);
        IMAGE[0].1.store((here & !0xfff_ffff) + 0x2000_0000, SeqCst);
        let before = COUNT[C::ReadTitle as usize].load(Relaxed);
        let head = TITLE_RING.reserved.load(SeqCst);
        assert!(arm_page(v.0, PAGE_READWRITE));
        assert_eq!(read(v.0 + 1), 0);
        IMAGE[0].1.store(0, SeqCst);
        IMAGE[0].0.store(saved.0, SeqCst);
        IMAGE[0].1.store(saved.1, SeqCst);
        assert_eq!(COUNT[C::ReadTitle as usize].load(Relaxed), before + 1);
        assert!(TITLE_RING.reserved.load(SeqCst) > head);
        clean();
    }

    #[test]
    fn untouched_pages_are_given_back_at_the_window_end() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        clean();
        let v = View::new(4);
        for p in 0..4 {
            assert!(arm_page(v.0 + p * KP, PAGE_READWRITE));
        }
        assert_eq!(LIVE.load(SeqCst), 4);
        assert_eq!(end_window(), 4);
        for p in 0..4 {
            assert_eq!(v.protection(p), PAGE_READWRITE);
        }
        assert_eq!(LIVE.load(SeqCst), 0);
        // A page the title protected itself meanwhile is left as it made it.
        assert!(arm_page(v.0, PAGE_READWRITE));
        let mut old = 0;
        // SAFETY: the test's view.
        unsafe { VirtualProtect(v.0 as *const _, KP, PAGE_READONLY, &mut old) };
        assert_eq!(old, PAGE_NOACCESS);
        end_window();
        assert_eq!(v.protection(0), PAGE_READONLY);
    }

    #[test]
    fn a_page_the_guards_or_the_emulator_come_for_is_given_back_and_dropped() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        clean();
        let v = View::new(32);
        // The guard's block (64 KiB) holds pages 0..16: one read-only page, one read-write.
        let mut old = 0;
        // SAFETY: the test's view.
        unsafe { VirtualProtect((v.0 + 2 * KP) as *const _, KP, PAGE_READONLY, &mut old) };
        assert!(arm_page(v.0 + 2 * KP, PAGE_READONLY));
        assert!(arm_page(v.0 + 9 * KP, PAGE_READWRITE));
        assert!(arm_page(v.0 + 20 * KP, PAGE_READWRITE));
        let dropped = COUNT[C::DroppedWatch as usize].load(Relaxed);
        let base = v.0 / BLOCK64;
        // (The view may not start on a 64 KiB block: the block of the first pages is what holds them.)
        release_range(base * BLOCK64, (base + 1) * BLOCK64, Why::Watch);
        let held = |p: usize| (v.0 + p * KP) / BLOCK64 == base;
        for (p, want) in [(2, PAGE_READONLY), (9, PAGE_READWRITE), (20, PAGE_READWRITE)] {
            let prot = v.protection(p);
            if held(p) {
                assert_eq!(prot, want, "page {p} given back with the protection it had");
            } else {
                assert_eq!(prot, PAGE_NOACCESS, "page {p} is another block's");
            }
        }
        assert!(COUNT[C::DroppedWatch as usize].load(Relaxed) > dropped || !held(2));
        // The emulator is about to read all of it.
        // (`emulator_reading` does this when the switch is on; the tests run with it off.)
        release_range(v.0, v.0 + 32 * KP, Why::Emulator);
        for p in [2, 9, 20] {
            assert_ne!(v.protection(p), PAGE_NOACCESS);
        }
        assert_eq!(LIVE.load(SeqCst), 0);
        // Not armed while a reader holds [BUSY].
        BUSY.fetch_add(1, SeqCst);
        assert!(!arm_page(v.0 + 4 * KP, PAGE_READWRITE));
        BUSY.fetch_sub(1, SeqCst);
        assert_eq!(v.protection(4), PAGE_READWRITE);
        assert!(arm_page(v.0 + 4 * KP, PAGE_READWRITE));
        clean();
    }

    #[test]
    fn a_window_arms_committed_view_pages_and_not_private_ones() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        clean();
        let v = View::new(64);
        let range = [(v.0, v.0 + 64 * KP)];
        let n = sample(&range, 8, 0, &|_| false);
        assert_eq!(n, 8);
        assert_eq!(LIVE.load(SeqCst), 8);
        assert_eq!(end_window(), 8);
        // A block left alone is not touched.
        assert_eq!(sample(&range, 8, 0, &|_| true), 0);
        // Pages that are not committed are left as they are, found by the protection call failing.
        // SAFETY: a reservation of the test's.
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), 16 * KP, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(p, 0);
            assert_eq!(sample(&[(p, p + 16 * KP)], 4, 0, &|_| false), 0);
            assert_eq!(LIVE.load(SeqCst), 0);
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
        }
        // Pages the title protected otherwise (guard) are left as they were.
        let v2 = View::new(4);
        let mut old = 0;
        // SAFETY: the test's view.
        unsafe { VirtualProtect(v2.0 as *const _, 4 * KP, PAGE_READONLY | PAGE_GUARD, &mut old) };
        assert_eq!(sample(&[(v2.0, v2.0 + 4 * KP)], 4, 0, &|_| false), 0);
        assert_eq!(v2.protection(0), PAGE_READONLY | PAGE_GUARD);
        clean();
    }

    #[test]
    fn a_fault_that_is_not_ours_goes_on_to_the_guards() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        clean();
        let v = View::new(4);
        assert!(arm_page(v.0, PAGE_READWRITE));
        // Not a page armed, nor in the range armed.
        assert!(!on_fault(0, v.0 + KP, 0x1000, 0));
        assert!(!on_fault(1, 0x10, 0x1000, 0));
        assert!(!on_fault(0, v.0 + 100 * KP, 0x1000, 0));
        assert_eq!(v.protection(0), PAGE_NOACCESS, "still armed");
        // A fault in the page: ours to give back.
        assert!(on_fault(0, v.0 + 7, 0x1000, 0));
        assert_eq!(v.protection(0), PAGE_READWRITE);
        // Late to the handler (the page was given back meanwhile): the instruction runs again.
        assert!(on_fault(0, v.0 + 7, 0x1000, 0), "given back already");
        // Unless it cannot succeed: a write into a page read-only now is the guards' to see.
        let mut old = 0;
        // SAFETY: the test's view.
        unsafe { VirtualProtect(v.0 as *const _, KP, PAGE_READONLY, &mut old) };
        assert!(!on_fault(1, v.0 + 7, 0x1000, 0));
        assert!(on_fault(0, v.0 + 7, 0x1000, 0));
        clean();
    }

    #[test]
    fn nothing_is_restored_over_what_was_done_to_a_page_since() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        clean();
        let v = View::new(4);
        assert!(arm_page(v.0, PAGE_READWRITE));
        assert_eq!(read(v.0), 0);
        // Touched and given back; now the title itself makes the page inaccessible.
        let mut old = 0;
        // SAFETY: the test's view.
        unsafe { VirtualProtect(v.0 as *const _, KP, PAGE_NOACCESS, &mut old) };
        // Neither the end of the window nor a release nor a stale fault flips it.
        assert_eq!(end_window(), 0);
        assert_eq!(v.protection(0), PAGE_NOACCESS);
        release_range(v.0, v.0 + KP, Why::Watch);
        assert!(!on_fault(0, v.0, 0x1000, 0));
        assert_eq!(v.protection(0), PAGE_NOACCESS);
        // SAFETY: the test's view.
        unsafe { VirtualProtect(v.0 as *const _, KP, PAGE_READWRITE, &mut old) };
        // A page that is changed under an armed slot is put back as it was changed.
        assert!(arm_page(v.0 + KP, PAGE_READWRITE));
        let conflicts = COUNT[C::Conflicts as usize].load(Relaxed);
        // SAFETY: the test's view.
        unsafe { VirtualProtect((v.0 + KP) as *const _, KP, PAGE_READONLY, &mut old) };
        end_window();
        assert_eq!(v.protection(1), PAGE_READONLY);
        assert!(COUNT[C::Conflicts as usize].load(Relaxed) > conflicts);
        clean();
    }

    #[test]
    fn a_read_through_the_titles_copies_is_the_titles() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        let saved_image = (IMAGE[0].0.load(SeqCst), IMAGE[0].1.load(SeqCst));
        let saved_helper = (HELPER[0].0.load(SeqCst), HELPER[0].1.load(SeqCst));
        IMAGE[0].1.store(0, SeqCst);
        IMAGE[0].0.store(0x1000_0000, SeqCst);
        IMAGE[0].1.store(0x1100_0000, SeqCst);
        HELPER[0].1.store(0, SeqCst);
        HELPER[0].0.store(0x5000, SeqCst);
        HELPER[0].1.store(0x5100, SeqCst);
        // The title's own code.
        assert_eq!(classify_with(0x1000_0040, &[]), (true, 0x1000_0040, false));
        // In the helper, called from the title (the return address on the stack).
        assert_eq!(classify_with(0x5010, &[0x7777, 0x1000_0500, 0x1000_0900]), (true, 0x1000_0500, true));
        // Deeper in ours, under the helper's frame.
        assert_eq!(classify_with(0x9000, &[0x5050, 0x9100, 0x1000_0500]), (true, 0x1000_0500, true));
        // Ours, called by the title through something else: ours.
        assert_eq!(classify_with(0x9000, &[0x9100, 0x1000_0500]), (false, 0x9000, false));
        // Ours with no title on the stack at all.
        assert_eq!(classify_with(0x5010, &[0x9100]), (false, 0x5010, false));
        IMAGE[0].1.store(0, SeqCst);
        IMAGE[0].0.store(saved_image.0, SeqCst);
        IMAGE[0].1.store(saved_image.1, SeqCst);
        HELPER[0].1.store(0, SeqCst);
        HELPER[0].0.store(saved_helper.0, SeqCst);
        HELPER[0].1.store(saved_helper.1, SeqCst);
    }

    #[test]
    fn the_ring_gives_entries_in_order_and_stops_at_one_not_written() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        let ring = Ring::new();
        let mut tail = 0;
        let mut got = Vec::new();
        for v in 1..=5 {
            ring.push(v);
        }
        ring.drain(&mut tail, |v| got.push(v));
        assert_eq!(got, [1, 2, 3, 4, 5]);
        // A slot reserved and not yet written stops the walk; nothing after it is skipped.
        ring.reserved.fetch_add(1, SeqCst);
        ring.push(7);
        got.clear();
        ring.drain(&mut tail, |v| got.push(v));
        assert!(got.is_empty());
        ring.buf[tail % RING].store(6, SeqCst);
        ring.drain(&mut tail, |v| got.push(v));
        assert_eq!(got, [6, 7]);
        // A wrapped ring loses the oldest and counts it.
        let lost = COUNT[C::RingLost as usize].load(Relaxed);
        for v in 1..=RING + 10 {
            ring.push(v);
        }
        got.clear();
        ring.drain(&mut tail, |v| got.push(v));
        assert_eq!(got.len(), RING);
        assert_eq!(*got.last().unwrap(), RING + 10);
        assert_eq!(COUNT[C::RingLost as usize].load(Relaxed) - lost, 10);
    }

    #[test]
    fn a_late_fault_finds_its_page_among_the_recent_ones_and_any_kind_of_access() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        clean();
        let v = View::new(4);
        assert!(arm_page(v.0, PAGE_READWRITE));
        end_window();
        // The slot is armed again, for another page: the first is no longer in any slot.
        assert!(arm_page(v.0 + 2 * KP, PAGE_READWRITE));
        assert!(on_fault(0, v.0 + 5, 0x1000, 0), "a read that came late to the handler");
        assert!(on_fault(1, v.0 + 5, 0x1000, 0), "a write, too");
        assert!(!on_fault(8, v.0 + 5, 0x1000, 0), "not an execute of data");
        end_window();
        // Executable memory: an execute that came late can run.
        // SAFETY: a private allocation of the test's.
        unsafe {
            use windows_sys::Win32::System::Memory::PAGE_EXECUTE_READ;
            let p = VirtualAlloc(std::ptr::null(), KP, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READ) as usize;
            assert_ne!(p, 0);
            assert!(accessible_for(8, p) && accessible_for(0, p) && !accessible_for(1, p));
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
        }
        clean();
    }

    #[test]
    fn no_page_is_armed_while_a_write_is_in_flight() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        clean();
        let v = View::new(8);
        assert!(arm_page(v.0 + KP, PAGE_READWRITE));
        BUSY.fetch_add(1, SeqCst);
        // The writer's release: given back, and nothing is armed until it is done.
        release_range(v.0, v.0 + 2 * KP, Why::Watch);
        assert_ne!(v.protection(1), PAGE_NOACCESS, "given back");
        let refused = COUNT[C::ArmBusy as usize].load(Relaxed);
        assert!(!arm_page(v.0, PAGE_READWRITE));
        assert!(!arm_page(v.0 + 4 * KP, PAGE_READWRITE), "anywhere");
        assert!(COUNT[C::ArmBusy as usize].load(Relaxed) >= refused + 2);
        BUSY.fetch_sub(1, SeqCst);
        assert!(arm_page(v.0 + 4 * KP, PAGE_READWRITE), "armed again when it is over");
        clean();
    }

    #[test]
    fn a_slot_whose_owner_was_killed_is_taken_over() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        clean();
        let v = View::new(2);
        assert!(arm_page(v.0, PAGE_READWRITE));
        let slot = SLOT.iter().find(|s| s.addr.load(SeqCst) == v.0 && s.st() == ARMED).unwrap();
        // A thread claims it and ends without giving the page back.
        std::thread::scope(|t| {
            t.spawn(|| assert!(slot.claim().is_some())).join().unwrap();
        });
        assert_eq!(v.protection(0), PAGE_NOACCESS);
        assert_eq!(end_window(), 1, "found, after a wait, that its owner ended");
        assert_eq!(v.protection(0), PAGE_READWRITE);
        clean();
    }

    #[test]
    fn the_hooks_do_nothing_with_the_switch_off() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        // (`VICTORY_HEAP_PROBE` is not set in the tests.)
        assert!(!on());
        allocated(0x4000_0000, 0xae0_0000, true);
        pool_mapped(0x4000_0000, 0xae0_0000);
        assert!(targets().iter().all(|t| t.1 == 0));
        let before: Vec<u64> = COUNT.iter().map(|c| c.load(Relaxed)).collect();
        note_call(Call::Map, 0x4000_0000, 0x10000);
        note_calls(Call::Scatter, std::iter::once((0x4000_0000, 0x10000)));
        view(true, 0, 1);
        dstorage(0, 1);
        flight_fetched(0, 1);
        guard_fault(0);
        tick();
        assert!(emulator_reading(0, 100).0 == false);
        let after: Vec<u64> = COUNT.iter().map(|c| c.load(Relaxed)).collect();
        assert_eq!(before, after);
    }
}
