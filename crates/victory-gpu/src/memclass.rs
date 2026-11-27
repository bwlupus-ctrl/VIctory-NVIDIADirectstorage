//! LOCAL (gpu-first): title memory by allocation class, for step 0/1 of the GPU-first memory
//! plan. All of it is off unless asked for; with the switches off nothing is recorded and no
//! behaviour changes (each hook is one relaxed load and a branch).
//!
//! - `VICTORY_CLASS_STATS=1`: a registry of the title's allocations (start, size, the protection
//!   it asked for, kind, XMem flags; filled by `xmem`'s `XMemVirtualAlloc` and by the pool pages
//!   mapped into title memory, [`crate::inplace::title_memory_view`]), each labelled with a
//!   [`Class`]. The per-frame costs that already exist (write-watch askings and bytes, guard
//!   faults and waits, protection changes, refreshes, uploads to copies in the card's memory,
//!   copy-back, texture refreshes) are added up per class and logged every ~5 s, one line per
//!   class ([`tick`]).
//! - Reads of write-combined title memory by the emulator (`VICTORY_CLASS_STATS=1` or
//!   `VICTORY_HONOUR_WC=1`): counted per reading site (file:line, by `#[track_caller]` through
//!   `crate::host`'s title-memory readers and [`wc_read`]) and logged with the class lines.
//!   With `VICTORY_HONOUR_WC=1` (`xmem`) the title's big write-combined blocks really are
//!   write-combined on the host, where a CPU read is slow: these counts say who reads them.
//! - Streaming (`VICTORY_CLASS_STATS=1`): DirectStorage destination bytes per class
//!   ([`streamed`]), and how many bytes of what was streamed in the last 5-10 s the emulator
//!   touches again before the card has them (write-watch askings, refresh checks, uploads to
//!   copies in the card's memory, texture refreshes), by a bitmap of streamed 64 KiB blocks.
//! - Console pools: no XMem bit that picks GPU-optimal against standard memory is known (xmem
//!   knows bits 0-1 graphics, bit 2 32-bit, bits 12-14 address bits; the title's 0x283 and 0x23
//!   have bits 5, 7, 9 unexplained): the registry keeps the raw protection, kind and flags and
//!   the lines group bytes by them, beside the console's budget (`crate::console`).

use std::cell::Cell;
use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{OnceLock, RwLock};

/// What title memory is, by what the title asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Class {
    /// Write-combined (or uncached) and GPU-reachable, private committed memory (the 1277 MB
    /// block).
    WcGpuPrivate = 0,
    /// Write-combined (or uncached) and GPU-reachable, filled with the title's physical pages.
    WcGpuPool = 1,
    /// GPU-reachable, cached by the CPU (the TileHeap, `0x2004`/`0x22004`).
    GpuCached = 2,
    /// No GPU access bit.
    CpuOnly = 3,
    /// Not in the registry (the host's heap, images, pool pages outside any allocation).
    Other = 4,
}

pub const CLASSES: usize = 5;

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::WcGpuPrivate => "wc+gpu private",
            Class::WcGpuPool => "wc+gpu pool-mapped",
            Class::GpuCached => "gpu cached",
            Class::CpuOnly => "cpu-only",
            Class::Other => "other",
        }
    }

    fn from_index(i: usize) -> Class {
        [Class::WcGpuPrivate, Class::WcGpuPool, Class::GpuCached, Class::CpuOnly, Class::Other][i.min(4)]
    }

    pub fn write_combined(self) -> bool {
        matches!(self, Class::WcGpuPrivate | Class::WcGpuPool)
    }
}

/// The console's protection bits: 0x200 uncached, 0x400 write-combined, 0x800 and up to
/// 0x40000 the GPU's access.
pub const CONSOLE_NOCACHE: u32 = 0x200;
pub const CONSOLE_WRITECOMBINE: u32 = 0x400;
const CONSOLE_GPU: u32 = 0x7_f800;

/// The class of memory the title asked for with `protect`; `pool`: filled with physical pages.
pub fn classify(protect: u32, pool: bool) -> Class {
    let wc = protect & (CONSOLE_WRITECOMBINE | CONSOLE_NOCACHE) != 0;
    let gpu = protect & CONSOLE_GPU != 0;
    match (wc && gpu, gpu, pool) {
        (true, _, false) => Class::WcGpuPrivate,
        (true, _, true) => Class::WcGpuPool,
        (false, true, _) => Class::GpuCached,
        (false, false, _) if protect == 0 => Class::Other,
        _ => Class::CpuOnly,
    }
}

/// `VICTORY_CLASS_STATS=1`.
pub fn stats_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_CLASS_STATS"))
}

/// `VICTORY_HONOUR_WC=1` (`xmem` passes write-combining to the host).
pub fn honour_wc() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_HONOUR_WC"))
}

/// `VICTORY_HONOUR_WC_MIN_MB=<n>`: only blocks of at least that size (0: all).
pub fn honour_wc_min_bytes() -> usize {
    static MIN: OnceLock<usize> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("VICTORY_HONOUR_WC_MIN_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0) << 20
    })
}

/// Whether the registry is kept (either switch).
pub fn registry_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // LOCAL (page-kinds): retain allocation metadata without enabling other meters.
    *ON.get_or_init(|| stats_on() || honour_wc() || crate::pagekinds::on())
}

/// Whether reads of write-combined memory are counted (either switch).
pub fn reads_on() -> bool {
    // LOCAL (page-kinds): registry-only measurement does not enable WC read instrumentation.
    stats_on() || honour_wc()
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    start: usize,
    end: usize,
    protect: u32,
    kind: u32,
    flags: u64,
    pool: bool,
    /// The host allocation is write-combined (or uncached) itself (`VICTORY_HONOUR_WC`).
    host_wc: bool,
    class: Class,
}

/// Sorted by start, not overlapping.
#[derive(Default)]
struct Registry {
    entries: Vec<Entry>,
}

impl Registry {
    /// The index of the entry holding `at`.
    fn find(&self, at: usize) -> Option<usize> {
        let i = self.entries.partition_point(|e| e.start <= at).checked_sub(1)?;
        (at < self.entries[i].end).then_some(i)
    }

    /// Takes out what overlaps `[start, end)`, then inserts `e`.
    fn insert(&mut self, e: Entry) {
        self.remove_overlapping(e.start, e.end);
        let i = self.entries.partition_point(|x| x.start < e.start);
        self.entries.insert(i, e);
    }

    fn remove_overlapping(&mut self, start: usize, end: usize) {
        self.entries.retain(|x| x.end <= start || x.start >= end);
    }

    /// The entry starting at `start` (a whole release).
    fn release(&mut self, start: usize) -> Option<Entry> {
        let i = self.find(start).filter(|&i| self.entries[i].start == start)?;
        Some(self.entries.remove(i))
    }

    /// `f(class, host_wc, bytes)` for each piece of `[at, at + len)`, `Other` between entries.
    fn split(&self, at: usize, len: usize, mut f: impl FnMut(Class, bool, usize)) {
        let end = at.saturating_add(len);
        let mut here = at;
        let mut i = self.entries.partition_point(|e| e.end <= at);
        while here < end {
            match self.entries.get(i) {
                Some(e) if e.start < end => {
                    if e.start > here {
                        f(Class::Other, false, e.start - here);
                        here = e.start;
                    }
                    let to = e.end.min(end);
                    f(e.class, e.host_wc, to - here);
                    here = to;
                    i += 1;
                }
                _ => {
                    f(Class::Other, false, end - here);
                    here = end;
                }
            }
        }
    }
}

static REGISTRY: RwLock<Registry> = RwLock::new(Registry { entries: Vec::new() });

/// LOCAL (page-kinds): owned metadata snapshot; release the lock before title reads.
pub(crate) fn page_kind_ranges() -> Vec<(usize, usize, Class)> {
    REGISTRY.read().unwrap_or_else(|e| e.into_inner()).entries.iter()
        .map(|e| (e.start, e.end - e.start, e.class)).collect()
}
/// Bumped at every change of the registry (the threads' last hits go stale).
static GENERATION: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// This thread's last hit: (generation, start, end, class | host_wc << 7).
    static LAST: Cell<(u64, usize, usize, u8)> = const { Cell::new((0, 0, 0, 0)) };
}

fn changed(mut f: impl FnMut(&mut Registry)) {
    let mut r = REGISTRY.write().unwrap_or_else(|e| e.into_inner());
    f(&mut r);
    GENERATION.fetch_add(1, Relaxed);
}

/// The class of the memory at `at`, and whether its host allocation is write-combined. Never
/// blocks (a registry being changed answers `Other`).
fn lookup(at: usize) -> (Class, bool) {
    let generation = GENERATION.load(Relaxed);
    let hit = LAST.try_with(|last| {
        let (g, start, end, packed) = last.get();
        (g == generation && start <= at && at < end).then_some(packed)
    });
    if let Ok(Some(packed)) = hit {
        return (Class::from_index((packed & 0x7f) as usize), packed & 0x80 != 0);
    }
    let Ok(r) = REGISTRY.try_read() else {
        return (Class::Other, false);
    };
    let (start, end, class, host_wc) = match r.find(at) {
        Some(i) => {
            let e = &r.entries[i];
            (e.start, e.end, e.class, e.host_wc)
        }
        None => {
            // The gap between entries.
            let i = r.entries.partition_point(|e| e.start <= at);
            let start = i.checked_sub(1).map_or(0, |j| r.entries[j].end);
            let end = r.entries.get(i).map_or(usize::MAX, |e| e.start);
            (start, end, Class::Other, false)
        }
    };
    let _ = LAST.try_with(|last| last.set((generation, start, end, class as u8 | u8::from(host_wc) << 7)));
    (class, host_wc)
}

/// The class of the memory at `at` (diagnostics, tests).
pub fn class_of(at: usize) -> Class {
    if !registry_on() {
        return Class::Other;
    }
    lookup(at).0
}

/// `xmem`: the title's allocation `[at, at + size)`, as asked for (`protect` the console's
/// protection, `kind` and `flags` as passed to the host and XMem); `pool`: a reservation for
/// physical pages; `host_wc`: the host allocation was made write-combined or uncached.
pub fn allocated(at: usize, size: usize, protect: u32, kind: u32, flags: u64, pool: bool, host_wc: bool) {
    // LOCAL (heap-probe): a reservation of the size followed (`VICTORY_HEAP_PROBE=1`; off: nothing).
    if kind & 0x2000 != 0 {
        crate::heapprobe::allocated(at, size, pool);
    }
    // LOCAL (gpu-first): neither reserve nor commit (e.g. MEM_RESET) changes no allocation;
    // inserting it would replace the containing entry with a fragment.
    if !registry_on() || at == 0 || size == 0 || kind & 0x3000 == 0 {
        return;
    }
    const MEM_COMMIT: u32 = 0x1000;
    // A commit inside a reservation the registry knows: the reservation's class stands (a
    // commit with another protection is said once).
    if kind & 0x2000 == 0 && kind & MEM_COMMIT != 0 {
        let known = {
            let r = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
            r.find(at).map(|i| r.entries[i])
        };
        if let Some(e) = known {
            if e.protect != protect && stats_on() {
                static SAID: AtomicUsize = AtomicUsize::new(0);
                if SAID.fetch_add(1, Relaxed) < 8 {
                    tracing::info!(
                        "class registry: commit {at:#x}+{size:#x} as protection {protect:#x} inside {:#x}+{:#x} ({}, protection {:#x})",
                        e.start,
                        e.end - e.start,
                        e.class.name(),
                        e.protect
                    );
                }
            }
            return;
        }
    }
    let class = classify(protect, pool);
    let e = Entry { start: at, end: at.saturating_add(size), protect, kind, flags, pool, host_wc, class };
    changed(|r| r.insert(e));
    if stats_on() && size >= 1 << 20 {
        tracing::info!(
            "class registry: {} {at:#x}+{size:#x} ({} MB) protection {protect:#x} kind {kind:#x} flags {flags:#x}{}",
            class.name(),
            size >> 20,
            if host_wc { " (host write-combined)" } else { "" }
        );
    }
}

/// Pool pages were mapped at `[at, at + len)` (a view of the title's physical pages): the
/// allocation holding it is pool-mapped from now on; one the registry does not know is
/// noted as such.
pub fn pool_mapped(at: usize, len: usize) {
    // LOCAL (heap-probe): a view of the size followed (`VICTORY_HEAP_PROBE=1`; off: nothing).
    crate::heapprobe::pool_mapped(at, len);
    if !registry_on() || len == 0 {
        return;
    }
    add_span(M::PoolMapped, at, len);
    let state = REGISTRY.try_read().ok().map(|r| r.find(at).map(|i| r.entries[i].pool));
    match state {
        Some(Some(true)) => {}
        Some(Some(false)) => changed(|r| {
            if let Some(i) = r.find(at) {
                let e = &mut r.entries[i];
                e.pool = true;
                e.class = classify(e.protect, true);
                if stats_on() && e.end - e.start >= 1 << 20 {
                    tracing::info!(
                        "class registry: {:#x}+{:#x} ({} MB, protection {:#x}) gets pool pages: {}",
                        e.start,
                        e.end - e.start,
                        (e.end - e.start) >> 20,
                        e.protect,
                        e.class.name()
                    );
                }
            }
        }),
        Some(None) => {
            // Pool pages where no allocation is known: noted as their view, unclassed.
            let e = Entry { start: at, end: at + len, protect: 0, kind: 0, flags: 0, pool: true, host_wc: false, class: Class::Other };
            changed(|r| {
                if r.find(at).is_none() && r.find(at + len - 1).is_none() {
                    r.insert(e)
                }
            });
        }
        // Being changed: the next mapping says it.
        None => {}
    }
}

/// The title released the allocation at `at` (`VirtualFree(.., MEM_RELEASE)`).
pub fn released(at: usize) {
    // LOCAL (heap-probe): (`VICTORY_HEAP_PROBE=1`; off: nothing).
    crate::heapprobe::released(at);
    if !registry_on() {
        return;
    }
    changed(|r| {
        r.release(at);
    });
}

// ---------------------------------------------------------------- costs by class

/// The costs added up per class.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum M {
    /// Write-watch askings, bytes asked about, nanoseconds (diagnostics builds).
    WatchAsks,
    WatchBytes,
    WatchNs,
    /// Guard faults (writes into guarded blocks) and nanoseconds waited for the guards' lock in
    /// them (diagnostics builds).
    GuardFaults,
    GuardWaitNs,
    /// Protection changes (blocks made read-only or writable again) and their nanoseconds
    /// (diagnostics builds).
    Protects,
    ProtectNs,
    /// Refreshes (`Shadow::update`): bytes checked, bytes copied, nanoseconds.
    Checked,
    Refreshed,
    RefreshNs,
    /// Uploads to copies in the card's memory (`Host::card_before`): bytes and ranges.
    CardBytes,
    CardRanges,
    /// Copy-back (`land::piece`): pieces and bytes put into title memory.
    BackPieces,
    BackBytes,
    /// Texture refreshes: bytes of title memory read and nanoseconds.
    TexBytes,
    TexNs,
    /// DirectStorage destination bytes ([`streamed`]).
    Streamed,
    /// Pool pages mapped (bytes of views made: the remap rate).
    PoolMapped,
    /// Bytes of recently streamed blocks touched again: by write-watch askings, refresh checks,
    /// uploads to copies in the card's memory, texture refreshes.
    ReWatch,
    ReRefresh,
    ReCard,
    ReTex,
}

const METRICS: usize = 22;
static COSTS: [[AtomicU64; CLASSES]; METRICS] = [const { [const { AtomicU64::new(0) }; CLASSES] }; METRICS];

/// Adds `n` to `m` of the class of the memory at `at`.
#[inline]
pub fn add(m: M, at: usize, n: u64) {
    if !stats_on() || n == 0 {
        return;
    }
    let class = lookup(at).0;
    COSTS[m as usize][class as usize].fetch_add(n, Relaxed);
}

/// Adds the bytes of `[at, at + len)` to `m`, each piece to its own class.
pub fn add_span(m: M, at: usize, len: usize) {
    if !stats_on() || len == 0 {
        return;
    }
    let again = match m {
        M::WatchBytes => Some(M::ReWatch),
        M::Checked => Some(M::ReRefresh),
        M::CardBytes => Some(M::ReCard),
        M::TexBytes => Some(M::ReTex),
        _ => None,
    };
    if let Some(again) = again
        && STREAMED_ANY.load(Relaxed)
    {
        let n = streamed_within(at, len);
        if n != 0 {
            add(again, at, n as u64);
        }
    }
    // One class most of the time.
    let (class, _) = lookup(at);
    if class != Class::Other && lookup(at + len - 1).0 == class {
        if let Ok(r) = REGISTRY.try_read()
            && let Some(i) = r.find(at)
            && r.entries[i].end >= at + len
        {
            COSTS[m as usize][class as usize].fetch_add(len as u64, Relaxed);
            return;
        }
    }
    match REGISTRY.try_read() {
        Ok(r) => r.split(at, len, |class, _, n| {
            COSTS[m as usize][class as usize].fetch_add(n as u64, Relaxed);
        }),
        Err(_) => {
            COSTS[m as usize][Class::Other as usize].fetch_add(len as u64, Relaxed);
        }
    }
}

/// A timer for `add(m, at, ns)`: `None` when the stats are off.
#[inline]
pub fn timer() -> Option<std::time::Instant> {
    stats_on().then(std::time::Instant::now)
}

/// Adds the nanoseconds since `started` (from [`timer`]) to `m` of the class at `at`.
#[inline]
pub fn add_since(m: M, at: usize, started: Option<std::time::Instant>) {
    if let Some(started) = started {
        add(m, at, started.elapsed().as_nanos() as u64);
    }
}

// ---------------------------------------------------------------- reads of write-combined memory

/// Reading sites, by the caller's location: open addressing on the location's address.
const SITES: usize = 512;
struct Site {
    at: AtomicPtr<Location<'static>>,
    /// Calls and bytes read of write-combined classes; calls and bytes of those whose host
    /// allocation is write-combined (`VICTORY_HONOUR_WC`).
    counts: [AtomicU64; 4],
}
static SITE: [Site; SITES] =
    [const { Site { at: AtomicPtr::new(std::ptr::null_mut()), counts: [const { AtomicU64::new(0) }; 4] } }; SITES];
static SITES_FULL: AtomicBool = AtomicBool::new(false);

fn site(loc: &'static Location<'static>) -> Option<&'static Site> {
    let key = loc as *const Location<'static> as *mut Location<'static>;
    let mut i = (key as usize >> 3).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - 9);
    for _ in 0..SITES {
        let s = &SITE[i % SITES];
        let now = s.at.load(Relaxed);
        if now == key {
            return Some(s);
        }
        if now.is_null() {
            match s.at.compare_exchange(std::ptr::null_mut(), key, Relaxed, Relaxed) {
                Ok(_) => return Some(s),
                Err(other) if other == key => return Some(s),
                Err(_) => {}
            }
        }
        i += 1;
    }
    SITES_FULL.store(true, Relaxed);
    None
}

/// The emulator reads `[at, at + len)` of title memory (counted per caller when that is
/// write-combined memory).
#[track_caller]
#[inline]
pub fn wc_read(at: usize, len: usize) {
    if !reads_on() || len == 0 {
        return;
    }
    wc_read_at(Location::caller(), at, len);
}

/// [`wc_read`] with the caller given.
pub fn wc_read_at(loc: &'static Location<'static>, at: usize, len: usize) {
    if !reads_on() || len == 0 {
        return;
    }
    let (class, host_wc) = lookup(at);
    let (mut wc, mut hw) = (0u64, 0u64);
    if class.write_combined() && (lookup(at + len - 1).0 == class) {
        wc = len as u64;
        if host_wc {
            hw = len as u64;
        }
    } else if class.write_combined() || lookup(at + len - 1).0.write_combined() {
        if let Ok(r) = REGISTRY.try_read() {
            r.split(at, len, |class, host_wc, n| {
                if class.write_combined() {
                    wc += n as u64;
                    if host_wc {
                        hw += n as u64;
                    }
                }
            });
        }
    }
    if wc == 0 {
        return;
    }
    if let Some(s) = site(loc) {
        s.counts[0].fetch_add(1, Relaxed);
        s.counts[1].fetch_add(wc, Relaxed);
        if hw != 0 {
            s.counts[2].fetch_add(1, Relaxed);
            s.counts[3].fetch_add(hw, Relaxed);
        }
    }
}

// ---------------------------------------------------------------- streaming

/// 64 KiB blocks of title memory below 2^40 that DirectStorage wrote, in this period and the
/// one before (`STREAM_NOW` says which is this one): 2 MiB each, untouched pages until used.
const STREAM_BLOCK: usize = 1 << 16;
const STREAM_LIMIT: usize = 1 << 40;
const STREAM_WORDS: usize = STREAM_LIMIT / STREAM_BLOCK / 64;
static STREAM_BITS: [[AtomicU64; STREAM_WORDS]; 2] = [const { [const { AtomicU64::new(0) }; STREAM_WORDS] }; 2];
static STREAM_NOW: AtomicUsize = AtomicUsize::new(0);
static STREAMED_ANY: AtomicBool = AtomicBool::new(false);
/// Bytes streamed into physical pages the title shows nowhere, and above 2^40 (not tracked).
static STREAMED_UNPLACED: AtomicU64 = AtomicU64::new(0);

/// DirectStorage wrote `[at, at + len)` of title memory (`VICTORY_CLASS_STATS`).
pub fn streamed(at: usize, len: usize) {
    // LOCAL (page-kinds): destinations are retained until the next table sample.
    crate::pagekinds::streamed(at, len);
    if !stats_on() || len == 0 {
        return;
    }
    add_span(M::Streamed, at, len);
    let end = at.saturating_add(len);
    if end > STREAM_LIMIT {
        STREAMED_UNPLACED.fetch_add((end - at.max(STREAM_LIMIT)) as u64, Relaxed);
    }
    if at >= STREAM_LIMIT {
        return;
    }
    let bits = &STREAM_BITS[STREAM_NOW.load(Relaxed) & 1];
    let (first, last) = (at / STREAM_BLOCK, (end.min(STREAM_LIMIT) - 1) / STREAM_BLOCK);
    for b in first..=last {
        let bit = 1u64 << (b % 64);
        let w = &bits[b / 64];
        if w.load(Relaxed) & bit == 0 {
            w.fetch_or(bit, Relaxed);
        }
    }
    STREAMED_ANY.store(true, Relaxed);
}

/// DirectStorage wrote `len` bytes of physical pages the title shows nowhere now.
pub fn streamed_unplaced(len: usize) {
    // LOCAL (page-kinds): physical destinations without a title VA are separate, not outside.
    crate::pagekinds::streamed_unplaced(len);
    if stats_on() {
        STREAMED_UNPLACED.fetch_add(len as u64, Relaxed);
    }
}

/// The bytes of `[at, at + len)` in blocks streamed in this period or the one before.
fn streamed_within(at: usize, len: usize) -> usize {
    if at >= STREAM_LIMIT || len == 0 {
        return 0;
    }
    let end = at.saturating_add(len).min(STREAM_LIMIT);
    let (first, last) = (at / STREAM_BLOCK, (end - 1) / STREAM_BLOCK);
    let mut n = 0;
    let mut b = first;
    while b <= last {
        let i = b / 64;
        let word = STREAM_BITS[0][i].load(Relaxed) | STREAM_BITS[1][i].load(Relaxed);
        let to = (i * 64 + 63).min(last);
        if word != 0 {
            for k in b..=to {
                if word >> (k % 64) & 1 != 0 {
                    let (lo, hi) = ((k * STREAM_BLOCK).max(at), ((k + 1) * STREAM_BLOCK).min(end));
                    n += hi - lo;
                }
            }
        }
        b = to + 1;
    }
    n
}

/// A new period: the one before last is forgotten.
fn stream_period() {
    if !STREAMED_ANY.load(Relaxed) {
        return;
    }
    let old = (STREAM_NOW.load(Relaxed) + 1) & 1;
    let mut any = false;
    for w in &STREAM_BITS[old] {
        if w.load(Relaxed) != 0 {
            w.store(0, Relaxed);
        }
    }
    for w in &STREAM_BITS[old ^ 1] {
        if w.load(Relaxed) != 0 {
            any = true;
            break;
        }
    }
    STREAM_NOW.store(old, Relaxed);
    STREAMED_ANY.store(any, Relaxed);
}

// ---------------------------------------------------------------- the lines

static FRAMES: AtomicU64 = AtomicU64::new(0);
static LOGGED_MS: AtomicU64 = AtomicU64::new(0);
static START: OnceLock<std::time::Instant> = OnceLock::new();
const PERIOD_MS: u64 = 5000;

/// A frame was presented: every ~5 s the lines.
pub fn tick() {
    // LOCAL (page-kinds): independent 5 s cadence, including when class stats are off.
    crate::pagekinds::tick();
    if !stats_on() && !honour_wc() {
        return;
    }
    FRAMES.fetch_add(1, Relaxed);
    let start = START.get_or_init(std::time::Instant::now);
    let now = start.elapsed().as_millis() as u64;
    let last = LOGGED_MS.load(Relaxed);
    if now < last + PERIOD_MS || LOGGED_MS.compare_exchange(last, now, Relaxed, Relaxed).is_err() {
        return;
    }
    let frames = FRAMES.swap(0, Relaxed).max(1);
    let secs = (now - last) as f64 / 1000.0;
    log(frames, secs);
}

fn log(frames: u64, secs: f64) {
    let per = frames as f64;
    let mb = |b: u64| b as f64 / per / 1e6;
    let ms = |ns: u64| ns as f64 / per / 1e6;
    let n = |c: u64| c as f64 / per;
    if stats_on() {
        // The registry's sizes.
        let mut held = [(0u64, 0u64); CLASSES];
        let mut host_wc = 0u64;
        if let Ok(r) = REGISTRY.try_read() {
            for e in &r.entries {
                let h = &mut held[e.class as usize];
                h.0 += 1;
                h.1 += (e.end - e.start) as u64;
                if e.host_wc {
                    host_wc += (e.end - e.start) as u64;
                }
            }
        }
        // By what the title asked for (no GPU-optimal/standard bit is known: the raw values),
        // committed and reserved apart, beside the console's budget.
        if let Ok(r) = REGISTRY.try_read() {
            let mut by: std::collections::BTreeMap<(u8, u32, u64), [u64; 3]> = Default::default();
            for e in &r.entries {
                let k = by.entry((e.class as u8, e.protect, e.flags)).or_default();
                k[0] += 1;
                let size = (e.end - e.start) as u64;
                // 0x1000: MEM_COMMIT at allocation (pool blocks: a reservation whatever pages
                // come later).
                if e.kind & 0x1000 != 0 && !e.pool { k[1] += size } else { k[2] += size }
            }
            let (optimal, standard) = crate::console::current().title_memory();
            tracing::info!(
                "class registry by request ({}: budget {} MB GPU-optimal + {} MB standard; committed/reserved): {}",
                crate::console::current().name(),
                optimal >> 20,
                standard >> 20,
                by.iter()
                    .filter(|(_, v)| v[1] + v[2] >= 1 << 20)
                    .map(|((c, p, f), v)| format!(
                        "{} protection {p:#x} flags {f:#x}: {} blocks {} MB committed {} MB reserved",
                        Class::from_index(*c as usize).name(),
                        v[0],
                        v[1] >> 20,
                        v[2] >> 20
                    ))
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
        tracing::info!(
            "class registry: {} (host write-combined {} MB)",
            (0..CLASSES)
                .filter(|&c| held[c].0 > 0)
                .map(|c| format!("{} {} blocks {} MB", Class::from_index(c).name(), held[c].0, held[c].1 >> 20))
                .collect::<Vec<_>>()
                .join("; "),
            host_wc >> 20
        );
        for c in 0..CLASSES {
            let v: Vec<u64> = (0..METRICS).map(|m| COSTS[m][c].swap(0, Relaxed)).collect();
            if v.iter().all(|&x| x == 0) {
                continue;
            }
            tracing::info!(
                "class stats {} per frame ({frames} frames, {secs:.1} s): watch {:.1} askings {:.1} MB {:.2} ms; \
                 guards {:.1} faults {:.2} ms waited; protection changes {:.1} {:.2} ms; \
                 refresh {:.1} MB checked {:.1} MB copied {:.2} ms; card uploads {:.1} MB in {:.1} ranges; \
                 copy-back {:.1} pieces {:.2} MB; textures {:.1} MB {:.2} ms",
                Class::from_index(c).name(),
                n(v[M::WatchAsks as usize]),
                mb(v[M::WatchBytes as usize]),
                ms(v[M::WatchNs as usize]),
                n(v[M::GuardFaults as usize]),
                ms(v[M::GuardWaitNs as usize]),
                n(v[M::Protects as usize]),
                ms(v[M::ProtectNs as usize]),
                mb(v[M::Checked as usize]),
                mb(v[M::Refreshed as usize]),
                ms(v[M::RefreshNs as usize]),
                mb(v[M::CardBytes as usize]),
                n(v[M::CardRanges as usize]),
                n(v[M::BackPieces as usize]),
                mb(v[M::BackBytes as usize]),
                mb(v[M::TexBytes as usize]),
                ms(v[M::TexNs as usize]),
            );
            let streamed = v[M::Streamed as usize];
            let again = [M::ReWatch, M::ReRefresh, M::ReCard, M::ReTex].map(|m| v[m as usize]);
            if streamed != 0 || again.iter().any(|&x| x != 0) || v[M::PoolMapped as usize] != 0 {
                let rate = |b: u64| b as f64 / secs / 1e6;
                let touched: u64 = again.iter().sum();
                tracing::info!(
                    "class streaming {}: streamed {:.1} MB/s; pool pages mapped {:.1} MB/s; recently streamed bytes touched again by the CPU {:.1} MB/s \
                     (watch {:.1}, refresh checks {:.1}, card uploads {:.1}, textures {:.1}); {:.2} touches per streamed byte",
                    Class::from_index(c).name(),
                    rate(streamed),
                    rate(v[M::PoolMapped as usize]),
                    rate(touched),
                    rate(again[0]),
                    rate(again[1]),
                    rate(again[2]),
                    rate(again[3]),
                    if streamed != 0 { touched as f64 / streamed as f64 } else { 0.0 }
                );
            }
        }
        let unplaced = STREAMED_UNPLACED.swap(0, Relaxed);
        if unplaced != 0 {
            tracing::info!(
                "class streaming: {:.1} MB/s into physical pages shown nowhere (or above 2^40), not classed",
                unplaced as f64 / secs / 1e6
            );
        }
        stream_period();
    }
    // Reads of write-combined memory, by site, most bytes first.
    let mut sites: Vec<(&'static Location<'static>, [u64; 4])> = SITE
        .iter()
        .filter_map(|s| {
            let at = s.at.load(Relaxed);
            if at.is_null() {
                return None;
            }
            let c = [0, 1, 2, 3].map(|i| s.counts[i].swap(0, Relaxed));
            // SAFETY: only `&'static Location`s are stored.
            (c[0] != 0).then(|| (unsafe { &*at } as &'static Location<'static>, c))
        })
        .collect();
    sites.sort_by_key(|s| std::cmp::Reverse(s.1[1]));
    if sites.is_empty() {
        tracing::info!("wc reads per frame: none ({frames} frames, {secs:.1} s)");
    }
    for (loc, c) in sites.iter().take(24) {
        tracing::info!(
            "wc reads per frame at {}:{}: {:.1} calls {:.1} KB; in host write-combined blocks {:.1} calls {:.1} KB",
            loc.file(),
            loc.line(),
            n(c[0]),
            c[1] as f64 / per / 1024.0,
            n(c[2]),
            c[3] as f64 / per / 1024.0
        );
    }
    if SITES_FULL.load(Relaxed) {
        tracing::info!("wc reads: the site table is full; reads at new sites are not counted");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(start: usize, end: usize, class: Class, host_wc: bool) -> Entry {
        Entry { start, end, protect: 0, kind: 0, flags: 0, pool: false, host_wc, class }
    }

    #[test]
    fn classes_from_protection() {
        // The 1277 MB block: read-write, write-combined, GPU access.
        assert_eq!(classify(0x2404, false), Class::WcGpuPrivate);
        assert_eq!(classify(0x2404, true), Class::WcGpuPool);
        assert_eq!(classify(0x22404, true), Class::WcGpuPool);
        // The TileHeap.
        assert_eq!(classify(0x2004, false), Class::GpuCached);
        assert_eq!(classify(0x22004, false), Class::GpuCached);
        assert_eq!(classify(0x4, false), Class::CpuOnly);
        // Uncached and GPU.
        assert_eq!(classify(0x2204, false), Class::WcGpuPrivate);
        // Write-combined without GPU bits: CPU-only.
        assert_eq!(classify(0x404, false), Class::CpuOnly);
        assert_eq!(classify(0, true), Class::Other);
    }

    #[test]
    fn registry_finds_inserts_and_releases() {
        let mut r = Registry::default();
        r.insert(entry(0x1000, 0x3000, Class::GpuCached, false));
        r.insert(entry(0x8000, 0x9000, Class::WcGpuPrivate, true));
        r.insert(entry(0x4000, 0x5000, Class::CpuOnly, false));
        assert_eq!(r.find(0xfff), None);
        assert_eq!(r.find(0x1000), Some(0));
        assert_eq!(r.find(0x2fff), Some(0));
        assert_eq!(r.find(0x3000), None);
        assert_eq!(r.find(0x4800).map(|i| r.entries[i].class), Some(Class::CpuOnly));
        assert_eq!(r.find(0x8000).map(|i| r.entries[i].class), Some(Class::WcGpuPrivate));
        // A new allocation over stale ones replaces them.
        r.insert(entry(0x2000, 0x4800, Class::WcGpuPool, false));
        assert_eq!(r.entries.len(), 2);
        assert_eq!(r.find(0x1000), None);
        assert_eq!(r.find(0x2000), Some(0));
        assert_eq!(r.entries[0].class, Class::WcGpuPool);
        // Only a release at the start takes an entry out.
        assert!(r.release(0x8800).is_none());
        assert!(r.release(0x8000).is_some());
        assert_eq!(r.find(0x8000), None);
    }

    #[test]
    fn spans_split_by_class() {
        let mut r = Registry::default();
        r.insert(entry(0x1000, 0x2000, Class::GpuCached, false));
        r.insert(entry(0x3000, 0x4000, Class::WcGpuPrivate, true));
        let mut got = Vec::new();
        r.split(0x800, 0x4000, |c, h, n| got.push((c, h, n)));
        assert_eq!(
            got,
            vec![
                (Class::Other, false, 0x800),
                (Class::GpuCached, false, 0x1000),
                (Class::Other, false, 0x1000),
                (Class::WcGpuPrivate, true, 0x1000),
                (Class::Other, false, 0x800),
            ]
        );
        let mut got = Vec::new();
        r.split(0x3800, 0x100, |c, h, n| got.push((c, h, n)));
        assert_eq!(got, vec![(Class::WcGpuPrivate, true, 0x100)]);
        let total: usize = {
            let mut t = 0;
            r.split(0, 0x10000, |_, _, n| t += n);
            t
        };
        assert_eq!(total, 0x10000);
    }

    #[test]
    fn streamed_blocks_are_found() {
        // (The bitmap is global; addresses picked apart from other tests'.)
        let base = 0x77_0000_0000usize;
        let now = STREAM_NOW.load(Relaxed) & 1;
        let bits = &STREAM_BITS[now];
        for b in base / STREAM_BLOCK..(base + 0x2_0000) / STREAM_BLOCK {
            bits[b / 64].fetch_or(1 << (b % 64), Relaxed);
        }
        assert_eq!(streamed_within(base, 0x2_0000), 0x2_0000);
        assert_eq!(streamed_within(base - 0x1000, 0x2000), 0x1000);
        assert_eq!(streamed_within(base + 0x1_8000, 0x2_0000), 0x8000);
        assert_eq!(streamed_within(base + 0x10_0000, 0x10_0000), 0);
        assert_eq!(streamed_within(STREAM_LIMIT, 0x1000), 0);
    }

    #[test]
    fn sites_are_found_again() {
        #[track_caller]
        fn here() -> &'static Location<'static> {
            Location::caller()
        }
        let a = here();
        let b = here();
        let sa = site(a).unwrap() as *const Site;
        let sb = site(b).unwrap() as *const Site;
        assert_ne!(sa, sb);
        assert_eq!(site(a).unwrap() as *const Site, sa);
    }
}
