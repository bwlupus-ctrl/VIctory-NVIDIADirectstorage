//! LOCAL (wc-vram): stage 1 of the GPU-first memory plan. `VICTORY_WC_VRAM=1` (off by default;
//! off, nothing here is called but [`touches`], one relaxed load that finds nothing) puts the
//! title's big write-combined, GPU-reachable private blocks (`xmem`'s `XMemVirtualAlloc` with
//! the selection of `VICTORY_HONOUR_WC`, of at least `VICTORY_WC_VRAM_MIN_MB`, default 64) in
//! the card's own memory, mapped for the CPU through the resizable BAR
//! (`D3D12_HEAP_TYPE_GPU_UPLOAD`). The address handed to the title IS the mapping: the title's
//! CPU writes land in video memory (31 GB/s measured, write-combined, as on the console), and
//! the card reads them at video-memory speed (1.58 TB/s). Shaders read and write it through
//! buffers placed in the block's heap ([`place`], bound by `crate::shadow` like the in-place
//! buffers of `crate::inplace`): no write watch, no guards, no refresh compares, no copies to
//! the card, no copy-back, no landing for that memory.
//!
//! What the probes said (scratchpad `probes\heapprobe`) and what follows from it:
//!
//! - A committed GPU_UPLOAD buffer (or heap) of up to 4095 MB; bigger is refused, so is a block
//!   that would need more ([`MOST`]): the title's memory is then made as before.
//! - The mapping is `MEM_PRIVATE`, protection `0x404` (read-write, write-combined).
//!   `VirtualProtect`, `VirtualFree(MEM_DECOMMIT)` and `VirtualFree(MEM_RELEASE)` on it fail with
//!   error 87 and the kernel's write watch refuses it: `xmem` takes the title's calls over
//!   ([`decommit`] writes zeros, [`release`] parks the block, protection changes are no-ops), and
//!   `crate::watch` never asks about it nor guards it ([`touches`]).
//! - Released, a GPU_UPLOAD resource leaves its addresses committed and D3D12 maps the next
//!   such buffer there (the host's own windows in card memory, `VICTORY_WINDOWS_BAR_MB`, are
//!   such buffers): a block the title releases is never released here, only parked and given
//!   to the next block the title asks for that fits ([`allocate`]).
//! - D3D12 picks the mapping's address top-down above 1 TiB, when it makes the heap; the
//!   title's GPU memory must lie below 2^40 (`xmem`'s `address_range`). While the heap is made,
//!   every free range from 1 TiB to the top of the address space is reserved ([`below_tib`]),
//!   so the mapping lands just below 1 TiB (0xFF80000000 and down), and released at once: the
//!   fence stands ~38 ms for a 1277 MB block here (~10 ms reserving, ~23 ms the heap); the
//!   mapping (~160 ms, the BAR's page tables) and the zeroing (~45 ms) come after it. Another
//!   thread that allocates address space in that moment gets it below 1 TiB too: harmless but
//!   for the address space it takes there. Should a driver pick the address at the mapping,
//!   the heap is made again and mapped inside the fence.
//! - A CPU read of the mapping crosses the bus a few bytes at a time (14 MB/s plain, 1.4 GB/s
//!   with streaming loads). The emulator's reads of title memory that pass through
//!   `crate::host`'s readers come here ([`read`]): pulled by a copy queue of our own into a
//!   read-back buffer when they are long ([`pull_bytes`]), else streaming loads; each counted
//!   by the reading place for the 5-second line ([`tick`]).
//! - The mapping survives the device's removal (probe c4): reads stay right after a TDR.
//! - The heap is made with `CREATE_NOT_ZEROED` (shorter in the fence) and zeroed by us, write-only,
//!   before the title or a shader sees it; a parked block given again has its title part zeroed
//!   (tested below).

use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

/// `VICTORY_WC_VRAM=1`.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_WC_VRAM"))
}

/// LOCAL (vram-tune): `VICTORY_WC_VRAM_GPU_FILLS=1` (off by default): the fills of upload
/// memory out of a block (constant views off 256 bytes) are done by a copy on the card in front
/// of their submission's lists (`crate::host::card_steps`), not with streaming loads on the CPU.
pub fn gpu_fills() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on() && crate::switch::on("VICTORY_WC_VRAM_GPU_FILLS"))
}

/// `VICTORY_WC_VRAM_MIN_MB=<n>` (64): only blocks of at least that size.
pub fn min_bytes() -> usize {
    static MIN: OnceLock<usize> = OnceLock::new();
    *MIN.get_or_init(|| std::env::var("VICTORY_WC_VRAM_MIN_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(64) << 20)
}

/// Reads of at least this many bytes are pulled by the copy queue (`VICTORY_WC_VRAM_PULL_KB`,
/// 64); shorter ones are streaming loads (a pull costs a queue round trip, ~50-100 us).
fn pull_bytes() -> usize {
    static AT: OnceLock<usize> = OnceLock::new();
    *AT.get_or_init(|| std::env::var("VICTORY_WC_VRAM_PULL_KB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(64) << 10)
}

const K64: usize = 64 << 10;
/// The title's address in a block lies on this line, and the block reaches the next one past
/// the title's end: the windows of title memory outside buffers (`crate::shadow`, 4 MiB cells)
/// that touch the title's memory then lie wholly in the block and are placed, not copied. It
/// also gives the 2 MiB alignment of `PAGES_2M` requests.
const GRID: usize = 4 << 20;
/// The largest GPU_UPLOAD resource the driver makes (probe: 4096 MB refused).
pub const MOST: usize = 4095 << 20;
const TIB: usize = 1 << 40;

/// A block in the card's memory.
struct Block {
    /// The title's allocation in it (address and bytes); 0 while parked.
    title: usize,
    len: usize,
    /// The mapping and the heap's bytes.
    base: usize,
    bytes: usize,
    heap: ID3D12Heap,
    /// One buffer over the whole heap: the mapping, and what pulls copy from.
    whole: ID3D12Resource,
}

static BLOCKS: Mutex<Vec<Block>> = Mutex::new(Vec::new());

/// LOCAL (page-kinds): live title ranges only; never maps or pulls BAR memory.
pub(crate) fn page_kind_ranges() -> Vec<(usize, usize)> {
    blocks().iter().filter(|b| b.title != 0).map(|b| (b.title, b.len)).collect()
}

/// The live blocks' mappings `[start, end)`, read without a lock by every hook: a slot is set
/// before it is counted in [`USED`], and cleared (0, 0) when its block is parked.
const SLOTS: usize = 32;
static FAST: [(AtomicUsize, AtomicUsize); SLOTS] = [const { (AtomicUsize::new(0), AtomicUsize::new(0)) }; SLOTS];
static USED: AtomicUsize = AtomicUsize::new(0);

/// Parked blocks' mappings (released by the title, still mapped and ours): reads of them fail
/// at once ([`read`]) instead of plain loads across the bus. Same rules as [`FAST`].
static PARKED: [(AtomicUsize, AtomicUsize); SLOTS] = [const { (AtomicUsize::new(0), AtomicUsize::new(0)) }; SLOTS];
static PARKED_USED: AtomicUsize = AtomicUsize::new(0);

fn parked_touches(at: usize, len: usize) -> bool {
    let n = PARKED_USED.load(Relaxed);
    if n == 0 {
        return false;
    }
    let end = at.saturating_add(len.max(1));
    PARKED[..n.min(SLOTS)].iter().any(|(s, e)| {
        let (s, e) = (s.load(Relaxed), e.load(Relaxed));
        s < e && at < e && s < end
    })
}

fn park_slot(start: usize, end: usize) {
    let n = PARKED_USED.load(Relaxed);
    let slot = PARKED[..n].iter().position(|(s, e)| s.load(Relaxed) == 0 && e.load(Relaxed) == 0).or((n < SLOTS).then_some(n));
    if let Some(slot) = slot {
        PARKED[slot].0.store(start, Relaxed);
        PARKED[slot].1.store(end, Relaxed);
        PARKED_USED.fetch_max(slot + 1, std::sync::atomic::Ordering::Release);
    }
}

fn unpark_slot(start: usize) {
    for (s, e) in &PARKED {
        if s.load(Relaxed) == start {
            e.store(0, Relaxed);
            s.store(0, Relaxed);
        }
    }
}

// ---------------------------------------------------------------- what was written there

/// Writes into blocks that are known (DirectStorage's, ours into title memory, what flights'
/// shaders were declared to write once they have run): `(from, to, stamp)`, the newest last,
/// at most [`JOURNAL_MOST`]; what fell out is answered "written" for any older stamp. Title CPU
/// writes are not seen (no write watch on the BAR): textures are hashed in turns for them
/// ([`texture_every`]).
static JOURNAL: Mutex<std::collections::VecDeque<(usize, usize, u64)>> = Mutex::new(std::collections::VecDeque::new());
static STAMP: AtomicU64 = AtomicU64::new(1);
/// The newest stamp that fell out of the journal.
static FORGOTTEN: AtomicU64 = AtomicU64::new(0);
const JOURNAL_MOST: usize = 8192;

/// The stamp now: what is noted written after this has a higher one.
pub(crate) fn stamp_now() -> u64 {
    STAMP.load(Relaxed)
}

/// `[from, to)` of title memory was written (noted if it touches a block).
pub(crate) fn note_written(from: usize, to: usize) {
    if from >= to || !touches(from, to - from) {
        return;
    }
    let stamp = STAMP.fetch_add(1, Relaxed) + 1;
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.len() >= JOURNAL_MOST
        && let Some((_, _, old)) = j.pop_front()
    {
        FORGOTTEN.fetch_max(old, Relaxed);
    }
    j.push_back((from, to, stamp));
}

/// Whether something noted written in `[from, to)` has a stamp above `since`.
pub(crate) fn written_since(from: usize, to: usize, since: u64) -> bool {
    if since < FORGOTTEN.load(Relaxed) {
        return true;
    }
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.iter().rev().take_while(|e| e.2 > since).any(|&(a, b, _)| a < to && from < b)
}

/// A texture in a block whose memory nothing known wrote is hashed again every this many
/// frames, by its address, the textures taking turns (`VICTORY_WC_VRAM_TEX_EVERY`, 8; 1: at
/// every check): title CPU writes to it are not seen otherwise.
pub(crate) fn texture_every() -> u64 {
    static EVERY: OnceLock<u64> = OnceLock::new();
    *EVERY.get_or_init(|| std::env::var("VICTORY_WC_VRAM_TEX_EVERY").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(8).max(1))
}

/// What else is counted, since the start ([`count`]).
#[derive(Clone, Copy)]
pub enum Count {
    /// Placed buffers refused (a copy made instead).
    PlaceFailed = 0,
    /// Constant views of descriptor tables bound to the block itself; and those filled on the
    /// CPU (off 256 bytes, or not in one copy).
    ConstantsBound = 1,
    ConstantFills = 2,
    /// Title copies (`memcpy`/`memmove`) out of a block and their bytes (streamed).
    TitleCopies = 3,
    TitleCopyBytes = 4,
    /// Title `VirtualAlloc` commits inside a block.
    TitleCommits = 5,
    /// Title `ReadFile`s into a block: through a buffer of ours; left to the kernel (overlapped).
    ReadsBounced = 6,
    ReadsAsync = 7,
    /// Reads of parked blocks, failed.
    ParkedReads = 8,
    /// Texture checks in a block: passed over (nothing known written, not its turn); hashed.
    TexSkipped = 9,
    TexHashed = 10,
    /// Reads that waited for flights writing what they read (card-side for pulls).
    WriterWaits = 11,
    /// LOCAL (vram-tune): reads that waited for every submission sent so far, the flights' line
    /// being held (`crate::host::writer_fence` with `VICTORY_WC_VRAM_AIR=0`).
    WriterAll = 12,
    /// LOCAL (vram-tune): fills of upload memory out of a block done on the card
    /// (`VICTORY_WC_VRAM_GPU_FILLS`), and submissions where they were left to the CPU.
    GpuFills = 13,
    GpuFillsLeft = 14,
    /// LOCAL (vram-tune): submissions whose CPU steps waited for such a copy of an earlier one
    /// to have read the bytes they write (`crate::host::card_reads_wait`).
    GpuFillWaits = 15,
}
static COUNTS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];

/// Counts `n` of `what`.
pub fn count(what: Count, n: u64) {
    COUNTS[what as usize].fetch_add(n, Relaxed);
}

fn blocks() -> std::sync::MutexGuard<'static, Vec<Block>> {
    BLOCKS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether any byte of `[at, at + len)` lies in a live block (its mapping, the padding beside
/// the title's memory included). One relaxed load when there is none.
#[inline]
pub fn touches(at: usize, len: usize) -> bool {
    let n = USED.load(Relaxed);
    if n == 0 {
        return false;
    }
    let end = at.saturating_add(len.max(1));
    FAST[..n.min(SLOTS)].iter().any(|(s, e)| {
        let (s, e) = (s.load(Relaxed), e.load(Relaxed));
        s < e && at < e && s < end
    })
}

/// Whether all of `[at, at + len)` lies in one live block.
#[inline]
pub fn holds(at: usize, len: usize) -> bool {
    span_of(at, len).is_some()
}

/// The mapping `[start, end)` of the live block holding all of `[at, at + len)`.
fn span_of(at: usize, len: usize) -> Option<(usize, usize)> {
    let n = USED.load(Relaxed);
    if n == 0 {
        return None;
    }
    let end = at.checked_add(len.max(1))?;
    FAST[..n.min(SLOTS)].iter().find_map(|(s, e)| {
        let (s, e) = (s.load(Relaxed), e.load(Relaxed));
        (s < e && s <= at && end <= e).then_some((s, e))
    })
}

/// The title's allocation (address, bytes) of the live block whose title memory holds `at`
/// (for `xmem`'s `VirtualQuery` and `VirtualFree`).
pub fn title_block(at: usize) -> Option<(usize, usize)> {
    if USED.load(Relaxed) == 0 {
        return None;
    }
    blocks().iter().find(|b| b.title != 0 && b.title <= at && at < b.title + b.len).map(|b| (b.title, b.len))
}

// ---------------------------------------------------------------- counters

/// Since the start: blocks made, parked blocks given again, title releases, decommits and their
/// bytes, protection changes passed over, queries answered, commits inside a block.
static EVENTS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
mod event {
    pub const MADE: usize = 0;
    pub const REUSED: usize = 1;
    pub const RELEASED: usize = 2;
    pub const DECOMMITS: usize = 3;
    pub const DECOMMIT_BYTES: usize = 4;
    pub const PROTECTS: usize = 5;
    pub const QUERIES: usize = 6;
    pub const COMMITS: usize = 7;
}

/// Fallbacks to the old path, by reason ([`Fallback`]).
static FALLBACKS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Why a block the title asked for is made as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// More than [`MOST`] with its padding.
    TooBig = 0,
    /// The host device is not there (yet).
    NoDevice = 1,
    /// The device offers no GPU_UPLOAD heaps (no resizable BAR, or the system's or driver's).
    NoUploadHeaps = 2,
    /// The driver refused the heap, the buffer or the mapping.
    Refused = 3,
    /// The mapping did not land where the title's flags allow (below 2^40).
    Address = 4,
    /// No slot left in the table of live blocks.
    Slots = 5,
}

impl Fallback {
    fn name(self) -> &'static str {
        ["too big", "no device yet", "no GPU_UPLOAD heaps", "refused by the driver", "mapped outside the allowed addresses", "slots full"][self as usize]
    }
}

/// GPU binds (placed buffers made), placed buffers alive and their bytes.
static BINDS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Per period: bytes pulled and pulls, bytes streamed and streaming reads, nanoseconds of
/// both, pull waits for the copy queue's lock; DirectStorage bytes decoded aside and streamed
/// in; frames.
static PERIOD: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
mod per {
    pub const PULLED: usize = 0;
    pub const PULLS: usize = 1;
    pub const STREAMED: usize = 2;
    pub const STREAMS: usize = 3;
    pub const NS: usize = 4;
    pub const PULL_WAITS: usize = 5;
    pub const DSTORAGE: usize = 6;
    pub const FRAMES: usize = 7;
    pub const PULL_FAILED: usize = 8;
    /// LOCAL (vram-tune): nanoseconds streaming reads stood waiting for writing flights.
    pub const WRITER_NS: usize = 9;
}

/// Reading sites (open addressing on the location), each: calls, bytes, nanoseconds.
const SITES: usize = 256;
struct Site {
    at: AtomicPtr<Location<'static>>,
    counts: [AtomicU64; 3],
}
static SITE: [Site; SITES] = [const { Site { at: AtomicPtr::new(std::ptr::null_mut()), counts: [const { AtomicU64::new(0) }; 3] } }; SITES];

#[allow(clippy::explicit_counter_loop)]
fn site(loc: &'static Location<'static>) -> Option<&'static Site> {
    let key = loc as *const Location<'static> as *mut Location<'static>;
    let mut i = (key as usize >> 3).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 56;
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
    None
}

// ---------------------------------------------------------------- making blocks

/// What [`allocate`] needs of the host: its device, and whether it offers GPU_UPLOAD heaps.
fn host_device() -> Option<(ID3D12Device, bool)> {
    let host = crate::host::host()?;
    Some((host.device.clone(), host.mapped_card_memory()))
}

/// A block for the title's `size` bytes of write-combined GPU memory, at an address in
/// `[lowest, highest]` on a 4 MiB line (zeroed, committed, read-write): the title's address,
/// or why it is to be made as before. Creates the host device if it is not made yet.
pub fn allocate(size: usize, lowest: usize, highest: usize) -> Result<usize, Fallback> {
    allocate_with(host_device, size, lowest, highest)
}

/// [`allocate`] with the device given by `device` (tests: none, the fallback).
pub(crate) fn allocate_with(
    device: impl FnOnce() -> Option<(ID3D12Device, bool)>,
    size: usize,
    lowest: usize,
    highest: usize,
) -> Result<usize, Fallback> {
    let result = allocate_inner(device, size, lowest, highest);
    match result {
        Ok(at) => tracing::info!(
            "wc-vram: title memory {at:#x}+{size:#x} ({} MB) is in the card's memory, mapped for the CPU (GPU_UPLOAD); no write watch, guards, refreshes, card copies or copy-back for it",
            size >> 20
        ),
        Err(why) => {
            FALLBACKS[why as usize].fetch_add(1, Relaxed);
            tracing::warn!("wc-vram: title memory of {size:#x} bytes ({} MB) made as before: {}", size >> 20, why.name());
        }
    }
    result
}

fn allocate_inner(
    device: impl FnOnce() -> Option<(ID3D12Device, bool)>,
    size: usize,
    lowest: usize,
    highest: usize,
) -> Result<usize, Fallback> {
    let bytes = GRID + size.max(1).next_multiple_of(GRID);
    if bytes > MOST {
        return Err(Fallback::TooBig);
    }
    // A parked block that fits: given again, its title memory zeroed.
    {
        let mut all = blocks();
        if let Some(b) = all.iter_mut().find(|b| {
            let title = b.base.next_multiple_of(GRID);
            b.title == 0 && title + size <= b.base + b.bytes && title >= lowest && title + size - 1 <= highest
        }) {
            let title = b.base.next_multiple_of(GRID);
            let Some(slot) = free_slot() else {
                return Err(Fallback::Slots);
            };
            // SAFETY: our own mapping, `size` bytes from `title` inside it.
            unsafe { zero(title as *mut u8, size) };
            b.title = title;
            b.len = size;
            unpark_slot(b.base);
            set_slot(slot, b.base, b.base + b.bytes);
            EVENTS[event::REUSED].fetch_add(1, Relaxed);
            return Ok(title);
        }
    }
    let Some((device, upload)) = device() else {
        return Err(Fallback::NoDevice);
    };
    if !upload {
        return Err(Fallback::NoUploadHeaps);
    }
    // One block made at a time: the fence is the whole address space above 1 TiB.
    static MAKING: Mutex<()> = Mutex::new(());
    let _making = MAKING.lock().unwrap_or_else(|e| e.into_inner());
    // D3D12 picks the mapping's address when it makes the heap (tried: a heap made before the
    // fence and mapped inside it still landed above 1 TiB; one made inside it and mapped after
    // lands below), so the fence stands around the heap's making alone (~10 ms to reserve and
    // ~23 ms for 1277 MB here); the mapping (~156 ms for 1277 MB: page tables of the BAR) and
    // the zeroing (~44 ms: the heap is made without, see `unfenced`) come after it.
    let made = below_tib(|| {
        let t = std::time::Instant::now();
        let made = unfenced(&device, bytes);
        FENCE[3].store(t.elapsed().as_micros() as u64, Relaxed);
        made
    })
    .and_then(|(heap, whole)| {
        let t = std::time::Instant::now();
        let base = map(&whole)?;
        FENCE[4].store(t.elapsed().as_micros() as u64, Relaxed);
        Ok((heap, whole, base))
    });
    // A driver that picks the address at the mapping: made again and mapped inside the fence.
    let made = match made {
        Ok((heap, whole, base)) if base + bytes > TIB => {
            tracing::info!("wc-vram: the mapping landed at {base:#x}, above 1 TiB: the heap is made again and mapped inside the fence");
            drop((heap, whole));
            below_tib(|| {
                let (heap, whole) = unfenced(&device, bytes)?;
                let base = map(&whole)?;
                windows::core::Result::Ok((heap, whole, base))
            })
        }
        other => other,
    };
    let (heap, whole, base) = made.map_err(|e| {
        tracing::warn!("wc-vram: a GPU_UPLOAD heap of {} MB: {e}", bytes >> 20);
        Fallback::Refused
    })?;
    let title = base.next_multiple_of(GRID);
    // (Dropping `heap` and `whole` gives the mapping back to D3D12, which may hand it to its
    // next GPU_UPLOAD buffer: nothing of ours points there yet.)
    if base % K64 != 0 || title < lowest || title + size - 1 > highest || title + size > base + bytes {
        tracing::warn!("wc-vram: the block's mapping {base:#x}+{bytes:#x} is not where the title's memory may be ({lowest:#x}..={highest:#x})");
        return Err(Fallback::Address);
    }
    // The heap was made without zeroing: all of it zeroed now (write-only), before the title
    // or a placed buffer sees any of it.
    let zeroing = std::time::Instant::now();
    // SAFETY: the whole mapping, ours.
    unsafe { zero(base as *mut u8, bytes) };
    FENCE[2].store(zeroing.elapsed().as_micros() as u64, Relaxed);
    crate::host::name(&whole, || format!("wc-vram block {title:#x}+{size:#x}"));
    let mut all = blocks();
    let Some(slot) = free_slot() else {
        return Err(Fallback::Slots);
    };
    all.push(Block { title, len: size, base, bytes, heap, whole });
    set_slot(slot, base, base + bytes);
    EVENTS[event::MADE].fetch_add(1, Relaxed);
    Ok(title)
}

/// A slot of [`FAST`] that is free (the caller holds [`BLOCKS`]).
fn free_slot() -> Option<usize> {
    let n = USED.load(Relaxed);
    FAST[..n].iter().position(|(s, e)| s.load(Relaxed) == 0 && e.load(Relaxed) == 0).or((n < SLOTS).then_some(n))
}

fn set_slot(slot: usize, start: usize, end: usize) {
    FAST[slot].0.store(start, Relaxed);
    FAST[slot].1.store(end, Relaxed);
    // (Counted after it is set: a reader that sees the count sees the slot.)
    USED.fetch_max(slot + 1, std::sync::atomic::Ordering::Release);
}

fn clear_slot(start: usize) {
    for (s, e) in &FAST {
        if s.load(Relaxed) == start {
            e.store(0, Relaxed);
            s.store(0, Relaxed);
        }
    }
}

fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// A GPU_UPLOAD heap of `bytes` and one buffer over all of it, not mapped yet.
fn unfenced(device: &ID3D12Device, bytes: usize) -> windows::core::Result<(ID3D12Heap, ID3D12Resource)> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: bytes as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() },
        Alignment: 0,
        // Zeroed by [`allocate`] after the fence (the title's committed memory reads as zeros).
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED,
    };
    let mut heap: Option<ID3D12Heap> = None;
    // SAFETY: plain creation with a valid description.
    unsafe { device.CreateHeap(&desc, &mut heap)? };
    let heap = heap.ok_or_else(windows::core::Error::empty)?;
    let mut whole: Option<ID3D12Resource> = None;
    // SAFETY: a buffer over the whole of a live heap.
    unsafe {
        device.CreatePlacedResource(&heap, 0, &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON, None, &mut whole)?
    };
    Ok((heap, whole.ok_or_else(windows::core::Error::empty)?))
}

/// Maps the block's buffer: the mapping's address. Mapped once and never unmapped while the
/// block lives.
fn map(whole: &ID3D12Resource) -> windows::core::Result<usize> {
    let mut data = std::ptr::null_mut();
    // SAFETY: a CPU-visible buffer.
    unsafe { whole.Map(0, None, Some(&mut data))? };
    if data.is_null() {
        return Err(windows::core::Error::empty());
    }
    Ok(data as usize)
}

/// The last fence: microseconds to reserve, microseconds held in all; then microseconds the
/// last block's zeroing took.
static FENCE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Runs `f` with every free range of the address space from 1 TiB up reserved, so that a
/// mapping D3D12 places top-down lands below 1 TiB; the reservations go right after.
fn below_tib<T>(f: impl FnOnce() -> T) -> T {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualAlloc, VirtualFree, VirtualQuery,
    };
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
    let started = std::time::Instant::now();
    // SAFETY: plain data filled in by the call.
    let top = unsafe {
        let mut info: SYSTEM_INFO = std::mem::zeroed();
        GetSystemInfo(&mut info);
        info.lpMaximumApplicationAddress as usize
    };
    // Released when this goes, a panic in `f` included.
    struct Held(Vec<usize>);
    impl Drop for Held {
        fn drop(&mut self) {
            for p in &self.0 {
                // SAFETY: our own reservations.
                unsafe { VirtualFree(*p as *mut _, 0, MEM_RELEASE) };
            }
        }
    }
    let mut held = Held(Vec::new());
    let mut at = TIB;
    while at < top {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
            break;
        }
        let end = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if info.State == MEM_FREE {
            let (a, e) = ((info.BaseAddress as usize).max(at).next_multiple_of(K64), end.min(top + 1) & !(K64 - 1));
            if e > a {
                // SAFETY: a reservation of free address space, released below.
                let p = unsafe { VirtualAlloc(a as *const _, e - a, MEM_RESERVE, PAGE_NOACCESS) };
                if !p.is_null() {
                    held.0.push(p as usize);
                }
            }
        }
        if end <= at {
            break;
        }
        at = end;
    }
    let fenced = started.elapsed();
    let result = f();
    let ranges = held.0.len();
    drop(held);
    FENCE[0].store(fenced.as_micros() as u64, Relaxed);
    FENCE[1].store(started.elapsed().as_micros() as u64, Relaxed);
    tracing::info!(
        "wc-vram: {} free ranges above 1 TiB held while the block was made ({:.2} ms to reserve, {:.2} ms in all)",
        ranges,
        fenced.as_secs_f64() * 1e3,
        started.elapsed().as_secs_f64() * 1e3
    );
    result
}

// ---------------------------------------------------------------- the title's calls

/// The title releases its allocation at `at`: the block is parked (kept, with its mapping, for
/// the next block the title asks for that fits) and what is placed over it is let go. `false`
/// if `at` is no live block's title address (then the host's call answers, and fails).
pub fn release(at: usize) -> bool {
    let span = {
        let mut all = blocks();
        let Some(b) = all.iter_mut().find(|b| b.title != 0 && b.title == at) else {
            return false;
        };
        b.title = 0;
        b.len = 0;
        clear_slot(b.base);
        park_slot(b.base, b.base + b.bytes);
        (b.base, b.bytes)
    };
    EVENTS[event::RELEASED].fetch_add(1, Relaxed);
    // What lists still hold stays alive until they are done with it.
    crate::shadow::in_place_let_go(span.0 as u64, span.1 as u64);
    tracing::info!("wc-vram: the title released its block at {at:#x}: parked (its mapping stays ours)");
    true
}

/// The title decommits `[at, at + len)` (or with no length the whole allocation from `at`) of
/// a block: it reads as zeros when committed again, so it is zeroed now. `false` if the range
/// is not title memory of one live block.
pub fn decommit(at: usize, len: usize) -> bool {
    let Some((title, size)) = title_block(at) else {
        return false;
    };
    let len = if len == 0 { title + size - at } else { len };
    let (from, to) = (at & !0xfff, (at + len).next_multiple_of(0x1000).min(title + size));
    if from < title || from >= to {
        return false;
    }
    // SAFETY: inside the block's mapping, which stays while it is ours.
    unsafe { zero(from as *mut u8, to - from) };
    EVENTS[event::DECOMMITS].fetch_add(1, Relaxed);
    EVENTS[event::DECOMMIT_BYTES].fetch_add((to - from) as u64, Relaxed);
    true
}

/// A protection change the title asked for in a block, passed over (the mapping refuses them).
pub fn protect_passed_over(at: usize, len: usize, protect: u32) {
    let n = EVENTS[event::PROTECTS].fetch_add(1, Relaxed);
    if n < 8 || (n + 1).is_power_of_two() {
        tracing::info!("wc-vram: the title's protection change {protect:#x} of {at:#x}+{len:#x} in card memory passed over ({} so far)", n + 1);
    }
}

/// A query the title made of a block, answered as for the allocation it asked for.
pub fn query_answered() {
    EVENTS[event::QUERIES].fetch_add(1, Relaxed);
}

/// A commit inside a block (committed already).
pub fn commit_inside() {
    EVENTS[event::COMMITS].fetch_add(1, Relaxed);
}

/// Zeros `len` bytes at `to` (write-only: the mapping is never read for it).
///
/// # Safety
/// `[to, to + len)` is writable memory.
unsafe fn zero(to: *mut u8, len: usize) {
    // SAFETY: the caller's.
    unsafe { to.write_bytes(0, len) };
    fence_stores();
}

/// A store fence: the title's write-combined stores reach the card before what comes next
/// (its lists' submission, a value written behind a flight).
pub fn fence_stores() {
    // SAFETY: a store fence, no operands.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence()
    };
}

/// Writes `bytes` at `to` without reading `to` (DirectStorage's destinations in a block).
///
/// # Safety
/// `[to, to + bytes.len())` is writable memory.
pub unsafe fn write_stream(to: *mut u8, bytes: &[u8]) {
    // SAFETY: the caller's; `copy_nonoverlapping` only stores to its destination.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), to, bytes.len()) };
    fence_stores();
    PERIOD[per::DSTORAGE].fetch_add(bytes.len() as u64, Relaxed);
    note_written(to as usize, to as usize + bytes.len());
}

// ---------------------------------------------------------------- the GPU side

/// A buffer placed over `[start, start + len)` of a live block (cut to its 64 KiB pages), for
/// `crate::shadow` to bind as the title's memory itself; `None` if the range is not all in one
/// live block (the caller then copies, as before).
pub(crate) fn place(start: u64, len: u64) -> Option<crate::inplace::Placed> {
    let (start, len) = (start as usize, len as usize);
    span_of(start, len)?;
    let Some(host) = crate::host::host() else {
        count(Count::PlaceFailed, 1);
        return None;
    };
    let all = blocks();
    let at = start & !(K64 - 1);
    let bytes = (start + len.max(1)).next_multiple_of(K64) - at;
    let Some(b) = all.iter().find(|b| b.title != 0 && b.base <= at && at + bytes <= b.base + b.bytes) else {
        // (In a block, but its 64 KiB pages not: the copy, as before, counted.)
        count(Count::PlaceFailed, 1);
        return None;
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: a buffer at a 64 KiB offset inside a live heap.
    let made = unsafe {
        host.device.CreatePlacedResource(
            &b.heap,
            (at - b.base) as u64,
            &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    let buffer = match made.ok().and(buffer) {
        Some(buffer) => buffer,
        None => {
            count(Count::PlaceFailed, 1);
            tracing::warn!("wc-vram: no buffer placed over {start:#x}+{len:#x}: a copy instead");
            return None;
        }
    };
    let heap = b.heap.clone();
    drop(all);
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x} in card memory (wc-vram)"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (start - at) as u64;
    BINDS[0].fetch_add(1, Relaxed);
    BINDS[1].fetch_add(1, Relaxed);
    BINDS[2].fetch_add(len as u64, Relaxed);
    Some(crate::inplace::Placed { buffer, heap, gpu, bias: (start - at) as u64, view: false })
}

/// LOCAL (vram-tune): the live blocks' mappings `[base, base + bytes)` and the buffer over each
/// whole heap (offset 0 at `base`), for copies on the card out of them
/// (`crate::host::card_steps`, `VICTORY_WC_VRAM_GPU_FILLS`).
pub(crate) fn wholes() -> Vec<(usize, usize, ID3D12Resource)> {
    if USED.load(Relaxed) == 0 {
        return Vec::new();
    }
    blocks().iter().filter(|b| b.title != 0).map(|b| (b.base, b.bytes, b.whole.clone())).collect()
}

/// What [`place`] gave is gone (`crate::shadow`'s copy of `len` bytes dropped).
pub(crate) fn unbound(len: u64) {
    BINDS[1].fetch_sub(1, Relaxed);
    BINDS[2].fetch_sub(len, Relaxed);
}

// ---------------------------------------------------------------- the emulator's reads

/// `crate::host`'s read of title memory `[src, src + len)` into `dst`, if any of it lies in a
/// live block: the block's parts pulled or streamed, the rest by `other`. `None` if none of it
/// does (the caller reads as before).
pub(crate) fn read(
    loc: &'static Location<'static>,
    src: usize,
    dst: *mut u8,
    len: usize,
    mut other: impl FnMut(usize, *mut u8, usize) -> bool,
) -> Option<bool> {
    if len == 0 {
        return None;
    }
    if !touches(src, len) {
        // A parked block (released by the title, still mapped): not title memory; fails at once
        // with zeros, instead of plain loads across the bus.
        if parked_touches(src, len) {
            count(Count::ParkedReads, 1);
            // SAFETY: `dst` holds `len` bytes.
            unsafe { dst.write_bytes(0, len) };
            return Some(false);
        }
        return None;
    }
    let started = std::time::Instant::now();
    let end = src + len;
    let mut at = src;
    let mut whole = true;
    let mut inside = 0usize;
    while at < end {
        // The live block holding `at`, if one does, else the next one's start.
        let (block, next) = {
            let n = USED.load(Relaxed).min(SLOTS);
            let mut block = None;
            let mut next = end;
            for (s, e) in &FAST[..n] {
                let (s, e) = (s.load(Relaxed), e.load(Relaxed));
                if s >= e {
                    continue;
                }
                if s <= at && at < e {
                    block = Some(e);
                } else if s > at {
                    next = next.min(s);
                }
            }
            (block, next)
        };
        match block {
            Some(e) => {
                let to = e.min(end);
                // SAFETY: `dst` holds `len` bytes, `[at, to)` lies in a live block.
                unsafe { fetch(at, dst.add(at - src), to - at) };
                inside += to - at;
                at = to;
            }
            None => {
                let to = next.min(end);
                // (The caller's own reader; a page that is not there fails it.)
                whole &= other(at, dst.wrapping_add(at - src), to - at);
                at = to;
            }
        }
    }
    let ns = started.elapsed().as_nanos() as u64;
    PERIOD[per::NS].fetch_add(ns, Relaxed);
    if let Some(s) = site(loc) {
        s.counts[0].fetch_add(1, Relaxed);
        s.counts[1].fetch_add(inside as u64, Relaxed);
        s.counts[2].fetch_add(ns, Relaxed);
    }
    Some(whole)
}

/// Reads `out.len()` bytes of title memory at `at` as `crate::host`'s readers do (a block's
/// part pulled or streamed, the rest through the kernel); false if a page is not there.
#[track_caller]
pub fn copy_out(at: usize, out: &mut [u8]) -> bool {
    let len = out.len();
    // LOCAL (heap-probe): the kernel's read fails on a page the probe made inaccessible
    // (`VICTORY_HEAP_PROBE=1`; off: nothing).
    let _probe = crate::heapprobe::emulator_reading(at, len);
    let plain = |s: usize, d: *mut u8, n: usize| {
        use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut done = 0;
        // SAFETY: `d` holds `n` bytes; the kernel checks the source.
        unsafe { ReadProcessMemory(GetCurrentProcess(), s as *const _, d.cast(), n, &mut done) != 0 }
    };
    read(Location::caller(), at, out.as_mut_ptr(), len, plain).unwrap_or_else(|| plain(at, out.as_mut_ptr(), len))
}

/// A `T` at `at` of title memory if it lies (wholly or in part) in a live block, read as
/// [`read`] reads (streaming loads; the part outside plainly), counted by the caller's place;
/// `None` elsewhere (the caller reads it as before). For the descriptor records the emulator
/// reads at the title's descriptor slots.
#[track_caller]
#[inline]
pub(crate) fn read_value<T: Copy>(at: usize) -> Option<T> {
    if !touches(at, size_of::<T>()) {
        return None;
    }
    let mut value = std::mem::MaybeUninit::<T>::uninit();
    read(Location::caller(), at, value.as_mut_ptr().cast(), size_of::<T>(), |s, d, n| {
        // SAFETY: the rest of the caller's `T`, which it vouches is readable title memory.
        unsafe { std::ptr::copy_nonoverlapping(s as *const u8, d, n) };
        true
    })?;
    // SAFETY: every byte was written above.
    Some(unsafe { value.assume_init() })
}

/// Reads `[src, src + len)` of a live block into `dst`: a pull for long reads, else (or if
/// the pull fails) streaming loads. Behind the flights in the air that were declared to write
/// any of it (`crate::host::writer_fence`): the pull waits for them on the card (the copy
/// queue stands, not the CPU), streaming loads wait for them here (the host's queue waits on
/// nothing the CPU signals, so it always gets there).
///
/// # Safety
/// `dst` holds `len` bytes; the range lies in one live block.
unsafe fn fetch(src: usize, dst: *mut u8, len: usize) {
    let writer = crate::host::writer_fence(src, len);
    if writer.is_some() {
        count(Count::WriterWaits, 1);
    }
    if len >= pull_bytes() && pull(src, dst, len, writer.as_ref()) {
        PERIOD[per::PULLED].fetch_add(len as u64, Relaxed);
        PERIOD[per::PULLS].fetch_add(1, Relaxed);
        return;
    }
    if let Some((fence, value)) = &writer {
        let waited = std::time::Instant::now();
        // SAFETY: a live fence; no event: the call returns once the value is reached (or the
        // device is gone).
        let _ = unsafe { fence.SetEventOnCompletion(*value, HANDLE::default()) };
        PERIOD[per::WRITER_NS].fetch_add(waited.elapsed().as_nanos() as u64, Relaxed);
    }
    // SAFETY: the caller's.
    unsafe { stream_load(src as *const u8, dst, len) };
    PERIOD[per::STREAMED].fetch_add(len as u64, Relaxed);
    PERIOD[per::STREAMS].fetch_add(1, Relaxed);
}

/// Copies `len` bytes from a block's mapping with streaming loads (`MOVNTDQA`, 1.4 GB/s from
/// the bus against 14 MB/s for plain loads), whole 64-byte lines (they lie in the block: its
/// mapping is 64 KiB aligned).
///
/// # Safety
/// `[src & !63, (src + len).next_multiple_of(64))` is readable; `dst` holds `len` bytes.
pub(crate) unsafe fn stream_load(src: *const u8, dst: *mut u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        // SAFETY: the caller's; the feature is there.
        unsafe { stream_load_sse41(src, dst, len) };
        return;
    }
    // SAFETY: the caller's.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn stream_load_sse41(src: *const u8, dst: *mut u8, len: usize) {
    use std::arch::x86_64::*;
    if len == 0 {
        return;
    }
    // Streaming loads may be served from a fill buffer of an earlier one: fenced first.
    _mm_mfence();
    let (start, end) = (src as usize, src as usize + len);
    let mut line = start & !63;
    let mut tmp = [0u8; 64];
    while line < end {
        let p = line as *mut __m128i;
        // SAFETY: a whole 64-byte line of readable memory (the caller's); `tmp` holds 64 bytes.
        unsafe {
            let a = _mm_stream_load_si128(p);
            let b = _mm_stream_load_si128(p.add(1));
            let c = _mm_stream_load_si128(p.add(2));
            let d = _mm_stream_load_si128(p.add(3));
            let t = tmp.as_mut_ptr().cast::<__m128i>();
            _mm_storeu_si128(t, a);
            _mm_storeu_si128(t.add(1), b);
            _mm_storeu_si128(t.add(2), c);
            _mm_storeu_si128(t.add(3), d);
        }
        let (from, to) = (line.max(start), (line + 64).min(end));
        // SAFETY: `[from, to)` lies in the line and in `[start, end)`, so in `dst` at its offset.
        unsafe { std::ptr::copy_nonoverlapping(tmp.as_ptr().add(from - line), dst.add(from - start), to - from) };
        line += 64;
    }
}

/// The copy queue that pulls.
struct Puller {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    value: u64,
    event: HANDLE,
    back: ID3D12Resource,
    data: *const u8,
}

// SAFETY: used under its mutex only; `data` is the read-back buffer's mapping, alive with it.
unsafe impl Send for Puller {}

const BACK: usize = 8 << 20;

impl Puller {
    fn new(device: &ID3D12Device) -> windows::core::Result<Puller> {
        use windows::Win32::System::Threading::CreateEventW;
        // SAFETY: plain creation of objects of our own.
        unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COPY, ..Default::default() })?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            let mut back: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(BACK as u64, D3D12_RESOURCE_FLAG_NONE),
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut back,
            )?;
            let back = back.ok_or_else(windows::core::Error::empty)?;
            let mut data = std::ptr::null_mut();
            back.Map(0, None, Some(&mut data))?;
            crate::host::name(&back, || "wc-vram pulls".into());
            Ok(Puller { device: device.clone(), queue, allocator, list, fence, value: 0, event, back, data: data.cast() })
        }
    }

    /// Copies `len` (at most [`BACK`]) bytes at `offset` of `from` into `dst`; false if the
    /// card did not (removed, or ten seconds passed).
    fn pull(&mut self, from: &ID3D12Resource, offset: u64, dst: *mut u8, len: usize, writer: Option<&(ID3D12Fence, u64)>) -> bool {
        use windows::Win32::System::Threading::WaitForSingleObject;
        // SAFETY: our own list, queue and fence, one pull at a time (the caller's lock); the
        // copy's range lies in both buffers.
        unsafe {
            if self.allocator.Reset().is_err() || self.list.Reset(&self.allocator, None).is_err() {
                return false;
            }
            self.list.CopyBufferRegion(&self.back, 0, from, offset, len as u64);
            if self.list.Close().is_err() {
                return false;
            }
            let Ok(list) = self.list.cast::<ID3D12CommandList>() else {
                return false;
            };
            // Behind the flights that write it, on the card.
            if let Some((fence, value)) = writer
                && self.queue.Wait(fence, *value).is_err()
            {
                return false;
            }
            self.queue.ExecuteCommandLists(&[Some(list)]);
            self.value += 1;
            if self.queue.Signal(&self.fence, self.value).is_err() {
                return false;
            }
            if self.fence.GetCompletedValue() < self.value {
                if self.fence.SetEventOnCompletion(self.value, self.event).is_err() {
                    return false;
                }
                // (Two seconds: a pull that does not come back breaks pulls for good.)
                if WaitForSingleObject(self.event, 2_000).0 != 0 {
                    return false;
                }
            }
            if self.device.GetDeviceRemovedReason().is_err() {
                return false;
            }
            std::ptr::copy_nonoverlapping(self.data, dst, len);
        }
        true
    }
}

static PULLER: Mutex<Option<Puller>> = Mutex::new(None);
static PULL_BROKEN: AtomicBool = AtomicBool::new(false);

/// Pulls `[src, src + len)` of a live block into `dst` by the copy queue; false if it cannot
/// (the caller streams).
fn pull(src: usize, dst: *mut u8, len: usize, writer: Option<&(ID3D12Fence, u64)>) -> bool {
    if PULL_BROKEN.load(Relaxed) {
        return false;
    }
    let (whole, base) = {
        let all = blocks();
        let Some(b) = all.iter().find(|b| b.base <= src && src + len <= b.base + b.bytes) else {
            return false;
        };
        (b.whole.clone(), b.base)
    };
    let mut guard = match PULLER.try_lock() {
        Ok(g) => g,
        Err(_) => {
            PERIOD[per::PULL_WAITS].fetch_add(1, Relaxed);
            PULLER.lock().unwrap_or_else(|e| e.into_inner())
        }
    };
    if guard.is_none() {
        let made = crate::host::host().ok_or_else(windows::core::Error::empty).and_then(|h| Puller::new(&h.device));
        match made {
            Ok(p) => *guard = Some(p),
            Err(e) => {
                PULL_BROKEN.store(true, Relaxed);
                tracing::warn!("wc-vram: no copy queue for reads of card memory ({e}): streaming loads instead");
                return false;
            }
        }
    }
    let Some(puller) = guard.as_mut() else {
        return false;
    };
    let mut done = 0;
    while done < len {
        let n = (len - done).min(BACK);
        // SAFETY: `dst` holds `len` bytes.
        if !puller.pull(&whole, (src - base + done) as u64, unsafe { dst.add(done) }, n, writer) {
            PERIOD[per::PULL_FAILED].fetch_add(1, Relaxed);
            // Broken for good on the first failure: no second wait of seconds per read.
            PULL_BROKEN.store(true, Relaxed);
            tracing::warn!("wc-vram: a pull by the copy queue failed: streaming loads from now on");
            if let Some((fence, value)) = writer {
                // SAFETY: as in `fetch`.
                let _ = unsafe { fence.SetEventOnCompletion(*value, HANDLE::default()) };
            }
            // What was pulled is right; the rest is streamed.
            // SAFETY: the rest of the range, in the block; `dst` holds it.
            unsafe { stream_load((src + done) as *const u8, dst.add(done), len - done) };
            return true;
        }
        done += n;
    }
    true
}

// ---------------------------------------------------------------- the line

/// Per presented frame: every ~5 s the line, once a block was ever made or refused.
pub(crate) fn tick() {
    if EVENTS[event::MADE].load(Relaxed) == 0 && FALLBACKS.iter().all(|f| f.load(Relaxed) == 0) {
        return;
    }
    PERIOD[per::FRAMES].fetch_add(1, Relaxed);
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms();
    let last = LAST.load(Relaxed);
    if last == 0 {
        let _ = LAST.compare_exchange(0, now, Relaxed, Relaxed);
        return;
    }
    if now < last + 5000 || LAST.compare_exchange(last, now, Relaxed, Relaxed).is_err() {
        return;
    }
    tracing::info!("{}", line());
}

/// The line: blocks, binds, the emulator's reads per frame by site, fallbacks.
pub(crate) fn line() -> String {
    let p: Vec<u64> = PERIOD.iter().map(|c| c.swap(0, Relaxed)).collect();
    let frames = p[per::FRAMES].max(1) as f64;
    let (mut live, mut live_bytes, mut parked, mut parked_bytes) = (0u64, 0u64, 0u64, 0u64);
    for b in blocks().iter() {
        if b.title != 0 {
            live += 1;
            live_bytes += b.len as u64;
        } else {
            parked += 1;
            parked_bytes += b.bytes as u64;
        }
    }
    let mut sites: Vec<(u64, u64, u64, &'static Location<'static>)> = SITE
        .iter()
        .filter_map(|s| {
            let at = s.at.load(Relaxed);
            // SAFETY: a `&'static Location` stored by `site`.
            let loc = unsafe { at.cast_const().as_ref() }?;
            let c = [0, 1, 2].map(|i| s.counts[i].swap(0, Relaxed));
            (c[0] != 0).then_some((c[0], c[1], c[2], loc))
        })
        .collect();
    sites.sort_unstable_by_key(|s| std::cmp::Reverse(s.1));
    let by_site = sites
        .iter()
        .take(12)
        .map(|(n, b, ns, loc)| {
            let file = loc.file().rsplit(['/', '\\']).next().unwrap_or(loc.file());
            format!("{file}:{} {:.1} calls {:.2} MB {:.2} ms", loc.line(), *n as f64 / frames, *b as f64 / frames / 1e6, *ns as f64 / frames / 1e6)
        })
        .collect::<Vec<_>>()
        .join(", ");
    let fallbacks = FALLBACKS
        .iter()
        .enumerate()
        .filter(|(_, f)| f.load(Relaxed) != 0)
        .map(|(i, f)| {
            let why = [Fallback::TooBig, Fallback::NoDevice, Fallback::NoUploadHeaps, Fallback::Refused, Fallback::Address, Fallback::Slots][i];
            format!("{} {}", f.load(Relaxed), why.name())
        })
        .collect::<Vec<_>>();
    let e: Vec<u64> = EVENTS.iter().map(|c| c.load(Relaxed)).collect();
    let c: Vec<u64> = COUNTS.iter().map(|c| c.load(Relaxed)).collect();
    format!(
        "wc-vram (VICTORY_WC_VRAM): {live} blocks {} MB in card memory ({parked} parked, {} MB); GPU binds {} made, {} alive ({} MB); \
         per frame: emulator CPU reads of it {:.2} MB in {:.1} reads, {:.2} ms ({:.2} MB pulled by the copy queue in {:.1}, {:.2} MB streaming loads in {:.1}; \
         {:.1} pulls waited for the queue, {:.1} failed and streamed), DirectStorage {:.2} MB decoded aside and streamed in; title CPU writes into it not measurable \
         (no write watch on the BAR); since the start: {} blocks made, {} given again, {} released, {} decommits ({} MB zeroed), {} protection changes passed over, \
         {} queries answered, {} commits inside; placed buffers refused {}; table constants bound in place {}, filled on the CPU {}; title copies out of it {} \
         ({} MB, streamed); title VirtualAlloc commits inside {}; title ReadFiles into it {} through a buffer, {} overlapped left to the kernel; parked reads failed {}; \
         textures passed over {}, hashed {}; reads behind writing flights {} (LOCAL vram-tune: {} of them behind every submission, the line being held; streaming reads stood {:.2} ms a frame for them, VICTORY_WC_VRAM_AIR={}); fills of upload memory out of it done on the card {}, submissions that left them to the CPU {}, CPU steps that waited for them to be read {} (VICTORY_WC_VRAM_GPU_FILLS={}); fallbacks: {}; reads by site: {}",
        live_bytes >> 20,
        parked_bytes >> 20,
        BINDS[0].load(Relaxed),
        BINDS[1].load(Relaxed),
        BINDS[2].load(Relaxed) >> 20,
        (p[per::PULLED] + p[per::STREAMED]) as f64 / frames / 1e6,
        (p[per::PULLS] + p[per::STREAMS]) as f64 / frames,
        p[per::NS] as f64 / frames / 1e6,
        p[per::PULLED] as f64 / frames / 1e6,
        p[per::PULLS] as f64 / frames,
        p[per::STREAMED] as f64 / frames / 1e6,
        p[per::STREAMS] as f64 / frames,
        p[per::PULL_WAITS] as f64 / frames,
        p[per::PULL_FAILED] as f64 / frames,
        p[per::DSTORAGE] as f64 / frames / 1e6,
        e[event::MADE],
        e[event::REUSED],
        e[event::RELEASED],
        e[event::DECOMMITS],
        e[event::DECOMMIT_BYTES] >> 20,
        e[event::PROTECTS],
        e[event::QUERIES],
        e[event::COMMITS],
        c[Count::PlaceFailed as usize],
        c[Count::ConstantsBound as usize],
        c[Count::ConstantFills as usize],
        c[Count::TitleCopies as usize],
        c[Count::TitleCopyBytes as usize] >> 20,
        c[Count::TitleCommits as usize],
        c[Count::ReadsBounced as usize],
        c[Count::ReadsAsync as usize],
        c[Count::ParkedReads as usize],
        c[Count::TexSkipped as usize],
        c[Count::TexHashed as usize],
        c[Count::WriterWaits as usize],
        c[Count::WriterAll as usize],
        p[per::WRITER_NS] as f64 / frames / 1e6,
        u8::from(crate::host::air_switch()),
        c[Count::GpuFills as usize],
        c[Count::GpuFillsLeft as usize],
        c[Count::GpuFillWaits as usize],
        u8::from(gpu_fills()),
        if fallbacks.is_empty() { "none".into() } else { fallbacks.join(", ") },
        if by_site.is_empty() { "none".into() } else { by_site },
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The card's device, if the default adapter is one that offers GPU_UPLOAD heaps (the
    /// tests below skip without it).
    pub(crate) fn card_with_bar() -> bool {
        let Some(host) = crate::host::host() else {
            eprintln!("wc-vram tests: no host device; skipped");
            return false;
        };
        if !host.mapped_card_memory() {
            eprintln!("wc-vram tests: no GPU_UPLOAD heaps on this device; skipped");
            return false;
        }
        true
    }

    /// One test at a time makes and releases blocks (they share the tables).
    pub(crate) fn turn() -> std::sync::MutexGuard<'static, ()> {
        static TURN: Mutex<()> = Mutex::new(());
        TURN.lock().unwrap_or_else(|e| e.into_inner())
    }

    const LOW: usize = 1 << 32;
    const HIGH: usize = (1 << 40) - 1;

    fn read_all(at: usize, len: usize) -> Vec<u8> {
        let mut v = vec![0xa5u8; len];
        // SAFETY: a live block's memory, `len` bytes.
        unsafe { stream_load(at as *const u8, v.as_mut_ptr(), len) };
        v
    }

    /// A block lands below 2^40 on a 4 MiB line, reads as zeros, is told apart by the tables,
    /// and is parked and given again (zeroed) when released.
    #[test]
    fn a_block_lies_below_2_40_zeroed_and_is_given_again() {
        let _turn = turn();
        if !card_with_bar() {
            return;
        }
        let size = 64 << 20;
        let at = allocate(size, LOW, HIGH).expect("a block");
        eprintln!("wc-vram test block at {at:#x}+{size:#x}");
        assert!(at >= LOW && at + size <= 1 << 40, "{at:#x}");
        assert_eq!(at % GRID, 0);
        assert!(touches(at, 1) && holds(at, size) && !holds(at, size + GRID + 1));
        assert_eq!(title_block(at + 5), Some((at, size)));
        let bytes = read_all(at, size);
        assert!(bytes.iter().all(|&b| b == 0), "a new block reads as zeros");
        // SAFETY: the block's memory.
        unsafe { std::ptr::write_bytes((at + 4096) as *mut u8, 0x5c, 3 * 4096) };
        assert!(decommit(at + 8192, 4096));
        let page = read_all(at + 4096, 3 * 4096);
        assert!(page[..4096].iter().all(|&b| b == 0x5c) && page[4096..8192].iter().all(|&b| b == 0) && page[8192..].iter().all(|&b| b == 0x5c));
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes(at as *mut u8, 0x77, 4096) };
        assert!(!release(at + 1));
        assert!(release(at));
        assert!(!touches(at, size) && title_block(at).is_none());
        let made = EVENTS[event::MADE].load(Relaxed);
        let again = allocate(size / 2, LOW, HIGH).expect("the parked block again");
        assert_eq!(EVENTS[event::MADE].load(Relaxed), made, "a parked block that fits is given again, none made");
        assert!(read_all(again, 4096).iter().all(|&b| b == 0), "given again zeroed");
        assert!(release(again));
    }

    /// A block of the title's size (1277 MB) lands below 2^40 too, its far end zeroed and read
    /// by the copy queue.
    #[test]
    fn a_block_of_the_titles_size_lies_below_2_40() {
        let _turn = turn();
        if !card_with_bar() {
            return;
        }
        let size = 1277 << 20;
        let started = std::time::Instant::now();
        let at = allocate(size, LOW, HIGH).expect("a block of 1277 MB");
        eprintln!(
            "wc-vram test: 1277 MB block at {at:#x} in {:.1} ms; the fence held {} us ({} us to reserve, heap {} us); after it map {} us, zeroing {} us",
            started.elapsed().as_secs_f64() * 1e3,
            FENCE[1].load(Relaxed),
            FENCE[0].load(Relaxed),
            FENCE[3].load(Relaxed),
            FENCE[4].load(Relaxed),
            FENCE[2].load(Relaxed)
        );
        assert!(at + size <= 1 << 40);
        let mut tail = vec![0xa5u8; 1 << 20];
        assert_eq!(read(Location::caller(), at + size - tail.len(), tail.as_mut_ptr(), tail.len(), |_, _, _| false), Some(true));
        assert!(tail.iter().all(|&b| b == 0));
        assert!(release(at));
    }

    /// A buffer placed over part of a block (off its 64 KiB lines) is what a shader reads and
    /// writes: the title's CPU bytes in, the shader's store seen at the title's address.
    #[test]
    fn a_placed_buffer_is_read_and_written_by_a_shader() {
        let _turn = turn();
        if !card_with_bar() {
            return;
        }
        let size = 64 << 20;
        let at = allocate(size, LOW, HIGH).expect("a block");
        let start = at + 3 * K64 + 0x40;
        // SAFETY: the block's memory, as the title writes it.
        unsafe {
            (start as *mut u32).write(40);
            (start as *mut u32).add(1).write(2);
        }
        fence_stores();
        let placed = place(start as u64, 256).expect("placed");
        assert_eq!(placed.bias, 0x40);
        let device = crate::host::host().unwrap().device.clone();
        let mut rig = crate::inplace::Rig::new(&device).expect("rig");
        rig.shade(placed.gpu).expect("the shader ran");
        let mut word = [0u8; 4];
        assert_eq!(read(Location::caller(), start + 8, word.as_mut_ptr(), 4, |_, _, _| false), Some(true));
        assert_eq!(u32::from_le_bytes(word), 42, "the shader's store, at the title's address");
        drop(placed);
        assert!(release(at));
    }

    /// Reads of a block through `crate::host`'s readers are pulled (long) or streamed (short),
    /// never plain loads, counted by site; a read across its edge takes the rest as before.
    #[test]
    fn the_emulators_reads_are_pulled_or_streamed() {
        let _turn = turn();
        if !card_with_bar() {
            return;
        }
        let size = 64 << 20;
        let at = allocate(size, LOW, HIGH).expect("a block");
        let pattern: Vec<u8> = (0..(1usize << 20)).map(|i| (i * 7 + 3) as u8).collect();
        // SAFETY: the block's memory.
        unsafe { write_stream((at + 12345) as *mut u8, &pattern) };
        PERIOD.iter().for_each(|c| c.store(0, Relaxed));
        let mut long = vec![0u8; pattern.len()];
        assert!(crate::host::read_into((at + 12345) as *const u8, &mut long));
        assert_eq!(long, pattern);
        let mut short = vec![0u8; 100];
        assert!(crate::host::read_into((at + 12345 + 5) as *const u8, &mut short));
        assert_eq!(short, pattern[5..105]);
        assert!(PERIOD[per::PULLED].load(Relaxed) >= pattern.len() as u64, "the long read was pulled");
        assert_eq!(PERIOD[per::STREAMED].load(Relaxed), 100, "the short one streamed");
        // Across the block's end: the part past it is not in a block and reads as before.
        let mut plain = vec![1u8; 64];
        let past = [7u8; 32];
        let mut both = vec![0u8; 96];
        let end = {
            let all = blocks();
            let b = all.iter().find(|b| b.title == at).unwrap();
            b.base + b.bytes
        };
        let r = read(Location::caller(), end - 64, both.as_mut_ptr(), 96, |s, d, n| {
            assert_eq!((s, n), (end, 32));
            // SAFETY: `d` holds the rest.
            unsafe { std::ptr::copy_nonoverlapping(past.as_ptr(), d, n) };
            true
        });
        assert_eq!(r, Some(true));
        plain.fill(0);
        assert_eq!(&both[..64], &plain[..], "the block's last bytes (zeros)");
        assert_eq!(&both[64..], &past[..]);
        assert!(line().contains("reads by site: wc_vram.rs") || line().contains("wc-vram"));
        assert!(release(at));
    }

    /// A title buffer, and a window outside buffers, in a block get the block's memory itself
    /// from `crate::shadow`: in place, nothing to refresh, no snapshot, nothing to copy back.
    #[test]
    fn title_buffers_and_windows_in_a_block_are_the_block_itself() {
        let _turn = turn();
        if !card_with_bar() {
            return;
        }
        let size = 64 << 20;
        let at = allocate(size, LOW, HIGH).expect("a block");
        let binds = BINDS[0].load(Relaxed);
        let (start, len) = ((at + 0x12340) as u64, 0x8000u64);
        crate::shadow::register(0x5ead, start, len);
        let shadow = crate::shadow::find(start + 0x100, 64).expect("the buffer's copy");
        assert!(shadow.in_place() && shadow.wc_vram(), "the block itself");
        assert_eq!(shadow.span(), (start, len));
        assert_eq!(shadow.offset(start + 0x100), 0x2340 + 0x100, "views count from the 64 KiB line");
        assert!(shadow.nothing_new(start as usize, len as usize));
        assert_eq!(shadow.update(start as usize, len as usize), 0);
        assert_eq!(shadow.copy_back(start as usize, len as usize), 0);
        // A shader write declared on it: no copy-back step, but the range is in the list's
        // written set (for the flights' order).
        let list = crate::host::host().unwrap().new_list().expect("a list");
        assert!(list.untouched());
        shadow.gpu_writes(&list, start + 0x200, 0x100);
        assert_eq!(list.wc_written(), vec![((start + 0x200) as usize, (start + 0x300) as usize)]);
        drop(list);
        // A window outside buffers, on the 4 MiB grid, deep in the block.
        let window = crate::shadow::find((at + (8 << 20) + 0x40) as u64, 256).expect("a window");
        assert!(window.in_place() && window.wc_vram());
        assert_eq!(BINDS[0].load(Relaxed), binds + 2);
        crate::shadow::unregister(0x5ead, start);
        drop((shadow, window));
        assert!(release(at));
    }

    /// Writes noted into a block are found by a later look and not by one stamped after them;
    /// a parked block's memory is no title memory: reads of it fail at once (zeros), no plain
    /// loads.
    #[test]
    fn writes_are_journaled_and_parked_blocks_fail_reads() {
        let _turn = turn();
        if !card_with_bar() {
            return;
        }
        let size = 64 << 20;
        let at = allocate(size, LOW, HIGH).expect("a block");
        let before = stamp_now();
        // SAFETY: the block's memory.
        unsafe { write_stream((at + 0x10000) as *mut u8, &[9u8; 256]) };
        assert!(written_since(at, at + 0x20000, before));
        assert!(!written_since(at + 0x20000, at + 0x30000, before), "elsewhere nothing");
        assert!(!written_since(at, at + 0x20000, stamp_now()), "a look after it");
        note_written(0x1000, 0x2000);
        assert!(!written_since(0x1000, 0x2000, before), "outside blocks nothing is noted");
        assert!(release(at));
        let parked = COUNTS[Count::ParkedReads as usize].load(Relaxed);
        let mut out = [1u8; 64];
        assert!(!copy_out(at + 0x10000, &mut out), "a parked block reads as not there");
        assert_eq!(out, [0u8; 64]);
        assert_eq!(COUNTS[Count::ParkedReads as usize].load(Relaxed), parked + 1);
        // Given again: title memory again, zeroed, readable.
        let again = allocate(size, LOW, HIGH).expect("given again");
        assert!(copy_out(again, &mut out) && out == [0u8; 64]);
        assert!(release(again));
    }

    /// Without the device a block is refused and counted as such (the title's memory is then
    /// made as before), as it is when bigger than the driver makes.
    #[test]
    fn without_a_device_the_block_falls_back() {
        let _turn = turn();
        let before = FALLBACKS[Fallback::NoDevice as usize].load(Relaxed);
        // (Bigger than any block the other tests park, which would be given again.)
        assert_eq!(allocate_with(|| None, 4000 << 20, LOW, HIGH), Err(Fallback::NoDevice));
        assert_eq!(FALLBACKS[Fallback::NoDevice as usize].load(Relaxed), before + 1);
        assert_eq!(allocate_with(|| None, 4095 << 20, LOW, HIGH), Err(Fallback::TooBig));
    }
}
