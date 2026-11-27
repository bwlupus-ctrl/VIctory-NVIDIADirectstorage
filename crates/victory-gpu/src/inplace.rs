//! Title memory as the card's own memory: a host heap opened on memory
//! the title already has, and a buffer placed in it, so that shaders read
//! and write the title's bytes themselves and no host copy stands between
//! (the console's one memory). This file holds the two openings Direct3D
//! has, a probe that says what this PC's driver makes of them
//! (`VICTORY_INPLACE_PROBE`), and a census of which title memory could be
//! opened.
//!
//! What the Direct3D runtime and the kernel answer, measured on this PC
//! with the software adapter (the rules are theirs, not the adapter's;
//! what the card's driver adds is what the probe is for):
//!
//! - `OpenExistingHeapFromAddress` takes the base address of a whole
//!   `VirtualAlloc` allocation and nothing inside it; every page must be
//!   committed and have one protection. Opening locks every page into the
//!   PC's memory at once (256 MiB never touched: 70 ms and 256 MB of
//!   working set).
//! - `OpenExistingHeapFromAddress1` (`ID3D12Device13`) takes any committed
//!   part of an allocation, with the same rule for its pages; parts may
//!   overlap, and the whole may be opened beside them. It also takes a
//!   committed view of a section.
//! - `OpenExistingHeapFromFileMapping` takes a section made with
//!   `SEC_COMMIT` and refuses one made with `SEC_RESERVE` (how `xmem`
//!   makes the title's page pool today). The heap is the section, not a
//!   view of it: views may be unmapped, mapped elsewhere or made
//!   read-only, and the section's handle closed, with the heap alive.
//!   Opening it locks nothing (256 MiB: 0.2 ms, no working set).
//! - The heap is a custom heap in the PC's memory (write-back pages, pool
//!   L0), shared and cross-adapter shared, 64 KiB aligned. A buffer in it
//!   must have `ALLOW_CROSS_ADAPTER`; unordered access beside it is taken.
//!   A texture must be row-major with one level. Tile mappings into such
//!   a heap are refused and REMOVE THE DEVICE: never call them.
//! - While a heap opened from an address stands, the title cannot
//!   decommit, release or protect the pages in it (`VirtualFree` and
//!   `VirtualProtect` fail with error 87), nor unmap a view in it. Pages
//!   outside the opened part are free to change.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, s};
use windows_sys::Win32::System::Memory as mem;

/// The unit heaps and placed buffers are cut on, and the console's page.
const K64: usize = 64 << 10;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// `VICTORY_INPLACE_PROBE`: 0 (unset) nothing; 1 the probe on memory of
/// its own at the first sweep, and the census 30 and 150 seconds later;
/// 2 also opens one part of the title's own memory for a moment at the
/// first census.
pub(crate) fn probe_level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| std::env::var("VICTORY_INPLACE_PROBE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// `VICTORY_INPLACE`: 0 (unset) off. 1: a window of title memory outside
/// buffers that lies in private memory committed whole is not copied: a
/// part of the title's allocation is opened as a host heap and the window
/// is a buffer placed over it ([`place_for`]). 2: title buffers too,
/// except the big ones in the card's memory.
///
/// What stage 1 covers is private memory (`XMemVirtualAlloc`'s). View
/// memory (the page pool) is turned down here: its section is made
/// reserved, which the host refuses, and what a view shows changes when
/// the title maps pages. LOCAL: unless [`views_on`] (with
/// `VICTORY_POOL_COMMIT=1`), for windows that are one run of a section.
///
/// What it assumes, for 2: nothing reads title memory that a submission
/// not yet run writes through such a buffer. With every submission waited
/// for (the default) that holds; with `VICTORY_LATE_WAIT` it does not,
/// since the flights' rules know shader writes only by the ranges copies
/// declare. Do not combine 2 with late wait.
///
/// shortcut: while a window is in place the title cannot decommit, free
/// or protect its pages (the call fails with error 87; the probe shows
/// it). The title's heaps are allocated once and kept, and nothing of the
/// kind was seen; [`title_memory_freeing`] is there for the runtime to
/// call before such a call once it takes them over.
pub(crate) fn level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| std::env::var("VICTORY_INPLACE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// The most title memory in place at once, in bytes
/// (`VICTORY_INPLACE_MB`, 4096): opening locks the pages into the PC's
/// memory, also those the title never touched.
fn most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| std::env::var("VICTORY_INPLACE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(4096) << 20)
}

/// In place now: windows and their bytes, title buffers and their bytes;
/// then, since the start: asks turned down because the memory is a view
/// or not private memory committed whole, because the host refused,
/// because of [`most`]; and what was let go because its memory changed.
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// A buffer placed over title memory, and its heap.
pub(crate) struct Placed {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) heap: ID3D12Heap,
    /// The host GPU address of the title memory's first byte.
    pub(crate) gpu: u64,
    /// Bytes from the buffer's first byte to the title memory's first.
    pub(crate) bias: u64,
    /// LOCAL: placed over a section of physical pages ([`place_view`]), not
    /// over a part of private memory.
    pub(crate) view: bool,
}

/// LOCAL: `VICTORY_INPLACE_VIEWS=1`, with [`level`] 1 or more: a window
/// outside buffers whose bytes are all one run of one section of the
/// title's physical pages ([`Cover::Run`]) is not copied either, if the
/// section was made with `SEC_COMMIT` (`VICTORY_POOL_COMMIT=1` on Windows):
/// the section is opened as a host heap once ([`From::Section`], kept per
/// section) and a buffer is placed over the run's offset in it
/// ([`place_view`]). The GPU reads the section's pages, which every view of
/// them shows, so what the title writes through any address of a page is
/// there with no write watching and no guard. A window over scattered
/// pages, or over unmapped ones, keeps its copy; so does a title buffer
/// over a view.
///
/// Such a window shows the pages the title had at its addresses when it
/// was placed. It is let go before the runtime changes any view in its
/// range (`crate::watch::title_memory_unmapping` and `_remapped` call
/// [`title_memory_freeing`]; every map and unmap of `xmem` goes through
/// them), and heap slots made on it are made anew ([`slots_stale`]). A
/// page shown at another address too changes nothing for it: the bytes
/// are the section's, wherever they are seen, and the section's pages are
/// never given out again while a view shows them.
///
/// `VICTORY_INPLACE_MB` does not count these windows: it bounds memory
/// locked by opening, and a section's heap locks nothing when opened (the
/// probe: 0.2 ms, no working set); it is opened whole, once, whatever the
/// windows over it.
pub(crate) fn views_on() -> bool {
    #[cfg(test)]
    if tests::VIEWS_ON.get() {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| level() >= 1 && std::env::var_os("VICTORY_INPLACE_VIEWS").is_some_and(|v| v == "1"))
}

/// LOCAL: of windows over views ([`views_on`]): in place now and their
/// bytes; since the start, turned down because their pages are scattered,
/// because some are not mapped, because the section is reserved, because
/// the section or the run is not known whole, because the host refused (the
/// heap or the buffer); sections opened as heaps; windows let go because
/// their views changed; and of those, ones heap slots showed.
static VIEW_STATS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

/// LOCAL: asks of [`place_for`] under way, from before they look at the
/// views until what they placed is counted in [`VIEW_STATS`]: a change of
/// mapping in between must not skip the let go ([`title_memory_freeing`]).
static PLACING: AtomicU64 = AtomicU64::new(0);

/// LOCAL: the longest window placed over a view, for the let go's search
/// (`crate::shadow::in_place_let_go`).
static LONGEST: AtomicU64 = AtomicU64::new(0);

struct Placing;

impl Placing {
    fn begin() -> Placing {
        PLACING.fetch_add(1, Ordering::SeqCst);
        Placing
    }
}

impl Drop for Placing {
    fn drop(&mut self) {
        PLACING.fetch_sub(1, Ordering::SeqCst);
    }
}

/// LOCAL: the longest window placed over a view so far (0: none).
pub(crate) fn longest() -> u64 {
    LONGEST.load(Ordering::Relaxed)
}

/// LOCAL: why a window was not placed in place ([`views_on`] only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// Views hold it, from more than one place in their sections.
    Scattered,
    /// Views hold only a part of it.
    Holes,
    /// One run of a section made reserved (`VICTORY_POOL_COMMIT` unset).
    Reserved,
    /// One run of a section not known (whole).
    Unknown,
    /// The host refused the heap or the buffer.
    Host,
    /// A title buffer over a view: only windows are placed there.
    Buffer,
    /// Not a view, and not private memory committed whole.
    NotPrivate,
    /// Over `VICTORY_INPLACE_MB`.
    Most,
}

impl Refusal {
    fn why(self) -> &'static str {
        match self {
            Refusal::Scattered => "view memory, scattered pages (a copy)",
            Refusal::Holes => "view memory with unmapped pages (a copy)",
            Refusal::Reserved => "one run of a section made reserved (VICTORY_POOL_COMMIT=1 makes it committed)",
            Refusal::Unknown => "one run of a section the runtime did not tell of whole",
            Refusal::Host => "the host refused the section's heap or a buffer in it",
            Refusal::Buffer => "a title buffer over view memory (only windows are placed over views)",
            Refusal::NotPrivate => "not a view, and not private memory committed whole",
            Refusal::Most => "over VICTORY_INPLACE_MB",
        }
    }
}

/// LOCAL: the first few refusals of each kind, said with their reason.
fn turned_down(start: u64, len: u64, why: Refusal, error: Option<&windows::core::Error>) {
    static SAID: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
    if SAID[why as usize].fetch_add(1, Ordering::Relaxed) < 4 {
        let error = error.map(|e| format!(": {:#010x} {}", e.code().0, e.message().trim())).unwrap_or_default();
        tracing::info!(start = format_args!("{start:#x}"), len, "title memory window not in place: {}{error}", why.why());
    }
}

/// LOCAL: sections opened as heaps, by handle and device; `None`: the host
/// refused, not asked again. Taken after [`VIEWS`] where both are.
struct Heaps(Vec<(usize, usize, Option<ID3D12Heap>)>);
// SAFETY: heaps are free-threaded; the list is behind a lock.
unsafe impl Send for Heaps {}

static HEAPS: std::sync::Mutex<Heaps> = std::sync::Mutex::new(Heaps(Vec::new()));

fn heaps() -> std::sync::MutexGuard<'static, Heaps> {
    HEAPS.lock().unwrap_or_else(|e| e.into_inner())
}

/// LOCAL: the heap of `section`, opened on `device` at the first asking.
/// With [`VIEWS`] held: a view the title sees is of the section, so its
/// handle is open and no other object's.
fn section_heap(device: &ID3D12Device, section: usize, bytes: u64) -> Result<ID3D12Heap, Option<windows::core::Error>> {
    let key = device.as_raw() as usize;
    let mut heaps = heaps();
    if let Some(known) = heaps.0.iter().find(|h| h.0 == section && h.1 == key) {
        return known.2.clone().ok_or(None);
    }
    let started = std::time::Instant::now();
    let opened = open(device, From::Section(section));
    match &opened {
        Ok(heap) => {
            VIEW_STATS[7].fetch_add(1, Ordering::Relaxed);
            tracing::info!(
                mb = bytes >> 20,
                ms = format_args!("{:.1}", started.elapsed().as_secs_f64() * 1000.0),
                "a section of the title's physical pages opened as a host heap, for windows over its views in place; {}",
                describe(heap)
            );
        }
        Err(e) => tracing::warn!(mb = bytes >> 20, error = %e, "a section of the title's physical pages refused as a host heap: its windows keep their copies"),
    }
    heaps.0.push((section, key, opened.as_ref().ok().cloned()));
    opened.map_err(Some)
}

/// LOCAL: the section's heap goes from the list (it lives on while a
/// buffer placed in it does).
fn section_heap_gone(section: usize) {
    let gone: Vec<_> = {
        let mut heaps = heaps();
        let (gone, kept) = std::mem::take(&mut heaps.0).into_iter().partition(|h| h.0 == section);
        heaps.0 = kept;
        gone
    };
    drop(gone);
}

/// LOCAL: title memory `[start, start + len)` in place over the section its
/// views show, if it is all one run of a section made committed: a buffer
/// placed at the run's offset in the section's heap, cut on 64 KiB lines.
pub(crate) fn place_view(device: &ID3D12Device, start: u64, len: u64) -> Result<Placed, (Refusal, Option<windows::core::Error>)> {
    let (heap, offset) = {
        let v = views();
        let (section, offset) = match cover_in(&v.views, start as usize, len as usize) {
            Cover::Run(section, offset) => (section, offset),
            Cover::Scattered => return Err((Refusal::Scattered, None)),
            Cover::Holes => return Err((Refusal::Holes, None)),
            Cover::None => return Err((Refusal::NotPrivate, None)),
        };
        let Some(&(_, bytes, committed)) = v.sections.iter().find(|s| s.0 == section) else {
            return Err((Refusal::Unknown, None));
        };
        if !committed {
            return Err((Refusal::Reserved, None));
        }
        let first = offset & !(K64 as u64 - 1);
        if (offset + len.max(1)).next_multiple_of(K64 as u64) > bytes {
            return Err((Refusal::Unknown, None));
        }
        (section_heap(device, section, bytes).map_err(|e| (Refusal::Host, e))?, (offset, first))
    };
    let (offset, first) = offset;
    let bytes = (offset + len.max(1)).next_multiple_of(K64 as u64) - first;
    let buffer = place(device, &heap, first, bytes).map_err(|e| (Refusal::Host, Some(e)))?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (offset - first);
    Ok(Placed { buffer, heap, gpu, bias: offset - first, view: true })
}

/// LOCAL: [`place_view`] on the host's device, counted.
fn place_over_view(start: u64, len: u64) -> Option<Placed> {
    let host = crate::host::host()?;
    place_view_counted(&host.device, start, len)
}

/// LOCAL: [`place_view`], counted in place ([`gone_view`] when it goes) or
/// turned down with its reason.
pub(crate) fn place_view_counted(device: &ID3D12Device, start: u64, len: u64) -> Option<Placed> {
    match place_view(device, start, len) {
        Ok(placed) => {
            // (Before the ask stops counting as under way: [`PLACING`].)
            VIEW_STATS[0].fetch_add(1, Ordering::SeqCst);
            VIEW_STATS[1].fetch_add(len, Ordering::Relaxed);
            LONGEST.fetch_max(len, Ordering::Relaxed);
            shown_note(&placed.buffer);
            Some(placed)
        }
        Err((why, error)) => {
            let at = match why {
                Refusal::Scattered => 2,
                Refusal::Holes => 3,
                Refusal::Reserved => 4,
                Refusal::Host => 6,
                _ => 5,
            };
            VIEW_STATS[at].fetch_add(1, Ordering::Relaxed);
            turned_down(start, len, why, error.as_ref());
            None
        }
    }
}

/// LOCAL: what [`place_view`] gave is gone (its buffer, by pointer). Slots
/// still made on the buffer (it outlives the window while a slot holds
/// it) are made anew as on a let go ([`let_go_view`]).
pub(crate) fn gone_view(bytes: u64, buffer: usize) {
    VIEW_STATS[0].fetch_sub(1, Ordering::SeqCst);
    VIEW_STATS[1].fetch_sub(bytes, Ordering::Relaxed);
    let mut s = slots();
    if s.0.get(&buffer).is_some_and(|shown| shown.gone == 0) {
        s.retire(buffer);
    }
}

// LOCAL: heap slots made on a window over a view. A slot keeps the buffer
// it was made on until its record changes (`HostHeap::mirror`), so a window
// let go because its view changed would go on being read there, showing
// the pages from before; its slots are made anew instead.

/// What is known of a buffer of a window over a view.
struct Shown {
    /// The heap slots made on it: (host heap, slot).
    slots: Vec<(usize, u32)>,
    /// When it was let go (`crate::host::now_ms`, at least 1); 0 while in
    /// place. Kept a while after: a slot may be made on it by a thread that
    /// found the window just before ([`slot_shows`]).
    gone: u64,
}

/// How long a buffer let go is remembered, in milliseconds.
const GONE_MS: u64 = 2000;

/// Buffers of windows over views, by pointer; slots to make anew, by host
/// heap; when buffers let go long ago were last forgotten.
struct Slots(std::collections::HashMap<usize, Shown>, std::collections::HashMap<usize, Vec<u32>>, u64);

impl Slots {
    /// The buffer at `buffer` is let go: its slots are to be made anew, and
    /// any slot made on it from now on too.
    fn retire(&mut self, buffer: usize) {
        let now = crate::host::now_ms().max(1);
        let shown = self.0.entry(buffer).or_insert(Shown { slots: Vec::new(), gone: 0 });
        shown.gone = now;
        let slots = std::mem::take(&mut shown.slots);
        if !slots.is_empty() {
            VIEW_STATS[9].fetch_add(1, Ordering::Relaxed);
            for (heap, slot) in slots {
                self.1.entry(heap).or_default().push(slot);
            }
            SLOTS_STALE.store(true, Ordering::Release);
        }
        if now >= self.2 + GONE_MS / 2 {
            self.2 = now;
            self.0.retain(|_, shown| shown.gone == 0 || now < shown.gone + GONE_MS);
        }
        ENTRIES.store(self.0.len(), Ordering::Relaxed);
    }
}

static SLOTS: std::sync::LazyLock<std::sync::Mutex<Slots>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Slots(Default::default(), Default::default(), 0)));
/// Whether [`Slots`]' second part has any.
static SLOTS_STALE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// How many buffers [`Slots`]' first part knows (asked without the lock).
static ENTRIES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn slots() -> std::sync::MutexGuard<'static, Slots> {
    SLOTS.lock().unwrap_or_else(|e| e.into_inner())
}

fn shown_note(buffer: &ID3D12Resource) {
    let mut s = slots();
    s.0.insert(buffer.as_raw() as usize, Shown { slots: Vec::new(), gone: 0 });
    ENTRIES.store(s.0.len(), Ordering::Relaxed);
}

/// LOCAL: slot `slot` of host heap `heap` is made on `buffer`
/// (`HostHeap::keep`). Nothing unless it is a window over a view; at once
/// to be made anew if that was let go meanwhile.
pub(crate) fn slot_shows(heap: usize, slot: u32, buffer: &ID3D12Resource) {
    if !views_on() || ENTRIES.load(Ordering::Relaxed) == 0 {
        return;
    }
    let mut s = slots();
    let Some(shown) = s.0.get_mut(&(buffer.as_raw() as usize)) else {
        return;
    };
    if shown.gone != 0 {
        s.1.entry(heap).or_default().push(slot);
        SLOTS_STALE.store(true, Ordering::Release);
    } else if !shown.slots.contains(&(heap, slot)) {
        shown.slots.push((heap, slot));
    }
}

/// LOCAL: the window over a view on `buffer` was let go
/// (`crate::shadow::in_place_let_go`): the slots made on it are to be made
/// anew ([`slots_stale`]).
pub(crate) fn let_go_view(buffer: &ID3D12Resource) {
    VIEW_STATS[8].fetch_add(1, Ordering::Relaxed);
    slots().retire(buffer.as_raw() as usize);
}

/// LOCAL: the slots of host heap `heap` made on windows over views that
/// were let go since the last asking: to be made anew from what their
/// records show now. None while the runtime changes a mapping (the views
/// it maps are not told yet: a slot made now would get a copy): the next
/// asking takes them.
pub(crate) fn slots_stale(heap: usize) -> Vec<u32> {
    if !SLOTS_STALE.load(Ordering::Acquire) || !crate::watch::changes().is_multiple_of(2) {
        return Vec::new();
    }
    let mut s = slots();
    let stale = s.1.remove(&heap).unwrap_or_default();
    if s.1.is_empty() {
        SLOTS_STALE.store(false, Ordering::Release);
    }
    stale
}

/// LOCAL: host heap `heap` is gone: nothing is to be made in it.
pub(crate) fn heap_gone(heap: usize) {
    if ENTRIES.load(Ordering::Relaxed) == 0 && !SLOTS_STALE.load(Ordering::Acquire) {
        return;
    }
    let mut s = slots();
    s.1.remove(&heap);
    for shown in s.0.values_mut() {
        shown.slots.retain(|slot| slot.0 != heap);
    }
    if s.1.is_empty() {
        SLOTS_STALE.store(false, Ordering::Release);
    }
}

/// Opens the 64 KiB pages that hold title memory `[start, start + len)`
/// as a heap of their own and places one buffer over all of them. `Err`
/// with nothing if the memory is not private memory committed whole
/// (nothing was asked of the host), with the host's error if it refused.
pub(crate) fn place_part(device: &ID3D12Device, start: u64, len: u64) -> Result<Placed, Option<windows::core::Error>> {
    let at = start & !(K64 as u64 - 1);
    let bytes = (start + len.max(1)).next_multiple_of(K64 as u64) - at;
    if !private_whole(at as usize, bytes as usize) {
        return Err(None);
    }
    let heap = open(device, From::Part(at as usize, bytes as usize)).map_err(Some)?;
    let buffer = place(device, &heap, 0, bytes).map_err(Some)?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (start - at);
    Ok(Placed { buffer, heap, gpu, bias: start - at, view: false })
}

/// Title memory `[start, start + len)` in place, for a window outside
/// buffers or a title buffer, if [`level`] asks for it and the memory can
/// be opened; `None` leaves the caller to make a copy, as before.
pub(crate) fn place_for(start: u64, len: u64, window: bool) -> Option<Placed> {
    if level() < if window { 1 } else { 2 } {
        return None;
    }
    // LOCAL: under way from before the look at the views until what is
    // placed is counted ([`PLACING`]).
    let _placing = views_on().then(Placing::begin);
    // View memory: known from the runtime without asking Windows.
    if cover(start as usize, len as usize) != Cover::None {
        // LOCAL: a window over one run of a committed section, in place.
        if views_on() {
            if window {
                return place_over_view(start, len);
            }
            turned_down(start, len, Refusal::Buffer, None);
        }
        STATS[4].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if STATS[1].load(Ordering::Relaxed) + STATS[3].load(Ordering::Relaxed) + len > most() {
        STATS[6].fetch_add(1, Ordering::Relaxed);
        if views_on() {
            turned_down(start, len, Refusal::Most, None);
        }
        return None;
    }
    let host = crate::host::host()?;
    match place_part(&host.device, start, len) {
        Ok(placed) => {
            let at = if window { 0 } else { 2 };
            STATS[at].fetch_add(1, Ordering::Relaxed);
            STATS[at + 1].fetch_add(len, Ordering::Relaxed);
            Some(placed)
        }
        Err(None) => {
            STATS[4].fetch_add(1, Ordering::Relaxed);
            if views_on() {
                turned_down(start, len, Refusal::NotPrivate, None);
            }
            None
        }
        Err(Some(e)) => {
            if STATS[5].fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::warn!(start = format_args!("{start:#x}"), len, error = %e, "title memory not opened in place: a copy instead");
            }
            None
        }
    }
}

/// What [`place_for`] gave is gone.
pub(crate) fn gone(window: bool, bytes: u64) {
    let at = if window { 0 } else { 2 };
    STATS[at].fetch_sub(1, Ordering::Relaxed);
    STATS[at + 1].fetch_sub(bytes, Ordering::Relaxed);
}

/// Title memory `[at, at + len)` is about to show something else, or the
/// title is about to decommit, free or protect it: what is in place there
/// is let go (the tables forget it; its heap goes when the last list and
/// slot that hold it do). Called for memory the runtime maps anew; for
/// the runtime to call before the title's own `VirtualFree` once it takes
/// that call over.
pub fn title_memory_freeing(at: usize, len: usize) {
    // LOCAL: windows over views, or one being placed. (Under way first: one
    // placed is counted in place before it stops counting as under way.)
    let views = views_on() && (PLACING.load(Ordering::SeqCst) != 0 || VIEW_STATS[0].load(Ordering::SeqCst) != 0);
    if !views && STATS[0].load(Ordering::Relaxed) + STATS[2].load(Ordering::Relaxed) == 0 {
        return;
    }
    let gone = crate::shadow::in_place_let_go(at as u64, len as u64);
    STATS[7].fetch_add(gone, Ordering::Relaxed);
}

/// The line of what is in place, every ten seconds while [`level`] is on.
fn say() {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 10_000 || LAST.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    tracing::info!(
        "title memory in place: {} windows {} MB, {} title buffers {} MB (no copy, no refresh, no copy-back); turned down: {} view memory or not private memory committed whole, \
         {} refused by the host, {} over VICTORY_INPLACE_MB; {} let go because their memory changed",
        s[0],
        s[1] >> 20,
        s[2],
        s[3] >> 20,
        s[4],
        s[5],
        s[6],
        s[7]
    );
    // LOCAL: windows over views, apart.
    if views_on() {
        let v: Vec<u64> = VIEW_STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
        tracing::info!(
            "title memory in place over the page pool (VICTORY_INPLACE_VIEWS): {} windows {} MB over {} sections opened; turned down: {} scattered pages, \
             {} with unmapped pages, {} of a reserved section (VICTORY_POOL_COMMIT), {} not known whole, {} refused by the host; \
             {} let go because their views changed, {} of them shown in heap slots (made anew)",
            v[0],
            v[1] >> 20,
            v[7],
            v[2],
            v[3],
            v[4],
            v[5],
            v[6],
            v[8],
            v[9]
        );
    }
}

/// What a heap is opened on.
pub(crate) enum From {
    /// A whole `VirtualAlloc` allocation, by its base address.
    Region(usize),
    /// A committed part of one: address and bytes.
    Part(usize, usize),
    /// A section made with `SEC_COMMIT`, by its handle.
    Section(usize),
}

/// Opens a heap on memory the process has.
pub(crate) fn open(device: &ID3D12Device, from: From) -> windows::core::Result<ID3D12Heap> {
    // SAFETY: the calls take an address or a handle and check both
    // themselves; a refusal is an error code.
    unsafe {
        match from {
            From::Region(at) => device.cast::<ID3D12Device3>()?.OpenExistingHeapFromAddress(at as *const c_void),
            From::Part(at, bytes) => device.cast::<ID3D12Device13>()?.OpenExistingHeapFromAddress1(at as *const c_void, bytes),
            From::Section(handle) => device.cast::<ID3D12Device3>()?.OpenExistingHeapFromFileMapping(HANDLE(handle as *mut c_void)),
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

/// A buffer of `size` bytes at `offset` (a multiple of 64 KiB) of an
/// opened heap, which shaders may read and write, in the common state.
pub(crate) fn place(device: &ID3D12Device, heap: &ID3D12Heap, offset: u64, size: u64) -> windows::core::Result<ID3D12Resource> {
    let mut placed: Option<ID3D12Resource> = None;
    // The heap is cross-adapter shared, so the buffer must say so too.
    let flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    // SAFETY: plain creation in a live heap.
    unsafe { device.CreatePlacedResource(heap, offset, &buffer_desc(size, flags), D3D12_RESOURCE_STATE_COMMON, None, &mut placed)? };
    placed.ok_or_else(windows::core::Error::empty)
}

// The title's views of its physical pages, as the runtime maps them: the
// census needs them to say which view memory is one run of a section (one
// placed buffer could show it) and which is scattered pages.

struct Views {
    /// By address: bytes, the section's handle, the offset in it.
    views: BTreeMap<usize, (usize, usize, u64)>,
    /// By section and byte offset of a 64 KiB page: the address of that
    /// page in every view that shows it. A write through one view of a
    /// page is a write of the bytes every other view shows
    /// (`crate::watch` guards them all).
    pages: BTreeMap<(usize, u64), Vec<usize>>,
    /// Of those, how many are shown more than once.
    pages_twice: usize,
    /// Sections by handle: bytes, and whether made committed.
    sections: Vec<(usize, u64, bool)>,
    /// Our own view of a whole section, by its handle (0: the host gave
    /// none), made when first asked for ([`beside`]) and kept.
    whole: Vec<(usize, usize)>,
}

/// Read by many threads at once (write faults, landings), changed when the
/// runtime maps or unmaps.
static VIEWS: std::sync::RwLock<Views> = std::sync::RwLock::new(Views {
    views: BTreeMap::new(),
    pages: BTreeMap::new(),
    pages_twice: 0,
    sections: Vec::new(),
    whole: Vec::new(),
});

fn views() -> std::sync::RwLockReadGuard<'static, Views> {
    VIEWS.read().unwrap_or_else(|e| e.into_inner())
}

fn views_mut() -> std::sync::RwLockWriteGuard<'static, Views> {
    VIEWS.write().unwrap_or_else(|e| e.into_inner())
}

/// The runtime made a section of `bytes` for physical pages; `committed`
/// if with `SEC_COMMIT` (only such a section can be opened as a heap).
pub fn title_memory_section(section: usize, bytes: u64, committed: bool) {
    let mut v = views_mut();
    v.sections.retain(|s| s.0 != section);
    // A handle's number given out again: the view was of the section that
    // had it before (it stays mapped: someone may be writing through it).
    v.whole.retain(|w| w.0 != section);
    v.sections.push((section, bytes, committed));
    // LOCAL: so is a heap opened on the one before.
    section_heap_gone(section);
}

/// The runtime lets go of a section, and of its handle: no view the title
/// sees is of it any more, and [`beside`]'s view goes.
pub fn title_memory_section_gone(section: usize) {
    let whole = {
        let mut v = views_mut();
        let at = v.whole.iter().position(|w| w.0 == section);
        // LOCAL: and its heap, while the views are held (no window is placed
        // over the section meanwhile: no view the title sees is of it).
        section_heap_gone(section);
        at.map(|i| v.whole.swap_remove(i).1)
    };
    if let Some(base) = whole.filter(|&base| base != 0) {
        // SAFETY: a view of ours. Who writes through it found the section
        // in a view the title saw and holds off its unmapping
        // (`crate::watch::title_memory_reading`), which came before this.
        unsafe { mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: base as *mut c_void }) };
    }
}

/// The runtime mapped `[at, at + len)` as a view of `section` from
/// `offset` on. Call after [`crate::title_memory_remapped`] of the range.
pub fn title_memory_view(at: usize, len: usize, section: usize, offset: u64) {
    // LOCAL (gpu-first): the allocation holding it is pool-mapped (`crate::memclass`; off unless
    // asked for).
    crate::memclass::pool_mapped(at, len);
    {
        let mut v = views_mut();
        v.views.insert(at, (len, section, offset));
        for k in 0..len.div_ceil(K64) {
            let shown = v.pages.entry((section, offset + (k * K64) as u64)).or_default();
            shown.push(at + k * K64);
            let twice = shown.len() == 2;
            v.pages_twice += usize::from(twice);
        }
    }
    // This view is writable from now on: a guard over the same bytes at
    // another view must look at them again, and the pages are noted written
    // there (`crate::watch`).
    crate::watch::views_mapped(section, offset, len);
}

/// What `[at, at + len)` showed is gone ([`crate::title_memory_remapped`]):
/// the views that touch it are forgotten (the runtime tells again those it
/// maps anew).
pub(crate) fn views_gone(at: usize, len: usize) {
    let mut v = views_mut();
    if v.views.is_empty() {
        return;
    }
    let end = at.saturating_add(len);
    let first = v.views.range(..=at).next_back().filter(|(base, view)| *base + view.0 > at).map(|(base, _)| *base);
    let gone: Vec<usize> = first.into_iter().chain(v.views.range(at.saturating_add(1)..end).map(|(base, _)| *base)).collect();
    for base in gone {
        let Some((bytes, section, offset)) = v.views.remove(&base) else {
            continue;
        };
        for k in 0..bytes.div_ceil(K64) {
            let key = (section, offset + (k * K64) as u64);
            if let Some(pages) = v.pages.get_mut(&key) {
                let page = base + k * K64;
                let before = pages.len();
                pages.retain(|&shown| shown != page);
                let after = pages.len();
                if after == 0 {
                    v.pages.remove(&key);
                }
                v.pages_twice = v.pages_twice.saturating_sub(usize::from(before >= 2 && after < 2));
            }
        }
    }
}

/// The 64 KiB pages of `[start, end)` that the title shows at other
/// addresses too, in address order, each with those; not the ones it shows
/// in very many views (`crate::watch::MANY_VIEWS`).
pub(crate) fn shown_elsewhere(start: usize, end: usize) -> Vec<(usize, Vec<usize>)> {
    let v = views();
    let mut out = Vec::new();
    if start >= end || v.pages_twice == 0 {
        return out;
    }
    let first = v.views.range(..=start).next_back().filter(|(base, view)| *base + view.0 > start).map(|(base, _)| *base);
    for (&base, &(bytes, section, offset)) in v.views.range(first.unwrap_or(start)..end) {
        let from = start.max(base) & !(K64 - 1);
        for page in (from..end.min(base + bytes)).step_by(K64) {
            let Some(shown) = v.pages.get(&(section, offset + (page - base) as u64)) else {
                continue;
            };
            if (2..=crate::watch::MANY_VIEWS).contains(&shown.len()) {
                out.push((page, shown.iter().copied().filter(|&at| at != page).collect()));
            }
        }
    }
    out
}

/// The address of the 64 KiB page shown at `page` in every title view that
/// shows it, `page`'s own address included; empty if `page` is not a view.
pub(crate) fn pages_of(page: usize) -> Vec<usize> {
    let v = views();
    let Some((&base, &(bytes, section, offset))) = v.views.range(..=page).next_back() else {
        return Vec::new();
    };
    let within = page.wrapping_sub(base);
    if within >= bytes || !within.is_multiple_of(K64) {
        return Vec::new();
    }
    v.pages.get(&(section, offset + within as u64)).cloned().unwrap_or_default()
}

/// The title views that show any of the physical bytes `[offset, offset +
/// len)` of `section`, as `(address, bytes)` ranges in address order; not
/// those of a page shown in very many views (`MANY_VIEWS`).
pub(crate) fn views_of_section(section: usize, offset: u64, len: usize) -> Vec<(usize, usize)> {
    if len == 0 {
        return Vec::new();
    }
    let v = views();
    let end = offset.saturating_add(len as u64);
    let mut pages: Vec<usize> = Vec::new();
    let mut at = offset - offset % K64 as u64;
    while at < end {
        if let Some(shown) = v.pages.get(&(section, at))
            && shown.len() <= crate::watch::MANY_VIEWS
        {
            pages.extend(shown.iter().copied());
        }
        match at.checked_add(K64 as u64) {
            Some(next) => at = next,
            None => break,
        }
    }
    pages.sort_unstable();
    pages.dedup();
    let mut runs: Vec<(usize, usize)> = Vec::with_capacity(pages.len());
    for page in pages {
        match runs.last_mut() {
            Some((first, bytes)) if *first + *bytes == page => *bytes += K64,
            _ => runs.push((page, K64)),
        }
    }
    runs
}

/// How many bytes of `[at, at + len)` views of physical pages show; `None`
/// if no view touches it (nothing is mapped there, or memory of another
/// kind). The runtime's own record of what it mapped: no question to the
/// host.
pub(crate) fn shown_bytes(at: usize, len: usize) -> Option<u64> {
    let end = at.saturating_add(len);
    let v = views();
    let first = v.views.range(..=at).next_back();
    let rest = v.views.range(at.saturating_add(1)..end.max(at.saturating_add(1)));
    let (mut there, mut any) = (0u64, false);
    for (&base, view) in first.into_iter().chain(rest) {
        let (from, to) = (base.max(at), base.saturating_add(view.0).min(end));
        if from < to {
            there += (to - from) as u64;
            any = true;
        }
    }
    any.then_some(there)
}

/// What of `[at, at + len)` views of physical pages show, in order, pieces
/// that touch as one: (address, bytes). Empty: nothing, or memory of another
/// kind.
pub(crate) fn shown_runs(at: usize, len: usize) -> Vec<(usize, usize)> {
    let end = at.saturating_add(len);
    let v = views();
    let first = v.views.range(..=at).next_back();
    let rest = v.views.range(at.saturating_add(1)..end.max(at.saturating_add(1)));
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (&base, view) in first.into_iter().chain(rest) {
        let (from, to) = (base.max(at), base.saturating_add(view.0).min(end));
        if from >= to {
            continue;
        }
        match runs.last_mut() {
            Some((run, bytes)) if *run + *bytes == from => *bytes += to - from,
            _ => runs.push((from, to - from)),
        }
    }
    runs
}

/// How the views cover `[at, at + len)`.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum Cover {
    /// No view holds its first byte.
    None,
    /// Views hold a part of it only.
    Holes,
    /// Views hold all of it, from more than one place.
    Scattered,
    /// All of it is one run of one section: the section and the offset of
    /// `at` in it.
    Run(usize, u64),
}

fn cover_in(views: &BTreeMap<usize, (usize, usize, u64)>, at: usize, len: usize) -> Cover {
    let end = at + len.max(1);
    let Some((&base, &(bytes, section, offset))) = views.range(..=at).next_back().filter(|(base, view)| *base + view.0 > at) else {
        return Cover::None;
    };
    let first = (section, offset + (at - base) as u64);
    let (mut reached, mut one) = (base + bytes, true);
    while reached < end {
        let Some(&(bytes, s, o)) = views.get(&reached) else {
            return Cover::Holes;
        };
        one &= s == first.0 && o == first.1 + (reached - at) as u64;
        reached += bytes;
    }
    if one { Cover::Run(first.0, first.1) } else { Cover::Scattered }
}

/// [`Cover`] of `[at, at + len)` by the title's views as they are now.
pub(crate) fn cover(at: usize, len: usize) -> Cover {
    cover_in(&views().views, at, len)
}

/// Another address of the title memory `[at, at + len)`, if it is a view of
/// the title's pages, all of it one run of a section: in a view of ours of
/// the whole section, which is never guarded (`crate::watch`) nor changed.
///
/// The section's pages are reserved and the runtime commits each before it
/// shows it, which holds for every view of the section (LOCAL: or the
/// section is made committed, `VICTORY_POOL_COMMIT`). The caller holds
/// `crate::watch::title_memory_reading`: until it lets go the title sees
/// the same pages at `at`.
pub(crate) fn beside(at: usize, len: usize) -> Option<usize> {
    let known = {
        let v = views();
        let Cover::Run(section, offset) = cover_in(&v.views, at, len) else {
            return None;
        };
        v.whole.iter().find(|w| w.0 == section).map(|w| (w.1, offset))
    };
    if let Some((base, offset)) = known {
        return (base != 0).then(|| base + offset as usize);
    }
    let mut v = views_mut();
    let Cover::Run(section, offset) = cover_in(&v.views, at, len) else {
        return None;
    };
    let base = match v.whole.iter().find(|w| w.0 == section) {
        Some(w) => w.1,
        None => {
            let &(_, bytes, _) = v.sections.iter().find(|s| s.0 == section)?;
            // SAFETY: the whole of a section the runtime made; a view the
            // title sees is of it, so its handle is open.
            let base = unsafe { mem::MapViewOfFile(section as _, mem::FILE_MAP_WRITE, 0, 0, bytes as usize) }.Value as usize;
            if base == 0 {
                tracing::warn!(error = %std::io::Error::last_os_error(), "no view of the title's pages beside its own");
            }
            v.whole.push((section, base));
            base
        }
    };
    (base != 0).then(|| base + offset as usize)
}

/// Our own address of the 64 KiB page the title sees at `page` ([`beside`]),
/// and every address the title sees it at (0: no more); `None` if the page is
/// in no view of a section, or in very many. While the caller holds
/// `crate::watch::title_memory_reading`.
pub(crate) fn beside_page(page: usize) -> Option<(usize, [usize; 8])> {
    for asked in [false, true] {
        {
            let v = views();
            let (&base, &(bytes, section, offset)) = v.views.range(..=page).next_back()?;
            let within = page.wrapping_sub(base);
            if within >= bytes || !within.is_multiple_of(K64) {
                return None;
            }
            let offset = offset + within as u64;
            if let Some(whole) = v.whole.iter().find(|w| w.0 == section) {
                let mut shown = [0; 8];
                let all = v.pages.get(&(section, offset))?;
                shown.get_mut(..all.len())?.copy_from_slice(all);
                return (whole.1 != 0).then(|| (whole.1 + offset as usize, shown));
            }
        }
        if asked || beside(page, K64).is_none() {
            return None;
        }
    }
    None
}

/// Whether `[at, at + len)` is private memory, all committed and
/// read-write: what [`From::Part`] takes. One question to Windows (it
/// walks the page tables to the end of the pages that share the first
/// one's state: a millisecond in the title's big blocks, so ask when a
/// copy is made, not when it is used).
pub(crate) fn private_whole(at: usize, len: usize) -> bool {
    // SAFETY: plain data filled in by the query.
    let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == mem::MEM_COMMIT
        && info.Type == mem::MEM_PRIVATE
        && info.Protect == mem::PAGE_READWRITE
        && (info.BaseAddress as usize).saturating_add(info.RegionSize) >= at.saturating_add(len)
}

fn hr<T>(result: &windows::core::Result<T>) -> String {
    match result {
        Ok(_) => "ok".to_owned(),
        Err(e) => format!("REFUSED {:#010x} ({})", e.code().0, e.message().trim()),
    }
}

/// A queue, a list and a fence of the probe's own on `device`: nothing of
/// the host's submissions is touched. LOCAL (wc-vram): also for `crate::wc_vram`'s tests.
pub(crate) struct Rig {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    done: u64,
    event: HANDLE,
}

impl Drop for Rig {
    fn drop(&mut self) {
        // SAFETY: our event.
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.event);
        }
    }
}

impl Rig {
    pub(crate) fn new(device: &ID3D12Device) -> windows::core::Result<Rig> {
        // SAFETY: plain creation of objects of our own.
        unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            Ok(Rig { device: device.clone(), queue, allocator, list, fence, done: 0, event })
        }
    }

    /// Records with `record`, runs it and waits (ten seconds at most).
    fn run(&mut self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> windows::core::Result<()> {
        // SAFETY: our own list and queue, one run at a time.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            record(&self.list);
            self.list.Close()?;
            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
            self.done += 1;
            self.queue.Signal(&self.fence, self.done)?;
            self.fence.SetEventOnCompletion(self.done, self.event)?;
            if WaitForSingleObject(self.event, 10_000).0 != 0 {
                return Err(windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL));
            }
            self.device.GetDeviceRemovedReason()
        }
    }

    fn committed(&self, kind: D3D12_HEAP_TYPE, size: u64, state: D3D12_RESOURCE_STATES) -> windows::core::Result<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            self.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(size, D3D12_RESOURCE_FLAG_NONE), state, None, &mut buffer)?
        };
        buffer.ok_or_else(windows::core::Error::empty)
    }

    /// What the GPU reads of `size` bytes of `resource` from `offset`.
    fn fetch(&mut self, resource: &ID3D12Resource, offset: u64, size: usize) -> windows::core::Result<Vec<u8>> {
        let back = self.committed(D3D12_HEAP_TYPE_READBACK, size as u64, D3D12_RESOURCE_STATE_COPY_DEST)?;
        // SAFETY: two live buffers, the range inside both.
        self.run(|list| unsafe { list.CopyBufferRegion(&back, 0, resource, offset, size as u64) })?;
        let mut data = std::ptr::null_mut();
        // SAFETY: the mapping holds `size` bytes until unmapped.
        unsafe {
            back.Map(0, None, Some(&mut data))?;
            let bytes = std::slice::from_raw_parts(data.cast::<u8>(), size).to_vec();
            back.Unmap(0, None);
            Ok(bytes)
        }
    }

    /// The GPU writes `bytes` at `offset` of `resource` (a copy).
    fn store(&mut self, resource: &ID3D12Resource, offset: u64, bytes: &[u8]) -> windows::core::Result<()> {
        let up = self.committed(D3D12_HEAP_TYPE_UPLOAD, bytes.len() as u64, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        let mut data = std::ptr::null_mut();
        // SAFETY: the mapping holds the buffer's bytes until unmapped.
        unsafe {
            up.Map(0, None, Some(&mut data))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.cast::<u8>(), bytes.len());
            up.Unmap(0, None);
        }
        // SAFETY: two live buffers, the range inside both.
        self.run(|list| unsafe { list.CopyBufferRegion(resource, offset, &up, 0, bytes.len() as u64) })
    }

    /// A compute shader stores, through a root unordered-access view at
    /// `gpu`, the sum of the buffer's first two words in its third.
    pub(crate) fn shade(&mut self, gpu: u64) -> windows::core::Result<()> {
        const SOURCE: &str = "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n[RootSignature(RS)]\n[numthreads(1, 1, 1)]\nvoid main() { b.Store(8, b.Load(0) + b.Load(4)); }\n";
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; then the
        // blob's own bytes, which hold the root signature too.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("inplace"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None)?;
            let code = code.ok_or_else(windows::core::Error::empty)?;
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            })?;
            (root, pso)
        };
        // SAFETY: an open list of ours; the view names a live buffer.
        self.run(|list| unsafe {
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            list.SetComputeRootUnorderedAccessView(0, gpu);
            list.Dispatch(1, 1, 1);
        })
    }

    /// Milliseconds the card takes to copy `bytes` of `from` into its own
    /// memory, the best of three.
    fn copy_ms(&mut self, from: &ID3D12Resource, bytes: u64) -> windows::core::Result<f64> {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut to: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            self.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(bytes, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON, None, &mut to)?
        };
        let to = to.ok_or_else(windows::core::Error::empty)?;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let started = std::time::Instant::now();
            // SAFETY: two live buffers of `bytes` bytes or more.
            self.run(|list| unsafe { list.CopyBufferRegion(&to, 0, from, 0, bytes) })?;
            best = best.min(started.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(best)
    }
}

fn fill(at: *mut u8, len: usize, seed: u8) {
    for i in 0..len {
        // SAFETY: the caller's own committed memory of `len` bytes.
        unsafe { at.add(i).write((i as u8).wrapping_mul(31).wrapping_add(seed)) };
    }
}

fn same(at: *const u8, bytes: &[u8]) -> bool {
    // SAFETY: the caller's own committed memory of that length.
    unsafe { std::slice::from_raw_parts(at, bytes.len()) == bytes }
}

/// The process's working set in MB.
fn working_set_mb() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    let mut counters = PROCESS_MEMORY_COUNTERS { cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32, ..Default::default() };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut counters, counters.cb) };
    counters.WorkingSetSize as u64 >> 20
}

/// What the probe found. The lines are the whole account; the flags are
/// what the stages build on.
#[derive(Default)]
pub(crate) struct Probe {
    pub(crate) lines: Vec<String>,
    /// A part of private, write-watched memory opens, a buffer in it shows
    /// the memory's bytes to the GPU, also those the CPU wrote afterwards.
    pub(crate) part_reads: bool,
    /// What the GPU writes there (a copy, and a shader through unordered
    /// access) the CPU reads at the memory's address.
    pub(crate) part_writes: bool,
    /// The same two for a committed section, through a view of it.
    pub(crate) section_reads: bool,
    pub(crate) section_writes: bool,
    /// The section's heap outlives the view: the GPU's write with no view
    /// mapped shows in a view mapped afterwards.
    pub(crate) section_outlives_views: bool,
    /// A section made with `SEC_RESERVE` (as `xmem` makes the page pool)
    /// was refused.
    pub(crate) reserved_section_refused: bool,
}

/// Tries the openings on memory of the probe's own, made the way `xmem`
/// makes title memory, with a queue of its own on `device`; `memory`
/// gives the card's memory account (`Host::video_memory`) where there is
/// one. Nothing of the title's is touched. `mb` is the size of the cost
/// measurement (0 leaves it out).
pub(crate) fn probe(device: &ID3D12Device, memory: &dyn Fn() -> Option<[u64; 4]>, mb: usize) -> Probe {
    let mut p = Probe::default();
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => {
            p.lines.push(format!("no queue of its own: {e}"));
            return p;
        }
    };
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: a plain query that fills our structure.
    let _ = unsafe { device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, (&raw mut options).cast(), size_of_val(&options) as u32) };
    p.lines.push(format!(
        "the device: ID3D12Device3 {}, ID3D12Device13 {} (parts of an allocation); cross-adapter row-major textures {}; heap tier {}",
        device.cast::<ID3D12Device3>().is_ok(),
        device.cast::<ID3D12Device13>().is_ok(),
        options.CrossAdapterRowMajorTextureSupported.as_bool(),
        options.ResourceHeapTier.0
    ));
    // SAFETY: memory of the probe's own, made and released here; every
    // address used lies inside it.
    unsafe {
        private(&mut p, &mut rig);
        section(&mut p, &mut rig);
        if mb > 0 {
            cost(&mut p, &mut rig, memory, mb);
        }
    }
    p.lines.push(format!(
        "VERDICT: private memory in place: reads {}, writes {}; the page pool's section in place: reads {}, writes {}, whatever the views do {}; \
         a reserved section is refused {} (the pool must be made committed to be opened)",
        p.part_reads, p.part_writes, p.section_reads, p.section_writes, p.section_outlives_views, p.reserved_section_refused
    ));
    p
}

fn describe(heap: &ID3D12Heap) -> String {
    // SAFETY: a plain query.
    let d = unsafe { heap.GetDesc() };
    format!(
        "heap of {} KiB: type {} (4 = custom), CPU pages {} (3 = write-back), pool {} (1 = the PC's memory), alignment {}, flags {:#x} (0x1 shared, 0x20 cross-adapter, 0x200 write watch, 0x400 shader atomics)",
        d.SizeInBytes >> 10,
        d.Properties.Type.0,
        d.Properties.CPUPageProperty.0,
        d.Properties.MemoryPoolPreference.0,
        d.Alignment,
        d.Flags.0
    )
}

/// Private memory, reserved as `xmem` reserves the title's heaps
/// (committed, read-write, write-watched): a part of it opened.
unsafe fn private(p: &mut Probe, rig: &mut Rig) {
    const SIZE: usize = 64 * K64;
    // SAFETY (the whole function): the allocation below, `SIZE` bytes.
    unsafe {
        let base = mem::VirtualAlloc(std::ptr::null(), SIZE, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE).cast::<u8>();
        if base.is_null() {
            p.lines.push("private memory: none to try with".to_owned());
            return;
        }
        // The part: 16 pages from the 16th on.
        let (at, bytes) = (base.add(16 * K64), 16 * K64);
        let whole = open(&rig.device, From::Region(base as usize));
        p.lines.push(format!("private memory, the whole allocation (OpenExistingHeapFromAddress): {}", hr(&whole)));
        drop(whole);
        let opened = open(&rig.device, From::Part(at as usize, bytes));
        p.lines.push(format!("private memory, a part of it (OpenExistingHeapFromAddress1): {}", hr(&opened)));
        if let Ok(heap) = &opened {
            p.lines.push(format!("  {}", describe(heap)));
            let placed = place(&rig.device, heap, K64 as u64, 4 * K64 as u64);
            p.lines.push(format!("  a buffer placed in it (cross-adapter, unordered access): {}", hr(&placed)));
            if let Ok(buffer) = &placed {
                let memory = at.add(K64);
                fill(memory, 4 * K64, 1);
                let first = rig.fetch(buffer, 0, 4 * K64).is_ok_and(|got| same(memory, &got));
                fill(memory, 4 * K64, 77);
                let second = rig.fetch(buffer, 0, 4 * K64).is_ok_and(|got| same(memory, &got));
                p.part_reads = first && second;
                p.lines.push(format!("  the GPU reads the memory's bytes: {first}; and those the CPU wrote afterwards, with no call in between: {second}"));
                mem::ResetWriteWatch(base.cast(), SIZE);
                let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x5a).collect();
                let copied = rig.store(buffer, 4096, &new).is_ok() && same(memory.add(4096), &new);
                let (mut found, mut count, mut granularity) = ([0usize; 64], 64usize, 0u32);
                let watched = mem::GetWriteWatch(0, base.cast(), SIZE, found.as_mut_ptr().cast(), &mut count, &mut granularity) == 0;
                memory.cast::<[u32; 3]>().write([40, 2, 0]);
                let shaded = rig.shade(buffer.GetGPUVirtualAddress());
                let shader = shaded.is_ok() && memory.cast::<[u32; 3]>().read()[2] == 42;
                p.part_writes = copied && shader;
                p.lines.push(format!(
                    "  the CPU reads at the memory's address what the GPU wrote: by a copy {copied}; by a shader through unordered access {shader} ({}); \
                     write watching saw {} of the copy's 16 pages{}",
                    hr(&shaded),
                    if watched { count } else { 0 },
                    if watched { "" } else { " (the asking failed)" }
                ));
            }
            // What the title can still do with the pages while the heap stands.
            let mut old = 0;
            let protect_in = mem::VirtualProtect(at.add(8 * K64).cast(), K64, mem::PAGE_READONLY, &mut old) != 0;
            if protect_in {
                mem::VirtualProtect(at.add(8 * K64).cast(), K64, mem::PAGE_READWRITE, &mut old);
            }
            let decommit_in = mem::VirtualFree(at.add(9 * K64).cast(), K64, mem::MEM_DECOMMIT) != 0;
            let decommit_out = mem::VirtualFree(base.add(40 * K64).cast(), K64, mem::MEM_DECOMMIT) != 0;
            p.lines.push(format!(
                "  with the heap standing, the title's own calls: read-only inside the part {protect_in}, decommit inside {decommit_in}, decommit outside {decommit_out} \
                 (false inside means the heap must go before the title frees or guards those pages)"
            ));
            drop(placed);
        }
        drop(opened);
        let freed = mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE) != 0;
        p.lines.push(format!("  released after the heap went: {freed}"));
    }
}

/// A section and a view of it in a placeholder, as `xmem` maps physical
/// pages: reserved (as it is made today) and committed.
unsafe fn section(p: &mut Probe, rig: &mut Rig) {
    const SIZE: usize = 32 * K64;
    // SAFETY (the whole function): the sections, placeholder and views
    // made below.
    unsafe {
        let process = windows_sys::Win32::System::Threading::GetCurrentProcess();
        let invalid = windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        let reserved = mem::CreateFileMappingW(invalid, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_RESERVE, 0, SIZE as u32, std::ptr::null());
        if !reserved.is_null() {
            let opened = open(&rig.device, From::Section(reserved as usize));
            p.reserved_section_refused = opened.is_err();
            p.lines.push(format!("a section made reserved, as the page pool is today (OpenExistingHeapFromFileMapping): {}", hr(&opened)));
            drop(opened);
            windows_sys::Win32::Foundation::CloseHandle(reserved);
        }
        let handle = mem::CreateFileMappingW(invalid, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_COMMIT, 0, SIZE as u32, std::ptr::null());
        let hole = mem::VirtualAlloc2(process, std::ptr::null(), SIZE, mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER, mem::PAGE_NOACCESS, std::ptr::null_mut(), 0).cast::<u8>();
        if handle.is_null() || hole.is_null() {
            p.lines.push("a committed section: none to try with".to_owned());
            return;
        }
        // Pages 4..12 of the section at the placeholder's start.
        mem::VirtualFree(hole.cast(), 8 * K64, mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER);
        let view = mem::MapViewOfFile3(handle, process, hole.cast(), (4 * K64) as u64, 8 * K64, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
        let opened = open(&rig.device, From::Section(handle as usize));
        p.lines.push(format!("a section made committed (OpenExistingHeapFromFileMapping): {}", hr(&opened)));
        let mut unmapped = false;
        if let (Ok(heap), false) = (&opened, view.Value.is_null()) {
            p.lines.push(format!("  {}", describe(heap)));
            // The section's page 5 is the view's second page.
            let placed = place(&rig.device, heap, 5 * K64 as u64, 2 * K64 as u64);
            p.lines.push(format!("  a buffer placed at the section's offset of a view's pages: {}", hr(&placed)));
            if let Ok(buffer) = &placed {
                let memory = hole.add(K64);
                fill(memory, 2 * K64, 3);
                p.section_reads = rig.fetch(buffer, 0, 2 * K64).is_ok_and(|got| same(memory, &got));
                let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x33).collect();
                p.section_writes = rig.store(buffer, 0, &new).is_ok() && same(memory, &new);
                // The guard of `watch.rs`: the view read-only.
                let mut old = 0;
                let guarded = mem::VirtualProtect(memory.cast(), K64, mem::PAGE_READONLY, &mut old) != 0;
                let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x44).collect();
                let through = rig.store(buffer, 0, &new).is_ok() && same(memory, &new);
                // The view goes; the GPU writes; another view shows it.
                unmapped = mem::UnmapViewOfFile2(process, view, mem::MEM_PRESERVE_PLACEHOLDER) != 0;
                let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x55).collect();
                let stored = rig.store(buffer, 0, &new).is_ok();
                let other = mem::MapViewOfFile(handle, mem::FILE_MAP_WRITE, 0, (5 * K64) as u32, K64);
                p.section_outlives_views = unmapped && stored && !other.Value.is_null() && same(other.Value.cast(), &new);
                p.lines.push(format!(
                    "  the GPU reads the view's bytes {}; the CPU reads through the view what the GPU wrote {}; the view made read-only under the heap {guarded}, \
                     and the GPU writes it all the same {through}; the view unmapped under the heap {unmapped}, the GPU's write then shows in a view mapped afterwards {}",
                    p.section_reads, p.section_writes, p.section_outlives_views
                ));
                if !other.Value.is_null() {
                    mem::UnmapViewOfFile(other);
                }
            }
        }
        drop(opened);
        if !unmapped && !view.Value.is_null() {
            mem::UnmapViewOfFile2(process, view, mem::MEM_PRESERVE_PLACEHOLDER);
        }
        windows_sys::Win32::Foundation::CloseHandle(handle);
        // The placeholder was split in two: each half is released.
        mem::VirtualFree(hole.cast(), 0, mem::MEM_RELEASE);
        mem::VirtualFree(hole.add(8 * K64).cast(), 0, mem::MEM_RELEASE);
    }
}

/// What opening costs, on `mb` MB each of private memory never touched
/// and of a committed section never touched: the time, the working set,
/// the card's account of the PC's memory shared with it; and how fast the
/// card copies out of such a heap against a shared buffer of today's kind.
unsafe fn cost(p: &mut Probe, rig: &mut Rig, memory: &dyn Fn() -> Option<[u64; 4]>, mb: usize) {
    let bytes = mb << 20;
    let shared = || memory().map_or(0, |m| m[3] >> 20);
    let ms = |speed: Option<f64>| speed.map_or("?".to_owned(), |ms| format!("{ms:.1}"));
    // SAFETY (the whole function): the allocation and the section below.
    unsafe {
        let base = mem::VirtualAlloc(std::ptr::null(), bytes, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE).cast::<u8>();
        if !base.is_null() {
            let (set, account, started) = (working_set_mb(), shared(), std::time::Instant::now());
            let opened = open(&rig.device, From::Part(base as usize, bytes));
            let took = started.elapsed().as_secs_f64() * 1000.0;
            let placed = opened.as_ref().ok().and_then(|heap| place(&rig.device, heap, 0, bytes as u64).ok());
            let speed = placed.as_ref().and_then(|buffer| rig.copy_ms(buffer, bytes as u64).ok());
            p.lines.push(format!(
                "cost, {mb} MB of private memory never touched: {}; opened in {took:.1} ms; working set {set} -> {} MB; shared with the card {account} -> {} MB; \
                 the card copies it out in {} ms",
                hr(&opened),
                working_set_mb(),
                shared(),
                ms(speed)
            ));
            drop((placed, opened));
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
        let invalid = windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        let handle = mem::CreateFileMappingW(invalid, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_COMMIT, (bytes as u64 >> 32) as u32, bytes as u32, std::ptr::null());
        if !handle.is_null() {
            let (set, account, started) = (working_set_mb(), shared(), std::time::Instant::now());
            let opened = open(&rig.device, From::Section(handle as usize));
            let took = started.elapsed().as_secs_f64() * 1000.0;
            let (set_opened, account_opened) = (working_set_mb(), shared());
            let placed = opened.as_ref().ok().and_then(|heap| place(&rig.device, heap, 0, bytes as u64).ok());
            let speed = placed.as_ref().and_then(|buffer| rig.copy_ms(buffer, bytes as u64).ok());
            p.lines.push(format!(
                "cost, {mb} MB of a committed section never touched: {}; opened in {took:.1} ms; working set {set} -> {set_opened} -> after the card read all of it {} MB; \
                 shared with the card {account} -> {account_opened} -> {} MB; the card copies it out in {} ms",
                hr(&opened),
                working_set_mb(),
                shared(),
                ms(speed)
            ));
            drop((placed, opened));
            windows_sys::Win32::Foundation::CloseHandle(handle);
        }
        // Today's kind of host copy, for the comparison.
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let made = rig.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        );
        if let (Ok(()), Some(buffer)) = (made, buffer) {
            let speed = rig.copy_ms(&buffer, bytes as u64).ok();
            p.lines.push(format!("cost, for comparison: the card copies {mb} MB out of a shared buffer of today's kind in {} ms", ms(speed)));
        }
    }
}

// The census: which of the title memory that has host copies today could
// be opened, and by which opening.

/// Counts and bytes by kind, in the order of [`KINDS`].
#[derive(Default, Clone, Copy)]
struct Tally([(u64, u64); 6]);

const KINDS: [&str; 6] = [
    "private, committed whole (a part can be opened)",
    "view memory, one run of a section (one placed buffer)",
    "view memory, scattered pages",
    "view memory with unmapped pages",
    "private but not committed whole, or guarded",
    "other",
];

fn kind_of(at: usize, len: usize, known: &mut Vec<(usize, usize)>) -> usize {
    match cover(at, len) {
        Cover::Run(..) => return 1,
        Cover::Scattered => return 2,
        Cover::Holes => return 3,
        Cover::None => {}
    }
    let end = at + len;
    if known.iter().any(|&(from, to)| from <= at && end <= to) {
        return 0;
    }
    // SAFETY: plain data filled in by the query.
    let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
    if n == 0 {
        return 5;
    }
    let reach = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    if info.Type == mem::MEM_PRIVATE && info.State == mem::MEM_COMMIT && info.Protect == mem::PAGE_READWRITE {
        known.push((info.BaseAddress as usize, reach));
        return if reach >= end { 0 } else { 4 };
    }
    match info.Type {
        mem::MEM_PRIVATE => 4,
        mem::MEM_MAPPED => 3,
        _ => 5,
    }
}

/// The census lines for `spans` (start, bytes, whether a window outside
/// buffers).
fn census_of(spans: &[(u64, u64, bool)]) -> Vec<String> {
    let mut known = Vec::new();
    let (mut buffers, mut windows) = (Tally::default(), Tally::default());
    for &(start, len, window) in spans {
        let kind = kind_of(start as usize, len as usize, &mut known);
        let tally = if window { &mut windows } else { &mut buffers };
        tally.0[kind].0 += 1;
        tally.0[kind].1 += len;
    }
    let mut lines = Vec::new();
    for (name, tally) in [("title buffers", buffers), ("windows outside buffers", windows)] {
        let all = tally.0.iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        lines.push(format!("census, {name}: {} of {} MB", all.0, all.1 >> 20));
        for (kind, &(n, bytes)) in KINDS.iter().zip(&tally.0) {
            if n > 0 {
                lines.push(format!("census,   {kind}: {n} of {} MB", bytes >> 20));
            }
        }
    }
    let v = views();
    let (count, bytes) = (v.views.len(), v.views.values().map(|view| view.0 as u64).sum::<u64>());
    let pages = v.views.values().filter(|view| view.0 == K64).count();
    let big = v.views.values().map(|view| view.0).max().unwrap_or(0);
    lines.push(format!(
        "census, the title's views of physical pages: {count} of {} MB; {pages} of them one 64 KiB page; the longest {} MB",
        bytes >> 20,
        big >> 20
    ));
    for &(_, bytes, committed) in &v.sections {
        lines.push(format!(
            "census, a section of {} MB, {}",
            bytes >> 20,
            if committed { "committed: it can be opened as one heap" } else { "reserved: refused as a heap until it is made with SEC_COMMIT" }
        ));
    }
    lines
}

/// With [`probe_level`] 2: one part of the title's own private memory
/// (the start of the biggest such title buffer, 4 MiB at most) is opened
/// for a moment, a buffer placed, 64 KiB read by the GPU and compared
/// with what the CPU reads there (the title may be writing it meanwhile:
/// the count of equal bytes is given), and all of it released. While the
/// heap stands (milliseconds) a `VirtualFree` of the title's on those
/// pages would fail.
fn real_part(spans: &[(u64, u64, bool)]) -> Vec<String> {
    let Some(host) = crate::host::host() else {
        return Vec::new();
    };
    let mut known = Vec::new();
    let found = spans
        .iter()
        .filter(|s| !s.2 && s.1 >= 2 * K64 as u64)
        .filter(|s| kind_of(s.0 as usize, s.1 as usize, &mut known) == 0)
        .max_by_key(|s| s.1);
    let Some(&(start, len, _)) = found else {
        return vec!["the title's memory: no title buffer in private, committed memory to try".to_owned()];
    };
    let at = (start as usize).next_multiple_of(K64);
    let bytes = ((start + len) as usize).saturating_sub(at).min(4 << 20) & !(K64 - 1);
    if bytes == 0 || !private_whole(at, bytes) {
        return vec!["the title's memory: the buffer found holds no whole 64 KiB page that is still committed".to_owned()];
    }
    let mut lines = Vec::new();
    let (set, started) = (working_set_mb(), std::time::Instant::now());
    let opened = open(&host.device, From::Part(at, bytes));
    lines.push(format!(
        "the title's memory, {} KiB at {at:#x} (of the title buffer {start:#x}+{len:#x}): {}; in {:.1} ms; working set {set} -> {} MB",
        bytes >> 10,
        hr(&opened),
        started.elapsed().as_secs_f64() * 1000.0,
        working_set_mb()
    ));
    if let Ok(heap) = &opened {
        let placed = place(&host.device, heap, 0, bytes as u64);
        lines.push(format!("  a buffer placed over it: {}", hr(&placed)));
        if let (Ok(buffer), Ok(mut rig)) = (&placed, Rig::new(&host.device)) {
            let mut before = vec![0u8; K64];
            crate::host::copy_mapped(before.as_mut_ptr(), at as *const u8, K64, false);
            let got = rig.fetch(buffer, 0, K64);
            let mut after = vec![0u8; K64];
            crate::host::copy_mapped(after.as_mut_ptr(), at as *const u8, K64, false);
            match got {
                Ok(got) => {
                    let equal = (0..K64).filter(|&i| got[i] == before[i] || got[i] == after[i]).count();
                    let zeros = got.iter().filter(|&&b| b == 0).count();
                    lines.push(format!(
                        "  the GPU read 64 KiB of the title's own bytes: {equal} of {K64} equal to what the CPU read just before or just after ({zeros} of them zero)"
                    ));
                }
                Err(e) => lines.push(format!("  the GPU's read failed: {e}")),
            }
        }
    }
    lines
}

/// Called once a second by the thread that submits, before a submission
/// (`host::sweep`): with [`probe_level`], the probe at the first call, the
/// census (on a thread of its own: it asks Windows about every copy) 30
/// and 150 seconds later.
pub(crate) fn tick() {
    if level() > 0 {
        say();
    }
    if probe_level() == 0 {
        return;
    }
    static STARTED: AtomicU64 = AtomicU64::new(0);
    static DONE: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms().max(1);
    let started = match STARTED.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => {
            let Some(host) = crate::host::host() else {
                return;
            };
            let mb = std::env::var("VICTORY_INPLACE_PROBE_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64usize);
            let took = std::time::Instant::now();
            let probe = probe(&host.device, &|| host.video_memory(), mb.min(1024));
            for line in &probe.lines {
                tracing::info!("in place probe: {line}");
            }
            tracing::info!("in place probe: took {:.0} ms", took.elapsed().as_secs_f64() * 1000.0);
            now
        }
        Err(started) => started,
    };
    let due = match DONE.load(Ordering::Relaxed) {
        0 => now >= started + 30_000,
        1 => now >= started + 150_000,
        _ => false,
    };
    if due {
        let which = DONE.fetch_add(1, Ordering::Relaxed);
        std::thread::spawn(move || {
            let spans = crate::shadow::spans();
            for line in census_of(&spans) {
                tracing::info!("in place {line}");
            }
            if which == 0 && probe_level() >= 2 {
                for line in real_part(&spans) {
                    tracing::info!("in place probe: {line}");
                }
            }
        });
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    thread_local! {
        /// LOCAL: [`views_on`] for this thread's test, whatever the switches.
        pub(crate) static VIEWS_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// LOCAL: [`views_on`] on this thread while held.
    pub(crate) struct ViewsOn;

    impl Drop for ViewsOn {
        fn drop(&mut self) {
            VIEWS_ON.set(false);
        }
    }

    pub(crate) fn views_on_here() -> ViewsOn {
        VIEWS_ON.set(true);
        ViewsOn
    }

    /// LOCAL: a section made as `xmem` makes the page pool
    /// (`VICTORY_POOL_COMMIT=1`: committed), with views at two addresses:
    /// `pages` pages from the start, and its page 2 again elsewhere (an
    /// alias). The section, the two addresses. Told to the runtime's record
    /// as `xmem` tells it.
    pub(crate) fn committed_section(pages: usize) -> (usize, usize, usize) {
        // SAFETY: a section and views of the test's own.
        unsafe {
            let invalid = windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
            let size = 8 * K64;
            let section = mem::CreateFileMappingW(invalid, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_COMMIT, 0, size as u32, std::ptr::null());
            assert!(!section.is_null());
            let a = mem::MapViewOfFile(section, mem::FILE_MAP_WRITE, 0, 0, pages * K64).Value as usize;
            let b = mem::MapViewOfFile(section, mem::FILE_MAP_WRITE, 0, (2 * K64) as u32, K64).Value as usize;
            assert!(a != 0 && b != 0);
            title_memory_section(section as usize, size as u64, true);
            title_memory_view(a, pages * K64, section as usize, 0);
            title_memory_view(b, K64, section as usize, (2 * K64) as u64);
            (section as usize, a, b)
        }
    }

    /// LOCAL: what [`committed_section`] made goes, as `xmem` lets it go.
    pub(crate) fn committed_section_gone(section: usize, a: usize, b: usize, pages: usize) {
        views_gone(a, pages * K64);
        views_gone(b, K64);
        title_memory_section_gone(section);
        // SAFETY: the test's own views and section.
        unsafe {
            mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: a as *mut c_void });
            mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: b as *mut c_void });
            windows_sys::Win32::Foundation::CloseHandle(section as _);
        }
    }

    /// LOCAL: the default adapter if it is a card (as `indirect`'s tests
    /// take it), for the calls the card's driver answers.
    fn card() -> Option<ID3D12Device> {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::*;
        // SAFETY: plain DXGI queries and device creation.
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
            let device = device?;
            let desc = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter1>(device.GetAdapterLuid()))
                .ok()?
                .GetDesc1()
                .ok()?;
            (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0).then_some(device)
        }
    }

    /// LOCAL: a window over one run of a committed section, off the 64 KiB
    /// lines, is placed over the run's offset in the section's heap: the GPU
    /// reads what the CPU wrote through another address of the same page,
    /// and its write is seen at both. What is not one run of a committed
    /// section is turned down without asking the host, and so is the run
    /// once its view is gone. On the software adapter and on the card.
    #[test]
    fn a_window_over_a_committed_section_shows_its_pages_warp_and_card() {
        let _turn = crate::indirect::tests::warp_turn();
        let devices: Vec<(&str, ID3D12Device)> = [("warp", warp()), ("card", card())].into_iter().filter_map(|(n, d)| Some((n, d?))).collect();
        for (name, device) in devices {
            eprintln!("a window over a committed section, on the {name}");
            over_a_committed_section(&device);
        }
    }

    fn over_a_committed_section(device: &ID3D12Device) {
        let device = device.clone();
        let (section, a, b) = committed_section(4);
        let (start, len) = ((a + K64 + 0x40) as u64, 2 * K64 as u64);
        let placed = match place_view(&device, start, len) {
            Ok(placed) => placed,
            Err((Refusal::Host, Some(e))) => {
                // A driver that refuses a section as a heap: nothing to try.
                eprintln!("the section refused as a heap: {e}");
                committed_section_gone(section, a, b, 4);
                return;
            }
            Err(e) => panic!("not placed: {e:?}"),
        };
        assert!(placed.view);
        assert_eq!(placed.bias, 0x40);
        // SAFETY: a live buffer.
        unsafe {
            assert_eq!(placed.gpu, placed.buffer.GetGPUVirtualAddress() + 0x40);
            assert_eq!(placed.buffer.GetDesc().Width, 3 * K64 as u64, "pages 1 to 3 of the section");
        }
        let mut rig = Rig::new(&device).unwrap();
        // Page 2 is at `a + 2 pages`, at `b`, and at the buffer's second page.
        fill(b as *mut u8, K64, 9);
        let got = rig.fetch(&placed.buffer, K64 as u64, K64).unwrap();
        assert!(same(b as *const u8, &got), "the GPU reads what the CPU wrote through the alias");
        assert!(same((a + 2 * K64) as *const u8, &got));
        rig.store(&placed.buffer, placed.bias, &[0x5a; 64]).unwrap();
        assert!(same(start as *const u8, &[0x5a; 64]), "the GPU's write is at the window's address");
        rig.store(&placed.buffer, K64 as u64 + 8, &[0xc3; 16]).unwrap();
        assert!(same((b + 8) as *const u8, &[0xc3; 16]), "and at the alias");
        // The heap is opened once per section.
        let again = place_view(&device, start, K64 as u64).unwrap_or_else(|e| panic!("not placed: {e:?}"));
        assert_eq!(again.heap.as_raw(), placed.heap.as_raw());
        drop(again);
        // (Records only, at addresses nothing has.) Over a view's end into
        // nothing: holes; over two views of pages that do not follow each
        // other in the section: scattered.
        let holes = 0x7_9200_0000usize;
        title_memory_view(holes, K64, section, 0);
        assert!(matches!(place_view(&device, holes as u64, 2 * K64 as u64), Err((Refusal::Holes, None))));
        views_gone(holes, K64);
        let scattered = 0x7_9000_0000usize;
        title_memory_view(scattered, K64, section, 0);
        title_memory_view(scattered + K64, K64, section, (5 * K64) as u64);
        assert!(matches!(place_view(&device, scattered as u64, 2 * K64 as u64), Err((Refusal::Scattered, None))));
        assert!(place_view(&device, (scattered + K64) as u64, K64 as u64).is_ok(), "a page of its own is one run");
        views_gone(scattered, 2 * K64);
        // A reserved section is not opened.
        let reserved = 0x7_9100_0000usize;
        title_memory_section(0x5151, 8 * K64 as u64, false);
        title_memory_view(reserved, K64, 0x5151, 0);
        assert!(matches!(place_view(&device, reserved as u64, K64 as u64), Err((Refusal::Reserved, None))));
        views_gone(reserved, K64);
        title_memory_section_gone(0x5151);
        // Its view gone (the runtime unmaps it): turned down.
        views_gone(a, 4 * K64);
        assert!(place_view(&device, start, len).is_err());
        drop(placed);
        committed_section_gone(section, a, b, 4);
        assert!(heaps().0.iter().all(|h| h.0 != section), "the section's heap is forgotten with it");
    }

    /// LOCAL: slots made on a window over a view are told to be made anew
    /// once it is let go, once, for their own heap.
    #[test]
    fn slots_on_a_window_let_go_are_made_anew() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = warp() else { return };
        let _on = views_on_here();
        let (section, a, b) = committed_section(4);
        let placed = place_view_counted(&device, a as u64, K64 as u64).expect("placed");
        slot_shows(0x1000, 7, &placed.buffer);
        slot_shows(0x1000, 7, &placed.buffer);
        slot_shows(0x2000, 9, &placed.buffer);
        assert!(slots_stale(0x1000).is_empty(), "nothing let go yet");
        let_go_view(&placed.buffer);
        // A slot made on it by a thread that found the window just before.
        slot_shows(0x1000, 8, &placed.buffer);
        {
            // Not while the runtime changes a mapping: the views it maps
            // are not told yet.
            let _changing = crate::title_memory_remapping();
            assert!(slots_stale(0x1000).is_empty());
        }
        assert_eq!(slots_stale(0x1000), vec![7, 8]);
        assert!(slots_stale(0x1000).is_empty());
        // A heap that goes has nothing made anew.
        heap_gone(0x2000);
        assert!(slots_stale(0x2000).is_empty());
        gone_view(K64 as u64, placed.buffer.as_raw() as usize);
        drop(placed);
        // A window that goes without a let go, its buffer held by a slot:
        // the slot is made anew too.
        let placed = place_view_counted(&device, a as u64, K64 as u64).expect("placed");
        slot_shows(0x1000, 3, &placed.buffer);
        gone_view(K64 as u64, placed.buffer.as_raw() as usize);
        assert_eq!(slots_stale(0x1000), vec![3]);
        drop(placed);
        committed_section_gone(section, a, b, 4);
    }

    /// Views as the runtime tells them: a range inside one view, or over
    /// views that continue each other in one section, is one run; over
    /// views from elsewhere it is scattered; over a gap it has holes.
    #[test]
    fn the_views_say_how_much_of_a_range_is_there() {
        const K64: usize = 64 << 10;
        // Addresses nothing has: only the record is asked.
        let at = 0x7ff1_0000_0000_usize;
        for (block, blocks) in [(0, 1), (1, 1), (4, 2)] {
            super::title_memory_view(at + block * K64, blocks * K64, 7, 0);
        }
        let shown = |from: usize, len: usize| super::shown_bytes(at + from, len);
        assert_eq!(shown(0, 8 * K64), Some(4 * K64 as u64));
        assert_eq!(shown(100, K64), Some(K64 as u64));
        assert_eq!(shown(K64 + 100, 2 * K64), Some(K64 as u64 - 100));
        assert_eq!(shown(5 * K64 + 1, 10), Some(10));
        assert_eq!(shown(2 * K64, 2 * K64), None);
        assert_eq!(shown(6 * K64, 2 * K64), None);
        // The same as runs: views that touch are one, cut to the range.
        let runs = |from: usize, len: usize| super::shown_runs(at + from, len);
        assert_eq!(runs(0, 8 * K64), vec![(at, 2 * K64), (at + 4 * K64, 2 * K64)]);
        assert_eq!(runs(100, K64), vec![(at + 100, K64)]);
        assert_eq!(runs(5 * K64 + 1, 10), vec![(at + 5 * K64 + 1, 10)]);
        assert_eq!(runs(2 * K64, 2 * K64), vec![]);
        super::views_gone(at + K64, K64);
        assert_eq!(shown(0, 2 * K64), Some(K64 as u64));
        assert_eq!(runs(0, 2 * K64), vec![(at, K64)]);
        super::views_gone(at, 8 * K64);
        assert_eq!(shown(0, 8 * K64), None);
    }

    #[test]
    fn views_say_what_one_placed_buffer_could_show() {
        let mut views = BTreeMap::new();
        let at = 0x7_0000_0000usize;
        views.insert(at, (4 * K64, 9, 16 * K64 as u64));
        // Continues the first in the section.
        views.insert(at + 4 * K64, (2 * K64, 9, 20 * K64 as u64));
        // Follows in address, from elsewhere in the section.
        views.insert(at + 6 * K64, (K64, 9, 3 * K64 as u64));
        // After a gap of one page.
        views.insert(at + 8 * K64, (K64, 9, 4 * K64 as u64));
        assert_eq!(cover_in(&views, at + K64, 2 * K64), Cover::Run(9, 17 * K64 as u64));
        assert_eq!(cover_in(&views, at + K64, 5 * K64), Cover::Run(9, 17 * K64 as u64));
        assert_eq!(cover_in(&views, at, 7 * K64), Cover::Scattered);
        assert_eq!(cover_in(&views, at + 6 * K64, 3 * K64), Cover::Holes);
        assert_eq!(cover_in(&views, at + 7 * K64, K64), Cover::None);
        assert_eq!(cover_in(&views, at - K64, 2 * K64), Cover::None);
    }

    /// What the runtime forgets when memory is mapped anew.
    #[test]
    fn remapped_memory_forgets_its_views() {
        let at = 0x7_4000_0000usize;
        title_memory_view(at, 4 * K64, 77, 0);
        title_memory_view(at + 4 * K64, K64, 77, 4 * K64 as u64);
        assert_eq!(cover(at, 5 * K64), Cover::Run(77, 0));
        // One page inside the first view goes: the whole view is forgotten
        // (the runtime maps and tells again what stays of it).
        views_gone(at + K64, K64);
        assert_eq!(cover(at, K64), Cover::None);
        assert_eq!(cover(at + 4 * K64, K64), Cover::Run(77, 4 * K64 as u64));
        views_gone(at + 4 * K64, K64);
        assert_eq!(cover(at + 4 * K64, K64), Cover::None);
    }

    /// Only private memory committed whole, read-write, is taken for a part.
    #[test]
    fn only_private_memory_committed_whole_is_a_part() {
        // SAFETY: an allocation of the test's own.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 8 * K64, mem::MEM_RESERVE, mem::PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert!(!private_whole(base as usize, K64), "reserved only");
            mem::VirtualAlloc(base.cast(), 4 * K64, mem::MEM_COMMIT, mem::PAGE_READWRITE);
            assert!(private_whole(base as usize, 4 * K64));
            assert!(private_whole(base as usize + K64, 2 * K64));
            assert!(!private_whole(base as usize, 5 * K64), "reaches into the reserved half");
            let mut old = 0;
            mem::VirtualProtect(base.cast(), K64, mem::PAGE_READONLY, &mut old);
            assert!(!private_whole(base as usize, 2 * K64), "a guarded page");
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
    }

    pub(crate) fn warp() -> Option<ID3D12Device> {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::*;
        // SAFETY: plain DXGI queries and device creation on the software
        // adapter; a card is never opened.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
            let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                return None;
            }
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
            device
        }
    }

    /// The probe, whole, on the software adapter: the same calls the card
    /// gets. The runtime's and the kernel's rules hold there; the card's
    /// driver is what a run with `VICTORY_INPLACE_PROBE=1` adds.
    #[test]
    fn warp_probe_opens_title_like_memory_and_the_gpu_sees_its_bytes() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = warp() else { return };
        if device.cast::<ID3D12Device13>().is_err() {
            eprintln!("no ID3D12Device13 here: parts of an allocation cannot be opened");
            return;
        }
        let probe = probe(&device, &|| None, 4);
        for line in &probe.lines {
            eprintln!("in place probe: {line}");
        }
        assert!(probe.part_reads, "the GPU reads private memory in place");
        assert!(probe.part_writes, "GPU writes land in private memory");
        assert!(probe.section_reads && probe.section_writes, "a committed section in place");
        assert!(probe.section_outlives_views, "the section's heap does not depend on views");
        assert!(probe.reserved_section_refused, "a reserved section is refused");
    }

    /// Stage 1's own call on the software adapter: a title buffer that
    /// begins and ends off the 64 KiB lines is placed over its pages, the
    /// GPU reads the buffer's bytes at the offset the views use, and its
    /// write lands at the title's address. Memory that is not committed
    /// whole is turned down without asking the host.
    #[test]
    fn warp_a_title_buffer_in_place_shows_its_bytes_at_its_offset() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = warp() else { return };
        if device.cast::<ID3D12Device13>().is_err() {
            return;
        }
        // SAFETY: an allocation of the test's own.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 8 * K64, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE).cast::<u8>();
            let (start, len) = (base as u64 + K64 as u64 + 0x1230, 2 * K64 as u64 + 5);
            let placed = place_part(&device, start, len).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(placed.bias, 0x1230);
            assert_eq!(placed.gpu, placed.buffer.GetGPUVirtualAddress() + 0x1230);
            assert_eq!(placed.buffer.GetDesc().Width, 3 * K64 as u64, "the pages that hold the buffer");
            let mut rig = Rig::new(&device).unwrap();
            fill(start as *mut u8, len as usize, 7);
            let got = rig.fetch(&placed.buffer, placed.bias, len as usize).unwrap();
            assert!(same(start as *const u8, &got), "the GPU reads the title buffer's bytes");
            rig.store(&placed.buffer, placed.bias + 16, &[0xa5; 64]).unwrap();
            assert!(same((start + 16) as *const u8, &[0xa5; 64]), "the GPU's write is at the title's address");
            drop(placed);
            // Half of it decommitted: turned down, the host not asked.
            // The heap lets go of the pages a moment after its last
            // reference, if the adapter is busy with other tests' work.
            let gone = (0..500).any(|_| {
                let done = mem::VirtualFree(base.add(2 * K64).cast(), K64, mem::MEM_DECOMMIT) != 0;
                if !done {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                done
            });
            assert!(gone, "the heap keeps the pages");
            assert!(matches!(place_part(&device, start, len), Err(None)));
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
    }

    /// The census names each kind of memory.
    #[test]
    fn census_tells_private_memory_from_views() {
        // SAFETY: an allocation of the test's own.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 8 * K64, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE) as u64;
            let view = 0x7_8000_0000u64;
            title_memory_view(view as usize, 2 * K64, 5, 0);
            title_memory_view(view as usize + 2 * K64, K64, 5, 9 * K64 as u64);
            let lines = census_of(&[(base, 2 * K64 as u64, false), (base + 4 * K64 as u64, K64 as u64, true), (view, K64 as u64, true), (view, 3 * K64 as u64, false)]);
            let text = lines.join("\n");
            assert!(text.contains("census, title buffers: 2 of"), "{text}");
            assert!(text.contains("census,   private, committed whole (a part can be opened): 1 of"), "{text}");
            assert!(text.contains("census,   view memory, scattered pages: 1 of"), "{text}");
            assert!(text.contains("census,   view memory, one run of a section (one placed buffer): 1 of"), "{text}");
            views_gone(view as usize, 3 * K64);
            mem::VirtualFree(base as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }
}
