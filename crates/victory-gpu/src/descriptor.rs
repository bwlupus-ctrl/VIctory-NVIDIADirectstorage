//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, DESCRIPTOR_HEAP, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid};
use crate::resource::{Memory, PAGEABLE_IIDS};

#[cfg(windows)]
pub(crate) mod host;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    kind: u32,
    count: u32,
    flags: u32,
    nodes: u32,
}

/// Newer titles (the title) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    memory: Option<Memory>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (decision 0007, step
    /// 3); slots written before wait in `pending`.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<host::HostHeap>>,
    #[cfg(windows)]
    pending: std::sync::Mutex<Vec<u32>>,
    /// Changed pages of a placed shader-visible heap. Some titles copy
    /// descriptor records into it as bytes, without a descriptor API call.
    #[cfg(windows)]
    written: Option<crate::watch::Tracked>,
}

pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of the title's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    #[cfg(windows)]
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS)
        {
            return None;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Publish only after replaying the queued writes.
        self.host
            .get_or_init(|| {
                let step = u64::from(increment(self.desc.kind)?);
                let h = host::HostHeap::new(self.samplers(), self.desc.count)?;
                for i in pending.drain(..) {
                    h.mirror(i, self.start + u64::from(i) * step, false);
                }
                Some(h)
            })
            .as_ref()
    }

    /// The handle of descriptor `index` of this heap.
    #[cfg(windows)]
    pub(crate) fn handle(&self, index: u32) -> Option<u64> {
        let step = u64::from(increment(self.desc.kind)?);
        (index < self.desc.count).then(|| self.start + u64::from(index) * step)
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap.
#[cfg(windows)]
///
/// Held shared while a heap found here is used (by every thread that writes
/// descriptors, and by [`sync_heaps`]), alone only where heaps come and go.
static SHADER_HEAPS: std::sync::RwLock<Vec<(u64, u64, usize)>> = std::sync::RwLock::new(Vec::new());

/// Makes the host descriptor for what the title wrote at `handle`, if it
/// is in a shader-visible heap. Keeps `SHADER_HEAPS` locked for the duration
/// so `Heap::drop` cannot free the heap concurrently.
#[cfg(windows)]
fn mirror(handle: u64) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    let Some(&(_, _, at)) = heaps
        .iter()
        .find(|&&(start, end, _)| (start..end).contains(&handle))
    else {
        return;
    };
    // SAFETY: registered heaps stay alive until `Heap::drop`, which takes
    // the `SHADER_HEAPS` lock we hold here.
    let heap = unsafe { &*(at as *const Heap) };
    let Some(index) = heap.index(handle) else {
        return;
    };
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                host.mirror(index, handle, false);
            }
        }
    }
}

/// Whether [`sync_heaps`] is done (`VICTORY_HEAP_SYNC=0`: not).
#[cfg(windows)]
fn sync_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC").is_none_or(|v| v != "0"))
}

/// Whether a submission's [`sync_heaps`] is done by a thread of its own
/// beside the rest of the preparing (`VICTORY_HEAP_SYNC_ASIDE=0`: in front of
/// it, as before).
///
/// It took 13 ms a frame of the submitting thread (session play-1125-0203:
/// 230 changed heap pages and 470,000 slots in 30 submissions). What it makes
/// are the host heap's descriptors, which the card reads: it has to be done
/// before the lists go to the queue, and nothing of the preparing looks at
/// them (what a draw reads through a table it made itself, when recorded).
#[cfg(windows)]
fn sync_aside() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| sync_on() && std::env::var_os("VICTORY_HEAP_SYNC_ASIDE").is_none_or(|v| v != "0"))
}

/// How long that thread waits for a submission before it makes a round of its
/// own (`VICTORY_HEAP_SYNC_EARLY=<ms>`, 4 if not given; `0`: none). The title
/// copies its records while it records, and the first submission of a frame
/// found all of them: 8 ms waited for (run cpu-2). A round made meanwhile
/// leaves it those written since.
#[cfg(windows)]
fn sync_early() -> Option<std::time::Duration> {
    static EVERY: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let ms = std::env::var("VICTORY_HEAP_SYNC_EARLY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4u64);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    })
}

/// The thread's work: rounds asked for, rounds done.
#[cfg(windows)]
static SYNC_ROUNDS: (std::sync::Mutex<(u64, u64)>, std::sync::Condvar) = (std::sync::Mutex::new((0, 0)), std::sync::Condvar::new());

/// [`sync_heaps`], begun; done when it is dropped (with [`sync_aside`]), or
/// when this returns.
#[cfg(windows)]
pub(crate) struct SyncedHeaps(Option<u64>);

#[cfg(windows)]
pub(crate) fn begin_sync_heaps() -> SyncedHeaps {
    if !sync_aside() {
        sync_heaps();
        return SyncedHeaps(None);
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    static THREAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    STARTED.call_once(|| {
        let made = std::thread::Builder::new().name("victory-heap-sync".into()).spawn(|| {
            crate::cores::keep_on_fast_cores("victory-heap-sync");
            let rounds = || SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
            /// A round is over, however it ends.
            struct Over;
            impl Drop for Over {
                fn drop(&mut self) {
                    SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner()).1 += 1;
                    SYNC_ROUNDS.1.notify_all();
                }
            }
            // Between the rounds asked for, while submissions come.
            let mut last = std::time::Instant::now();
            loop {
                let mut r = rounds();
                let mut asked = r.0 != r.1;
                while !asked {
                    let Some(every) = sync_early().filter(|_| last.elapsed().as_millis() < 500) else {
                        r = SYNC_ROUNDS.1.wait(r).unwrap_or_else(|e| e.into_inner());
                        asked = r.0 != r.1;
                        continue;
                    };
                    let (back, over) = SYNC_ROUNDS.1.wait_timeout(r, every).unwrap_or_else(|e| e.into_inner());
                    r = back;
                    asked = r.0 != r.1;
                    if over.timed_out() {
                        break;
                    }
                }
                drop(r);
                let _over = asked.then(|| {
                    last = std::time::Instant::now();
                    Over
                });
                let _round =
                    crate::timeline::span(if asked { "the heaps' records, for a submission" } else { "the heaps' records, early" });
                let _ = std::panic::catch_unwind(sync_heaps);
            }
        });
        THREAD.store(made.is_ok(), std::sync::atomic::Ordering::Release);
    });
    if !THREAD.load(std::sync::atomic::Ordering::Acquire) {
        sync_heaps();
        return SyncedHeaps(None);
    }
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    rounds.0 += 1;
    let mine = rounds.0;
    drop(rounds);
    SYNC_ROUNDS.1.notify_all();
    SyncedHeaps(Some(mine))
}

/// [`begin_sync_heaps`] for a submission that needs round `mark` (one that
/// began after what it reads was written, `crate::heap_marks`): asked for if
/// it has not been. 0: as [`begin_sync_heaps`]. Ported from
/// Belobug/VIctory-Belobug-AMD 673be0e and a609a2f.
#[cfg(windows)]
pub(crate) fn begin_sync_heaps_after(mark: u64) -> SyncedHeaps {
    if mark == 0 || !sync_aside() {
        return begin_sync_heaps();
    }
    // (The thread is started by the first `begin_sync_heaps`.)
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    if rounds.1 == 0 && rounds.0 == 0 {
        drop(rounds);
        return begin_sync_heaps();
    }
    let asked = rounds.0 < mark;
    rounds.0 = rounds.0.max(mark);
    drop(rounds);
    use crate::heap_marks::{EARLY_ROUNDS, LATE_ROUNDS, STATS};
    if asked {
        SYNC_ROUNDS.1.notify_all();
        STATS[LATE_ROUNDS].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        STATS[EARLY_ROUNDS].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    SyncedHeaps(Some(mark))
}

/// The number of a round of the heaps' records that begins after this call,
/// asked for if none is waiting to begin (at the title's submit: the round
/// runs while the submission waits in the line). 0 if the records are not
/// synced by a thread of their own (or it has not started yet).
#[cfg(windows)]
pub(crate) fn round_after_now() -> u64 {
    if !sync_aside() {
        return 0;
    }
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    if rounds.1 == 0 && rounds.0 == 0 {
        return 0;
    }
    let (round, ask) = crate::heap_marks::round_after(rounds.0, rounds.1);
    if ask {
        rounds.0 = round;
        drop(rounds);
        SYNC_ROUNDS.1.notify_all();
        crate::heap_marks::STATS[crate::heap_marks::ROUNDS_ASKED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    round
}

/// The number of the first round that begins after this call, not asked for
/// ([`begin_sync_heaps_after`] asks for it if nobody did).
#[cfg(windows)]
pub(crate) fn round_mark() -> u64 {
    if !sync_aside() {
        return 0;
    }
    let rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    if rounds.1 == 0 && rounds.0 == 0 {
        return 0;
    }
    rounds.0 + 1
}

/// Whether `[at, at + len)` touches the title memory of a placed
/// shader-visible heap of descriptors.
#[cfg(windows)]
pub(crate) fn in_shader_heap(at: usize, len: usize) -> bool {
    let (from, to) = (at as u64, at as u64 + len as u64);
    SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner()).iter().any(|&(start, end, _)| from < end && start < to)
}

#[cfg(windows)]
impl Drop for SyncedHeaps {
    fn drop(&mut self) {
        let Some(mine) = self.0 else {
            return;
        };
        let began = std::time::Instant::now();
        let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
        while rounds.1 < mine {
            rounds = SYNC_ROUNDS.1.wait(rounds).unwrap_or_else(|e| e.into_inner());
        }
        drop(rounds);
        SYNCED[6].fetch_add(began.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Mirrors records copied directly into placed shader-visible heaps. Calls
/// to `Create*View` are already mirrored by [`mirror`]; raw CPU copies leave
/// only write-watch pages to reveal which slots changed.
///
/// Upstream VIctory 86a5ed6 does the same and counts 1,300 to 14,400 such
/// slots a frame in the world of the title this fork works on, against a
/// few hundred views made by calls; without it those host slots stayed
/// null, or went on showing what they showed before.
/// `VICTORY_HEAP_SYNC=0`: not done, as before. [`SYNCED`] counts.
#[cfg(windows)]
pub(crate) fn sync_heaps() {
    use std::sync::atomic::Ordering::Relaxed;
    if !sync_on() {
        return;
    }
    let started = std::time::Instant::now();
    let (mut pages, mut looked, mut mirrored, mut tabled, mut later) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut views, mut bytes) = (0u64, 0u64);
    let mut spent = [0u64; 3];
    let since = |at: std::time::Instant| at.elapsed().as_nanos() as u64;
    let mut kinds = [0u64; 5];
    let mut short = 0u64;
    let mut wait: Vec<u32> = Vec::new();
    let frame = crate::frame::current();
    let (pages_a_job, records_a_job) = sync_jobs();
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    for &(start, end, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        let (Some(Some(host)), Some(written), Some(step)) =
            (heap.host.get(), &heap.written, increment(heap.desc.kind))
        else {
            continue;
        };
        let step = u64::from(step);
        // Capped host heaps have no counterparts for the title slots reserved
        // for per-draw tables; raw writes there must not be mirrored.
        let mirrored_count = heap.desc.count.min(host.count());
        // The records of some frames ago that no draw read through a table
        // since: a shader may pick them by index.
        let before = host.copied_before(frame);
        let of_tables = std::sync::atomic::AtomicU64::new(0);
        crate::land::each_view(true, before.len().div_ceil(records_a_job), &|n| {
            for &index in before.chunks(records_a_job).nth(n).unwrap_or_default() {
                if host.copied_tabled(index, frame) {
                    of_tables.fetch_add(1, Relaxed);
                } else {
                    host.mirror(index, start + u64::from(index) * step, false);
                }
            }
        });
        let of_tables = of_tables.into_inner();
        tabled += of_tables;
        mirrored += before.len() as u64 - of_tables;
        kinds[4] += before.len() as u64 - of_tables;
        // Those of the changed pages, made together at the end.
        let mut make: Vec<u32> = Vec::new();
        let began = std::time::Instant::now();
        let taken = written.take(start as usize, end as usize);
        spent[0] += since(began);
        let mut parts: Vec<(usize, usize)> = Vec::new();
        for (range_from, range_end) in taken {
            let mut page = range_from & !(crate::watch::PAGE - 1);
            while page < range_end {
                parts.push((page.max(range_from), (page + crate::watch::PAGE).min(range_end)));
                page += crate::watch::PAGE;
            }
        }
        // A few pages a thread at a time: read, compared with what was last
        // seen there and sorted out, beside one another (14 ms a frame of one
        // thread at 160 pages, run faults-6).
        let began = std::time::Instant::now();
        let copied = host.copied_begin(step as usize);
        let samplers = heap.samplers();
        let in_place = sync_in_place() && step == 32 && crate::safecopy::on();
        let found: Vec<std::sync::Mutex<SyncPart>> = parts.chunks(pages_a_job).map(|_| Default::default()).collect();
        crate::land::each_view(true, found.len(), &|n| {
            let mut out = SyncPart::default();
            let mut page_bytes: Vec<u8> = Vec::new();
            let mut changed: Vec<u32> = Vec::new();
            let mut bits: Vec<u64> = Vec::new();
            for &(from, to) in parts.chunks(pages_a_job).nth(n).unwrap_or_default() {
                let slots = changed_slots(from, to, start as usize, step, mirrored_count);
                // A record that straddles the range's ends is left
                // to the range that holds its first byte.
                let first = ((from - start as usize) as u64).div_ceil(step).max(slots.start);
                let lead = (start + first * step) as usize;
                let records: &[u8] = if in_place {
                    // Read where they are, each record once, and compared
                    // as it is read (`host::Copied::changes_in_place`).
                    let count = (to.saturating_sub(lead) as u64 / step).min(slots.end.saturating_sub(first));
                    changed.clear();
                    // SAFETY: as below; title memory is read by a copy that
                    // fails where a page is not there.
                    let whole =
                        unsafe { copied.changes_in_place(first as u32, lead as *const u8, count as usize, &mut bits, &mut changed) };
                    out.pages += 1;
                    out.views += u64::from(crate::watch::guarded(from, to));
                    out.bytes += (to - from) as u64;
                    out.looked += count;
                    out.short += u64::from(!whole);
                    // What was read of the changed records is in the records
                    // last seen now.
                    // SAFETY: this part's records alone, which nothing else
                    // writes meanwhile.
                    unsafe { copied.seen(first as u32, count as usize) }
                } else {
                    // The page's records are read through the kernel, into a
                    // copy: a page the title unmaps meanwhile fails the read
                    // where 2,048 reads in place would fault.
                    page_bytes.resize(to - from, 0);
                    if !crate::host::read_into(from as *const u8, &mut page_bytes) {
                        continue;
                    }
                    out.pages += 1;
                    out.views += u64::from(crate::watch::guarded(from, to));
                    out.bytes += (to - from) as u64;
                    let Some(records) = page_bytes.get(lead.wrapping_sub(from)..) else {
                        continue;
                    };
                    let count = (records.len() as u64 / step).min(slots.end.saturating_sub(first));
                    out.looked += count;
                    // Only the records that are not the ones last seen
                    // there (`host::Copied::changes`).
                    changed.clear();
                    // SAFETY: the parts are ranges of the heap apart from one
                    // another, and only records that lie whole in one are its.
                    unsafe { copied.changes(first as u32, &records[..(count * step) as usize], &mut changed) };
                    records
                };
                for &index in &changed {
                    let at = (u64::from(index) - first) as usize * step as usize;
                    let word = |n: usize| {
                        records
                            .get(at + 4 * n..at + 4 * n + 4)
                            .and_then(|b| b.try_into().ok())
                            .map_or(0, u32::from_le_bytes)
                    };
                    if !samplers && word(0) == VIEW_TAG {
                        // A draw's own table: the draw makes what it
                        // reads of it (`draw.rs` `tables`).
                        if host.copied_tabled(index, frame) {
                            out.tabled += 1;
                            continue;
                        }
                        // Buffer views and placed textures get no host
                        // slot from this anyway, they wait for a draw
                        // or a pass: noted some frames on, unless a
                        // draw's table turns out to hold them.
                        if matches!(word(2) >> KIND_SHIFT, CONSTANT | PLACED | LOD_PLACED) && host.copied_waits() {
                            out.wait.push(index);
                            continue;
                        }
                    }
                    // No view, in a slot that shows none: nothing to make or
                    // unmake (32,000 of 41,000 a frame on the water, run
                    // b2-water).
                    if word(0) != VIEW_TAG && host.shows_none(index) {
                        out.nothing += 1;
                        continue;
                    }
                    out.make.push(index);
                    out.kinds[if samplers {
                        3
                    } else if word(0) != VIEW_TAG {
                        0
                    } else if word(2) >> KIND_SHIFT == 0 {
                        1
                    } else {
                        2
                    }] += 1;
                }
            }
            *found[n].lock().unwrap_or_else(|e| e.into_inner()) = out;
        });
        drop(copied);
        for part in found {
            let mut part = part.into_inner().unwrap_or_else(|e| e.into_inner());
            pages += part.pages;
            views += part.views;
            bytes += part.bytes;
            looked += part.looked;
            short += part.short;
            tabled += part.tabled;
            kinds[0] += part.nothing;
            later += part.wait.len() as u64;
            mirrored += part.make.len() as u64;
            wait.append(&mut part.wait);
            make.append(&mut part.make);
            for (kind, n) in kinds.iter_mut().zip(part.kinds) {
                *kind += n;
            }
        }
        spent[1] += since(began);
        let began = std::time::Instant::now();
        crate::land::each_view(true, make.len().div_ceil(records_a_job), &|n| {
            for &index in make.chunks(records_a_job).nth(n).unwrap_or_default() {
                host.mirror(index, start + u64::from(index) * step, false);
            }
        });
        spent[2] += since(began);
        if !wait.is_empty() {
            host.copied_wait(frame, std::mem::take(&mut wait));
        }
    }
    drop(heaps);
    let micros = started.elapsed().as_micros() as u64;
    for (count, n) in SYNCED.iter().zip([pages, looked, mirrored, micros, tabled, later]) {
        count.fetch_add(n, Relaxed);
    }
    let [taking, reading, making] = spent;
    for (count, n) in SYNC_SPLIT.iter().zip([views, bytes, taking, reading, making]) {
        count.fetch_add(n, Relaxed);
    }
    for (count, n) in SYNCED_KINDS.iter().zip(kinds) {
        count.fetch_add(n, Relaxed);
    }
    SYNC_SHORT.fetch_add(short, Relaxed);
}

/// Pages [`sync_heaps`] read in place of which a part could not be read
/// ([`sync_in_place`]), since the last frame line.
#[cfg(windows)]
static SYNC_SHORT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether [`sync_heaps`] reads a changed page's records where they are, each
/// once, and compares each with the one last seen as it reads it, 32 bytes at
/// a time (`VICTORY_SYNC_IN_PLACE=1`; off by default: into a copy of the page
/// first, through the kernel, then compared, as before).
///
/// Ported from Belobug/VIctory-Belobug-AMD 15ab94e (measured there: the copy
/// and the compare were each a pass over all of the page, 7.4 and 5.3 ms a
/// frame of the six threads for 35 MB, run heapsync-ab1; on by default there
/// since 8822ed4). Here the round's "reading and comparing" was 2.6-3.1 ms a
/// frame (release run 233354), and the submitting thread waits for the
/// round. A page the title unmaps meanwhile fails where the copy failed; the
/// records before the place are taken as they were read.
#[cfg(windows)]
fn sync_in_place() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_SYNC_IN_PLACE") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SYNC_IN_PLACE").is_some_and(|v| v != "0"))
}

/// Of those mirrored, since the last frame line: no view, a view of an object,
/// another, a sampler; waited and no draw came.
#[cfg(windows)]
static SYNCED_KINDS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// What a thread found on its pages of a heap, for [`sync_heaps`].
#[cfg(windows)]
#[derive(Default)]
struct SyncPart {
    pages: u64,
    views: u64,
    bytes: u64,
    looked: u64,
    tabled: u64,
    nothing: u64,
    make: Vec<u32>,
    wait: Vec<u32>,
    kinds: [u64; 5],
    /// Pages read in place not all there ([`SYNC_SHORT`]).
    short: u64,
}

/// Pages [`sync_heaps`] gives a thread at a time.
#[cfg(windows)]
const SYNC_PAGES: usize = 8;

/// Records [`sync_heaps`] gives a thread at a time.
#[cfg(windows)]
const SYNC_AT_A_TIME: usize = 256;

/// With `VICTORY_SYNC_FINE=1`: pages and records [`sync_heaps`] gives a thread
/// at a time.
#[cfg(windows)]
const SYNC_PAGES_FINE: usize = 2;
#[cfg(windows)]
const SYNC_AT_A_TIME_FINE: usize = 64;

/// Pages and records [`sync_heaps`] gives a thread at a time
/// (`VICTORY_SYNC_FINE=1`: [`SYNC_PAGES_FINE`] and [`SYNC_AT_A_TIME_FINE`];
/// off by default: [`SYNC_PAGES`] and [`SYNC_AT_A_TIME`], as before).
///
/// Ported from Belobug/VIctory-Belobug-AMD a284ca7: a round for a submission
/// finds a dozen pages or so; by eight a job that was two jobs, on two of the
/// six threads, while the submission waited for the round (2 to 3.4 ms at a
/// time, their run city4).
#[cfg(windows)]
fn sync_jobs() -> (usize, usize) {
    #[cfg(feature = "diagnostics")]
    if let Some(fine) = crate::ab::pick("VICTORY_SYNC_FINE") {
        return if fine { (SYNC_PAGES_FINE, SYNC_AT_A_TIME_FINE) } else { (SYNC_PAGES, SYNC_AT_A_TIME) };
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("VICTORY_SYNC_FINE").is_some_and(|v| v != "0")) {
        (SYNC_PAGES_FINE, SYNC_AT_A_TIME_FINE)
    } else {
        (SYNC_PAGES, SYNC_AT_A_TIME)
    }
}

/// Since the last frame line, of [`sync_heaps`]: changed pages of placed
/// shader-visible heaps looked at, slots on them, slots whose record was
/// not the one last seen and was mirrored, the microseconds it took; then
/// new records left to the draws that read them through a table, and new
/// records that waited for one ([`host::HostHeap::copied_wait`]); and the
/// microseconds submissions waited for it ([`sync_aside`]).
#[cfg(windows)]
pub(crate) static SYNCED: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];

/// The same, of [`SYNCED`]'s pages: those in view memory that was guarded;
/// bytes read; nanoseconds asking what was written, reading it and comparing
/// it with what was last seen, making views.
#[cfg(windows)]
static SYNC_SPLIT: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// With the frame line: what [`sync_heaps`] did per frame over `frames`.
/// Nothing while no heap page changed.
#[cfg(windows)]
pub(crate) fn sync_line(frames: f64) {
    let [twins, moved, lost] = [0, 1, 2].map(|i| host::TWINS[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    if twins > 0 {
        tracing::info!(
            "raw twins of buffer views made per frame (VICTORY_RAW_VIEWS=0: none): {:.0}, of them {:.1} that do not begin on 16 bytes ({:.1} null)",
            twins as f64 / frames,
            moved as f64 / frames,
            lost as f64 / frames
        );
    }
    let [pages, looked, mirrored, micros, tabled, later, waited] = [0, 1, 2, 3, 4, 5, 6].map(|i| SYNCED[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    let split = [0, 1, 2, 3, 4].map(|i| SYNC_SPLIT[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64);
    let kinds = [0, 1, 2, 3, 4].map(|i| SYNCED_KINDS[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames);
    let short = SYNC_SHORT.swap(0, std::sync::atomic::Ordering::Relaxed);
    let quick = [0, 1].map(|i| host::SPANS_QUICK[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames);
    if quick[0] > 0.0 || sync_jobs().0 == SYNC_PAGES_FINE {
        tracing::info!(
            "writes of view slots per frame (VICTORY_SPANS_QUICK=1): {:.0} leaving the placed views' spans alone, {:.0} under \
             their lock; the heaps' records in jobs of {} pages (VICTORY_SYNC_FINE)",
            quick[0],
            quick[1],
            sync_jobs().0
        );
    }
    if pages == 0 {
        return;
    }
    if sync_in_place() {
        tracing::info!(
            "the heaps' records read where they are (VICTORY_SYNC_IN_PLACE=1): {:.1} pages per frame, {short} of them not all \
             there since the last line",
            pages as f64 / frames
        );
    }
    tracing::info!(
        "descriptor records the title copied into its heap as bytes (VICTORY_HEAP_SYNC=0: not looked for): per frame \
         {:.1} changed heap pages, {:.0} slots looked at, {:.1} whose record was new and was mirrored, {:.1} new ones of \
         draws' tables left to the draws, {:.1} that waited for a draw first, {:.2} ms ({:.2} ms waited for); of the mirrored \
         {:.0} no views (with those of slots that show none), {:.0} of objects, {:.0} others, {:.0} samplers, {:.0} that waited and no draw came; \
         {:.1} of the pages guarded views, {:.1} MB read; {:.2} ms asking, {:.2} reading and comparing, {:.2} making",
        pages as f64 / frames,
        looked as f64 / frames,
        mirrored as f64 / frames,
        tabled as f64 / frames,
        later as f64 / frames,
        micros as f64 / 1000.0 / frames,
        waited as f64 / 1000.0 / frames,
        kinds[0],
        kinds[1],
        kinds[2],
        kinds[3],
        kinds[4],
        split[0] / frames,
        split[1] / (1 << 20) as f64 / frames,
        split[2] / 1e6 / frames,
        split[3] / 1e6 / frames,
        split[4] / 1e6 / frames
    );
}

/// Title descriptor slots in a changed byte range that have host counterparts.
/// `range_end` is exclusive.
#[cfg(windows)]
fn changed_slots(
    range_from: usize,
    range_end: usize,
    heap_start: usize,
    step: u64,
    mirrored_count: u32,
) -> std::ops::Range<u64> {
    let from = (range_from - heap_start) as u64;
    let to = (range_end - heap_start) as u64;
    let first = from / step;
    let last = to.div_ceil(step).min(u64::from(mirrored_count));
    first.min(last)..last
}

/// The title's view heap that shaders of a list without one get their views
/// from (`draw.rs`, `bind`): the title's structure builder binds root views
/// only, and the host shows those, and memory by address, through heap slots.
/// The biggest there is. `VICTORY_LENT_HEAP=0`: none, such a dispatch is
/// dropped, as before.
#[cfg(windows)]
pub(crate) fn lent_heap() -> Option<usize> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_LENT_HEAP").is_none_or(|v| v != "0")) {
        return None;
    }
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        // SAFETY: registered heaps are live (removed when dropped).
        .map(|&(_, _, h)| (h, unsafe { &*(h as *const Heap) }))
        .filter(|(_, heap)| !heap.samplers() && matches!(heap.host.get(), Some(Some(_))))
        .max_by_key(|(_, heap)| heap.desc.count)
        .map(|(h, _)| h)
}

/// `f` of the title heap at `heap`, if that heap still exists; it stays
/// until `f` is done.
#[cfg(windows)]
pub(crate) fn with_heap<T>(heap: usize, f: impl FnOnce(&Heap) -> T) -> Option<T> {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    // SAFETY: registered heaps are live (removed when dropped).
    heaps.iter().any(|&(_, _, h)| h == heap).then(|| f(unsafe { &*(heap as *const Heap) }))
}

/// Gives back dynamic slots of the host heap behind the title heap at
/// `heap`, if that heap still exists.
#[cfg(windows)]
pub(crate) fn release_dynamic(heap: usize, first: u32, n: u32) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.release(first, n);
    }
}

/// Runs `f` with the host heap of the title's largest view heap, if there is
/// one (nothing is waited for).
#[cfg(windows)]
pub(crate) fn with_view_heap<T>(f: impl FnOnce(&host::HostHeap) -> T) -> Option<T> {
    let heaps = SHADER_HEAPS.try_read().ok()?;
    let host = heaps
        .iter()
        // SAFETY: registered heaps are live (removed when dropped).
        .map(|&(_, _, at)| unsafe { &*(at as *const Heap) })
        .filter(|heap| !heap.samplers())
        .filter_map(|heap| heap.host.get()?.as_ref())
        .max_by_key(|host| host.count())?;
    Some(f(host))
}

/// Makes dynamic slot `index` of the host heap behind the title heap at
/// `heap`, if that heap still exists, the constant buffer view of `size`
/// bytes of `buffer` at `gpu`.
#[cfg(windows)]
pub(crate) fn constant_slot(
    heap: usize,
    index: u32,
    buffer: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    gpu: u64,
    size: u32,
) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.constant_at(index, buffer, gpu, size);
    }
}

#[cfg(windows)]
impl Drop for Heap {
    fn drop(&mut self) {
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        heaps.retain(|&(_, _, h)| h != at);
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

impl RtvRecord {
    /// All the view it records is made from: records with equal keys have
    /// equal views (of the same host texture).
    #[cfg(windows)]
    pub(crate) fn key(&self) -> (usize, [u32; 7]) {
        let mut words = [self.described; 7];
        words[1..].copy_from_slice(&self.desc);
        (self.resource as usize, words)
    }

    /// The view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn depth_desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        const TEXTURE_2D: u32 = 3;
        self.host_desc(TEXTURE_2D)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    #[cfg(windows)]
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = if self.described != 0 {
            self.desc
        } else {
            [
                crate::resource::view_format(self.resource)?,
                dimension,
                0,
                0,
                0,
                0,
            ]
        };
        // A target or depth view: of a target, in its stand-in format.
        words[0] = crate::resource::target_format(crate::resource::pc_format(words[0]));
        // SAFETY: both host view descriptions are 24 bytes in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }
}

/// Device method: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: the title passes a description or null.
    let described = unsafe { desc.as_ref() };
    let record = RtvRecord {
        tag: RTV_TAG,
        described: u32::from(described.is_some()),
        resource,
        desc: described.copied().unwrap_or_default(),
    };
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// Device method: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(
    this: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    create_rtv(this, resource, desc, handle);
    if handle != 0 {
        // SAFETY: just written by `create_rtv`.
        unsafe { (handle as *mut u32).write_unaligned(DSV_TAG) };
    }
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

impl View {
    /// Whether the view reads buffer memory (constant buffers, buffer views
    /// of resources or of memory) or memory the title placed a texture in:
    /// such views get host slots only for draws.
    #[cfg(windows)]
    fn is_buffer(&self) -> bool {
        match self {
            View::Constant(..) | View::Placed(..) => true,
            View::Shader(r, _) | View::Unordered(r, _, _) => {
                crate::resource::buffer_bytes(*r as This).is_some()
            }
            View::Sampler(_) | View::ConsoleSampler(_) => false,
        }
    }
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
///
/// Views without a resource object have nothing that releases them, and a
/// title copies descriptors as bytes, so nothing says when the last
/// descriptor naming an entry is gone: such entries were kept for the run.
/// The title writes some 14,000 constant buffer views a frame,
/// each at the address its constants have that frame: 2,450,139 entries
/// after four minutes of play, 1,047 MB with the index, and 11,000 more
/// every second (run 65). So the two kinds that carry an address are not
/// entries any more: their records name them whole ([`constant_words`],
/// [`placed_words`]), and the table holds only what is bounded: views of
/// resource objects (freed with the resource), samplers (as many as there
/// are sampler states), and the shapes of placed views (their descriptions
/// without the address: as many as the title has kinds of textures and
/// buffers, however often it moves them).
/// `VICTORY_VIEW_INLINE=0`: every view is an entry, as before.
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    index: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
    /// The shapes of views of placed memory, and where each is.
    shapes: Vec<Shape>,
    shape_index: std::collections::BTreeMap<Shape, u32>,
    /// LOCAL (views): the most shapes there may be ([`SHAPES`]; a field so
    /// that tests can fill a small table).
    shape_cap: usize,
    /// LOCAL (views2): the distinct `ResourceMinLODClamp` words of placed
    /// shader views that a record names by number ([`LOD_PLACED`]), where
    /// each is, and the most there may be ([`LODS`]; a field for tests).
    lods: Vec<u32>,
    lod_index: std::collections::BTreeMap<u32, u32>,
    lod_cap: usize,
    /// LOCAL (views2): the bare shapes (without clamp) that [`LOD_PLACED`]
    /// records name, a table of their own (not the shapes' numbering), and
    /// the most there may be ([`LOD_SHAPES`]).
    lod_shapes: Vec<Shape>,
    lod_shape_index: std::collections::BTreeMap<Shape, u32>,
    lod_shape_cap: usize,
    /// LOCAL (views): when each entry was last looked at or written, in
    /// seconds since the first (parallel to `list`; [`Views::sweep`]).
    stamps: Vec<std::sync::atomic::AtomicU32>,
    /// LOCAL (views): live entries of constant buffer views and of views of
    /// placed memory (counters, so that the memory line scans nothing).
    live_constants: u64,
    live_placed: u64,
}

/// Read far more often than written: every slot a draw reads through a
/// table is looked up here, on several recording threads at once in the
/// world. Readers share it; only a view not seen before (a new table entry
/// or shape) and a released resource take it alone.
static VIEWS: std::sync::RwLock<Views> = std::sync::RwLock::new(Views {
    list: Vec::new(),
    index: std::collections::BTreeMap::new(),
    by_resource: std::collections::BTreeMap::new(),
    free: Vec::new(),
    shapes: Vec::new(),
    shape_index: std::collections::BTreeMap::new(),
    shape_cap: SHAPES,
    lods: Vec::new(),
    lod_index: std::collections::BTreeMap::new(),
    lod_cap: LODS,
    lod_shapes: Vec::new(),
    lod_shape_index: std::collections::BTreeMap::new(),
    lod_shape_cap: LOD_SHAPES,
    stamps: Vec::new(),
    live_constants: 0,
    live_placed: 0,
});
const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

/// A record is the tag and two words, `i` and `generation` (the host keeps
/// them as one 64-bit key per slot). The top two bits of `generation` say
/// what the other 62 bits are: 0, entry `i` of the table at that
/// generation; [`CONSTANT`], a constant buffer view, whole; [`PLACED`], a
/// view of placed memory: its address and the number of its shape.
const KIND_SHIFT: u32 = 30;
/// Table generations stay below the kind bits (and above 0).
const GENERATIONS: u32 = (1 << KIND_SHIFT) - 1;
const CONSTANT: u32 = 1;
const PLACED: u32 = 2;
/// Bits of a constant buffer view's address in its record: any address a
/// program can have. The 15 bits above them are its size.
const CONSTANT_ADDRESS_BITS: u32 = 47;
/// Bits of a placed view's address in its record: placed memory is title
/// memory, which lies below 1 TiB. The 22 bits above them number its shape.
const PLACED_ADDRESS_BITS: u32 = 40;
const SHAPES: usize = 1 << (62 - PLACED_ADDRESS_BITS);

/// LOCAL (views2): the fourth kind of record, [`LOD_PLACED`]: a view of
/// placed memory whose `ResourceMinLODClamp` is not 0. The title's texture
/// streaming moves that float continuously (one texture's views: 0.0067,
/// then 0.0155, ...), and as part of the shape it made some 4,000 new
/// shapes a second (2.59 million in 22 minutes, the table full after half an
/// hour). Here the shape holds the view without it, and the record names the
/// clamp by number in a table of distinct clamp words ([`LODS`]) as it names
/// the shape: the 62 bits are the address (on a 64 KiB boundary: 24 bits),
/// the bare shape's number in a table of its own (17 bits) and the clamp's
/// (21 bits: the clamps are 16-bit fractions, up to 65,536 for each mip).
/// The address is on a 64 KiB boundary (24 bits, as placed textures are).
/// Whatever does not fit is a shape that holds the clamp, as before.
const LOD_PLACED: u32 = 3;
const LOD_ALIGN_BITS: u32 = 16;
const LOD_ADDRESS_BITS: u32 = PLACED_ADDRESS_BITS - LOD_ALIGN_BITS;
const LOD_SHAPE_BITS: u32 = 17;
const LOD_SHAPES: usize = 1 << LOD_SHAPE_BITS;
const LODS: usize = 1 << (62 - LOD_ADDRESS_BITS - LOD_SHAPE_BITS);

/// LOCAL (views2): whether the shape of a placed view leaves out what is not
/// part of it ([`padding_cleared`], [`lod_split`]). On by default;
/// `VICTORY_VIEW_SHAPE_KEY=old` keeps every word, as before (the filler of
/// the resource description is [`filler_cleared`]'s, `VICTORY_PLACED_FILLER`).
fn split_keys() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VICTORY_VIEW_SHAPE_KEY").is_ok_and(|v| v.trim().eq_ignore_ascii_case("old")))
}

/// LOCAL (views2): the word of a shader resource view description (the
/// host's layout: format, dimension, component mapping, a word of padding
/// before the 8-aligned union, so the union's members begin at word 4) that
/// holds `ResourceMinLODClamp`, by view dimension; `None` where there is
/// none (buffers, multisampled, acceleration structures).
fn lod_word(dimension: u32) -> Option<usize> {
    match dimension {
        // 1D, 3D, cube: most detailed mip, mip levels, clamp.
        2 | 8 | 9 => Some(6),
        // 2D: most detailed mip, mip levels, plane, clamp.
        4 => Some(7),
        // 1D array, cube array: mip, levels, two more, clamp.
        3 | 10 => Some(8),
        // 2D array: mip, levels, first slice, slices, plane, clamp.
        5 => Some(9),
        _ => None,
    }
}

/// LOCAL (views2): `desc` of a shader view with word 3 cleared: the padding
/// that lies before the 8-aligned union of a `D3D12_SHADER_RESOURCE_VIEW_DESC`
/// (the unordered access description has none: its word 3 is the high half
/// of a buffer's first element or a texture's plane). It is what the title's
/// stack held (0x102, then 0x103, ...), shown to nothing, and as part of a
/// shape it made a new one for every value.
fn padding_cleared(unordered: bool, mut desc: [u32; 16]) -> [u32; 16] {
    if !unordered && split_keys() {
        desc[3] = 0;
        // Also the union words past the members of the view's dimension
        // (a 2D view has words 4-7: 8 and 9 are the union's tail, which the
        // device does not read); never the address (10, 11) nor the console
        // fields (12-15), and nothing for a dimension not known.
        let tail = match desc[1] {
            1 | 3 | 10 => 9, // buffer; 1D array, cube array
            2 | 8 | 9 => 7,  // 1D, 3D, cube
            4 => 8,          // 2D
            6 => 4,          // multisampled (one unused word)
            7 | 11 => 6,     // multisampled array; acceleration structure
            _ => 10,         // 2D array fills it; unknown: untouched
        };
        desc[tail..10].fill(0);
    }
    desc
}

/// LOCAL (views2): the description of a shader view of a texture without its
/// clamp, and the clamp's word, if it has one that is not 0 (0.0, the clamp
/// that is none, is what the words without a clamp hold: that view is its
/// shape as it is).
fn lod_split(unordered: bool, desc: &[u32; 16]) -> Option<([u32; 16], u32)> {
    if unordered || !split_keys() {
        return None;
    }
    let at = lod_word(desc[1])?;
    let bits = desc[at];
    let mut bare = *desc;
    bare[at] = 0;
    (bits != 0).then_some((bare, bits))
}

/// A view of placed memory without its address: whether it is unordered
/// access, the resource description, and the view's words but the two that
/// hold the address (10 and 11).
type Shape = (bool, [u32; 14], [u32; 14]);

/// Whether constant buffer views and views of placed memory are named whole
/// in their records ([`Views`]). On by default; `VICTORY_VIEW_INLINE=0`
/// makes every view a table entry, as before.
fn inline() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_INLINE").is_none_or(|v| v != "0"))
}

/// Views named whole in their records since the start: constant buffer
/// views, views of placed memory (for the memory line).
static NAMED_WHOLE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The two record words of the 62 bits `payload` of kind `kind`.
fn kind_words(kind: u32, payload: u64) -> [u32; 2] {
    [payload as u32, kind << KIND_SHIFT | (payload >> 32) as u32]
}

/// The 62 bits two record words carry beside their kind.
fn payload(i: u32, generation: u32) -> u64 {
    u64::from(generation & GENERATIONS) << 32 | u64::from(i)
}

/// The record words that name the constant buffer view of `size` bytes at
/// `address` whole; `None` if they cannot: an address past 47 bits, or a
/// size of 16 KiB or more that is no multiple of 256 or reaches 4 MiB (the
/// size has 15 bits: bytes below 16 KiB, else units of 256 bytes). Such a
/// view is a table entry.
fn constant_words(address: u64, size: u32) -> Option<[u32; 2]> {
    if address >> CONSTANT_ADDRESS_BITS != 0 {
        return None;
    }
    let code = if size < 1 << 14 {
        u64::from(size)
    } else if size.is_multiple_of(256) && size >> 8 < 1 << 14 {
        1 << 14 | u64::from(size >> 8)
    } else {
        return None;
    };
    Some(kind_words(CONSTANT, code << CONSTANT_ADDRESS_BITS | address))
}

/// The constant buffer view [`constant_words`] named.
fn constant_of(i: u32, generation: u32) -> View {
    let payload = payload(i, generation);
    let code = (payload >> CONSTANT_ADDRESS_BITS) as u32;
    let size = if code >> 14 == 0 { code } else { (code & 0x3fff) << 8 };
    View::Constant(payload & ((1 << CONSTANT_ADDRESS_BITS) - 1), size)
}

/// A placed view's shape and its address.
fn shape_of(unordered: bool, resource: &[u32; 14], desc: &[u32; 16]) -> (Shape, u64) {
    let resource = &filler_cleared(*resource);
    let mut rest = [0u32; 14];
    rest[..10].copy_from_slice(&desc[..10]);
    rest[10..].copy_from_slice(&desc[12..]);
    (
        (unordered, *resource, rest),
        u64::from(desc[10]) | u64::from(desc[11]) << 32,
    )
}

/// LOCAL (views): the resource description with its filler words (1, after
/// the dimension, and 13, at the end: padding of the title's layout, which
/// holds whatever its stack did) cleared. They are not part of what a
/// texture is (`resource::texture_at` leaves them out of its key too), but
/// they were part of a shape: the same texture, viewed again and again with
/// other stack garbage in them, made millions of shapes until the table was
/// full (a long run: 4,194,304 shapes for some 160,000 textures; 390,854
/// "filler twins" counted by `texture_at`), and every view after that a
/// table entry that nothing frees. `VICTORY_PLACED_FILLER=1` keeps them.
fn filler_cleared(mut resource: [u32; 14]) -> [u32; 14] {
    static KEPT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*KEPT.get_or_init(|| std::env::var_os("VICTORY_PLACED_FILLER").is_some_and(|v| v == "1")) {
        resource[1] = 0;
        resource[13] = 0;
    }
    resource
}

/// The record words that name the view of placed memory at `address` with
/// shape number `shape`.
fn placed_words(shape: u32, address: u64) -> [u32; 2] {
    kind_words(PLACED, u64::from(shape) << PLACED_ADDRESS_BITS | address)
}

/// The view of `shape` at `address`.
fn placed_view(&(unordered, resource, rest): &Shape, address: u64) -> View {
    let mut desc = [0u32; 16];
    desc[..10].copy_from_slice(&rest[..10]);
    (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
    desc[12..].copy_from_slice(&rest[10..]);
    View::Placed(unordered, resource, desc)
}

impl Views {
    /// As [`Views::shape`], and whether the shape is new.
    fn insert_shape(&mut self, shape: Shape) -> Option<(u32, bool)> {
        if let Some(&at) = self.shape_index.get(&shape) {
            return Some((at, false));
        }
        if self.shapes.len() >= self.shape_cap {
            return None;
        }
        let at = self.shapes.len() as u32;
        self.shapes.push(shape);
        self.shape_index.insert(shape, at);
        Some((at, true))
    }

    /// The view of placed memory [`placed_words`] named.
    fn placed_of(&self, i: u32, generation: u32) -> Option<View> {
        let payload = payload(i, generation);
        let shape = self.shapes.get((payload >> PLACED_ADDRESS_BITS) as usize)?;
        Some(placed_view(shape, payload & ((1 << PLACED_ADDRESS_BITS) - 1)))
    }

    /// LOCAL (views2): the view of placed memory [`LOD_PLACED`] named: the
    /// view of its shape at its address, with the clamp it names.
    fn lod_placed_of(&self, i: u32, generation: u32) -> Option<View> {
        let payload = payload(i, generation);
        let address = (payload & ((1 << LOD_ADDRESS_BITS) - 1)) << LOD_ALIGN_BITS;
        let shape = (payload >> LOD_ADDRESS_BITS) & ((1 << LOD_SHAPE_BITS) - 1);
        let bits = *self.lods.get((payload >> (LOD_ADDRESS_BITS + LOD_SHAPE_BITS)) as usize)?;
        let View::Placed(unordered, resource, mut desc) = placed_view(self.lod_shapes.get(shape as usize)?, address)
        else {
            return None;
        };
        desc[lod_word(desc[1])?] = bits;
        Some(View::Placed(unordered, resource, desc))
    }

    /// The record words of `view` as a table entry, if it has one.
    fn entry_of(&self, view: &View) -> Option<[u32; 2]> {
        let &i = self.index.get(view)?;
        Some([i, self.list[i as usize].1])
    }

    /// The record words of `view` as a table entry: its entry (a new or a
    /// freed one if the view is not there yet) and that entry's generation.
    fn entry(&mut self, view: View) -> [u32; 2] {
        let i = match self.index.get(&view) {
            Some(&i) => i,
            None => {
                let i = match self.free.pop() {
                    Some(i) => {
                        self.list[i as usize].0 = view.clone();
                        i
                    }
                    None => {
                        self.list.push((view.clone(), 1));
                        self.stamps.push(std::sync::atomic::AtomicU32::new(0));
                        self.list.len() as u32 - 1
                    }
                };
                match view {
                    View::Constant(..) => self.live_constants += 1,
                    View::Placed(..) => self.live_placed += 1,
                    _ => {}
                }
                for r in view.resources().into_iter().filter(|&r| r != 0) {
                    self.by_resource.entry(r).or_default().push(i);
                }
                self.index.insert(view, i);
                i
            }
        };
        self.touch(i);
        [i, self.list[i as usize].1]
    }

    /// LOCAL (views): entry `i` is in use now.
    fn touch(&self, i: u32) {
        use std::sync::atomic::Ordering::Relaxed;
        if let Some(stamp) = self.stamps.get(i as usize) {
            let now = now_seconds();
            // (Read first: no write to a line that readers share, unless
            // the second changed.)
            if stamp.load(Relaxed) != now {
                stamp.store(now, Relaxed);
            }
        }
    }

    /// LOCAL (views): frees the entries of views without a resource object
    /// (constant buffer views and views of placed memory: nothing releases
    /// them) that nothing has looked at or written for `idle` seconds, for
    /// reuse under a new generation; descriptor slots that still name one
    /// go stale, as those of a released resource do. Returns how many.
    /// The table holds such entries only for views that do not fit a record
    /// (and for all of them once the shapes are full).
    fn sweep(&mut self, now: u32, idle: u32) -> u64 {
        use std::sync::atomic::Ordering::Relaxed;
        let mut freed = 0;
        for i in 0..self.list.len() {
            let (view, generation) = &self.list[i];
            let constant = match view {
                View::Constant(..) => true,
                View::Placed(..) => false,
                _ => continue,
            };
            // Free entries keep their old view: only those the index names.
            if self.index.get(view) != Some(&(i as u32)) {
                continue;
            }
            if now.saturating_sub(self.stamps[i].load(Relaxed)) < idle {
                continue;
            }
            let next = ((generation + 1) & GENERATIONS).max(1);
            let view = view.clone();
            self.index.remove(&view);
            self.list[i].1 = next;
            self.free.push(i as u32);
            if constant {
                self.live_constants -= 1;
            } else {
                self.live_placed -= 1;
            }
            freed += 1;
        }
        freed
    }
}

/// LOCAL (views): seconds since the first call.
fn now_seconds() -> u32 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_secs() as u32
}

/// LOCAL (views): [`Views::shape`] on `views`, taking the lock alone only
/// to add a shape: a shape known, and a full table, are answered under the
/// lock shared (a full table is asked a new shape for ever, by several
/// threads at once, and it is the lock every slot a draw reads goes
/// through). The flag: the shape was new.
fn shape_in(views: &std::sync::RwLock<Views>, shape: Shape) -> Option<(u32, bool)> {
    {
        let shared = views.read().unwrap_or_else(|e| e.into_inner());
        if let Some(&at) = shared.shape_index.get(&shape) {
            return Some((at, false));
        }
        if shared.shapes.len() >= shared.shape_cap {
            return None;
        }
    }
    views.write().unwrap_or_else(|e| e.into_inner()).insert_shape(shape)
}

/// LOCAL (views2): why a placed view with a clamp is not named by a
/// [`LOD_PLACED`] record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LodMiss {
    /// Its address is no 64 KiB boundary, or lies past 1 TiB.
    Address,
    /// Its bare shape is not in the table of them, which is full.
    Shapes,
    /// Its clamp is not in the table of them, which is full.
    Clamps,
}

/// LOCAL (views2): the record words ([`LOD_PLACED`]) of the placed view of
/// bare `shape` (no clamp) at `address` with clamp word `bits`. The cheap
/// checks (address, room in the tables) come first: a bare shape or a clamp
/// that cannot be used is never added.
fn lod_named(views: &std::sync::RwLock<Views>, shape: Shape, address: u64, bits: u32) -> Result<[u32; 2], LodMiss> {
    if address >> PLACED_ADDRESS_BITS != 0 || address & ((1 << LOD_ALIGN_BITS) - 1) != 0 {
        return Err(LodMiss::Address);
    }
    let words = |shape: u32, lod: u32| {
        kind_words(
            LOD_PLACED,
            u64::from(lod) << (LOD_ADDRESS_BITS + LOD_SHAPE_BITS)
                | u64::from(shape) << LOD_ADDRESS_BITS
                | address >> LOD_ALIGN_BITS,
        )
    };
    {
        let shared = views.read().unwrap_or_else(|e| e.into_inner());
        let (known_shape, known_lod) = (shared.lod_shape_index.get(&shape), shared.lod_index.get(&bits));
        if known_shape.is_none() && shared.lod_shapes.len() >= shared.lod_shape_cap {
            return Err(LodMiss::Shapes);
        }
        if known_lod.is_none() && shared.lods.len() >= shared.lod_cap {
            return Err(LodMiss::Clamps);
        }
        if let (Some(&s), Some(&l)) = (known_shape, known_lod) {
            return Ok(words(s, l));
        }
    }
    let mut v = views.write().unwrap_or_else(|e| e.into_inner());
    // (Room for both, before either is added.)
    if !v.lod_shape_index.contains_key(&shape) && v.lod_shapes.len() >= v.lod_shape_cap {
        return Err(LodMiss::Shapes);
    }
    if !v.lod_index.contains_key(&bits) && v.lods.len() >= v.lod_cap {
        return Err(LodMiss::Clamps);
    }
    let s = match v.lod_shape_index.get(&shape) {
        Some(&s) => s,
        None => {
            let s = v.lod_shapes.len() as u32;
            v.lod_shapes.push(shape);
            v.lod_shape_index.insert(shape, s);
            s
        }
    };
    let l = match v.lod_index.get(&bits) {
        Some(&l) => l,
        None => {
            let l = v.lods.len() as u32;
            v.lods.push(bits);
            v.lod_index.insert(bits, l);
            l
        }
    };
    Ok(words(s, l))
}

/// LOCAL (views2): placed views with a clamp that were not named by a
/// [`LOD_PLACED`] record since the start, by [`LodMiss`].
static LOD_FALLBACKS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// LOCAL (views2): the record words of a placed view that has a clamp, named
/// whole in `views` (a [`LOD_PLACED`] record), or of the placed view in a
/// shape of the shapes (its clamp in the shape) when that does not fit, or
/// `None` when that is full too (the view is then a table entry). Counted,
/// and said at the first of each cause and at each power of two.
fn placed_words_in(views: &std::sync::RwLock<Views>, unordered: bool, resource: &[u32; 14], desc: &[u32; 16]) -> Option<[u32; 2]> {
    use std::sync::atomic::Ordering::Relaxed;
    let desc = &padding_cleared(unordered, *desc);
    if let Some((bare, bits)) = lod_split(unordered, desc) {
        let (shape, address) = shape_of(unordered, resource, &bare);
        match lod_named(views, shape, address, bits) {
            Ok(words) => {
                NAMED_WHOLE[1].fetch_add(1, Relaxed);
                return Some(words);
            }
            Err(miss) => {
                let n = LOD_FALLBACKS[miss as usize].fetch_add(1, Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::info!("views: {n} placed views with a min-LOD clamp did not fit a record that names it ({miss:?}): their shapes hold the clamp (past 1 TiB: a table entry)");
                }
            }
        }
    }
    let (shape, address) = shape_of(unordered, resource, desc);
    if address >> PLACED_ADDRESS_BITS != 0 {
        return None;
    }
    // Nearly every placed view has a shape seen before. (When the table is
    // full, the shared lock says so.)
    match shape_in(views, shape) {
        Some((at, _new)) => {
            #[cfg(feature = "diagnostics")]
            if _new {
                note_volatile(at);
            }
            NAMED_WHOLE[1].fetch_add(1, Relaxed);
            Some(placed_words(at, address))
        }
        None => {
            SHAPE_FALLBACKS.fetch_add(1, Relaxed);
            None
        }
    }
}

/// LOCAL (views): placed views that became table entries since the start
/// because the shapes were full; entries of such views freed for reuse
/// ([`Views::sweep`]).
static SHAPE_FALLBACKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RECYCLED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// LOCAL (views): how many entries of views without a resource object the
/// table may hold before [`maybe_sweep`] frees the idle ones. A healthy run
/// has next to none. `VICTORY_VIEW_ENTRY_CAP`, default 0 (never): Codex review
/// round 2 — idleness is not proof a retained descriptor slot no longer names
/// the entry; a swept entry turns such a slot null. Opt in with a cap.
fn entry_cap() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("VICTORY_VIEW_ENTRY_CAP").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0)
    })
}

/// LOCAL (views): when the entries of views without a resource object are
/// over [`entry_cap`], frees the ones idle for two minutes (and, if that
/// leaves over twice the cap, those idle for ten seconds); at most once in
/// five seconds. Called after such an entry was made.
fn maybe_sweep() {
    use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
    static LAST: AtomicU32 = AtomicU32::new(0);
    let cap = entry_cap();
    if cap == 0 {
        return;
    }
    {
        let v = views_read();
        if v.live_constants + v.live_placed <= cap {
            return;
        }
    }
    let now = now_seconds().max(1);
    let last = LAST.load(Relaxed);
    if (last != 0 && now < last + 5) || LAST.compare_exchange(last, now, Relaxed, Relaxed).is_err() {
        return;
    }
    let mut v = views_write();
    let now = now_seconds();
    let mut freed = v.sweep(now, 120);
    if v.live_constants + v.live_placed > cap * 2 {
        freed += v.sweep(now, 10);
    }
    let left = v.live_constants + v.live_placed;
    drop(v);
    RECYCLED.fetch_add(freed, Relaxed);
    if freed > 0 {
        tracing::info!("views: {freed} idle entries of views without a resource freed for reuse, {left} left (over {cap})");
    }
}

/// LOCAL (views, diagnostics): when a new shape was made, says once per
/// word which single word of it differs from a recent shape (the shape
/// numbered `at` is the newest): the word's number (0: unordered; 1-14 the
/// resource description; 15-28 the view description without the address),
/// and both values. Sampled: at most once in ten seconds, over the last
/// 20,000 shapes.
#[cfg(feature = "diagnostics")]
fn note_volatile(at: u32) {
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
    static LAST: AtomicU32 = AtomicU32::new(0);
    static SEEN: AtomicU64 = AtomicU64::new(0);
    let now = now_seconds().max(1);
    let last = LAST.load(Relaxed);
    if (last != 0 && now < last + 10) || LAST.compare_exchange(last, now, Relaxed, Relaxed).is_err() {
        return;
    }
    let views = views_read();
    let Some(new) = views.shapes.get(at as usize) else {
        return;
    };
    for old in views.shapes[..at as usize].iter().rev().take(20_000) {
        if let Some((word, was, is)) = single_difference(old, new) {
            if SEEN.fetch_or(1 << word, Relaxed) & (1 << word) == 0 {
                tracing::info!(
                    "views: shape {at} differs from an earlier one in word {word} only ({was:#x} then, {is:#x} now): a volatile word of a placed view?"
                );
            }
            return;
        }
    }
}

/// LOCAL (views): the number of the only word in which two shapes differ
/// (0: whether unordered; 1-14 the resource description; 15-28 the view's),
/// and its values; `None` if they differ in none or in several.
#[cfg(any(test, feature = "diagnostics"))]
fn single_difference(a: &Shape, b: &Shape) -> Option<(usize, u32, u32)> {
    let flat = |s: &Shape| -> Vec<u32> {
        std::iter::once(u32::from(s.0)).chain(s.1.iter().copied()).chain(s.2.iter().copied()).collect()
    };
    let (a, b) = (flat(a), flat(b));
    let mut differ = a.iter().zip(&b).enumerate().filter(|(_, (x, y))| x != y);
    let (word, (&x, &y)) = differ.next()?;
    differ.next().is_none().then_some((word, x, y))
}

/// The two words that name `view` in a record: the view itself where its
/// kind allows ([`Views`]), else its table entry.
fn words_for(view: View) -> [u32; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    if inline() {
        match &view {
            View::Constant(address, size) => {
                if let Some(words) = constant_words(*address, *size) {
                    NAMED_WHOLE[0].fetch_add(1, Relaxed);
                    return words;
                }
            }
            View::Placed(unordered, resource, desc) => {
                // LOCAL (views2): what is no part of the view is not in its
                // shape ([`padding_cleared`]); nor is a clamp that a record
                // can name ([`LOD_PLACED`]).
                if let Some(words) = placed_words_in(&VIEWS, *unordered, resource, desc) {
                    return words;
                }
            }
            _ => {}
        }
    }
    // LOCAL (views): the filler of a placed view's resource description is
    // no part of the view (see [`filler_cleared`]), here as in a shape.
    let view = match view {
        View::Placed(unordered, resource, desc) => {
            View::Placed(unordered, filler_cleared(resource), padding_cleared(unordered, desc))
        }
        view => view,
    };
    // A view written before keeps its entry and generation.
    {
        let views = views_read();
        if let Some(words) = views.entry_of(&view) {
            views.touch(words[0]);
            return words;
        }
    }
    let unowned = matches!(view, View::Constant(..) | View::Placed(..));
    let words = views_write().entry(view);
    if unowned {
        maybe_sweep();
    }
    words
}

fn views_read() -> std::sync::RwLockReadGuard<'static, Views> {
    VIEWS.read().unwrap_or_else(|e| e.into_inner())
}

fn views_write() -> std::sync::RwLockWriteGuard<'static, Views> {
    VIEWS.write().unwrap_or_else(|e| e.into_inner())
}

/// For the memory line: the distinct views kept ([`Views`]), those of them
/// free for reuse, and the bytes the table and its index hold for them (an
/// estimate: an entry of the table, and one of the index with its share of
/// a tree node; the shapes likewise); then the entries that are constant
/// buffer views and views of placed memory (those that did not fit a
/// record, or all of them with `VICTORY_VIEW_INLINE=0`), the shapes of
/// placed views kept, and the views named whole in their records since the
/// start: constant buffer views, views of placed memory.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn views_kept() -> [u64; 15] {
    use std::sync::atomic::Ordering::Relaxed;
    let views = views_read();
    let each = size_of::<(View, u32)>() as u64;
    let shape = size_of::<Shape>() as u64;
    // LOCAL (views): counters kept as entries come and go, not a scan of
    // the table under the lock every five seconds.
    let (constants, placed) = (views.live_constants, views.live_placed);
    [
        views.list.len() as u64,
        views.free.len() as u64,
        views.list.capacity() as u64 * each
            + views.index.len() as u64 * each * 3 / 2
            + views.shapes.capacity() as u64 * shape
            + views.shape_index.len() as u64 * shape * 3 / 2,
        constants,
        placed,
        views.shapes.len() as u64,
        NAMED_WHOLE[0].load(Relaxed),
        NAMED_WHOLE[1].load(Relaxed),
        SHAPE_FALLBACKS.load(Relaxed),
        RECYCLED.load(Relaxed),
        views.lods.len() as u64,
        views.lod_shapes.len() as u64,
        LOD_FALLBACKS[0].load(Relaxed),
        LOD_FALLBACKS[1].load(Relaxed),
        LOD_FALLBACKS[2].load(Relaxed),
    ]
}

impl View {
    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }

    /// The texture object the view is of, if that texture has no host copy
    /// yet (`crate::resource::host_waits`): a slot written with such a
    /// view is left null until the texture is first used.
    #[cfg(windows)]
    fn waiting_texture(&self) -> Option<usize> {
        match *self {
            View::Shader(r, _) | View::Unordered(r, _, _)
                if crate::resource::host_waits(r as This) =>
            {
                Some(r)
            }
            _ => None,
        }
    }
}

/// Title slots (their handles) written with a view of a texture object
/// that had no host copy then, by texture. A title texture gets its host
/// copy on first use (`resource.rs` `Hosted`), and the title
/// writes each texture's views into its view heap when it creates the
/// texture: making those host slots would make every host copy at once.
/// They are made when the texture is: translated shaders index the heap
/// directly, so a slot is not always made again by the draw that reads it.
#[cfg(windows)]
static WAITING: std::sync::Mutex<std::collections::BTreeMap<usize, Vec<u64>>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Textures given their host copy since [`make_waiting`] last ran.
#[cfg(windows)]
static MADE_LATE: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Title texture `resource` was given its host copy (called under that
/// texture's own lock: nothing else is locked here).
#[cfg(windows)]
pub(crate) fn texture_made(resource: usize) {
    MADE_LATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(resource);
}

/// The slot at `handle` was written with a view of texture `resource`,
/// which has no host copy: it waits for one.
#[cfg(windows)]
fn wait_for(resource: usize, handle: u64) {
    /// A texture with more slots waiting than this is in use for all we
    /// know (its views are being copied about): its host copy is made.
    const MOST: usize = 64;
    let many = {
        let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
        let slots = waiting.entry(resource).or_default();
        if !slots.contains(&handle) {
            slots.push(handle);
        }
        slots.len() > MOST
    };
    if many {
        // Makes it, and notes it made.
        let _ = crate::resource::host_copy(resource as This);
    } else if !crate::resource::host_waits(resource as This) {
        // Made by another thread since the slot was looked at: that making
        // may have been taken by `make_waiting` before this slot was noted.
        texture_made(resource);
    }
}

/// Makes the host slots that waited for a texture's host copy, for the
/// textures made since the last call. Called before a submission runs, by
/// the thread that submits: a slot is read when a list runs.
#[cfg(windows)]
pub(crate) fn make_waiting() {
    let made = std::mem::take(&mut *MADE_LATE.lock().unwrap_or_else(|e| e.into_inner()));
    for resource in made {
        let handles = WAITING
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&resource);
        for handle in handles.into_iter().flatten() {
            // Only if the slot still holds a view of that texture: the
            // title may have written it since (and a view of another
            // texture without a host copy must go on waiting).
            // (Looked up first: the handle is read only while its heap is
            // there.)
            // (The heap stays while the lock is held: `mirror`.)
            let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
            let Some(&(_, _, at)) = heaps.iter().find(|&&(start, end, _)| (start..end).contains(&handle)) else {
                continue;
            };
            // SAFETY: a registered heap, live while the lock is held.
            let heap = unsafe { &*(at as *const Heap) };
            let Some(index) = heap.index(handle) else {
                continue;
            };
            let same = match view(handle) {
                Some(View::Shader(r, _) | View::Unordered(r, _, _)) => r == resource,
                _ => false,
            };
            if same && let Some(Some(host)) = heap.host.get() {
                host.mirror(index, handle, true);
            }
        }
    }
}

fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    let [i, generation] = words_for(view);
    // SAFETY: the handle is a descriptor slot, at least 16 bytes.
    unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
    #[cfg(windows)]
    mirror(handle);
}

/// The title released `resource`: the views naming it go stale.
pub(crate) fn forget(resource: usize) {
    #[cfg(windows)]
    WAITING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&resource);
    let mut views = views_write();
    let Some(entries) = views.by_resource.remove(&resource) else {
        return;
    };
    for i in entries {
        let (view, generation) = views.list[i as usize].clone();
        // Entries freed through a view's other resource may hold another
        // view by now.
        if views.index.get(&view) != Some(&i) || !view.resources().contains(&resource) {
            continue;
        }
        views.index.remove(&view);
        // (Below the bits that say a record is no table entry.)
        views.list[i as usize].1 = ((generation + 1) & GENERATIONS).max(1);
        // The view's other resource (a counter) keeps a stale entry, which
        // `forget` of that resource skips.
        views.free.push(i);
    }
}

/// The view the record words `i` and `generation` name: the one they hold
/// whole, or the view of entry `i` if `generation` is still its own.
fn view_at(i: u32, generation: u32) -> Option<View> {
    match generation >> KIND_SHIFT {
        0 => {
            let views = views_read();
            let (view, current) = views.list.get(i as usize)?;
            (*current == generation).then(|| {
                views.touch(i);
                view.clone()
            })
        }
        CONSTANT => Some(constant_of(i, generation)),
        PLACED => views_read().placed_of(i, generation),
        LOD_PLACED => views_read().lod_placed_of(i, generation),
        _ => None,
    }
}

/// The words of the record at `handle`, if they are a view's.
pub(crate) fn record_at(handle: u64) -> Option<[u32; 3]> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    // LOCAL (wc-vram): a slot in card memory is read with streaming loads (`crate::wc_vram::read_value`).
    let words = crate::wc_vram::read_value::<[u32; 3]>(handle as usize).unwrap_or_else(|| unsafe { (handle as *const [u32; 3]).read_unaligned() });
    (words[0] == VIEW_TAG).then_some(words)
}

/// The view recorded at `handle`, if one is (and its resource is alive).
pub(crate) fn view(handle: u64) -> Option<View> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    // LOCAL (wc-vram): a slot in card memory is read with streaming loads (`crate::wc_vram::read_value`).
    let [tag, i, generation] = crate::wc_vram::read_value::<[u32; 3]>(handle as usize).unwrap_or_else(|| unsafe { (handle as *const [u32; 3]).read_unaligned() });
    if tag != VIEW_TAG {
        return None;
    }
    view_at(i, generation)
}

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

/// Device method.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    if let Some([lo, hi, size, _]) = words(desc) {
        let address = u64::from(lo) | u64::from(hi) << 32;
        record(handle, View::Constant(address, size));
    }
}

/// Device method.
pub(crate) extern "system" fn create_srv(
    _: This,
    resource: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    record(handle, View::Shader(resource as usize, words(desc)));
}

/// Device method.
pub(crate) extern "system" fn create_uav(
    _: This,
    resource: This,
    counter: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    let view = View::Unordered(resource as usize, counter as usize, words(desc));
    record(handle, view);
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`.
pub(crate) extern "system" fn create_placed_srv(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(false, resource, desc));
    }
}

pub(crate) extern "system" fn create_placed_uav(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(true, resource, desc));
    }
}

/// Device method.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let Some(step) = increment(kind).map(u64::from) else {
        return;
    };
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        // SAFETY: both ranges are descriptor slots the title owns.
        #[cfg(windows)]
        unsafe {
            crate::titlewrite::put(dst as *mut u8, src as *const u8, len as usize)
        };
        // SAFETY: as above.
        #[cfg(not(windows))]
        unsafe {
            std::ptr::copy(src as *const u8, dst as *mut u8, len as usize)
        };
        #[cfg(windows)]
        for i in 0..u64::from(count) {
            mirror(dst + i * step);
        }
    }
}

/// Device method: ranges of descriptors (start and count each) gathered
/// from the sources and scattered to the destinations, in order.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            .map(|i| {
                // SAFETY: the title passes `n` starts and sizes (or null sizes).
                unsafe {
                    (
                        *starts.add(i),
                        if sizes.is_null() { 1 } else { *sizes.add(i) },
                    )
                }
            })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    // LOCAL (wc-vram): a slot in card memory is read with streaming loads (`crate::wc_vram::read_value`).
    let record = crate::wc_vram::read_value::<RtvRecord>(handle as usize).unwrap_or_else(|| unsafe { (handle as *const RtvRecord).read_unaligned() });
    (record.tag == tag).then_some(record)
}

/// Device method.
pub(crate) extern "system" fn create_heap(
    _: This,
    desc_ptr: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1)))
    else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    tracing::debug!(?desc, placed, "descriptor heap");
    let (start, memory) = if placed {
        if placement.address.checked_add(size).is_none() {
            return E_INVALIDARG;
        }
        (placement.address, None)
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        memory,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        pending: Default::default(),
        #[cfg(windows)]
        written: (placed && visible)
            .then(|| crate::watch::track(start as usize, size as usize, true))
            .flatten(),
    });
    #[cfg(windows)]
    if visible {
        let at = std::ptr::from_ref::<Heap>(&heap) as usize;
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        heaps.push((start, start + size, at));
    }
    #[cfg(not(windows))]
    let _ = visible;
    object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &HEAP_IIDS).then_some(this);
    object::answer(DESCRIPTOR_HEAP, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
}
/// The same address for the CPU and the GPU start.
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<DESCRIPTOR_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = desc as *const (); // 0x40
    f[9] = start as *const (); // 0x48 CPU
    f[10] = start as *const (); // 0x50 GPU
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{S_OK, slot};

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    #[cfg(windows)]
    #[test]
    fn raw_heap_sync_stops_before_dynamic_host_slots() {
        let start = 0x1000usize;
        let step = 32;
        let capped = 5;
        assert_eq!(
            changed_slots(
                start + 3 * step + 4,
                start + 4 * step + 8,
                start,
                step as u64,
                capped
            ),
            3..5
        );
        assert_eq!(
            changed_slots(
                start + 6 * step,
                start + 9 * step,
                start,
                step as u64,
                capped
            ),
            5..5
        );
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc {
            kind: 0,
            count: 1000,
            flags: 1,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    /// A descriptor copied into a placed heap as bytes is mirrored before
    /// the next submission, even though no descriptor method saw the copy.
    #[cfg(windows)]
    #[test]
    fn records_copied_as_bytes_are_made_at_the_next_sync() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let memory = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                crate::watch::PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        };
        assert!(!memory.is_null());
        let desc = HeapDesc {
            kind: SAMPLERS,
            count: 64,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let placement = Placement {
            address: memory as u64,
            size: 64 * 16,
        };
        let mut h: This = std::ptr::null();
        assert_eq!(create(desc, placement, &mut h), S_OK);
        if let Some(host) = heap(h).host() {
            let mut elsewhere = [0u64; 2];
            let source = elsewhere.as_mut_ptr() as u64;
            let sampler = [21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
            create_sampler(std::ptr::null(), &sampler, source);
            // SAFETY: the heap slot and source both hold 16 descriptor bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source as *const u8,
                    memory.cast::<u8>().add(5 * 16),
                    16,
                )
            };
            assert!(!host.made(5));
            sync_heaps();
            assert!(host.made(5));
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        // SAFETY: this allocation belongs to the test.
        unsafe { VirtualFree(memory, 0, MEM_RELEASE) };
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 4 * 8]; // four 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3) = (at(0), at(1), at(2), at(3));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0), Some(View::Constant(0x1_0000_1000, 256)));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1), Some(View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 3]) }, unsafe {
            *(s2 as *const [u32; 3])
        });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(
            null,
            1,
            dst.as_ptr(),
            dst_n.as_ptr(),
            2,
            src.as_ptr(),
            src_n.as_ptr(),
            0,
        );
        assert_eq!(view(s2), view(s0));
        assert_eq!(view(s3), view(s1));
        assert_eq!(view(at(0) + 8), None);
        heap[0] = 0;
    }

    /// Constant buffer views are named whole in their records: any address
    /// a program can have, any size below 16 KiB, multiples of 256 bytes
    /// up to 4 MiB; what does not fit says so.
    #[test]
    fn constant_views_fit_their_records() {
        let top = (1u64 << CONSTANT_ADDRESS_BITS) - 1;
        for address in [0u64, 0x100, 0x2602_0000, 0xd_6b60_0000, 0x7ffd_a34f_0000, top] {
            for size in [0u32, 1, 255, 256, 4000, (1 << 14) - 1, 1 << 14, 65536, (1 << 22) - 256] {
                let [i, generation] = constant_words(address, size).expect("fits");
                assert_eq!(generation >> KIND_SHIFT, CONSTANT);
                assert_eq!(view_at(i, generation), Some(View::Constant(address, size)), "{address:#x} {size}");
            }
        }
        assert_eq!(constant_words(top + 1, 256), None, "an address past 47 bits");
        assert_eq!(constant_words(0x1000, (1 << 14) + 1), None, "16 KiB or more, no multiple of 256");
        assert_eq!(constant_words(0x1000, 1 << 22), None, "4 MiB");
        // Different views never share their words.
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1000, 512));
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1100, 256));
        assert_ne!(constant_words(0x1000, 1 << 14), constant_words(0x1000, 64));
    }

    /// The leak of run 65: a title that writes constant buffer views and
    /// views of placed memory at ever new addresses adds nothing to the
    /// table; one shape stands for all the placed views alike.
    #[test]
    fn views_at_new_addresses_add_no_entries() {
        if !inline() {
            return;
        }
        let null = std::ptr::null();
        let slots = [0u64; 4];
        let at = slots.as_ptr() as u64;
        // A resource description and a view no other test uses.
        let mut resource: [u32; 14] = std::array::from_fn(|k| 0x5ee0_0000 + k as u32);
        // (LOCAL (views): views come back with the filler words cleared.)
        (resource[1], resource[13]) = (0, 0);
        let mut desc: [u32; 16] = std::array::from_fn(|k| 0x7a70_0000 + k as u32);
        // (LOCAL (views2): word 3 of a shader view is padding, cleared in a shape.)
        desc[3] = 0;
        let mine = |v: &View| match v {
            View::Constant(address, _) => (0x6_0000_0000..0x6_1000_0000).contains(address),
            View::Placed(_, r, _) => *r == resource,
            _ => false,
        };
        let entries = || VIEWS.read().unwrap().list.iter().filter(|(v, _)| mine(v)).count();
        let shapes = || VIEWS.read().unwrap().shapes.iter().filter(|s| s.1 == resource).count();
        for n in 0..20_000u64 {
            let address = 0x6_0000_0000 + n * 0x100;
            create_cbv(null, &[address as u32, (address >> 32) as u32, 512, 0], at);
            assert_eq!(view(at), Some(View::Constant(address, 512)));
            let placed = 0xd_4940_0000 + n * 0x1_0000;
            (desc[10], desc[11]) = (placed as u32, (placed >> 32) as u32);
            create_placed_srv(null, &resource, &desc, at);
            assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
        }
        assert_eq!(entries(), 0, "no table entries");
        assert_eq!(shapes(), 1, "one shape for 20,000 placed views");
        // The same shape, unordered: another shape.
        create_placed_uav(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(true, resource, desc)));
        assert_eq!(shapes(), 2);
        // Placed memory past 1 TiB (not title memory) is a table entry.
        (desc[10], desc[11]) = (0, 0x100);
        create_placed_srv(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
        assert_eq!(entries(), 1);
        // A record's words are the same for the same view (the host knows
        // a slot's view by them), and differ with the address.
        // SAFETY: both handles are slots of 32 bytes above.
        let words = |handle: u64| unsafe { *(handle as *const [u32; 3]) };
        let other = [0u64; 4];
        create_cbv(null, &[0x40, 6, 512, 0], at);
        create_cbv(null, &[0x40, 6, 512, 0], other.as_ptr() as u64);
        assert_eq!(words(at), words(other.as_ptr() as u64));
        create_cbv(null, &[0x140, 6, 512, 0], other.as_ptr() as u64);
        assert_ne!(words(at), words(other.as_ptr() as u64));
    }

    /// LOCAL (views): a table to try on without the global one.
    fn fresh(shape_cap: usize) -> std::sync::RwLock<Views> {
        std::sync::RwLock::new(Views {
            list: Vec::new(),
            index: Default::default(),
            by_resource: Default::default(),
            free: Vec::new(),
            shapes: Vec::new(),
            shape_index: Default::default(),
            shape_cap,
            lods: Vec::new(),
            lod_index: Default::default(),
            lod_cap: LODS,
            lod_shapes: Vec::new(),
            lod_shape_index: Default::default(),
            lod_shape_cap: LOD_SHAPES,
            stamps: Vec::new(),
            live_constants: 0,
            live_placed: 0,
        })
    }

    /// LOCAL (views): the entries the index names, by kind, as a scan finds
    /// them.
    fn scanned(v: &Views) -> (u64, u64) {
        let (mut constants, mut placed) = (0, 0);
        for (i, (view, _)) in v.list.iter().enumerate() {
            if v.index.get(view) != Some(&(i as u32)) {
                continue;
            }
            match view {
                View::Constant(..) => constants += 1,
                View::Placed(..) => placed += 1,
                _ => {}
            }
        }
        (constants, placed)
    }

    /// LOCAL (views): views of one texture that differ only in the filler
    /// words of its resource description (the title's stack) share a shape
    /// and come back alike.
    #[test]
    fn filler_words_make_no_new_shapes() {
        if !inline() || std::env::var_os("VICTORY_PLACED_FILLER").is_some_and(|v| v == "1") {
            return;
        }
        let null = std::ptr::null();
        let slots = [0u64; 4];
        let at = slots.as_ptr() as u64;
        let mut resource: [u32; 14] = std::array::from_fn(|k| 0x5ee1_0000 + k as u32);
        let mut desc: [u32; 16] = std::array::from_fn(|k| 0x7a71_0000 + k as u32);
        desc[3] = 0;
        (desc[10], desc[11]) = (0x1000, 0);
        let shapes = |r: &[u32; 14]| {
            let mut r = *r;
            (r[1], r[13]) = (0, 0);
            VIEWS.read().unwrap().shapes.iter().filter(|s| s.1 == r).count()
        };
        let mut first = None;
        for n in 0..1000u32 {
            resource[1] = n.wrapping_mul(0x9e37_79b9);
            resource[13] = !n;
            create_placed_srv(null, &resource, &desc, at);
            // SAFETY: a slot above.
            let words = unsafe { *(at as *const [u32; 3]) };
            assert_eq!(*first.get_or_insert(words), words, "one record for the one view");
            let mut cleared = resource;
            (cleared[1], cleared[13]) = (0, 0);
            assert_eq!(view(at), Some(View::Placed(false, cleared, desc)));
        }
        assert_eq!(shapes(&resource), 1);
        // A word that matters still makes a shape of its own.
        resource[6] += 1;
        create_placed_srv(null, &resource, &desc, at);
        assert_eq!(shapes(&resource), 1, "(counted by its own description)");
        assert_eq!(VIEWS.read().unwrap().shapes.iter().filter(|s| s.1[6] == resource[6]).count(), 1);
    }

    /// LOCAL (views2): a 2D array shader view description (format 28,
    /// dimension 5: most detailed mip 0, 10 levels, slice 0, 1 slice, plane
    /// 0, clamp word `lod`) at `address`.
    fn array_view(lod: u32, address: u64) -> [u32; 16] {
        let mut desc = [0u32; 16];
        desc[..10].copy_from_slice(&[28, 5, 0x1688, 0, 0, 10, 0, 1, 0, lod]);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        desc
    }

    /// LOCAL (views2): views of one texture that differ only in the
    /// clamp (which the title's streaming moves all the time) and in the
    /// padding word make no new shapes; every view comes back exactly as it
    /// was made (but for the padding), and a word that matters still makes a
    /// shape of its own.
    #[test]
    fn min_lod_clamp_and_padding_make_no_new_shapes() {
        if !inline() || !split_keys() {
            return;
        }
        let null = std::ptr::null();
        let slots = [0u64; 4];
        let at = slots.as_ptr() as u64;
        let mut resource: [u32; 14] = std::array::from_fn(|k| 0x5ee2_0000 + k as u32);
        resource[0] = 3;
        let shapes = |r: &[u32; 14]| {
            let v = VIEWS.read().unwrap();
            v.shapes.iter().chain(&v.lod_shapes).filter(|s| s.1[2] == r[2]).collect::<std::collections::BTreeSet<_>>().len()
        };
        let mut seen = std::collections::BTreeMap::new();
        for n in 0..5000u32 {
            // Clamps from 0.0 (none) up: the first is no clamp at all.
            let lod = if n == 0 { 0 } else { (0.0067 + n as f32 * 0.00013).to_bits() };
            let address = 0x2_0000_0000 + u64::from(n % 50) * 0x1_0000;
            let mut desc = array_view(lod, address);
            desc[3] = 0x100 + n;
            create_placed_srv(null, &resource, &desc, at);
            // SAFETY: a slot above.
            let words = unsafe { *(at as *const [u32; 3]) };
            if let Some(&before) = seen.get(&(lod, address)) {
                assert_eq!(before, words, "the same view, the same record");
            }
            seen.insert((lod, address), words);
            let mut shown = desc;
            shown[3] = 0;
            let Some(View::Placed(false, r, d)) = view(at) else { panic!("not a placed shader view") };
            assert_eq!((r, d), (resource_cleared(resource), shown), "n = {n}");
            assert_eq!(f32::from_bits(d[9]), f32::from_bits(lod));
        }
        // Distinct views have distinct records.
        let distinct: std::collections::BTreeSet<_> = seen.values().collect();
        assert_eq!(distinct.len(), seen.len());
        assert_eq!(shapes(&resource), 1, "one shape for views of 5,000 clamps");
        // Words that matter: the width (word 5 of a shape), the mip range, the dimension.
        let mut wider = resource;
        wider[4] *= 2;
        create_placed_srv(null, &wider, &array_view(0x3c7d_3fff, 0x2_0000_0000), at);
        assert_eq!(shapes(&resource), 2, "(counted by resource word 2)");
        let mut mips = array_view(0x3c7d_3fff, 0x2_0000_0000);
        mips[5] = 4;
        create_placed_srv(null, &resource, &mips, at);
        assert_eq!(VIEWS.read().unwrap().lod_shapes.iter().filter(|s| s.1[2] == resource[2] && s.2[5] == 4).count(), 1);
        assert_eq!(view(at), Some(View::Placed(false, resource_cleared(resource), mips)));
        let mut first_slice = array_view(0x3c7d_3fff, 0x2_0000_0000);
        first_slice[6] = 3;
        create_placed_srv(null, &resource, &first_slice, at);
        assert_eq!(view(at), Some(View::Placed(false, resource_cleared(resource), first_slice)));
        // The padding of an unordered view is no padding: it is part of the view.
        let mut uav = array_view(0, 0x2_0000_0000);
        uav[1] = 4;
        let before = shapes(&resource);
        for pad in [1, 2] {
            uav[3] = pad;
            create_placed_uav(null, &resource, &uav, at);
            assert_eq!(view(at), Some(View::Placed(true, resource_cleared(resource), uav)));
        }
        assert_eq!(shapes(&resource), before + 2);
    }

    /// LOCAL (views2): `resource` as a view of it comes back (filler cleared).
    fn resource_cleared(mut resource: [u32; 14]) -> [u32; 14] {
        (resource[1], resource[13]) = (0, 0);
        resource
    }

    /// LOCAL (views2): the view a record's two words name in `views`.
    fn decoded(views: &std::sync::RwLock<Views>, [i, generation]: [u32; 2]) -> Option<View> {
        let v = views.read().unwrap();
        match generation >> KIND_SHIFT {
            PLACED => v.placed_of(i, generation),
            LOD_PLACED => v.lod_placed_of(i, generation),
            _ => None,
        }
    }

    /// LOCAL (views2): a placed view's record in `views` and what it names.
    fn through(views: &std::sync::RwLock<Views>, resource: &[u32; 14], desc: &[u32; 16]) -> (Option<[u32; 2]>, Option<View>) {
        let words = placed_words_in(views, false, resource, desc);
        (words, words.and_then(|w| decoded(views, w)))
    }

    /// LOCAL (views2): the clamps and the bare shapes that [`LOD_PLACED`]
    /// records name are tables of their own, and what does not fit them
    /// (address, bare shapes, clamps) falls back to a shape that holds the
    /// clamp: exact, and counted by cause. With the clamp table full the
    /// number of shapes stays flat while views decode exactly.
    #[test]
    fn min_lod_clamp_tables_fill_and_fall_back_exactly() {
        if !split_keys() {
            return;
        }
        let lock = fresh(SHAPES);
        {
            let mut v = lock.write().unwrap();
            (v.lod_cap, v.lod_shape_cap) = (50, 2);
        }
        let mut resource = [7u32; 14];
        (resource[1], resource[13]) = (0, 0);
        let (a, b, c) = (0x3bdf_7fff, 0x3c7d_3fff, 0x3c7d_4000);
        let exact = |desc: [u32; 16]| Some(View::Placed(false, resource, desc));
        // Named: the 64 KiB boundaries, up to 1 TiB.
        for (lod, address) in [(a, 0x1_0000), (b, 0x20_0000_0000), (a, 0xff_ffff_0000), (b, 0x1_0000)] {
            let desc = array_view(lod, address);
            let (words, view) = through(&lock, &resource, &desc);
            assert_eq!(words.unwrap()[1] >> KIND_SHIFT, LOD_PLACED);
            assert_eq!(view, exact(desc));
        }
        assert_eq!(lock.read().unwrap().lods, [a, b]);
        assert_eq!(lock.read().unwrap().lod_shapes.len(), 1);
        assert_eq!(lock.read().unwrap().shapes.len(), 0, "the global shapes are not used");
        // Not named, by cause (nothing is added for them), and still exact.
        let bare = |desc: &[u32; 16]| shape_of(false, &resource, &lod_split(false, desc).unwrap().0).0;
        for address in [0x1000, 0x1_0010, 0x1_8000, 1 << 40] {
            let desc = array_view(a, address);
            assert_eq!(lod_named(&lock, bare(&desc), address, a), Err(LodMiss::Address), "{address:#x}");
            let (words, view) = through(&lock, &resource, &desc);
            if address >> 40 == 0 {
                assert_eq!(words.unwrap()[1] >> KIND_SHIFT, PLACED, "a shape holds the clamp");
                assert_eq!(view, exact(desc));
            } else {
                assert_eq!(words, None, "past 1 TiB: a table entry");
            }
        }
        assert_eq!((lock.read().unwrap().lods.len(), lock.read().unwrap().lod_shapes.len()), (2, 1));
        // The clamps fill (cap 50: 48 more); the shapes of the global table stay as they are.
        let shapes_before = lock.read().unwrap().shapes.len();
        for n in 0..48u32 {
            let desc = array_view(0x3c00_0000 + n, 0x1_0000);
            assert_eq!(through(&lock, &resource, &desc).1, exact(desc));
        }
        assert_eq!(lock.read().unwrap().lods.len(), 50);
        assert_eq!(lock.read().unwrap().shapes.len(), shapes_before, "no shape for a clamp that fits");
        // A clamp past the cap: not named, nothing added, exact in a shape that holds it.
        let desc = array_view(c, 0x1_0000);
        assert_eq!(lod_named(&lock, bare(&desc), 0x1_0000, c), Err(LodMiss::Clamps));
        let (words, view) = through(&lock, &resource, &desc);
        assert_eq!(words.unwrap()[1] >> KIND_SHIFT, PLACED);
        assert_eq!(view, exact(desc));
        assert_eq!((lock.read().unwrap().lods.len(), lock.read().unwrap().lod_shapes.len()), (50, 1));
        // A second bare shape fits (2), a third is refused and not added, and
        // a known clamp does not matter then.
        let mut other = resource;
        other[4] += 1;
        let shown = |r: [u32; 14], d| Some(View::Placed(false, r, d));
        let d2 = array_view(a, 0x1_0000);
        assert_eq!(through(&lock, &other, &d2).1, shown(other, d2));
        let mut third = resource;
        third[4] += 2;
        let sh = shape_of(false, &third, &lod_split(false, &d2).unwrap().0).0;
        assert_eq!(lod_named(&lock, sh, 0x1_0000, a), Err(LodMiss::Shapes));
        assert_eq!(through(&lock, &third, &d2).1, shown(third, d2));
        assert_eq!(lock.read().unwrap().lod_shapes.len(), 2);
        // A known bare shape with a known clamp is named even with both tables full.
        let (words, view) = through(&lock, &resource, &array_view(b, 0x3_0000));
        assert_eq!(words.unwrap()[1] >> KIND_SHIFT, LOD_PLACED);
        assert_eq!(view, exact(array_view(b, 0x3_0000)));
        // A shape that does not name a clamp (none, a buffer, unordered access) is as it is.
        assert!(lod_split(false, &array_view(0, 0x1_0000)).is_none());
        let mut buffer = array_view(c, 0x1_0000);
        buffer[1] = 1;
        assert!(lod_split(false, &buffer).is_none());
        assert!(lod_split(true, &array_view(c, 0x1_0000)).is_none());
        // The clamp is where the dimension puts it.
        for (dimension, word) in [(2, 6), (3, 8), (4, 7), (5, 9), (8, 6), (9, 6), (10, 8)] {
            let mut desc = [0u32; 16];
            (desc[1], desc[word]) = (dimension, c);
            let (bare, bits) = lod_split(false, &desc).unwrap();
            assert_eq!((bits, bare[word]), (c, 0), "dimension {dimension}");
        }
        // A negative zero is a clamp word like another (it is not 0 bits).
        assert_eq!(lod_split(false, &array_view(0x8000_0000, 0x3_0000)).map(|s| s.1), Some(0x8000_0000));
        // The record has room for the numbers it holds.
        assert_eq!(LOD_ADDRESS_BITS + LOD_SHAPE_BITS + 21, 62);
        assert_eq!(LODS, 1 << 21);
    }

    /// LOCAL (views2): the union words past the members of a view's dimension
    /// are no part of a shape (a 2D view has words 4-7); the address and the
    /// console words are.
    #[test]
    fn union_tail_words_make_no_new_shapes() {
        if !split_keys() {
            return;
        }
        let lock = fresh(SHAPES);
        let resource = [9u32; 14];
        let mut desc = array_view(0x3c7d_3fff, 0x1_0000);
        (desc[1], desc[7], desc[9]) = (4, 0x3c7d_3fff, 0);
        desc[6] = 0; // plane
        let mut first = None;
        for n in 1..100u32 {
            let mut d = desc;
            (d[8], d[9]) = (n, n * 3);
            let (words, view) = through(&lock, &resource, &d);
            assert_eq!(*first.get_or_insert(words), words);
            assert_eq!(view, Some(View::Placed(false, resource_cleared(resource), desc)));
        }
        // The console words (12-15) do matter.
        let mut d = desc;
        d[12] = 1;
        assert_eq!(through(&lock, &resource, &d).1, Some(View::Placed(false, resource_cleared(resource), d)));
        assert_eq!(lock.read().unwrap().lod_shapes.len(), 2);
    }

    /// LOCAL (views): which word differs between two shapes.
    #[test]
    fn single_difference_names_the_word() {
        let a: Shape = (false, [1; 14], [2; 14]);
        let mut b = a;
        assert_eq!(single_difference(&a, &b), None, "alike");
        b.1[13] = 9;
        assert_eq!(single_difference(&a, &b), Some((14, 1, 9)));
        b.2[0] = 7;
        assert_eq!(single_difference(&a, &b), None, "two words");
        let mut c = a;
        c.2[0] = 5;
        assert_eq!(single_difference(&a, &c), Some((15, 2, 5)));
    }

    /// LOCAL (views): with the shapes full, a shape known or not is
    /// answered with the lock shared: a reader holding it does not hold
    /// the asker up (a write would wait for the reader).
    #[test]
    fn full_shapes_take_no_write_lock() {
        let shape = |n: u32| -> Shape { (false, [n; 14], [n; 14]) };
        let views = std::sync::Arc::new(fresh(2));
        assert_eq!(shape_in(&views, shape(1)), Some((0, true)));
        assert_eq!(shape_in(&views, shape(2)), Some((1, true)));
        assert_eq!(shape_in(&views, shape(1)), Some((0, false)));
        let reader = views.read().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let asked = views.clone();
        std::thread::spawn(move || {
            let answers = (shape_in(&asked, shape(3)), shape_in(&asked, shape(2)), shape_in(&asked, shape(4)));
            let _ = tx.send(answers);
        });
        let answers = rx.recv_timeout(std::time::Duration::from_secs(5)).expect("asked without a write lock");
        assert_eq!(answers, (None, Some((1, false)), None));
        drop(reader);
        assert_eq!(views.read().unwrap().shapes.len(), 2, "nothing was added");
    }

    /// LOCAL (views): a sweep frees only entries without a resource that
    /// nothing has used for long; one used lately keeps its entry and
    /// generation; the counters agree with a scan throughout.
    #[test]
    fn sweep_never_frees_an_entry_in_use() {
        use std::sync::atomic::Ordering::Relaxed;
        let lock = fresh(0);
        let mut v = lock.write().unwrap();
        let placed = |n: u32| View::Placed(false, [n; 14], [0; 16]);
        let old_c = v.entry(View::Constant(0x10, 4));
        let old_p = v.entry(placed(1));
        let used_c = v.entry(View::Constant(0x20, 4));
        let used_p = v.entry(placed(2));
        let texture = v.entry(View::Shader(0x4444, None));
        let sampler = v.entry(View::Sampler([5; 13]));
        assert_eq!(scanned(&v), (2, 2));
        assert_eq!((v.live_constants, v.live_placed), scanned(&v));
        // Stamps: two used a moment ago, the others long ago.
        for (words, stamp) in [(old_c, 100), (old_p, 100), (used_c, 990), (used_p, 995), (texture, 100), (sampler, 100)] {
            v.stamps[words[0] as usize].store(stamp, Relaxed);
        }
        assert_eq!(v.sweep(1000, 120), 2);
        assert_eq!((v.live_constants, v.live_placed), (1, 1));
        assert_eq!((v.live_constants, v.live_placed), scanned(&v));
        // The ones in use name the same entries as before.
        assert_eq!(v.entry_of(&View::Constant(0x20, 4)), Some(used_c));
        assert_eq!(v.entry_of(&placed(2)), Some(used_p));
        // The ones swept are gone from the index and stale (their generation moved on).
        assert_eq!(v.entry_of(&View::Constant(0x10, 4)), None);
        assert_eq!(v.entry_of(&placed(1)), None);
        assert_ne!(v.list[old_c[0] as usize].1, old_c[1]);
        assert_ne!(v.list[old_p[0] as usize].1, old_p[1]);
        // A texture and a sampler are not swept.
        assert_eq!(v.entry_of(&View::Shader(0x4444, None)), Some(texture));
        assert_eq!(v.entry_of(&View::Sampler([5; 13])), Some(sampler));
        // A sweep again frees nothing; a freed entry is reused under a new generation.
        assert_eq!(v.sweep(1000, 120), 0);
        let again = v.entry(View::Constant(0x30, 4));
        let before = if again[0] == old_c[0] { old_c } else { old_p };
        assert_eq!(again[0], before[0], "a freed entry is reused");
        assert_ne!(again[1], before[1]);
        assert_eq!((v.live_constants, v.live_placed), scanned(&v));
        // Everything idle for long enough goes (but the texture and the sampler).
        assert_eq!(v.sweep(5000, 120), 3);
        assert_eq!((v.live_constants, v.live_placed), (0, 0));
        assert_eq!(scanned(&v), (0, 0));
        assert_eq!(v.entry_of(&View::Shader(0x4444, None)), Some(texture));
    }

    /// A released resource's views go stale; the same view of a new
    /// resource at the same address is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = 0x7777_0000usize as This;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Shader(0x7777_0000, None)));
        forget(0x7777_0000);
        assert_eq!(view(a), None, "stale after release");
        create_srv(null, resource, std::ptr::null(), b);
        assert_eq!(view(b), Some(View::Shader(0x7777_0000, None)));
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 3]), *(b as *const [u32; 3])) };
        assert_ne!(old, new, "a new generation");
    }

    /// The title places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        assert_eq!(
            create(
                desc,
                Placement {
                    address,
                    size: 64 * 32
                },
                &mut h
            ),
            S_OK
        );
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
    }
}
