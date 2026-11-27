//! LOCAL (journal): the streaming journal (`VICTORY_STREAM_JOURNAL=1`; off by default, and then
//! nothing here runs and nothing else changes: every hook is one load and a branch).
//!
//! DirectStorage (`victory-hle` `dstorage`) writes every byte the title streams in, so it knows
//! what changed. When a request's bytes are in title memory, before its completion is told, the
//! ranges it wrote (title addresses; a page-list destination at every address the title shows its
//! pages, `xmem::shown_at`) go into this journal, stamped with a number of the write-watch askings'
//! count ([`crate::watch::stamp`]): a stamp above the asking a copy was taken after says the bytes
//! came after (or while) the copy read them. What reads title memory for the card asks the journal
//! instead of searching, but only ever to do *more* (copy without comparing, read now rather than
//! later), never to skip finding a write of the title's:
//!
//! - Guarded views (pool pages): the blocks under a page write are told as written *beside* their
//!   guards ([`crate::watch::writing_beside`]: read-only and clean still, the next asking takes
//!   the bit and tells them written, as for the title's bytes put in beside a guard,
//!   `watch::put_beside`) instead of opened with `VirtualProtect` and guarded again at the next
//!   asking. No write of the title's is lost: the block stays read-only, so its next write faults
//!   as before; the asking that comes before the mark came before the request's completion.
//! - Refreshes ([`fresh_parts`], `shadow::Shadow::copy_stale`): of the pages the watch or the
//!   guards found written, the 4 KiB parts the journal says DirectStorage wrote after the copy
//!   took them are copied whole, without the compare that would find which dwords changed
//!   (`VICTORY_SEND_CHANGED`, and the landing echo check). Copying bytes that are equal is no
//!   change, so this is safe whatever else wrote there; the watch is still asked.
//! - Textures only partly mapped (streaming in) were read again at most every 30 frames: one the
//!   journal says was written since its last read is read at its next refresh
//!   ([`any_fresh`]).
//! - Write watch (`VICTORY_STREAM_JOURNAL_RESET=1` as well; off by default): what DirectStorage
//!   wrote through an address of write-watched memory is taken off the kernel's dirty bits into
//!   the watch's own journal at once ([`crate::watch::take_over_written`]), so the submission's
//!   asking finds (and resets) fewer pages. Safe because it is an asking in all but name: the
//!   range is held against other askings, the kernel's bits are reset and the pages noted written
//!   under a new asking's number, so a write of the title's before the reset is in that note and
//!   one after it is in the kernel's bits.
//!
//! The producers are DirectStorage's 32 workers: each appends to one of [`SHARDS`] buffers (its
//! own, mostly); the buffers are drained into one map when a consumer asks or at the period's
//! line. A consumer that finds the map busy, or an entry pruned, does what it did without the
//! journal. Nothing here reads title memory.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock, RwLock};

/// `VICTORY_STREAM_JOURNAL=1` (off unless set); with `VICTORY_AB` naming it (diagnostics builds),
/// on and off while the title runs. Turning it either way is safe at any moment: what is not in
/// the journal is done as without it.
pub fn on() -> bool {
    #[cfg(test)]
    if FORCED.with(std::cell::Cell::get) {
        return true;
    }
    #[cfg(feature = "diagnostics")]
    {
        static BIT: OnceLock<Result<bool, u32>> = OnceLock::new();
        match BIT.get_or_init(|| match crate::ab::bit("VICTORY_STREAM_JOURNAL") {
            Some(bit) => Err(bit),
            None => Ok(crate::switch::on("VICTORY_STREAM_JOURNAL")),
        }) {
            Ok(on) => *on,
            Err(bit) => crate::ab::on(*bit),
        }
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| crate::switch::on("VICTORY_STREAM_JOURNAL"))
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: on for this thread whatever the switch says.
    pub(crate) static FORCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `VICTORY_STREAM_JOURNAL_RESET=1`: with the journal, DirectStorage's writes into write-watched
/// memory are taken off the kernel's bits at once ([`crate::watch::take_over_written`]). Off by
/// default: it moves the resets from the submitting thread to the workers, one call (and one
/// flush of the processors' address caches) per request instead of one per asking.
fn reset_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_STREAM_JOURNAL_RESET"))
}

// ---------------------------------------------------------------- the ranges

/// Written ranges of title memory, apart and in order: start → (end, stamp). Where two writes
/// overlap the later stamp stands; touching ranges of one stamp are one.
#[derive(Default)]
pub(crate) struct Ranges {
    map: BTreeMap<usize, (usize, u64)>,
}

impl Ranges {
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    /// `[start, end)` was written, stamped `stamp`.
    pub(crate) fn insert(&mut self, start: usize, end: usize, stamp: u64) {
        if start >= end {
            return;
        }
        // The entries that overlap or touch it, in order (apart, so only the first can begin
        // before `start` and only the last end after `end`).
        let begin = self.map.range(..=start).next_back().map_or(start, |(&s, _)| s);
        let keys: Vec<usize> =
            self.map.range(begin..=end).filter(|&(&s, &(e, _))| e >= start && s <= end).map(|(&s, _)| s).collect();
        let mut out: Vec<(usize, usize, u64)> = Vec::with_capacity(keys.len() + 2);
        let push = |out: &mut Vec<(usize, usize, u64)>, a: usize, b: usize, s: u64| {
            if a >= b {
                return;
            }
            match out.last_mut() {
                Some(last) if last.1 == a && last.2 == s => last.1 = b,
                _ => out.push((a, b, s)),
            }
        };
        let (mut at, mut tail) = (start, None);
        for k in keys {
            let Some((e, s)) = self.map.remove(&k) else { continue };
            push(&mut out, k, e.min(start), s);
            let (oa, ob) = (k.max(start), e.min(end));
            if oa < ob {
                push(&mut out, at, oa, stamp);
                push(&mut out, oa, ob, s.max(stamp));
                at = ob;
            }
            if e > end {
                tail = Some((end.max(k), e, s));
            }
        }
        push(&mut out, at, end, stamp);
        if let Some((a, e, s)) = tail {
            push(&mut out, a, e, s);
        }
        for (a, b, s) in out {
            self.map.insert(a, (b, s));
        }
    }

    /// The entries that overlap `[from, to)`, cut to it, in order.
    fn within(&self, from: usize, to: usize) -> impl Iterator<Item = (usize, usize, u64)> + '_ {
        let begin = self.map.range(..=from).next_back().map_or(from, |(&s, _)| s);
        self.map
            .range(begin..to)
            .filter(move |&(_, &(e, _))| e > from)
            .map(move |(&s, &(e, st))| (s.max(from), e.min(to), st))
    }

    /// The parts of `[from, to)` written with a stamp above `since`, in order.
    pub(crate) fn fresh(&self, from: usize, to: usize, since: u64) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::new();
        for (a, b, s) in self.within(from, to) {
            if s <= since || a >= b {
                continue;
            }
            match out.last_mut() {
                Some(last) if last.1 == a => last.1 = b,
                _ => out.push((a, b)),
            }
        }
        out
    }

    /// Whether a byte of `[from, to)` was written with a stamp above `since`.
    pub(crate) fn any_fresh(&self, from: usize, to: usize, since: u64) -> bool {
        self.within(from, to).any(|(a, b, s)| s > since && a < b)
    }

    /// Forgets the entries stamped below `floor`; how many.
    pub(crate) fn prune(&mut self, floor: u64) -> usize {
        let before = self.map.len();
        self.map.retain(|_, &mut (_, s)| s >= floor);
        before - self.map.len()
    }

    /// Forgets the older half (by stamp); how many went.
    fn halve(&mut self) -> usize {
        let mut stamps: Vec<u64> = self.map.values().map(|v| v.1).collect();
        if stamps.is_empty() {
            return 0;
        }
        let middle = stamps.len() / 2;
        let (_, &mut floor, _) = stamps.select_nth_unstable(middle);
        self.prune(floor)
    }
}

/// The 4 KiB parts (a bit each, of the sixteen of a 64 KiB page) lying wholly in `runs`, of the
/// page at `page_at`.
pub(crate) fn whole_parts(page_at: usize, runs: &[(usize, usize)]) -> u16 {
    const PART: usize = 4096;
    let mut mask = 0u16;
    for &(a, b) in runs {
        let (lo, hi) = (a.div_ceil(PART) * PART, b / PART * PART);
        let mut at = lo.max(page_at);
        while at < hi.min(page_at + crate::watch::PAGE) {
            mask |= 1 << ((at - page_at) / PART);
            at += PART;
        }
    }
    mask
}

// ---------------------------------------------------------------- the journal

/// Producers' buffers: a worker takes one by its thread's number.
const SHARDS: usize = 16;
static SHARD: [Mutex<Vec<(usize, usize, u64)>>; SHARDS] = [const { Mutex::new(Vec::new()) }; SHARDS];
/// Entries in the buffers, not yet in [`MERGED`].
static PENDING: AtomicUsize = AtomicUsize::new(0);
/// The highest stamp journaled (pending ones too): a copy taken after it has nothing newer.
static NEWEST: AtomicU64 = AtomicU64::new(0);
/// The journal.
static MERGED: RwLock<Ranges> = RwLock::new(Ranges { map: BTreeMap::new() });
/// Something was journaled (the line is written from then on, also while switched off).
static SEEN: AtomicBool = AtomicBool::new(false);
/// Pending entries past which a producer drains them itself, and the most the journal holds
/// before its older half is forgotten.
const PENDING_MOST: usize = 1 << 14;
const HELD_MOST: usize = 1 << 17;

thread_local! {
    static MINE: usize = {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        NEXT.fetch_add(1, Relaxed) % SHARDS
    };
}

fn record(at: usize, len: usize, stamp: u64) {
    let end = at.saturating_add(len);
    let i = MINE.try_with(|i| *i).unwrap_or(0);
    SHARD[i].lock().unwrap_or_else(|e| e.into_inner()).push((at, end, stamp));
    NEWEST.fetch_max(stamp, Relaxed);
    SEEN.store(true, Relaxed);
    count(C::Bytes, len as u64);
    count(C::Ranges, 1);
    // LOCAL (journal): a drain can subtract before this add; wrap, never panic.
    if PENDING.fetch_add(1, Relaxed).wrapping_add(1) >= PENDING_MOST
        && let Ok(mut m) = MERGED.try_write()
    {
        drain_into(&mut m);
    }
}

fn drain_into(m: &mut Ranges) {
    for shard in &SHARD {
        let taken = std::mem::take(&mut *shard.lock().unwrap_or_else(|e| e.into_inner()));
        if taken.is_empty() {
            continue;
        }
        PENDING.fetch_sub(taken.len(), Relaxed);
        for (a, b, s) in taken {
            m.insert(a, b, s);
        }
    }
    if m.len() > HELD_MOST {
        count(C::Pruned, m.halve() as u64);
    }
}

/// The journal, with what the producers appended since it was last asked (if nobody else is
/// putting that in now: then without, which only makes it say less).
fn merged() -> std::sync::RwLockReadGuard<'static, Ranges> {
    if PENDING.load(Relaxed) != 0
        && let Ok(mut m) = MERGED.try_write()
    {
        drain_into(&mut m);
    }
    MERGED.read().unwrap_or_else(|e| e.into_inner())
}

/// The stamp a consumer notes before it reads title memory: what is journaled later is newer
/// than what it read.
pub(crate) fn now() -> u64 {
    crate::watch::asked_now()
}

// ---------------------------------------------------------------- producers

/// DirectStorage's bytes are in title memory `[at, at + len)`, written through that address (a
/// destination given as an address; the guards were opened for it, `kernel_writing`, and that is
/// over). Call when the bytes are there, before the request's completion is told. The caller
/// still tells the GPU side as before (`kernel_will_write`).
pub fn wrote_raw(at: usize, len: usize) {
    if !on() || len == 0 {
        return;
    }
    record(at, len, crate::watch::stamp());
    count(C::RawBytes, len as u64);
    if reset_on() {
        use crate::watch::Refused;
        match crate::watch::take_over_written(at, len) {
            Ok(pages) => {
                count(C::Resets, 1);
                count(C::ResetPages, pages);
            }
            Err(Refused::Small) => count(C::RefusedSmall, 1),
            Err(Refused::Guarded) => count(C::RefusedGuarded, 1),
            Err(Refused::Busy) => count(C::RefusedBusy, 1),
            Err(Refused::NotWatched) => count(C::RefusedNotWatched, 1),
        }
    }
}

/// DirectStorage's bytes are in physical pages the title shows at `[at, at + len)`, written
/// through a view of the runtime's own (`xmem::pages_written`, for each address showing them).
/// Does what the caller did without the journal (`notice`: `xmem::stream_notice`, then
/// `kernel_will_write`, else `title_memory_written(.., true)`), with the guarded blocks under it
/// told as written beside their guards instead of opened, where that can be done.
pub fn wrote_pages(at: usize, len: usize, notice: bool) {
    if len == 0 {
        return;
    }
    record(at, len, crate::watch::stamp());
    count(C::PageBytes, len as u64);
    if notice {
        crate::watch::writing_beside(at, len);
    } else {
        crate::watch::title_memory_written_beside(at, len);
    }
}

// ---------------------------------------------------------------- consumers

/// Of pages of a copy found written (page number, the asking the copy took it after; in page
/// order, none never taken): the 4 KiB parts DirectStorage wrote wholly after that, a bit each,
/// for the pages that have any. Empty while switched off.
pub(crate) fn fresh_parts(pages: &[(usize, u64)]) -> Vec<(usize, u16)> {
    if !on() || pages.is_empty() {
        return Vec::new();
    }
    let newest = NEWEST.load(Relaxed);
    if pages.iter().all(|&(_, taken)| taken >= newest) {
        return Vec::new();
    }
    let m = merged();
    let mut out = Vec::new();
    for &(page, taken) in pages {
        if taken >= newest {
            continue;
        }
        let at = page * crate::watch::PAGE;
        let mask = whole_parts(at, &m.fresh(at, at + crate::watch::PAGE, taken));
        if mask != 0 {
            count(C::FreshBytes, u64::from(mask.count_ones()) * 4096);
            out.push((page, mask));
        }
    }
    out
}

/// The parts of [`fresh_parts`]'s answer for `page`.
pub(crate) fn parts_of(fresh: &[(usize, u16)], page: usize) -> u16 {
    fresh.binary_search_by_key(&page, |f| f.0).map_or(0, |i| fresh[i].1)
}

/// Whether DirectStorage wrote a byte of `[from, to)` with a stamp above `since` ([`now`] noted
/// before the last read). `false` while switched off.
pub(crate) fn any_fresh(from: usize, to: usize, since: u64) -> bool {
    if !on() || since >= NEWEST.load(Relaxed) {
        return false;
    }
    merged().any_fresh(from, to, since)
}

// ---------------------------------------------------------------- counts and the line

/// Counted since the last line.
#[derive(Clone, Copy)]
#[repr(usize)]
pub(crate) enum C {
    /// Bytes and ranges journaled; of them written through an address and into physical pages.
    Bytes,
    Ranges,
    RawBytes,
    PageBytes,
    /// Page writes whose guarded blocks were told as written beside their guards: blocks of the
    /// written range, views of their pages marked in all; writes with no guarded block; writes
    /// that fell back to opening the guards (switches off, or a mapping being changed).
    BesideBlocks,
    BesideViews,
    BesideNone,
    BesideFallback,
    /// Write faults of the title's in a block so marked, before an asking took the mark: faults
    /// opening the guards would have spared (the title fixing up what it streamed in).
    BesideFaulted,
    /// Write watch (`VICTORY_STREAM_JOURNAL_RESET`): resets, 4 KiB pages taken off the kernel's
    /// bits; refused: no whole 4 KiB page, view memory, the range being asked about, not
    /// write-watched.
    Resets,
    ResetPages,
    RefusedSmall,
    RefusedGuarded,
    RefusedBusy,
    RefusedNotWatched,
    /// Refreshes: bytes of written pages the journal said DirectStorage wrote after the copy took
    /// them; of those, bytes that would have been compared and were copied without.
    FreshBytes,
    CompareSkipped,
    /// Partly mapped textures read at once rather than up to 30 frames later.
    TexForced,
    /// Entries forgotten.
    Pruned,
}

const COUNTS: usize = 19;
static COUNTED: [AtomicU64; COUNTS] = [const { AtomicU64::new(0) }; COUNTS];

#[inline]
pub(crate) fn count(c: C, n: u64) {
    if n != 0 {
        COUNTED[c as usize].fetch_add(n, Relaxed);
    }
}

/// A frame was presented: every ~5 s the line (once anything was journaled), and entries older
/// than two lines are forgotten.
pub(crate) fn tick() {
    if !SEEN.load(Relaxed) {
        return;
    }
    const PERIOD_MS: u64 = 5000;
    static FRAMES: AtomicU64 = AtomicU64::new(0);
    static LOGGED_MS: AtomicU64 = AtomicU64::new(0);
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    /// The askings' count at the last two lines.
    static FLOORS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    FRAMES.fetch_add(1, Relaxed);
    let now = START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64;
    let last = LOGGED_MS.load(Relaxed);
    if now < last + PERIOD_MS || LOGGED_MS.compare_exchange(last, now, Relaxed, Relaxed).is_err() {
        return;
    }
    let frames = FRAMES.swap(0, Relaxed).max(1) as f64;
    let secs = (now - last).max(1) as f64 / 1000.0;
    let floor = FLOORS[0].swap(FLOORS[1].swap(crate::watch::asked_now(), Relaxed), Relaxed);
    let held = {
        let mut m = MERGED.write().unwrap_or_else(|e| e.into_inner());
        drain_into(&mut m);
        count(C::Pruned, m.prune(floor) as u64);
        m.len()
    };
    let v: [u64; COUNTS] = std::array::from_fn(|i| COUNTED[i].swap(0, Relaxed));
    let mbs = |c: C| v[c as usize] as f64 / secs / 1e6;
    let per = |c: C| v[c as usize] as f64 / frames;
    tracing::info!(
        "stream journal ({}, {frames:.0} frames, {secs:.1} s): journaled {:.1} MB/s in {:.0} ranges/s (through addresses {:.1}, into physical pages {:.1}); \
         guards: per frame {:.1} blocks told written beside their guards ({:.1} views marked; about twice as many protection changes saved), \
         {:.1} writes with none guarded, {:.1} fell back to opening them, {:.1} title write faults in marked blocks; \
         write watch{}: per frame {:.1} resets took {:.1} 4 KiB pages off the askings, refused {:.1} small, {:.1} view, {:.1} busy, {:.1} not watched; \
         refreshes: {:.2} MB a frame of written pages found journaled, {:.2} MB of them copied without compare; \
         textures: {:.1} partly mapped reads a frame brought forward; held {held} ranges, {} forgotten",
        if on() { "on" } else { "off now" },
        mbs(C::Bytes),
        v[C::Ranges as usize] as f64 / secs,
        mbs(C::RawBytes),
        mbs(C::PageBytes),
        per(C::BesideBlocks),
        per(C::BesideViews),
        per(C::BesideNone),
        per(C::BesideFallback),
        per(C::BesideFaulted),
        if reset_on() { "" } else { " (VICTORY_STREAM_JOURNAL_RESET unset: not taken)" },
        per(C::Resets),
        per(C::ResetPages),
        per(C::RefusedSmall),
        per(C::RefusedGuarded),
        per(C::RefusedBusy),
        per(C::RefusedNotWatched),
        v[C::FreshBytes as usize] as f64 / frames / 1e6,
        v[C::CompareSkipped as usize] as f64 / frames / 1e6,
        per(C::TexForced),
        v[C::Pruned as usize],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(r: &Ranges) -> Vec<(usize, usize, u64)> {
        r.map.iter().map(|(&s, &(e, st))| (s, e, st)).collect()
    }

    #[test]
    fn ranges_merge_where_they_touch_and_keep_the_later_stamp_where_they_overlap() {
        let mut r = Ranges::default();
        r.insert(0x1000, 0x2000, 5);
        r.insert(0x2000, 0x3000, 5);
        assert_eq!(all(&r), vec![(0x1000, 0x3000, 5)], "touching, one stamp: one range");
        r.insert(0x1800, 0x2800, 9);
        assert_eq!(all(&r), vec![(0x1000, 0x1800, 5), (0x1800, 0x2800, 9), (0x2800, 0x3000, 5)]);
        // An older stamp over a newer one leaves the newer.
        r.insert(0x1000, 0x3000, 7);
        assert_eq!(all(&r), vec![(0x1000, 0x1800, 7), (0x1800, 0x2800, 9), (0x2800, 0x3000, 7)]);
        // Covering everything and beyond, newest.
        r.insert(0x800, 0x4000, 11);
        assert_eq!(all(&r), vec![(0x800, 0x4000, 11)]);
        // Apart.
        r.insert(0x8000, 0x9000, 3);
        assert_eq!(all(&r), vec![(0x800, 0x4000, 11), (0x8000, 0x9000, 3)]);
        // Bridging the two with a stamp between.
        r.insert(0x3000, 0x8800, 6);
        assert_eq!(all(&r), vec![(0x800, 0x4000, 11), (0x4000, 0x8800, 6), (0x8800, 0x9000, 3)]);
        r.insert(0x5000, 0x5000, 99);
        assert_eq!(r.len(), 3, "an empty range changes nothing");
    }

    #[test]
    fn ranges_cover_every_byte_written_once_and_never_more() {
        // Against a byte map: random writes, the newest stamp a byte had.
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let mut r = Ranges::default();
        let mut bytes = vec![0u64; 4096];
        for stamp in 1..400u64 {
            let a = next(4096);
            let b = (a + 1 + next(300)).min(4096);
            let s = if next(4) == 0 { stamp / 2 } else { stamp };
            r.insert(a, b, s);
            for x in &mut bytes[a..b] {
                *x = (*x).max(s);
            }
        }
        let mut seen = vec![0u64; 4096];
        let mut at = 0;
        for (s, e, st) in all(&r) {
            assert!(s >= at && s < e, "apart and in order");
            at = e;
            for x in &mut seen[s..e] {
                *x = st;
            }
        }
        assert_eq!(seen, bytes);
        for since in [0, 50, 200, 398] {
            let fresh = r.fresh(100, 3000, since);
            for (i, &st) in bytes.iter().enumerate().take(3000).skip(100) {
                let inside = fresh.iter().any(|&(a, b)| (a..b).contains(&i));
                assert_eq!(inside, st > since, "byte {i} since {since}");
            }
            assert_eq!(r.any_fresh(100, 3000, since), bytes[100..3000].iter().any(|&st| st > since));
        }
    }

    #[test]
    fn whole_parts_take_only_parts_wholly_written() {
        let page = 0x7_0000;
        assert_eq!(whole_parts(page, &[(page, page + 0x1_0000)]), u16::MAX);
        assert_eq!(whole_parts(page, &[(page + 0x10, page + 0x2000)]), 0b10, "the first part only in part");
        assert_eq!(whole_parts(page, &[(page - 0x1000, page + 0x1000), (page + 0xf000, page + 0x2_0000)]), 1 | 1 << 15);
        assert_eq!(whole_parts(page, &[(page + 0x1000, page + 0x1fff)]), 0);
    }

    #[test]
    fn pruning_forgets_only_older_stamps() {
        let mut r = Ranges::default();
        r.insert(0, 10, 1);
        r.insert(20, 30, 5);
        r.insert(40, 50, 9);
        assert_eq!(r.prune(5), 1);
        assert!(!r.any_fresh(0, 10, 0), "forgotten: said as never written, which only does what is done without");
        assert!(r.any_fresh(20, 30, 4));
        assert_eq!(r.halve(), 1);
        assert_eq!(all(&r), vec![(40, 50, 9)]);
    }

    #[test]
    fn a_copy_taken_after_the_write_has_nothing_fresh() {
        // What a refresh asks: a page taken after asking 10 is not fresh for a write stamped 10
        // or before; one taken at 9 is, in the parts wholly written.
        let mut r = Ranges::default();
        let page = 0x30_0000;
        r.insert(page + 0x1000, page + 0x3800, 10);
        let parts = |taken| whole_parts(page, &r.fresh(page, page + 0x1_0000, taken));
        assert_eq!(parts(10), 0);
        assert_eq!(parts(9), 0b110, "the part at 0x3000 is only half written");
    }
}
