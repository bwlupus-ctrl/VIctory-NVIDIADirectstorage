//! LOCAL (page-kinds): read-only snapshots. No mapping, protection, residency or write hooks.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

pub const PAGE: usize = 64 << 10;
const USER_END: usize = 1 << 47;

/// LOCAL (page-kinds): exact opt-in, cached; all entry points return immediately otherwise.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PAGE_KINDS").is_some_and(|v| v == "1"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub cpu: usize,
    pub gpu: usize,
    pub bytes: usize,
    pub table: usize,
}

impl Region {
    /// LOCAL (page-kinds): Windows lower canonical user addresses; bounded by console RAM.
    pub fn validate(self) -> Result<Self, &'static str> {
        if self.bytes == 0 || self.bytes > 16usize << 30 || !self.bytes.is_multiple_of(PAGE) {
            return Err("invalid region size");
        }
        for base in [self.cpu, self.gpu] {
            if base < PAGE
                || !base.is_multiple_of(PAGE)
                || !base
                    .checked_add(self.bytes)
                    .is_some_and(|end| end <= USER_END)
            {
                return Err("invalid view base or extent");
            }
        }
        if self.cpu < self.gpu + self.bytes && self.gpu < self.cpu + self.bytes {
            return Err("overlapping views");
        }
        if self.table < PAGE
            || !self.table.is_multiple_of(8)
            || !self
                .table
                .checked_add(self.table_bytes())
                .is_some_and(|end| end <= USER_END)
        {
            return Err("invalid table pointer or extent");
        }
        Ok(self)
    }

    pub fn table_bytes(self) -> usize {
        self.bytes / PAGE * 8
    }

    pub fn index_of(self, at: usize) -> Option<(View, usize)> {
        for (view, base) in [(View::Cpu, self.cpu), (View::Gpu, self.gpu)] {
            if at
                .checked_sub(base)
                .is_some_and(|offset| offset < self.bytes)
            {
                return Some((view, (at - base) / PAGE));
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Cpu,
    Gpu,
    Outside,
    Mixed,
}

/// LOCAL (page-kinds): row/column order throughout the API and logs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Kind {
    GpuOnly,
    Both,
    CpuOnly,
    Unmapped,
    Outside,
}

pub fn state_class(word: u64) -> Kind {
    match word >> 62 {
        2 => Kind::GpuOnly,
        3 => Kind::Both,
        1 => Kind::CpuOnly,
        _ => Kind::Unmapped,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// LOCAL (page-kinds): 64 KiB virtual pages touched, partial end pages count once.
    pub pages: [u64; 5],
    /// LOCAL (page-kinds): exact overlap bytes, without rounding partial pages up.
    pub bytes: [u64; 5],
}

impl Counts {
    fn add(&mut self, other: &Self, times: u64) {
        for i in 0..5 {
            self.pages[i] += other.pages[i] * times;
            self.bytes[i] += other.bytes[i] * times;
        }
    }
}

pub struct RangeKinds {
    pub view: View,
    pub counts: Counts,
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub region: Region,
    states: Vec<Kind>,
}

impl Snapshot {
    pub fn new(region: Region, words: &[u64]) -> Result<Self, &'static str> {
        let region = region.validate()?;
        if words.len() != region.bytes / PAGE {
            return Err("incomplete table");
        }
        Ok(Self {
            region,
            states: words.iter().copied().map(state_class).collect(),
        })
    }

    /// LOCAL (page-kinds): overflow/empty ranges are unavailable; mixed views are explicit.
    pub fn query(&self, at: usize, len: usize) -> Option<RangeKinds> {
        let end = at.checked_add(len).filter(|_| len != 0)?;
        let mut here = at;
        let mut counts = Counts::default();
        let mut view = None;
        while here < end {
            let (part_view, kind, to) = match self.region.index_of(here) {
                Some((v, i)) => {
                    let base = if v == View::Cpu {
                        self.region.cpu
                    } else {
                        self.region.gpu
                    };
                    (v, self.states[i], end.min(base + (i + 1) * PAGE))
                }
                None => {
                    let to = [self.region.cpu, self.region.gpu]
                        .into_iter()
                        .filter(|&b| b > here)
                        .fold(end, usize::min);
                    (View::Outside, Kind::Outside, to)
                }
            };
            view = Some(match view {
                None => part_view,
                Some(v) if v == part_view => v,
                _ => View::Mixed,
            });
            counts.bytes[kind as usize] += (to - here) as u64;
            counts.pages[kind as usize] += ((to - 1) / PAGE - here / PAGE + 1) as u64;
            here = to;
        }
        Some(RangeKinds {
            view: view?,
            counts,
        })
    }

    fn totals(&self) -> [u64; 4] {
        let mut totals = [0; 4];
        for &state in &self.states {
            totals[state as usize] += 1;
        }
        totals
    }

    fn churn(&self, previous: &Self) -> Option<[u64; 2]> {
        if self.region != previous.region {
            return None;
        }
        let mut flips = [0; 2];
        for (&old, &new) in previous.states.iter().zip(&self.states) {
            match (old, new) {
                (Kind::GpuOnly, Kind::Both) => flips[0] += 1,
                (Kind::Both, Kind::GpuOnly) => flips[1] += 1,
                _ => {}
            }
        }
        Some(flips)
    }
}

/// LOCAL (page-kinds): HLE owns validation and fault-safe reads; GPU owns only copied data.
pub type Reader = fn() -> Result<Snapshot, &'static str>;
static READER: OnceLock<Reader> = OnceLock::new();
static LATEST: RwLock<Option<Snapshot>> = RwLock::new(None);

pub fn set_reader(reader: Reader) {
    if on() {
        let _ = READER.set(reader);
    }
}

/// LOCAL (page-kinds): latest successful tick (at most 5 s old during presents), not live truth.
/// No sample/off/failed read/invalid range returns None, never a fabricated unmapped result.
pub fn query(at: usize, len: usize) -> Option<RangeKinds> {
    if !on() {
        return None;
    }
    LATEST.read().ok()?.as_ref()?.query(at, len)
}

#[derive(Default)]
struct Streams {
    ranges: BTreeMap<(usize, usize), u64>,
    unplaced: u64,
    dropped: u64,
}

static STREAMS: Mutex<Streams> = Mutex::new(Streams {
    ranges: BTreeMap::new(),
    unplaced: 0,
    dropped: 0,
});

pub fn streamed(at: usize, len: usize) {
    if !on() || len == 0 {
        return;
    }
    let mut s = STREAMS.lock().unwrap_or_else(|e| e.into_inner());
    if at.checked_add(len).is_none()
        || (s.ranges.len() >= 65536 && !s.ranges.contains_key(&(at, len)))
    {
        s.dropped += len as u64;
    } else {
        *s.ranges.entry((at, len)).or_default() += 1;
    }
}

pub fn streamed_unplaced(len: usize) {
    if on() {
        STREAMS.lock().unwrap_or_else(|e| e.into_inner()).unplaced += len as u64;
    }
}

fn cross_tab(snapshot: &Snapshot, ranges: impl IntoIterator<Item = (usize, usize, u64)>) -> Counts {
    let mut counts = Counts::default();
    for (at, len, times) in ranges {
        if let Some(range) = snapshot.query(at, len) {
            counts.add(&range.counts, times);
        }
    }
    counts
}

/// LOCAL (page-kinds): no title VA accesses except the installed reader, every ~5 s at present.
pub fn tick() {
    if !on() {
        return;
    }
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let Ok(mut last) = LAST.try_lock() else {
        return;
    };
    let now = Instant::now();
    if last.is_none() {
        *last = Some(now);
        return;
    }
    if now.duration_since(last.unwrap()) < Duration::from_secs(5) {
        return;
    }
    *last = Some(now);
    let streams = std::mem::take(&mut *STREAMS.lock().unwrap_or_else(|e| e.into_inner()));
    let sample = READER
        .get()
        .ok_or("reader not registered")
        .and_then(|reader| reader());
    let mut latest = LATEST.write().unwrap_or_else(|e| e.into_inner());
    let snapshot = match sample {
        Ok(snapshot) => snapshot,
        Err(reason) => {
            *latest = None;
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(
                    "page kinds: unavailable ({reason}); failed periods discarded; retrying every 5 s"
                );
            }
            return;
        }
    };
    let churn = latest.as_ref().and_then(|old| snapshot.churn(old));
    let entries = crate::memclass::page_kind_ranges();
    let mut rows = Vec::new();
    for class in [
        crate::memclass::Class::WcGpuPrivate,
        crate::memclass::Class::WcGpuPool,
        crate::memclass::Class::GpuCached,
        crate::memclass::Class::CpuOnly,
        crate::memclass::Class::Other,
    ] {
        let mut counts = cross_tab(
            &snapshot,
            entries
                .iter()
                .filter(|e| e.2 == class)
                .map(|e| (e.0, e.1, 1)),
        );
        if class == crate::memclass::Class::Other {
            // LOCAL (page-kinds): include unregistered gaps in both views; outside has no finite universe.
            for base in [snapshot.region.cpu, snapshot.region.gpu] {
                let end = base + snapshot.region.bytes;
                let mut at = base;
                for &(start, len, _) in &entries {
                    let to = start.saturating_add(len);
                    if to <= at || start >= end {
                        continue;
                    }
                    if start > at {
                        counts.add(&snapshot.query(at, start - at).unwrap().counts, 1);
                    }
                    at = to.min(end);
                }
                if at < end {
                    counts.add(&snapshot.query(at, end - at).unwrap().counts, 1);
                }
            }
        }
        rows.push(format!("{}={:?}", class.name(), counts.bytes));
    }
    let wc = cross_tab(
        &snapshot,
        crate::wc_vram::page_kind_ranges()
            .into_iter()
            .map(|(a, n)| (a, n, 1)),
    );
    // LOCAL (page-kinds): identify the default heap-probe target from metadata; never enable its guards.
    // Fable review: also take the heap probe's own targets (pool-mapped views
    // inside a larger reservation make no registry entry), and log how many
    // ranges matched and where, so all-zero counts can't be mistaken for "found".
    let mut heap_ranges: Vec<(usize, usize)> = entries
        .iter()
        .filter(|e| {
            e.1 == 0xae00000 && e.0 != 0 && e.0.checked_add(e.1).is_some_and(|end| end <= 1usize << 32)
        })
        .map(|e| (e.0, e.1))
        .collect();
    for t in crate::heapprobe::target_ranges() {
        if !heap_ranges.iter().any(|h| h.0 < t.0 + t.1 && t.0 < h.0 + h.1) {
            heap_ranges.push(t);
        }
    }
    let heap_where = if heap_ranges.is_empty() {
        "not found".to_string()
    } else {
        heap_ranges.iter().map(|(a, n)| format!("{a:#x}+{n:#x}")).collect::<Vec<_>>().join(",")
    };
    let heap = cross_tab(&snapshot, heap_ranges.iter().map(|&(a, n)| (a, n, 1)));
    let ds = cross_tab(
        &snapshot,
        streams
            .ranges
            .into_iter()
            .map(|((a, n), count)| (a, n, count)),
    );
    let r = snapshot.region;
    tracing::info!(
        "page kinds: cpu={:#x} gpu={:#x} bytes={:#x} table={:#x}; totals_pages[gpu-only,both,cpu-only,unmapped]={:?}; \
         flips_pages[gpu-only->both,both->gpu-only]={:?} (None=baseline); \
         cross_bytes[gpu-only,both,cpu-only,unmapped,outside]: {}; wc_vram={:?}; heap174 ({})={:?}; \
         dstorage={:?}; dstorage_unplaced={} dropped_bytes={} (destinations sampled now; repeats/aliases counted)",
        r.cpu,
        r.gpu,
        r.bytes,
        r.table,
        snapshot.totals(),
        churn,
        rows.join("; "),
        wc.bytes,
        heap_where,
        heap.bytes,
        ds.bytes,
        streams.unplaced,
        streams.dropped
    );
    *latest = Some(snapshot);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region() -> Region {
        Region {
            cpu: 0x100000,
            gpu: 0x200000,
            bytes: 4 * PAGE,
            table: 0x300000,
        }
    }
    fn snapshot() -> Snapshot {
        Snapshot::new(region(), &[2 << 62, 3 << 62, 1 << 62, 0]).unwrap()
    }

    #[test]
    fn pagekinds_index_and_range_boundaries() {
        let r = region();
        assert_eq!(r.index_of(r.cpu - 1), None);
        assert_eq!(r.index_of(r.cpu), Some((View::Cpu, 0)));
        assert_eq!(r.index_of(r.gpu + r.bytes - 1), Some((View::Gpu, 3)));
        assert_eq!(r.index_of(r.gpu + r.bytes), None);
        assert_eq!(
            snapshot().query(r.cpu + PAGE - 1, 2).unwrap().counts.pages,
            [1, 1, 0, 0, 0]
        );
        assert_eq!(snapshot().query(r.gpu, r.bytes).unwrap().view, View::Gpu);
        assert_eq!(snapshot().query(r.cpu - 1, 2).unwrap().view, View::Mixed);
        assert!(snapshot().query(usize::MAX, 2).is_none());
        assert!(snapshot().query(r.cpu, 0).is_none());
    }

    #[test]
    fn pagekinds_state_bits_ignore_lower_bits() {
        for (bits, kind) in [
            (0, Kind::Unmapped),
            (1, Kind::CpuOnly),
            (2, Kind::GpuOnly),
            (3, Kind::Both),
        ] {
            assert_eq!(state_class((bits << 62) | ((1 << 62) - 1)), kind);
        }
    }

    #[test]
    fn pagekinds_invalid_geometry_is_unavailable() {
        let r = region();
        for bad in [
            Region {
                cpu: r.cpu + 1,
                ..r
            },
            Region { gpu: USER_END, ..r },
            Region { bytes: 0, ..r },
            Region {
                bytes: PAGE + 1,
                ..r
            },
            Region {
                bytes: (16usize << 30) + PAGE,
                ..r
            },
            Region { gpu: r.cpu, ..r },
            Region {
                table: usize::MAX - 7,
                ..r
            },
        ] {
            assert!(bad.validate().is_err());
        }
        assert!(Snapshot::new(r, &[0]).is_err());
    }

    #[test]
    fn pagekinds_synthetic_cross_tab_and_churn() {
        let s = snapshot();
        let r = s.region;
        let c = cross_tab(&s, [(r.cpu - 3, r.bytes + 6, 1), (r.gpu, PAGE, 2)]);
        assert_eq!(
            c.bytes,
            [3 * PAGE as u64, PAGE as u64, PAGE as u64, PAGE as u64, 6]
        );
        assert_eq!(c.pages, [3, 1, 1, 1, 2]);
        assert_eq!(s.totals(), [1, 1, 1, 1]);
        let next = Snapshot::new(r, &[3 << 62, 2 << 62, 1 << 62, 0]).unwrap();
        assert_eq!(next.churn(&s), Some([1, 1]));
        let changed = Snapshot::new(
            Region {
                table: r.table + 8,
                ..r
            },
            &[0; 4],
        )
        .unwrap();
        assert_eq!(changed.churn(&s), None);
    }
}
