//! Textures shaders write that the title's own code then reads on the CPU.
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it stays
//! in the texture's host copy. The title's ocean is one: its wave passes write
//! a 256x256 height map (R32_FLOAT) and a displacement map (R32G32_FLOAT), and
//! its CPU code samples both, which is what boats float on; it read memory
//! nothing had written (run rw1, `VICTORY_READ_WATCH=1`: about 30 places in
//! the title read those two kinds), and boats sank.
//!
//! For textures of the kinds in `VICTORY_TEXTURE_BACK` (comma-separated
//! `<width>x<height>f<format>`; unset or `1`: the four water formats found by
//! the upstream trace; `0`: none), a list in which shaders write one copies it
//! into a read-back buffer of ours when it closes, and once the list has run
//! the texels are tiled into the texture's title memory in the console's
//! layout, as the console's GPU would have left them.
//!
//! shortcut: by kind, not by texture: textures of those kinds the title never
//! reads are copied back too (about a megabyte per such list). The copy shows
//! what the list's last write left; a title read in between sees the frame
//! before, as it would before the console's GPU has run.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

/// The kinds copied back: (width, height, format).
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        const DEFAULT: &str = "256x256f41,256x256f16,64x64f16,128x128f36";
        let list = match std::env::var("VICTORY_TEXTURE_BACK").as_deref() {
            Ok("1") | Err(_) => DEFAULT.to_owned(),
            Ok(spec) => spec.to_owned(),
        };
        let kinds: Vec<(u64, u32, u32)> = list
            .split(',')
            .filter_map(|k| {
                let (size, format) = k.trim().split_once('f')?;
                let (w, h) = size.split_once('x')?;
                Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
            })
            .collect();
        if !kinds.is_empty() {
            tracing::info!(?kinds, "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK)");
        }
        kinds
    })
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty()
        && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// An outstanding write to a specific allocation. Mapping and resource
/// changes invalidate only the writes that could touch the replaced memory.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());
static TRACKING: AtomicBool = AtomicBool::new(false);

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket { owner, at, end, valid: AtomicBool::new(true) });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Relaxed);
    ticket
}

fn invalidate(matches: impl Fn(&Ticket) -> bool) {
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    TICKETS.lock().unwrap_or_else(|e| e.into_inner()).retain(|weak| {
        let Some(ticket) = weak.upgrade() else { return false };
        if matches(&ticket) {
            ticket.valid.store(false, Ordering::Release);
        }
        true
    });
}

pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner);
}

pub(crate) fn replaced(at: usize) {
    invalidate(|ticket| ticket.at == at);
}

pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Target {
    pub(crate) fn new(resource: This) -> Option<Self> {
        // Keep the host copy independently of the title's resource lifetime.
        let copy = crate::resource::host_copy(resource)?;
        let _mapping = crate::watch::title_memory_reading();
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        // Tiled back only where the layout has its address equation (the
        // tiling refuses others, `Layout::tile`), and a render-XOR one only
        // on its block (`crate::texture::render_xor_base_ok`).
        if !crate::texture::render_xor_base_ok(&layout, at) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(desc = ?layout.desc, at = format_args!("{at:#x}"), "render-XOR texture off its block not copied back")
            });
            return None;
        }
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        let ticket = track(resource as usize, at, end);
        Some(Self { copy, layout, ticket })
    }

    pub(crate) fn owner(&self) -> usize { self.ticket.owner }
}

/// The readback resources and the allocation the pixels belong to.
pub(crate) struct Back {
    buffer: ID3D12Resource,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Back {
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }
}

/// Records a host copy from a texture already in COPY_SOURCE.
pub(crate) fn record(list: &ID3D12GraphicsCommandList, target: Target) -> Option<Back> {
    if !target.ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: a live texture kept through the command list's state tracking.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let subresources = mips * u32::from(desc.DepthOrArraySize.max(1));
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays of `n` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        )
    };
    // The last footprint's advertised extent can omit row-end padding.
    // The tiler reads a pitched slice, so the mapped buffer must include it.
    for (footprint, &rows) in footprints.iter().zip(&rows) {
        let format = footprint.Footprint;
        let pitched = u64::from(format.RowPitch)
            .checked_mul(u64::from(rows))?
            .checked_mul(u64::from(format.Depth))?;
        total = total.max(footprint.Offset.checked_add(pitched)?);
    }
    let buffer = host.read_back_buffer(total)?;
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
        };
        // SAFETY: an open list; the texture is in the copy-source state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back { buffer, footprints, rows, mips, layout: target.layout, ticket: target.ticket })
}

// LOCAL (upstream-port 2d1dc38): texture read-backs written by a thread of their own.

/// How texture read-backs are written into title memory
/// (`VICTORY_TEXBACK_ASYNC`):
///
/// - `0`: where they land, on the thread that lands submissions (as before);
/// - `1` (the default): by [`writer`]'s thread, and the landing is held at
///   the points where it would tell the title, or a reader, that they are
///   there ([`flush`]);
/// - `2`: by [`writer`]'s thread, never waited for (upstream's way: the title
///   reads the frame before at most).
///
/// Writing one takes the lock against mapping changes
/// ([`crate::watch::title_memory_reading`]), and while the title streams the
/// world it changes mappings all the time: on the thread that lands
/// submissions the wait for that lock stood 100-400 ms in some frames
/// (upstream's 16 GB machine, RTX 3060: two to ten 350 KB water maps a
/// frame), and with it the landing of everything else. Here the landing
/// thread only copies the texels out of the read-back buffer; the waiting,
/// tiling and writing go to [`writer`]. Of several copies of one texture
/// waiting there only the newest is written.
///
/// Mode 1 keeps what the landing promised: the range of a flight's texture
/// read-back stays in the line's written set, which holds readers back
/// (`late_blocked`, settles), until the flight is landed, and the landing is
/// not done (value written, fence signalled, flight number finished, written
/// set cleared) until [`flush`] saw the writer idle. What overlaps with the
/// writer is the rest of the landing: the other flights' buffer read-backs,
/// card plans and own work.
pub(crate) fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_TEXBACK_ASYNC").as_deref() {
        Ok("0") => 0,
        Ok("2") => 2,
        _ => 1,
    })
}

/// A read-back's texels taken out of its buffer, to be tiled into title memory.
struct Landing {
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
    /// The read-back buffer's bytes, from 0 to the end of its last subresource.
    data: Vec<u8>,
}

/// Once the list has run, tile its texels back into the same title allocation
/// (on [`writer`]'s thread unless `VICTORY_TEXBACK_ASYNC=0`).
pub(crate) fn land(back: &Back) {
    if !back.ticket.valid.load(Ordering::Acquire) {
        stale();
        return;
    }
    let len = back
        .footprints
        .iter()
        .zip(&back.rows)
        .map(|(f, &rows)| f.Offset as usize + f.Footprint.RowPitch as usize * rows as usize * f.Footprint.Depth as usize)
        .max()
        .unwrap_or(0);
    let mut mapped = std::ptr::null_mut();
    // SAFETY: our read-back buffer, written by the list that has run; `len`
    // ends with its last subresource, which it holds.
    let data = unsafe {
        if back.buffer.Map(0, None, Some(&mut mapped)).is_err() || mapped.is_null() {
            return;
        }
        let data = std::slice::from_raw_parts(mapped.cast::<u8>().cast_const(), len).to_vec();
        back.buffer.Unmap(0, None);
        data
    };
    let landing = Landing {
        footprints: back.footprints.clone(),
        rows: back.rows.clone(),
        mips: back.mips,
        layout: back.layout.clone(),
        ticket: back.ticket.clone(),
        data,
    };
    match (mode() != 0).then(writer).flatten() {
        Some(writer) => {
            WRITER_SEEN.store(true, Ordering::Release);
            writer.send(landing)
        }
        None => write(&landing),
    }
}

fn stale() {
    static STALE: AtomicU64 = AtomicU64::new(0);
    let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
    if skipped.is_power_of_two() {
        tracing::info!(skipped, "skipping texture readbacks whose title allocation changed");
    }
}

/// Tiles a read-back's texels into its title allocation.
fn write(landing: &Landing) {
    use crate::land::{count, spent, stat};
    static COPIED: AtomicU64 = AtomicU64::new(0);
    let asked = std::time::Instant::now();
    let _mapping = crate::watch::title_memory_reading();
    let _ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    let working = spent(stat::TB_WAIT, asked);
    if !landing.ticket.valid.load(Ordering::Acquire) {
        stale();
        return;
    }
    let (at, end) = (landing.ticket.at, landing.ticket.end);
    let size = end - at;
    let mut tiled = vec![0u8; size];
    // Keep title bytes the subresources do not cover.
    if !crate::host::read_into(at as *const u8, &mut tiled) {
        return;
    }
    for (i, (footprint, &rows)) in landing.footprints.iter().zip(&landing.rows).enumerate() {
        let (mip, slice) = (i as u32 % landing.mips, i as u32 / landing.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let from = footprint.Offset as usize;
        let linear = &landing.data[from..from + pitch * rows as usize * f.Depth as usize];
        if let Err(e) = landing.layout.tile(&mut tiled, mip, slice, linear, pitch, pitch * rows as usize) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled"));
            return;
        }
    }
    crate::host::copy_mapped(at as *mut u8, tiled.as_ptr(), size, true);
    spent(stat::TB_WORK, working);
    count(stat::TB_WRITTEN, 1);
    let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
    if copied == 1 || copied.is_multiple_of(128) {
        tracing::info!(copied, bytes = size, "texture readback landed");
    }
}

/// The thread that writes read-backs into title memory, and what waits for it.
struct Writer {
    state: Mutex<Waiting>,
    more: std::sync::Condvar,
    /// Told when nothing waits and nothing is being written.
    idle: std::sync::Condvar,
}

#[derive(Default)]
struct Waiting {
    /// LOCAL (port-fix): in arrival order, oldest first, and written so: two
    /// landings whose ranges overlap without being the same must reach title
    /// memory in the order they were read back, or the older one lands over
    /// the newer.
    waiting: std::collections::VecDeque<Landing>,
    /// The title ranges of the batch being written.
    busy: Vec<(usize, usize)>,
}

/// LOCAL (port-fix): whether `a` and `b` are read-backs of the very same
/// thing: one allocation, one range, one layout and footprint set, so that
/// the newer covers all the older one did.
fn same_target(a: &Landing, b: &Landing) -> bool {
    a.ticket.owner == b.ticket.owner
        && a.ticket.at == b.ticket.at
        && a.ticket.end == b.ticket.end
        && a.mips == b.mips
        && a.rows == b.rows
        && a.layout == b.layout
        && a.footprints.len() == b.footprints.len()
        && a.footprints.iter().zip(&b.footprints).all(|(x, y)| {
            x.Offset == y.Offset
                && x.Footprint.Format == y.Footprint.Format
                && x.Footprint.Width == y.Footprint.Width
                && x.Footprint.Height == y.Footprint.Height
                && x.Footprint.Depth == y.Footprint.Depth
                && x.Footprint.RowPitch == y.Footprint.RowPitch
        })
}

/// LOCAL (port-fix): clears `busy` and tells the waiters when a batch is done
/// or its writing panicked, so that [`Writer::wait`] cannot wait for ever.
struct BatchDone(&'static Writer);

impl Drop for BatchDone {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        state.busy.clear();
        if state.waiting.is_empty() {
            self.0.idle.notify_all();
        }
    }
}

impl Writer {
    fn send(&self, landing: Landing) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        // Only an exact repeat is replaced (the newest wins, and takes the
        // newest's place in the order: what was read back in between must
        // not land over it).
        if let Some(old) = state.waiting.iter().position(|w| same_target(w, &landing)) {
            state.waiting.remove(old);
            crate::land::count(crate::land::stat::TB_REPLACED, 1);
        }
        state.waiting.push_back(landing);
        drop(state);
        self.more.notify_one();
    }

    /// Returns when everything sent before is written (or dropped as stale),
    /// if `overlapping` holds for what is still to be.
    fn wait(&self, overlapping: impl Fn(usize, usize) -> bool) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let pending = |s: &Waiting| {
            s.busy.iter().any(|&(at, end)| overlapping(at, end))
                || s.waiting.iter().any(|l| overlapping(l.ticket.at, l.ticket.end))
        };
        if !pending(&state) {
            return;
        }
        // The whole of it: what waits
        // behind it may be what a later wait is for.
        while !state.waiting.is_empty() || !state.busy.is_empty() {
            state = self.idle.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// The writer, if its thread could be started (if not, landings are written
/// where they are made, as before).
fn writer() -> Option<&'static Writer> {
    static WRITER: OnceLock<Option<&'static Writer>> = OnceLock::new();
    *WRITER.get_or_init(|| {
        let writer: &'static Writer = Box::leak(Box::new(Writer {
            state: Mutex::new(Waiting::default()),
            more: std::sync::Condvar::new(),
            idle: std::sync::Condvar::new(),
        }));
        let started = std::thread::Builder::new().name("texture read-backs".into()).spawn(move || loop {
            let batch: Vec<Landing> = {
                let mut state = writer.state.lock().unwrap_or_else(|e| e.into_inner());
                while state.waiting.is_empty() {
                    state = writer.more.wait(state).unwrap_or_else(|e| e.into_inner());
                }
                let batch: Vec<Landing> = state.waiting.drain(..).collect();
                state.busy = batch.iter().map(|l| (l.ticket.at, l.ticket.end)).collect();
                batch
            };
            // (Cleared on the way out even if a write panics.)
            let done = BatchDone(writer);
            for landing in &batch {
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| write(landing))).is_err() {
                    tracing::warn!("a texture read-back could not be written (panicked)");
                }
            }
            drop(batch);
            drop(done);
        });
        match started {
            Ok(_) => Some(writer),
            Err(e) => {
                tracing::warn!(error = %e, "no thread for texture read-backs: they are written where they land");
                None
            }
        }
    })
}

/// Returns when every texture read-back landed so far is in title memory
/// (or was dropped: its allocation changed, or a newer copy replaced it).
/// Called before the landing tells the title, or a reader, that they are
/// there. Nothing in modes 0 and 2, or if no read-back was ever landed.
pub(crate) fn flush() {
    if mode() == 1
        && let Some(writer) = writer_started()
    {
        writer.wait(|_, _| true);
    }
}

/// [`flush`], if a read-back not written yet is in `[from, to)` of title memory.
pub(crate) fn flush_range(from: usize, to: usize) {
    if mode() == 1
        && let Some(writer) = writer_started()
    {
        writer.wait(|at, end| at < to && from < end);
    }
}

/// The writer if one was started (a flush starts none).
fn writer_started() -> Option<&'static Writer> {
    WRITER_SEEN.load(Ordering::Acquire).then(writer).flatten()
}

/// Set once a landing went to [`writer`].
static WRITER_SEEN: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_readbacks_for_changed_allocations_are_invalidated() {
        let first = track(91, 0x1000, 0x2000);
        let next = track(92, 0x2000, 0x3000);
        let far = track(93, 0x4000, 0x5000);

        remapped(0x0f00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));
        replaced(0x2000);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        released(93);
        assert!(!far.valid.load(Ordering::Acquire));
        assert!(track(94, 0x2000, 0x3000).valid.load(Ordering::Acquire));
    }

    /// LOCAL (port-fix): a test landing over `[at, end)` of `owner`.
    fn landing(owner: usize, at: usize, end: usize, tag: u8) -> Landing {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 16,
            height: 16,
            depth: 1,
            mips: 1,
            element_bytes: 1,
            block: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        })
        .expect("a linear layout");
        Landing {
            footprints: vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default()],
            rows: vec![16],
            mips: 1,
            layout,
            ticket: track(owner, at, end),
            data: vec![tag],
        }
    }

    fn test_writer() -> Writer {
        Writer {
            state: Mutex::new(Waiting::default()),
            more: std::sync::Condvar::new(),
            idle: std::sync::Condvar::new(),
        }
    }

    /// Landings come out in the order they were read back, overlapping
    /// ranges with different starts included; only an exact repeat is
    /// replaced, by the newest, in the newest's place.
    #[test]
    fn landings_are_written_in_arrival_order_and_only_exact_repeats_coalesce() {
        let w = test_writer();
        w.send(landing(1, 0x10_000, 0x12_000, 1));
        w.send(landing(2, 0x10_800, 0x13_000, 2)); // overlaps, other start
        w.send(landing(1, 0x10_000, 0x12_000, 3)); // exact repeat of the first
        w.send(landing(3, 0x10_000, 0x11_000, 4)); // same start, other range
        let tags: Vec<u8> = w.state.lock().unwrap().waiting.iter().map(|l| l.data[0]).collect();
        assert_eq!(tags, [2, 3, 4]);
    }

    /// A panicking write does not leave the writer busy for ever.
    #[test]
    fn a_panicking_batch_clears_busy_and_wakes_the_waiters() {
        let w: &'static Writer = Box::leak(Box::new(test_writer()));
        w.state.lock().unwrap().busy = vec![(0x1000, 0x2000)];
        let r = std::panic::catch_unwind(|| {
            let _done = BatchDone(w);
            panic!("a write failed");
        });
        assert!(r.is_err());
        assert!(w.state.lock().unwrap().busy.is_empty());
        w.wait(|_, _| true);
    }
}
