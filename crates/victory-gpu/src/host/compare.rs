//! LOCAL (queues): the compare of copies in the card's memory on a compute
//! queue of its own (`VICTORY_COMPARE_QUEUE`, see [`level`]; off by default).
//!
//! Every flight ends with our compare ([`CARD_COMPARE`]: about 115 passes a
//! frame over card copies) behind a global UAV barrier, on the one direct
//! queue: the title's next submission stands behind it. With the switch the
//! compare goes to a compute queue instead, and runs beside what the direct
//! queue does next where nothing orders the two.
//!
//! What is ordered, and how (all GPU-side; no thread waits for the card
//! here):
//!
//! - A compare starts after its flight: the compute queue `Wait`s for the
//!   host fence's value behind the flight before the compare's list
//!   ([`send`]). Behind the flight are also the sends up and rehashes of
//!   that submission and of all before it, so the compare finds the copy as
//!   the flight left it, as it did on the direct queue.
//! - Compares stay in order among themselves: one queue, in the order they
//!   were sent, each its own `ExecuteCommandLists`. The compare chain (each
//!   stores its hashes, mode 2) flags what changed since the one before, as
//!   before.
//! - The thread that lands flights takes a flight only when the compare's
//!   fence has passed its compare too (`CardPlan::checked`, [`landable`]):
//!   the results in `differs` and the stored hashes are complete before
//!   they are read, and nothing the compare names is let go before it ran
//!   (`FINISHED` moves only at landings).
//! - What the direct queue does later and writes what a compare still in
//!   the air reads or writes waits for that compare (`Wait` on the direct
//!   queue for the compare's fence, [`hold`]): a send up (it writes the
//!   copy's bytes) and its rehash (it writes the copy's stored hashes, which
//!   the compare reads and writes: a rehash that went first would take the
//!   flight's writes for the baseline, and the compare would never flag
//!   them; that is a lost shader write); a flight that declares shader
//!   writes to the copy (including `crate::picked`'s and `card_steps`'
//!   copies, which are declared); a compare that goes to the direct queue
//!   while compares are in the air on the compute queue (the A/B switch;
//!   the results buffer is shared). A send up and its rehash wait per copy
//!   at every level: two queues never write the copy's hashes, nor its
//!   bytes, at once ("a resource cannot be written to from multiple command
//!   queues simultaneously", D3D12's "Executing and synchronizing command
//!   lists"). Declared shader writes wait per copy with [`level`] 1, and
//!   with 2 only where their pieces meet the pieces compared, widened by a
//!   piece on each side (the compare widens a piece's offsets to 16 bytes):
//!   the flight then writes the copy's bytes while the compare reads other
//!   pieces of the same buffer, a write beside a read on another queue,
//!   which that page's rule does not allow either; for buffers (no layout,
//!   no compression) the only effect is on what the compare reads, and it
//!   reads none of those bytes. Level 2 is for measuring that.
//! - With [`level`] 3, as 2, and a send up and its rehash wait only where
//!   their pieces (widened) meet pieces compared: with the ring of constants
//!   sent up into the heap copy before nearly every submission, a wait per
//!   copy there held almost every flight behind the compare before it.
//!   Why no byte or hash is lost that way (checked in the code): a compare
//!   reads only its pieces' bytes, each piece's offsets widened to 16 bytes
//!   (less than a piece into its neighbours, [`CARD_COMPARE`]), reads and
//!   stores only those pieces' 16-byte hash slots, at `piece * 16`, and
//!   writes `differs` (never written by a send-up list: its rehash runs with
//!   `compare == 0`). A send-up list writes the bytes of its ranges only (the
//!   copies; `send::Own`'s copies and pairs, the pairs' offsets lying in the
//!   ranges) and the hash slots of their pieces only (`card_run` of each
//!   range; `joined_piece_runs` keeps exactly that set of pieces). Pieces are
//!   numbered alike on both sides (`card_span_of`: piece i at `i * 64 KiB -
//!   skew`, as the shader's `piece * 65536 - skew`). Both sides are widened
//!   by a piece, so no byte and no slot one writes is one the other reads or
//!   writes. What remains is formal: the send-up list's COPY_DEST (or
//!   UNORDERED_ACCESS) to NON_PIXEL_SHADER_RESOURCE barrier on the whole
//!   copy, and its writes into the copy's bytes and its hash buffer, run
//!   beside the compare's reads (and its hash stores) of the same two buffers
//!   on the compute queue: two queues at one resource, one of them writing,
//!   which D3D12's rule above does not allow. A buffer has no layout or
//!   compression for the barrier to change; it orders caches, and the bytes
//!   the compare reads are not among those written. Level 3 is for measuring
//!   that; 1 is the one within the rule but for undeclared stores.
//! - Nothing waits for a value not yet signalled: the compute queue waits
//!   only for values the direct queue was given before, and the direct
//!   queue only for compares sent before, all under the submission lock. So
//!   no wait can stand for ever (`victory-nvidia-8gb`'s queue2 lanes keep
//!   the same rule).
//!
//! States: the compare's list is a compute list and names only buffers
//! (the copy as a root SRV, its hashes and the results as root UAVs), in
//! non-pixel-shader-resource and unordered-access states, both legal on a
//! compute queue. Buffers fall back to the common state when an
//! `ExecuteCommandLists` ends and are promoted from it on first use, on
//! either queue; the compute queue's wait is for the end of the flight's
//! lists, so the copy is common when the compare starts. No texture goes
//! this way.
//!
//! What is not ordered: shader writes nobody declared (the design never
//! lands those but by chance, through a later declared piece; `crate::picked`
//! declares what goes through views shaders pick, and `Shadow::keep_stores`
//! names copies known to get such stores). One of them beside a compare is
//! a write on the direct queue beside a read on the compute queue of the
//! same buffer (the rule above again; no layout or compression to break
//! for a buffer). The compare then reads half-written bytes there, which is what the fetch
//! on the copy queue (`VICTORY_RELAY_COPY=1`) already does beside later
//! flights; the landing takes the piece as fetched, and the next compare
//! flags whatever changed after this one hashed it (`torn`, with
//! `VICTORY_FETCH_CHECK=1`, catches a piece fetched other than hashed).
//!
//! Only flights of the relay go this way (`VICTORY_RELAY`; a submission
//! that is waited for keeps its compare on the direct queue, and it settles
//! everything in the air before it, so no compare is pending then).

use super::*;

/// `VICTORY_COMPARE_QUEUE`: 0 (off, unset), `1` (compares on the compute
/// queue; the direct queue waits for every compare in the air of a copy
/// before it writes that copy), `2` (as 1, but a flight's declared shader
/// writes wait only where their pieces touch the pieces compared; sends up
/// and rehashes still per copy), `3` (as 2, and sends up and rehashes wait
/// per piece too). Diagnostics builds: named in
/// `VICTORY_AB`, on (at the level asked, at least 1) and off while the
/// title runs; a compare in the air is waited for by whatever touches its
/// copy either way.
pub(super) fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    let asked = *LEVEL.get_or_init(|| {
        let level = match std::env::var("VICTORY_COMPARE_QUEUE").ok().as_deref().map(str::trim) {
            Some("1") => 1,
            Some("2") => 2,
            Some("3") => 3,
            _ => 0,
        };
        if level > 0 && !relay() {
            tracing::warn!(
                "VICTORY_COMPARE_QUEUE={level} needs the relay (VICTORY_RELAY): compares stay on the direct queue"
            );
        } else if level > 0 {
            tracing::info!(level, "compares of copies in the card's memory go to a compute queue of their own (VICTORY_COMPARE_QUEUE)");
        }
        level
    });
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_COMPARE_QUEUE") {
        return if on { asked.max(1) } else { 0 };
    }
    asked
}

/// Whether declared shader writes wait only for compares of the pieces they
/// write ([`level`] 2). While the switch is off (A/B), per copy: the
/// stricter.
pub(super) fn by_pieces() -> bool {
    level() >= 2
}

/// Whether sends up and their rehashes wait only for compares of the pieces
/// they write ([`level`] 3).
pub(super) fn sends_by_pieces() -> bool {
    level() == 3
}

/// The compute queue, its fence, and the last value signalled on it (under
/// the submission lock).
pub(super) struct CompareQueue {
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    sent: std::sync::atomic::AtomicU64,
}

// SAFETY: D3D12 queues and fences are free-threaded.
unsafe impl Send for CompareQueue {}
// SAFETY: as above.
unsafe impl Sync for CompareQueue {}

static QUEUE: OnceLock<Option<CompareQueue>> = OnceLock::new();

/// The queue, made at the first ask; `None` if it could not be made (the
/// compares then stay on the direct queue, said once).
pub(super) fn queue(host: &Host) -> Option<&'static CompareQueue> {
    QUEUE
        .get_or_init(|| {
            // SAFETY: plain object creation.
            let made = unsafe {
                (|| -> windows::core::Result<CompareQueue> {
                    let queue: ID3D12CommandQueue = host.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                        ..Default::default()
                    })?;
                    let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                    Ok(CompareQueue {
                        queue,
                        fence,
                        sent: std::sync::atomic::AtomicU64::new(0),
                    })
                })()
            };
            match made {
                Ok(made) => {
                    name(&made.queue, || "victory compare queue (compute)".into());
                    name(&made.fence, || "victory compare queue fence".into());
                    Some(made)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "no compute queue for compares (VICTORY_COMPARE_QUEUE): they stay on the direct queue");
                    None
                }
            }
        })
        .as_ref()
}

/// An allocator and a closed list of the compute type, for a compare.
pub(super) fn make_list(host: &Host) -> Option<Prelude> {
    // SAFETY: plain object creation.
    let made: windows::core::Result<Prelude> = unsafe {
        (|| {
            let allocator: ID3D12CommandAllocator = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)?;
            let list: ID3D12GraphicsCommandList =
                host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, None)?;
            list.Close()?;
            Ok((allocator, list))
        })()
    };
    LISTS[1].fetch_add(1, Ordering::Relaxed);
    match made {
        Ok(prelude) => Some(prelude),
        Err(e) => {
            tracing::warn!(error = %e, "no compute list for a compare");
            None
        }
    }
}

/// The value the compare queue's fence has reached; 0 without the queue
/// (every flight's `checked` is 0 then).
pub(super) fn completed() -> u64 {
    match QUEUE.get() {
        // SAFETY: a plain query.
        Some(Some(q)) => unsafe { q.fence.GetCompletedValue() },
        _ => 0,
    }
}

/// A compare sent to the compute queue and maybe not done: the copy
/// (`Shadow::id`), the pieces it reads and writes the hashes of, widened by
/// one on each side (from, to), and the value signalled behind it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pending {
    copy: u64,
    from: u32,
    to: u32,
    value: u64,
}

static PENDING: Mutex<Vec<Pending>> = Mutex::new(Vec::new());

/// The pieces of a compared or written span `first..first + pieces`, widened
/// by one on each side.
fn widened(first: u32, pieces: u32) -> (u32, u32) {
    (first.saturating_sub(1), first.saturating_add(pieces).saturating_add(1))
}

/// Of `pending`, the newest compare value that writing pieces `[from, to)`
/// of `copy` must wait for (per piece or per copy); 0 for none.
fn needed(pending: &[Pending], copy: u64, from: u32, to: u32, pieces: bool) -> u64 {
    pending
        .iter()
        .filter(|p| p.copy == copy && (!pieces || (p.from < to && from < p.to)))
        .map(|p| p.value)
        .max()
        .unwrap_or(0)
}

/// What the direct queue must wait for before it writes `touched` (copy,
/// first piece, pieces; `u32::MAX` pieces: all of it): the newest compare
/// value in the air of those copies (`pieces`: only of those pieces); 0 for
/// none. Compares the card has done are forgotten here. If any is in the
/// air, `looked` ([`STATS`]' place) is counted: what could have waited.
/// `touched` is not looked at when none is, so the caller may make it
/// lazily. (Per copy, the caller gives one entry per copy.)
pub(super) fn hazard(touched: impl Iterator<Item = (u64, u32, u32)>, pieces: bool, looked: usize) -> u64 {
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    if pending.is_empty() {
        return 0;
    }
    let done = completed();
    pending.retain(|p| p.value > done);
    if pending.is_empty() {
        return 0;
    }
    let mut touched = touched.peekable();
    if touched.peek().is_none() {
        return 0;
    }
    STATS[looked].fetch_add(1, Ordering::Relaxed);
    touched
        .map(|(copy, first, n)| {
            let (from, to) = if n == u32::MAX { (0, u32::MAX) } else { widened(first, n) };
            needed(&pending, copy, from, to, pieces)
        })
        .max()
        .unwrap_or(0)
}

/// The newest compare in the air at all; 0 for none.
pub(super) fn any_pending() -> u64 {
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    let done = completed();
    pending.retain(|p| p.value > done);
    pending.iter().map(|p| p.value).max().unwrap_or(0)
}

/// Why the direct queue waited for the compare queue ([`STATS`]' places):
/// before a send-up list, per copy or per piece ([`level`] 3); before a
/// flight's declared writes; before a compare on the direct queue.
pub(super) const BEFORE_SEND: usize = 2;
pub(super) const BEFORE_SEND_PIECES: usize = 7;
pub(super) const BEFORE_WRITES: usize = 3;
pub(super) const BEFORE_DIRECT_COMPARE: usize = 4;
/// Send-up lists, and submissions with declared writes to copies in the
/// card's memory, looked at while compares were in the air ([`hazard`]).
pub(super) const SENDS_LOOKED: usize = 8;
pub(super) const WRITES_LOOKED: usize = 9;

/// Since the last frame-rate line: compares sent to the compute queue,
/// their passes; GPU waits of the direct queue for it, by why
/// ([`BEFORE_SEND`]..); waits of the thread that lands flights for it, and
/// their microseconds; then [`BEFORE_SEND_PIECES`], [`SENDS_LOOKED`],
/// [`WRITES_LOOKED`].
pub(super) static STATS: [std::sync::atomic::AtomicU64; 10] = [const { std::sync::atomic::AtomicU64::new(0) }; 10];

/// The direct queue `queue` waits, on the card, until the compare queue has
/// passed `value` (0: nothing to wait for). Before an `ExecuteCommandLists`
/// on it, under the submission lock.
pub(super) fn hold(queue: &ID3D12CommandQueue, value: u64, why: usize) {
    if value == 0 {
        return;
    }
    let Some(Some(q)) = QUEUE.get() else {
        return;
    };
    // SAFETY: our queue and fence; the value was signalled (sent) before.
    if let Err(e) = unsafe { queue.Wait(&q.fence, value) } {
        fail(format_args!("the direct queue could not be made to wait for a compare (VICTORY_COMPARE_QUEUE, {e})"));
    }
    STATS[why].fetch_add(1, Ordering::Relaxed);
}

/// The compare `list` (a closed compute list) goes to the compute queue
/// behind the direct queue's `fence` reaching `after` (the value behind its
/// flight); `ranges` (copy, first piece, pieces) are what it compares.
/// Returns the value signalled behind it on the compare queue's fence.
pub(super) fn send(
    q: &CompareQueue,
    fence: &ID3D12Fence,
    after: u64,
    list: &ID3D12CommandList,
    ranges: impl Iterator<Item = (u64, u32, u32)>,
) -> u64 {
    let value = q.sent.fetch_add(1, Ordering::AcqRel) + 1;
    // SAFETY: a closed list of the compute type; our queue and fences;
    // `after` was signalled on the direct queue before this call.
    let sent = unsafe {
        q.queue.Wait(fence, after).and_then(|()| {
            q.queue.ExecuteCommandLists(&[Some(list.clone())]);
            q.queue.Signal(&q.fence, value)
        })
    };
    if let Err(e) = sent {
        fail(format_args!("a compare could not be sent to the compute queue (VICTORY_COMPARE_QUEUE, {e})"));
    }
    let mut passes = 0;
    {
        let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        for (copy, first, pieces) in ranges {
            let (from, to) = widened(first, pieces);
            pending.push(Pending { copy, from, to, value });
            passes += 1;
        }
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[1].fetch_add(passes, Ordering::Relaxed);
    value
}

/// On the thread that lands flights: waits until the compare queue has
/// passed `value`, with `event` (that thread's own).
pub(super) fn wait_landing(value: u64, event: HANDLE) {
    let Some(Some(q)) = QUEUE.get() else {
        return;
    };
    let waiting = std::time::Instant::now();
    // SAFETY: our fence and the caller's event.
    let waited = unsafe { q.fence.SetEventOnCompletion(value, event).map(|()| WaitForSingleObject(event, INFINITE)) };
    if waited.is_err() {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    STATS[5].fetch_add(1, Ordering::Relaxed);
    STATS[6].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// [`super::landable`] with the compare: of the line's things in order (a
/// flight: the values of the host's fence and of the compare queue's fence
/// behind it, 0 for no compare there; else `None`), how many from the front
/// can be landed when the fences have reached `completed`.
pub(super) fn landable(line: &[Option<(u64, u64)>], completed: (u64, u64)) -> usize {
    line.iter()
        .position(|f| f.is_some_and(|(fence, compare)| fence > completed.0 || compare > completed.1))
        .unwrap_or(line.len())
}

/// The frame-rate line's words for it.
pub(super) fn log(frames: f64) {
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if s.iter().all(|&s| s == 0.0) {
        return;
    }
    let level = level();
    tracing::info!(
        "LOCAL (queues) compares on the compute queue per frame (VICTORY_COMPARE_QUEUE={level}): {:.1} compare lists of {:.1} \
         passes; send-up lists while compares were in the air {:.1}: held for them {:.1} by copy + {:.1} by piece, beside them \
         {:.1}; submissions with declared writes to card copies while compares were in the air {:.1}: held {:.1} ({}), beside \
         them {:.1}; held before a compare on the direct queue {:.1}; the landing thread waited {:.1} times, {:.2} ms, for them",
        s[0],
        s[1],
        s[SENDS_LOOKED],
        s[BEFORE_SEND],
        s[BEFORE_SEND_PIECES],
        (s[SENDS_LOOKED] - s[BEFORE_SEND] - s[BEFORE_SEND_PIECES]).max(0.0),
        s[WRITES_LOOKED],
        s[BEFORE_WRITES],
        if level >= 2 { "by piece" } else { "by copy" },
        (s[WRITES_LOOKED] - s[BEFORE_WRITES]).max(0.0),
        s[BEFORE_DIRECT_COMPARE],
        s[5],
        s[6] / 1000.0,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(copy: u64, from: u32, to: u32, value: u64) -> Pending {
        Pending { copy, from, to, value }
    }

    /// Per copy, any compare of the copy is waited for; per piece only those
    /// whose widened pieces meet the ones written; other copies never.
    #[test]
    fn writes_wait_for_compares_of_what_they_touch() {
        let pending = [p(1, 9, 13, 5), p(1, 40, 43, 7), p(2, 0, 4, 9)];
        // Per copy.
        assert_eq!(needed(&pending, 1, 100, 101, false), 7);
        assert_eq!(needed(&pending, 2, 100, 101, false), 9);
        assert_eq!(needed(&pending, 3, 0, u32::MAX, false), 0);
        // Per piece: pieces 10..12 compared, widened to 9..13.
        assert_eq!(needed(&pending, 1, 13, 20, true), 0);
        assert_eq!(needed(&pending, 1, 12, 20, true), 5);
        assert_eq!(needed(&pending, 1, 0, 9, true), 0);
        assert_eq!(needed(&pending, 1, 0, 10, true), 5);
        assert_eq!(needed(&pending, 1, 0, u32::MAX, true), 7);
        assert_eq!(needed(&pending, 1, 20, 30, true), 0);
    }

    /// A span is widened by one piece on each side, and never wraps.
    #[test]
    fn spans_are_widened_by_a_piece() {
        assert_eq!(widened(10, 3), (9, 14));
        assert_eq!(widened(0, 1), (0, 2));
        assert_eq!(widened(u32::MAX - 1, 5), (u32::MAX - 2, u32::MAX));
        // A write of piece 14 meets the compare of pieces 10..13 (widened
        // to 9..14) only through its own widening.
        let (from, to) = widened(14, 1);
        assert_eq!(needed(&[p(1, 9, 14, 3)], 1, from, to, true), 3);
    }

    /// Level 3: a send up waits for a compare when its pieces come near the
    /// compared ones (the compare reads under 16 bytes into a neighbour; both
    /// sides are widened by a piece, so one piece between them still waits,
    /// two do not); a copy hashed whole waits for every compare of it.
    #[test]
    fn sends_by_piece_wait_only_near_compared_pieces() {
        // Pieces 10..13 compared (10, 11, 12), as `send` keeps them.
        let (from, to) = widened(10, 3);
        let pending = [p(1, from, to, 4)];
        let sent = |first: u32, pieces: u32| {
            let (a, b) = widened(first, pieces);
            needed(&pending, 1, a, b, true)
        };
        assert_eq!(sent(13, 1), 4);
        assert_eq!(sent(9, 1), 4);
        assert_eq!(sent(11, 1), 4);
        assert_eq!(sent(14, 1), 4);
        assert_eq!(sent(15, 1), 0);
        assert_eq!(sent(0, 8), 0);
        assert_eq!(sent(0, 9), 4);
        assert_eq!(needed(&pending, 1, 0, u32::MAX, true), 4);
        assert_eq!(needed(&pending, 2, 0, u32::MAX, true), 0);
    }

    /// A flight lands only when both its flight and its compare are done;
    /// one without a compare on the compute queue (0) as before; nothing
    /// overtakes a flight that waits.
    #[test]
    fn flights_land_when_flight_and_compare_are_done() {
        let line = [Some((5, 2)), None, Some((6, 0)), None, Some((7, 3))];
        assert_eq!(landable(&line, (4, 9)), 0);
        assert_eq!(landable(&line, (5, 1)), 0);
        assert_eq!(landable(&line, (5, 2)), 2);
        assert_eq!(landable(&line, (6, 2)), 4);
        assert_eq!(landable(&line, (9, 2)), 4);
        assert_eq!(landable(&line, (9, 3)), 5);
        // As `super::landable` where no compare went to the compute queue.
        let plain = [Some(5), None, None, Some(6), None];
        let both: Vec<Option<(u64, u64)>> = plain.iter().map(|f| f.map(|f| (f, 0))).collect();
        for completed in [0, 4, 5, 6, u64::MAX] {
            assert_eq!(landable(&both, (completed, 0)), super::super::landable(&plain, completed));
        }
    }
}
