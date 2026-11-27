//! LOCAL (queues): what the title's compute queue gives the host, counted
//! (`VICTORY_COMPUTE_QUEUE=1`, with `VICTORY_ASYNC_SUBMIT=1` and the relay;
//! off by default). Measurement only: every submission still goes to the
//! one direct queue, as without the switch.
//!
//! Why not a host compute queue for them (the switch's first aim). Each of
//! these stands in the way, and the first two alone make it unsafe in
//! general:
//!
//! - Host lists are made as direct lists when the title makes its list
//!   (`Host::make_list`, the pool in `crate::list`), and are recorded into
//!   as commands come: the queue they will go to is not known then, and a
//!   direct list cannot go to a compute queue. A compute host list would
//!   have to hold only what a compute queue may run, decided while
//!   recording, with no way back to the direct queue once a command that
//!   needs it (a texture transition to or from a pixel-shader or depth
//!   state in the prelude, a depth twin copy, a plane pass, which draws)
//!   has been recorded.
//! - D3D12 lets one queue at a time write a resource ("a resource cannot be
//!   written to from multiple command queues simultaneously", "Executing and
//!   synchronizing command lists"). The title's memory heap is one host
//!   buffer in the card's memory (174 MB, the skinning passes and the ring
//!   of constants in it), and shaders reach it through the bindless table:
//!   what a compute submission writes there is not known before it runs
//!   (`Shadow::keep_stores`: stores nobody declared). So "send it to the
//!   compute queue only when no direct work in the air writes what it
//!   names" cannot be decided: what it names is not known.
//! - Texture states are tracked once for all (`HostList::prelude` swaps a
//!   texture's host state at submission, in submission order). Two queues
//!   running at once would each transition from a state the other may not
//!   have reached, and a texture (unlike a buffer) is not promoted or
//!   decayed between queues.
//! - The relay's landing assumes one queue: `crate::land::Had` and the
//!   fetch's "flights it runs behind" are values of the direct queue's
//!   fence, `card_steps`' holder takes every writer for one ahead of it on
//!   the same queue, a wait met by a flight's bottom-of-pipe write is met
//!   by FIFO order (`late_met_known`, `queue::behind`'s `take_along`), and
//!   the compare chain and its rehashes are ordered by that one queue.
//!
//! What this counts tells how much of the frame such a queue could take at
//! best: the submissions that name no texture and declare no write into a
//! copy in the card's memory are the only candidates (an upper bound:
//! undeclared stores are not seen).

use super::*;

/// The title's queue type for compute (`D3D12_COMMAND_LIST_TYPE_COMPUTE`).
pub(crate) const COMPUTE: u32 = 2;

/// Whether the title's compute submissions are counted.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_COMPUTE_QUEUE").is_some_and(|v| v == "1");
        if asked && !relay() {
            tracing::warn!("VICTORY_COMPUTE_QUEUE=1 needs VICTORY_ASYNC_SUBMIT=1 and the relay (VICTORY_RELAY): not counted");
            return false;
        }
        if asked {
            tracing::warn!(
                "VICTORY_COMPUTE_QUEUE=1: the title's compute submissions are only counted (frame line); they stay on the direct \
                 queue (host lists are direct lists, the heap copy is one buffer written from both queues, texture states are \
                 tracked for one queue: see host/titlecompute.rs)"
            );
        }
        asked
    })
}

/// Since the last frame-rate line: compute submissions, their lists; of
/// the submissions, those that name textures, those that declare writes
/// into copies in the card's memory, those that declare writes into copies
/// in the PC's memory only, and those that do neither.
static STATS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// A submission of the title's compute queue, about to go to the host.
pub(crate) fn census(runs: &[&HostList]) {
    // (Not the prelude: with `VICTORY_LAZY_PRELUDE=0` every list has one.)
    let textures = runs.iter().any(|l| {
        !l.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            || !l.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    });
    let (mut card, mut pc) = (false, false);
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::GpuWrite(shadow, ..) = step {
                if shadow.carded() { card = true } else { pc = true }
            }
        }
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[1].fetch_add(runs.len() as u64, Ordering::Relaxed);
    let at = if textures {
        2
    } else if card {
        3
    } else if pc {
        4
    } else {
        5
    };
    STATS[at].fetch_add(1, Ordering::Relaxed);
}

/// The frame-rate line's words for it.
pub(super) fn log(frames: f64) {
    if !on() {
        return;
    }
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "LOCAL (queues) the title's compute-queue submissions per frame (VICTORY_COMPUTE_QUEUE=1, counted only, all on the direct \
         queue): {:.1} of {:.1} lists; {:.1} name textures, {:.1} declare writes into copies in the card's memory, {:.1} declare \
         writes into copies in the PC's memory only, {:.1} neither (the most a host compute queue could take)",
        s[0],
        s[1],
        s[2],
        s[3],
        s[4],
        s[5],
    );
}
