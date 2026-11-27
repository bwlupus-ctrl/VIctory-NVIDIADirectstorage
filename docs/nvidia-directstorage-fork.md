# VIctory-NVIDIA DirectStorage fork: what this branch adds

This branch (`nvidia-vram-wip`) is built on
[trustno1-coder/VIctory-NVIDIA](https://github.com/trustno1-coder/VIctory-NVIDIA)
at `3e5fb36`, plus selected later fixes from its `main`. It is tuned on one machine:
an RTX 5090 (32 GB), a Ryzen 9 9950X3D and 176 GB of RAM, running the Series S
profile at 1080p.

## The idea behind it

An Xbox Series console has one pool of fast GDDR6 shared by the CPU and the GPU.
A Series X has 10 GB at 560 GB/s plus 6 GB at 336 GB/s. A Series S has 8 GB at
224 GB/s plus 2 GB at 56 GB/s. The game streams the open world constantly and
expects that bus.

On a PC, emulating that shared memory costs about 35 ms of a 58 ms frame:
- write watch
- page guards
- copies up to the card
- copies back to the title

This fork works from one rule: **graphics data lives on the GPU, and the CPU
never shuttles it.** Producers say what they changed rather than having it
found by scanning.

## Results on the test machine

| | Average fps |
|---|---|
| Base fork, same settings | ~17.7 |
| With the write-combined block in VRAM (`VICTORY_WC_VRAM`) | ~23 |
| One-hour run before the view-table fix | 24.5 falling to 11.6 |
| One-hour run after the view-table fix | 22.8, no decay (24.7 → ~22) |

The frame is bound by the CPU and the submit thread, not the GPU: the card is
about 20% busy.

## What this branch adds

Everything is behind a switch and **off unless noted**.

### Memory and streaming (our own work)

| Switch | What it does |
|---|---|
| `VICTORY_WC_VRAM=1` | Puts the title's write-combined block (about 1.3 GB) in VRAM through `GPU_UPLOAD` heaps (resizable BAR). The title's CPU writes go straight to the card, so there is no copy up. See `crates/victory-gpu/src/wc_vram.rs`. |
| `VICTORY_WC_VRAM_GPU_FILLS=1` | With `WC_VRAM`, fills of that block are done on the card. |
| `VICTORY_LAND_ASIDE=1` | Copy-back landing runs beside the submitting thread. |
| `VICTORY_PAGE_KINDS=1` | Reads the title's own page-kind table: bit 62 means CPU-mapped, bit 63 GPU-mapped, one entry per 64 KiB page. In game, about 2 GB of the 16 GB streaming region is GPU-only. This is the base for moving GPU-only pages fully to the card. See `pagekinds.rs`. |
| `VICTORY_CLASS_STATS=1` | Per memory-class cost and streaming counters. See `memclass.rs`. |
| `VICTORY_STREAM_JOURNAL=1` | A journal of what producers changed, so it doesn't have to be found by scanning. See `stream_journal.rs`. |
| `VICTORY_HEAP_PROBE=1` | An experimental probe of the title's 174 MB GPU heap. Known issue: its target finder does not find the heap yet. |
| `VICTORY_VCACHE_CORES=2` | Pins the process to the CCD with the 3D V-Cache, using CPU Sets found from the largest L3. Process Lasso shows this as CPU sets, not affinity. See `cores.rs`. |

### Fixes

- **Descriptor view-table leak.** Placed texture views were recorded under a new
  shape whenever their padding or `ResourceMinLODClamp` changed. The table grew
  for hours and fps fell with it. Fixed with `LOD_PLACED` records and cleared
  padding (`descriptor.rs`). `VICTORY_VIEW_SHAPE_KEY=old` brings back the old
  keys for comparison.
- **Ports from other forks:**
  - [DexNom/VIctory](https://github.com/DexNom/VIctory): d16 facial animation fix, NaN scrub (`VICTORY_NAN_SCRUB=1`), half-precision handling.
  - [codec-d/cool_fox](https://github.com/codec-d/cool_fox): slot release and other fixes.
  - [polackangry/POLSKAGUROM](https://github.com/polackangry/POLSKAGUROM): selected fixes.
  - [Op-lite/victory-scarlett](https://github.com/Op-lite/victory-scarlett): page-kind globals and cheap fixes.
  - A paired wave64 lowering for 32-lane NVIDIA hardware.

### From trustno1-coder's `main` (ported 2026-11-27, reviewed)

| Switch | What it does |
|---|---|
| Water light (default on) | The two tiled water light programs (`5dc3b25986622949`, `6f225b641c5b0708`) are left out on 32-lane hosts, where they leave dark blocks on open water. `VICTORY_WATER_LIGHT=1` runs them. |
| `VICTORY_LEAVE_OUT=<hash>,<hash>` | Never translates or makes the named pipelines. Recommended: `731a39171b568298,02384a60fda6f2d0` (the heat-haze pass with its smeared blocks, and the DefragHeap pass, 11.9 ms of GPU). |
| `VICTORY_EARLY_DEPTH=1` | Pixel programs keep the console's early depth test. Upstream says this fixes the haze and white specks around transparent surfaces. |
| Submit-thread cuts (default on) | Only as many helper threads are woken as there are jobs, and a submission's batch is reused. |
| `VICTORY_HEAP_SYNC_FIRST=1`, `VICTORY_SPANS_QUICK=1`, `VICTORY_SYNC_FINE=1` | Heap-record sync speed-ups. |
| `VICTORY_HEAP_SYNC_AT_SUBMIT`, `VICTORY_SYNC_IN_PLACE` | **Keep these off.** Reviews found ordering bugs that can give stale descriptors. `SYNC_IN_PLACE` also bypasses the `WC_VRAM` writer fence. |

### Resolution

| Switch | What it does |
|---|---|
| `VICTORY_RENDER_SCALE=0.2..1.0` | Pins the title's internal render scale. |
| `VICTORY_OWN_UPSCALER=1` | Makes Series S use the title's own upscaler instead of FSR. |
| `VICTORY_DISPLAY_MODE=WxH` | The display size the console is told. **Don't use 2560x1440 on Series S:** the title stops at start on its scanout allocation check. Real 1440p needs the Series X path (`native_1440p.rs` upstream, not ported yet). |

## Recommended settings

These are the settings used for the numbers above, for the Series S profile:

```bash
export VICTORY_THIN=1 VICTORY_ASYNC_SUBMIT=1 VICTORY_OWN_UPSCALER=1 VICTORY_COMPOSE=1
export VICTORY_F16_FIX=1 VICTORY_DESC_CACHE=1 VICTORY_NULL_TEMPLATE=1 VICTORY_UPLOAD_POOL=1 \
  VICTORY_ASYNC_BURY=1 VICTORY_MAPPED_BATCH=1 VICTORY_BUFFER_TIERS=1 VICTORY_STREAM_JOURNAL=1 \
  VICTORY_WC_VRAM=1 VICTORY_WC_VRAM_GPU_FILLS=1 VICTORY_LAND_ASIDE=1 VICTORY_VCACHE_CORES=2 \
  VICTORY_NAN_SCRUB=1 VICTORY_PAGE_KINDS=1
export VICTORY_LEAVE_OUT=731a39171b568298,02384a60fda6f2d0 VICTORY_EARLY_DEPTH=1
export VICTORY_HEAP_SYNC_FIRST=1 VICTORY_SPANS_QUICK=1 VICTORY_SYNC_FINE=1
export VICTORY_PIPELINE_WAIT_MS=50   # on a cold cache: brief pop-in instead of multi-second freezes
victory.exe run "<game folder>\game_scarlett_master.exe" --execute --console series-s
```

Build it with:

```bash
cargo build --profile diag -p victory --features diagnostics
```

## DLSS files

`scripts/get-dlss.sh` fetches the NVIDIA Streamline files that DLSS and frame
generation need. It writes to `%LOCALAPPDATA%\victory-nvidia\nvidia-sdk\`. It is
stricter than upstream's script:
- It downloads one file only: the pinned `streamline-sdk-v2.14.16.zip` from
  NVIDIA's GitHub.
- It checks the zip's SHA-256 against the digest GitHub publishes, and stops on
  a mismatch.
- It takes out only the 8 DLLs and the licence texts.
- It refuses to install unless every DLL has a valid NVIDIA Corporation
  Authenticode signature.
- It never runs anything it downloads.

The DLSS and frame generation code itself (upstream `streamline.rs`) is
**not ported yet**.

## Known issues

- **Micro-stutter from memory remapping.** Each `MapViewOfFile3` and unmap of a
  title view goes through a kernel lock that the write-watch guards also take.
  Upstream's fix, `VICTORY_SOFT_VIEWS` (which needs lazy views), is not ported.
- **Water and volumetrics.** The three volumetric programs that only translate
  in the paired wave64 form can leave white specks indoors.
- **Texture streaming waits.** Thousands of waits for new card memory per
  session.
- **First run after a cache clear.** Until `victory shaders <shaders.rpf>`
  pre-translation is used, it compiles thousands of pipelines while playing.

## Roadmap

1. GPU-first remaps. Page-mapping copies whose pages are GPU-only (from
   `VICTORY_PAGE_KINDS`) become `UpdateTileMappings`/`CopyTileMappings` on the
   card, with no host view and no byte copy. Soft views cover CPU-mapped pages.
2. DLSS and frame generation with native 1440p (Series X). Frame generation
   suits a CPU-bound frame best.
3. DirectStorage with GPU GDeflate, and Sampler Feedback Streaming.
4. Stage 2 of the page kinds: GPU-only pages move to a VRAM pool.

Made with [Claude Code](https://claude.com/claude-code). Reviews by Fable and
Codex.
