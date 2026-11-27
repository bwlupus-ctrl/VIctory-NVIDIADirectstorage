# VIctory-NVIDIA DirectStorage

An Xbox Series compatibility layer for running the Xbox build of GTA VI on a
Windows PC with an **NVIDIA GeForce RTX** card. It is tuned and tested on an
RTX 5090 (32 GB) with a Ryzen 9 9950X3D, running the Series S profile at 1080p.

## The idea

An Xbox Series console has one pool of fast GDDR6 that the CPU and the GPU share:
- Series X: 10 GB at 560 GB/s plus 6 GB at 336 GB/s.
- Series S: 8 GB at 224 GB/s plus 2 GB at 56 GB/s.

The game streams its open world constantly and expects that bus. A PC has two
separate memories, so keeping them in step costs most of each frame.

This fork follows one rule: **graphics data lives on the GPU, and the CPU never
shuttles it.**

## What it adds

Everything is behind a switch.

- **Write-combined block in VRAM** (`VICTORY_WC_VRAM`): the game's CPU writes go
  straight to the card through resizable BAR, not through a copy. Average fps
  went from about 17.7 to about 23.
- **No slowdown over long sessions:** a descriptor view-table leak is fixed.
  Before the fix, fps fell from 24.5 to 11.6 over an hour.
- **The game's own page-kind table** (`VICTORY_PAGE_KINDS`): finds which memory
  only the GPU uses (about 2 GB). This is groundwork for keeping that memory on
  the card entirely.
- **X3D V-Cache pinning** (`VICTORY_VCACHE_CORES`): keeps the game on the cores
  with the large cache.
- **Picture fixes:**
  - the dark blocks on open water
  - early depth testing, against the haze and white specks
  - a leave-out list for broken passes
  - the facial animation fix
  - the NaN scrub
- **Submit-thread speed-ups.**
- **DLSS files, verified:** `scripts/get-dlss.sh` downloads NVIDIA's Streamline
  files. It checks the SHA-256 and NVIDIA's signature on every DLL and runs
  nothing it downloads. The DLSS code itself is not wired in yet.

The full list of switches, recommended settings, results, known issues and
roadmap is in **[docs/nvidia-directstorage-fork.md](docs/nvidia-directstorage-fork.md)**.

## Build

You need [Rust](https://rustup.rs) and the Visual Studio C++ build tools with a
Windows SDK.

```bash
cargo build --profile diag -p victory --features diagnostics
```

## Run

Use your own legally obtained copy of the game. The recommended settings are in
the docs. In short:

```bash
victory.exe run "<game folder>\game_scarlett_master.exe" --execute --console series-s
```

## Status

This is a work in progress. Known issues:
- Micro-stutter from memory remapping. The fix is planned: GPU tile remaps, with
  soft views as the fallback.
- Some volumetric effects show white specks indoors.
- 1440p is not available on the Series S profile; it needs the Series X path.

## Credits

This fork is built on code from these projects, under their licences:
- [Marvisak/VIctory](https://github.com/Marvisak/VIctory): the original
  compatibility layer.
- [trustno1-coder/VIctory-NVIDIA](https://github.com/trustno1-coder/VIctory-NVIDIA):
  the NVIDIA fork this one is based on, which brings in work from fxceless,
  avatarmanz and others.
- Ports from:
  - [DexNom/VIctory](https://github.com/DexNom/VIctory)
  - [codec-d/cool_fox](https://github.com/codec-d/cool_fox)
  - [polackangry/POLSKAGUROM](https://github.com/polackangry/POLSKAGUROM)
  - [Op-lite/victory-scarlett](https://github.com/Op-lite/victory-scarlett)

Licensed under the GNU GPL v3 (see [LICENSE](LICENSE)). Made with
[Claude Code](https://claude.com/claude-code).
