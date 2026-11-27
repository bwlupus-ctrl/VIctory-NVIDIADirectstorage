//! Which console the title runs on. Titles pick settings (resolution,
//! memory budgets) by it, so the user chooses; Series X unless told.

const GIB: usize = 1 << 30;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Console {
    #[default]
    SeriesX,
    SeriesS,
}

impl Console {
    /// Title memory in bytes: (GPU-optimal, standard). Series X in the
    /// advanced configuration; Series S has one range, counted as standard
    /// (7,776 MiB, the published console-mode figure).
    // shortcut: later system updates gave Series S titles more (amount not
    // published here); the title sizes nothing from this figure
    // (it goes by the device type), so update it if a title does.
    pub const fn title_memory(self) -> (usize, usize) {
        match self {
            Console::SeriesX => (10 * GIB, 7 * GIB / 2),
            Console::SeriesS => (0, 7_776 << 20),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Console::SeriesX => "Xbox Series X",
            Console::SeriesS => "Xbox Series S",
        }
    }
}

/// Set before the title starts (before it makes any resource: render-XOR
/// layouts take their addresses from the model, DexNom-VIctory 570a941).
/// The model is kept in one place, `victory_formats::texture`, which the
/// layouts read.
pub fn set(console: Console) {
    if !victory_formats::texture::set_series_s_tiling(console == Console::SeriesS) {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            tracing::error!(
                console = console.name(),
                "the console model changed after render-XOR textures were laid out for the other: \
                 those keep the other's addresses (set the console before the title starts)"
            );
        });
    }
}

pub fn current() -> Console {
    if victory_formats::texture::series_s_tiling() {
        Console::SeriesS
    } else {
        Console::SeriesX
    }
}
