//! LOCAL: the read map (`VICTORY_READ_MAP=<file>`): which parts of the
//! title's package files the title reads, kept over sessions, so that a
//! preloader can bring just those parts into the host's file cache before
//! the next session. Unset (the default) nothing here runs: [`on`] is one
//! load of a flag and every caller asks it first.
//!
//! What is recorded: every read of a file under the package folder (the
//! `G:` drive) that goes through DirectStorage (`dstorage`), the title's
//! `ReadFile` on a handle from `CreateFileW`/`CreateFileA`/`CreateFile2`
//! (at the file pointer, or at the `OVERLAPPED` offset), and the title's
//! `fread` on a stream from `fopen`/`_wfopen` and their kin. Files that are
//! only opened (read by `fgetc`, or by the C++ library after `_Fiopen`,
//! whose reads never pass through an import) are counted as opened, with no
//! chunks: a preloader takes such a file whole when it is small.
//!
//! Recording is one add per 1 MiB chunk on counters made when the file is
//! opened: no lock is held and nothing is allocated per read. A thread
//! writes the map every 30 s, and the ways out of the process write it once
//! more ([`flush`]). The file is written beside itself (`<file>.tmp`) and
//! renamed over the old one, so a crash leaves the old map or the new one.
//!
//! The file is JSON, one file of the title's per line:
//!
//! ```json
//! {"format":"victory-read-map","version":1,"chunk_bytes":1048576,
//!  "sessions":3,"session_times":[1795000000,1795003600,1795090000],
//!  "files":{
//!   "levels.rpf":{"size":76155065056,"opens":5,"last_session":3,
//!    "chunks":[[0,12,3,3],[1,4,2,3],[57,1,1,1]]}
//!  }}
//! ```
//!
//! - `sessions`: how many sessions have written to this map; sessions are
//!   numbered from 1, and `session_times[n - 1]` is when session `n`
//!   started (seconds since 1970).
//! - `files`: by path under the package folder, with `/` between parts, in
//!   the case the host stores. `size`: the file's size in bytes; a file
//!   whose size has changed starts over. `opens`: how often it was opened,
//!   over all sessions. `last_session`: the last session that opened or
//!   read it.
//! - `chunks`: `[index, reads, sessions, last_session]`, ascending by
//!   index, only chunks read at least once. Chunk `index` covers bytes
//!   `index * chunk_bytes ..` of the file; `reads`: the reads that touched
//!   it, over all sessions; `sessions`: in how many sessions it was read;
//!   `last_session`: the last session that read it.
//!
//! A session's numbers are added to what the file held when the session
//! started; two sessions writing one map at the same time lose one of the
//! two. A map that cannot be read is kept as `<file>.corrupt` and a new one
//! started.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

/// Bytes a chunk covers.
pub(crate) const CHUNK: u64 = 1 << 20;
/// Seconds between writes of the map.
const EVERY: u64 = 30;

// What a session counts.

/// One file's counters for this session. Lives as long as the process: the
/// title's handles and DirectStorage's file objects point at it.
pub(crate) struct FileCounts {
    /// Path under the package folder, `/` between parts.
    path: String,
    size: u64,
    opens: AtomicU32,
    /// Reads that touched each chunk.
    chunks: Box<[AtomicU32]>,
    /// The session's record count, to tell a write has something new.
    changes: &'static AtomicU64,
}

impl FileCounts {
    fn new(path: String, size: u64, changes: &'static AtomicU64) -> FileCounts {
        let chunks = (0..size.div_ceil(CHUNK)).map(|_| AtomicU32::new(0)).collect();
        FileCounts { path, size, opens: AtomicU32::new(0), chunks, changes }
    }

    /// Counts a read of `len` bytes at `offset`; what lies past the end of
    /// the file (as it was when opened) is not counted.
    pub(crate) fn record(&self, offset: u64, len: u64) {
        if len == 0 || offset >= self.size {
            return;
        }
        let last = (offset.saturating_add(len).min(self.size) - 1) / CHUNK;
        for chunk in &self.chunks[(offset / CHUNK) as usize..=last as usize] {
            chunk.fetch_add(1, Ordering::Relaxed);
        }
        self.changes.fetch_add(1, Ordering::Relaxed);
    }

    fn opened(&self) {
        self.opens.fetch_add(1, Ordering::Relaxed);
        self.changes.fetch_add(1, Ordering::Relaxed);
    }
}

/// A session's files, by lower-case path.
#[derive(Default)]
struct Session {
    files: Mutex<HashMap<String, &'static FileCounts>>,
}

impl Session {
    /// The counters of `path` (made on first sight), counted as opened once
    /// more. A file seen again with another size keeps its first counters.
    fn open(&self, path: &str, size: u64, changes: &'static AtomicU64) -> &'static FileCounts {
        let path = path.replace('\\', "/");
        let mut files = self.files.lock().unwrap_or_else(|e| e.into_inner());
        let file = *files
            .entry(path.to_lowercase())
            .or_insert_with(|| Box::leak(Box::new(FileCounts::new(path, size, changes))));
        drop(files);
        file.opened();
        file
    }

    fn all(&self) -> Vec<&'static FileCounts> {
        let mut all: Vec<_> =
            self.files.lock().unwrap_or_else(|e| e.into_inner()).values().copied().collect();
        all.sort_by(|a, b| a.path.cmp(&b.path));
        all
    }
}

// What the file holds.

/// A chunk over all sessions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Chunk {
    reads: u64,
    sessions: u32,
    last_session: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct FileRecord {
    path: String,
    size: u64,
    opens: u64,
    last_session: u32,
    chunks: BTreeMap<u32, Chunk>,
}

/// The map as stored; files by lower-case path.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Stored {
    sessions: u32,
    session_times: Vec<u64>,
    files: BTreeMap<String, FileRecord>,
}

impl Stored {
    /// This map with a session's counts added, as that session's write.
    /// `self` is what the file held when the session started, so writing
    /// again later in the session adds nothing twice.
    fn with(&self, session: &[&FileCounts], started: u64) -> Stored {
        let mut out = self.clone();
        out.sessions = self.sessions + 1;
        out.session_times.resize(self.sessions as usize, 0);
        out.session_times.push(started);
        let now = out.sessions;
        for counts in session {
            let record = out.files.entry(counts.path.to_lowercase()).or_default();
            if record.size != counts.size {
                // Another file than the one the offsets were learned on.
                record.chunks.clear();
                record.opens = 0;
            }
            record.path.clone_from(&counts.path);
            record.size = counts.size;
            record.opens += u64::from(counts.opens.load(Ordering::Relaxed));
            record.last_session = now;
            for (index, reads) in counts.chunks.iter().enumerate() {
                let reads = reads.load(Ordering::Relaxed);
                if reads != 0 {
                    let chunk = record.chunks.entry(index as u32).or_default();
                    chunk.reads += u64::from(reads);
                    chunk.sessions += 1;
                    chunk.last_session = now;
                }
            }
        }
        out
    }

    fn to_json(&self) -> String {
        let mut out = String::new();
        let _ = write!(
            out,
            "{{\"format\":\"victory-read-map\",\"version\":1,\"chunk_bytes\":{CHUNK},\n \"sessions\":{},\"session_times\":[",
            self.sessions
        );
        for (i, time) in self.session_times.iter().enumerate() {
            let _ = write!(out, "{}{time}", if i == 0 { "" } else { "," });
        }
        out.push_str("],\n \"files\":{");
        for (i, file) in self.files.values().enumerate() {
            out.push_str(if i == 0 { "\n  " } else { ",\n  " });
            quote(&mut out, &file.path);
            let _ = write!(
                out,
                ":{{\"size\":{},\"opens\":{},\"last_session\":{},\"chunks\":[",
                file.size, file.opens, file.last_session
            );
            for (j, (index, chunk)) in file.chunks.iter().enumerate() {
                let _ = write!(
                    out,
                    "{}[{index},{},{},{}]",
                    if j == 0 { "" } else { "," },
                    chunk.reads,
                    chunk.sessions,
                    chunk.last_session
                );
            }
            out.push_str("]}");
        }
        out.push_str("\n }}\n");
        out
    }

    fn from_json(text: &str) -> Result<Stored, String> {
        let root = json::parse(text)?;
        if root.get("format").and_then(Json::text) != Some("victory-read-map") {
            return Err("not a read map".into());
        }
        if root.number("version")? != 1 {
            return Err("a version this build does not know".into());
        }
        if root.number("chunk_bytes")? != CHUNK {
            return Err("another chunk size".into());
        }
        let mut out = Stored { sessions: root.number("sessions")? as u32, ..Stored::default() };
        if let Some(Json::Array(times)) = root.get("session_times") {
            out.session_times = times.iter().filter_map(Json::as_number).collect();
        }
        let Some(Json::Object(files)) = root.get("files") else {
            return Err("no files".into());
        };
        for (path, file) in files {
            let mut record = FileRecord {
                path: path.clone(),
                size: file.number("size")?,
                opens: file.number("opens")?,
                last_session: file.number("last_session")? as u32,
                chunks: BTreeMap::new(),
            };
            let Some(Json::Array(chunks)) = file.get("chunks") else {
                return Err(format!("{path}: no chunks"));
            };
            for chunk in chunks {
                let numbers: Option<Vec<u64>> = match chunk {
                    Json::Array(n) if n.len() == 4 => n.iter().map(Json::as_number).collect(),
                    _ => None,
                };
                let Some(n) = numbers else {
                    return Err(format!("{path}: a chunk that is not four numbers"));
                };
                record.chunks.insert(
                    n[0] as u32,
                    Chunk { reads: n[1], sessions: n[2] as u32, last_session: n[3] as u32 },
                );
            }
            out.files.insert(path.to_lowercase(), record);
        }
        Ok(out)
    }
}

fn quote(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

use json::Json;

/// As much of JSON as a read map needs (numbers are unsigned integers),
/// read without a dependency.
mod json {
    #[derive(Debug)]
    pub(super) enum Json {
        Number(u64),
        Text(String),
        Array(Vec<Json>),
        Object(Vec<(String, Json)>),
        Other,
    }

    impl Json {
        pub(super) fn get(&self, key: &str) -> Option<&Json> {
            match self {
                Json::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
                _ => None,
            }
        }

        pub(super) fn as_number(&self) -> Option<u64> {
            match self {
                Json::Number(n) => Some(*n),
                _ => None,
            }
        }

        pub(super) fn text(&self) -> Option<&str> {
            match self {
                Json::Text(t) => Some(t),
                _ => None,
            }
        }

        pub(super) fn number(&self, key: &str) -> Result<u64, String> {
            self.get(key).and_then(Json::as_number).ok_or_else(|| format!("no number `{key}`"))
        }
    }

    pub(super) fn parse(text: &str) -> Result<Json, String> {
        let mut at = Parser { bytes: text.as_bytes(), at: 0 };
        let value = at.value(0)?;
        at.space();
        if at.at != at.bytes.len() {
            return Err(at.error("text after the end"));
        }
        Ok(value)
    }

    struct Parser<'a> {
        bytes: &'a [u8],
        at: usize,
    }

    impl Parser<'_> {
        fn error(&self, what: &str) -> String {
            format!("{what} at byte {}", self.at)
        }

        fn space(&mut self) {
            while self.bytes.get(self.at).is_some_and(u8::is_ascii_whitespace) {
                self.at += 1;
            }
        }

        fn eat(&mut self, byte: u8) -> bool {
            self.space();
            let found = self.bytes.get(self.at) == Some(&byte);
            self.at += usize::from(found);
            found
        }

        fn value(&mut self, depth: u32) -> Result<Json, String> {
            if depth > 16 {
                return Err(self.error("nested too deep"));
            }
            self.space();
            match self.bytes.get(self.at) {
                Some(b'{') => {
                    self.at += 1;
                    let mut fields = Vec::new();
                    if self.eat(b'}') {
                        return Ok(Json::Object(fields));
                    }
                    loop {
                        self.space();
                        let key = self.string()?;
                        if !self.eat(b':') {
                            return Err(self.error("no `:`"));
                        }
                        fields.push((key, self.value(depth + 1)?));
                        if self.eat(b'}') {
                            return Ok(Json::Object(fields));
                        }
                        if !self.eat(b',') {
                            return Err(self.error("no `,` or `}`"));
                        }
                    }
                }
                Some(b'[') => {
                    self.at += 1;
                    let mut items = Vec::new();
                    if self.eat(b']') {
                        return Ok(Json::Array(items));
                    }
                    loop {
                        items.push(self.value(depth + 1)?);
                        if self.eat(b']') {
                            return Ok(Json::Array(items));
                        }
                        if !self.eat(b',') {
                            return Err(self.error("no `,` or `]`"));
                        }
                    }
                }
                Some(b'"') => self.string().map(Json::Text),
                Some(b'0'..=b'9') => {
                    let start = self.at;
                    while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
                        self.at += 1;
                    }
                    if matches!(self.bytes.get(self.at), Some(b'.' | b'e' | b'E')) {
                        return Err(self.error("a number that is not a whole one"));
                    }
                    std::str::from_utf8(&self.bytes[start..self.at])
                        .ok()
                        .and_then(|digits| digits.parse().ok())
                        .map(Json::Number)
                        .ok_or_else(|| self.error("a number too large"))
                }
                Some(_) => {
                    for word in ["true", "false", "null"] {
                        if self.bytes[self.at..].starts_with(word.as_bytes()) {
                            self.at += word.len();
                            return Ok(Json::Other);
                        }
                    }
                    Err(self.error("not a value"))
                }
                None => Err(self.error("ends early")),
            }
        }

        fn string(&mut self) -> Result<String, String> {
            if self.bytes.get(self.at) != Some(&b'"') {
                return Err(self.error("no string"));
            }
            self.at += 1;
            let mut out = Vec::new();
            loop {
                let Some(&byte) = self.bytes.get(self.at) else {
                    return Err(self.error("a string that does not end"));
                };
                self.at += 1;
                match byte {
                    b'"' => break,
                    b'\\' => {
                        let Some(&escape) = self.bytes.get(self.at) else {
                            return Err(self.error("a string that does not end"));
                        };
                        self.at += 1;
                        match escape {
                            b'n' => out.push(b'\n'),
                            b't' => out.push(b'\t'),
                            b'r' => out.push(b'\r'),
                            b'b' => out.push(8),
                            b'f' => out.push(12),
                            b'u' => {
                                let code = self
                                    .bytes
                                    .get(self.at..self.at + 4)
                                    .and_then(|hex| std::str::from_utf8(hex).ok())
                                    .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                                    .ok_or_else(|| self.error("a bad \\u"))?;
                                self.at += 4;
                                // Surrogate pairs are not ours to write; a lone one is replaced.
                                let c = char::from_u32(code).unwrap_or('\u{fffd}');
                                out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
                            }
                            other => out.push(other),
                        }
                    }
                    byte => out.push(byte),
                }
            }
            String::from_utf8(out).map_err(|_| self.error("a string that is not UTF-8"))
        }
    }
}

// Writing.

/// A map being recorded: where it is written, what it held, and what this
/// session has counted.
struct Recorder {
    path: PathBuf,
    /// What the file held when the session started.
    base: Stored,
    started: u64,
    session: Session,
    changes: &'static AtomicU64,
    /// `changes` at the last write; held while writing, so writes take turns.
    written: Mutex<u64>,
}

impl Recorder {
    /// Reads what `path` holds (nothing if there is no file; a file that
    /// cannot be read as a map is kept as `<path>.corrupt`).
    fn start(path: PathBuf) -> Recorder {
        let base = match std::fs::read_to_string(&path) {
            Ok(text) => Stored::from_json(&text).unwrap_or_else(|error| {
                let aside = beside(&path, "corrupt");
                let kept = std::fs::rename(&path, &aside).is_ok();
                tracing::warn!(path = %path.display(), error, kept, "read map: the map could not be read; a new one is started");
                Stored::default()
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Stored::default(),
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "read map: the map could not be opened; a new one is started");
                Stored::default()
            }
        };
        let started = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |t| t.as_secs());
        Recorder {
            path,
            base,
            started,
            session: Session::default(),
            changes: Box::leak(Box::new(AtomicU64::new(0))),
            written: Mutex::new(0),
        }
    }

    fn open(&self, path: &str, size: u64) -> &'static FileCounts {
        self.session.open(path, size, self.changes)
    }

    /// Writes the map if the session has counted anything since the last
    /// write: to `<path>.tmp`, then renamed over `path`.
    fn write(&self) -> std::io::Result<bool> {
        let mut written = self.written.lock().unwrap_or_else(|e| e.into_inner());
        let changes = self.changes.load(Ordering::Relaxed);
        if changes == *written {
            return Ok(false);
        }
        let text = self.base.with(&self.session.all(), self.started).to_json();
        let temp = beside(&self.path, "tmp");
        std::fs::write(&temp, text)?;
        std::fs::rename(&temp, &self.path)?;
        *written = changes;
        Ok(true)
    }
}

/// `path` with `.suffix` added to its name.
fn beside(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".");
    name.push(suffix);
    name.into()
}

// The process's map.

static ON: AtomicBool = AtomicBool::new(false);
static RECORDER: OnceLock<Option<Recorder>> = OnceLock::new();

fn recorder() -> Option<&'static Recorder> {
    RECORDER
        .get_or_init(|| {
            let path = std::env::var_os("VICTORY_READ_MAP").filter(|p| !p.is_empty())?;
            let recorder = Recorder::start(PathBuf::from(path));
            tracing::info!(
                path = %recorder.path.display(),
                sessions = recorder.base.sessions,
                files = recorder.base.files.len(),
                "read map: recording what the title reads (VICTORY_READ_MAP)"
            );
            ON.store(true, Ordering::Relaxed);
            let spawned = std::thread::Builder::new().name("read map".into()).spawn(|| {
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(EVERY));
                    flush();
                }
            });
            if let Err(error) = spawned {
                tracing::warn!(%error, "read map: no thread to write it; it is written at exit only");
            }
            Some(recorder)
        })
        .as_ref()
}

/// Whether a map is recorded. The first call reads the switch.
pub(crate) fn on() -> bool {
    static ASKED: std::sync::Once = std::sync::Once::new();
    ASKED.call_once(|| {
        let _ = recorder();
    });
    ON.load(Ordering::Relaxed)
}

/// Writes the map now, if one is recorded and has something new. Called
/// every 30 s and where the process ends.
pub fn flush() {
    let Some(Some(recorder)) = RECORDER.get() else {
        return;
    };
    if let Err(error) = recorder.write() {
        tracing::warn!(path = %recorder.path.display(), %error, "read map: writing the map failed");
    }
}

/// The package folder as the host names files in it (`\\?\X:\..\`, lower
/// case), set when the `G:` drive is mounted.
static ROOT: RwLock<Option<String>> = RwLock::new(None);

/// Names the package folder: files under it are the ones recorded.
pub(crate) fn set_root(root: &Path) {
    if !on() {
        return;
    }
    let root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_owned());
    let mut root = root.to_string_lossy().to_lowercase();
    if !root.ends_with('\\') {
        root.push('\\');
    }
    tracing::info!(root, "read map: the package folder");
    *ROOT.write().unwrap_or_else(|e| e.into_inner()) = Some(root);
}

/// `full` (a host path) as a path under the package folder.
fn under_root<'a>(root: &str, full: &'a str) -> Option<&'a str> {
    // Lower-casing can change a character's length; then the prefix is not ours.
    let head = full.get(..root.len())?;
    (head.to_lowercase() == root && full.len() > root.len()).then(|| &full[root.len()..])
}

#[cfg(windows)]
pub(crate) use win::*;

#[cfg(windows)]
mod win {
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::RwLock;

    use windows_sys::Win32::Storage::FileSystem::{
        FILE_CURRENT, GetFileSizeEx, GetFinalPathNameByHandleW, SetFilePointerEx,
    };

    use super::{FileCounts, ROOT, recorder, under_root};

    /// The title's open handles and streams on files of the package.
    static HANDLES: RwLock<Option<HashMap<usize, &'static FileCounts>>> = RwLock::new(None);
    static STREAMS: RwLock<Option<HashMap<usize, &'static FileCounts>>> = RwLock::new(None);

    type Table = RwLock<Option<HashMap<usize, &'static FileCounts>>>;

    fn set(table: &Table, key: usize, file: Option<&'static FileCounts>) {
        let mut table = table.write().unwrap_or_else(|e| e.into_inner());
        match file {
            Some(file) => drop(table.get_or_insert_with(HashMap::new).insert(key, file)),
            // A handle or stream reused for a file that is not ours.
            None => drop(table.as_mut().and_then(|t| t.remove(&key))),
        }
    }

    fn get(table: &Table, key: usize) -> Option<&'static FileCounts> {
        table.read().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&key).copied()
    }

    /// The counters of the file behind `handle`, if it is a file of the
    /// package; counted as opened once more.
    fn file_of(handle: *mut c_void) -> Option<&'static FileCounts> {
        let recorder = recorder()?;
        let mut name = [0u16; 1024];
        // SAFETY: a buffer of the length given; any handle value is allowed.
        let n = unsafe { GetFinalPathNameByHandleW(handle, name.as_mut_ptr(), name.len() as u32, 0) };
        if n == 0 || n as usize >= name.len() {
            return None;
        }
        let full = String::from_utf16_lossy(&name[..n as usize]);
        let root = ROOT.read().unwrap_or_else(|e| e.into_inner());
        let path = under_root(root.as_deref()?, &full)?;
        let mut size = 0i64;
        // SAFETY: an out pointer to a local.
        if unsafe { GetFileSizeEx(handle, &mut size) } == 0 {
            return None;
        }
        Some(recorder.open(path, size as u64))
    }

    /// Keeps the host's last error over `f`: the title asks for it after
    /// its own call, not ours.
    fn quietly<T>(f: impl FnOnce() -> T) -> T {
        use windows_sys::Win32::Foundation::{GetLastError, SetLastError};
        // SAFETY: plain calls.
        let error = unsafe { GetLastError() };
        let out = f();
        // SAFETY: as above.
        unsafe { SetLastError(error) };
        out
    }

    /// A handle `CreateFile*` gave the title.
    pub(crate) fn handle_opened(handle: *mut c_void) {
        if handle.is_null() || handle as isize == -1 {
            return;
        }
        quietly(|| set(&HANDLES, handle as usize, file_of(handle)));
    }

    /// The title closes `handle`.
    pub(crate) fn handle_closed(handle: *mut c_void) {
        // Nothing to take out of an empty table: no write lock for the
        // title's many other handles (events, threads).
        if get(&HANDLES, handle as usize).is_some() {
            set(&HANDLES, handle as usize, None);
        }
    }

    /// Takes over a handle `CreateFile*` gave: its reads are counted by the
    /// taker (DirectStorage), which also closes it without telling us.
    pub(crate) fn take_handle(handle: *mut c_void) -> Option<&'static FileCounts> {
        let file = get(&HANDLES, handle as usize)?;
        set(&HANDLES, handle as usize, None);
        Some(file)
    }

    /// The title's `ReadFile(handle, .., len, .., overlapped)`, before it
    /// is made: at the `OVERLAPPED` offset, or else at the file pointer.
    pub(crate) fn handle_read(handle: *mut c_void, len: u32, overlapped: *const c_void) {
        let Some(file) = get(&HANDLES, handle as usize) else {
            return;
        };
        let offset = if overlapped.is_null() {
            let mut at = 0i64;
            // SAFETY: moves the pointer by nothing, to learn where it is.
            if quietly(|| unsafe { SetFilePointerEx(handle, 0, &mut at, FILE_CURRENT) }) == 0 {
                return;
            }
            at as u64
        } else {
            // SAFETY: the title's `OVERLAPPED`; its offset is the two
            // 32-bit halves after the two pointer-sized status fields.
            let halves = unsafe { overlapped.cast::<u8>().add(16).cast::<[u32; 2]>().read_unaligned() };
            u64::from(halves[0]) | u64::from(halves[1]) << 32
        };
        file.record(offset, u64::from(len));
    }

    /// A stream `fopen` and its kin gave the title (`tracked`: its `fread`s
    /// reach us; otherwise the file is only counted as opened).
    pub(crate) fn stream_opened(stream: *mut c_void, tracked: bool) {
        if stream.is_null() {
            return;
        }
        let file = quietly(|| {
            let fileno = super::super::files::host!("ucrtbase.dll", "_fileno": extern "C" fn(*mut c_void) -> i32);
            let handle = super::super::files::host!("ucrtbase.dll", "_get_osfhandle": extern "C" fn(i32) -> isize);
            file_of(handle(fileno(stream)) as *mut c_void)
        });
        set(&STREAMS, stream as usize, file.filter(|_| tracked));
    }

    /// The title closes `stream`.
    pub(crate) fn stream_closed(stream: *mut c_void) {
        if get(&STREAMS, stream as usize).is_some() {
            set(&STREAMS, stream as usize, None);
        }
    }

    /// The title's `fread(.., size, count, stream)`, before it is made.
    pub(crate) fn stream_read(stream: *mut c_void, len: usize) {
        let Some(file) = get(&STREAMS, stream as usize) else {
            return;
        };
        let tell = super::super::files::host!("ucrtbase.dll", "_ftelli64": extern "C" fn(*mut c_void) -> i64);
        if let Ok(offset) = u64::try_from(tell(stream)) {
            file.record(offset, len as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn changes() -> &'static AtomicU64 {
        Box::leak(Box::new(AtomicU64::new(0)))
    }

    fn read(file: &FileCounts) -> Vec<u32> {
        file.chunks.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }

    #[test]
    fn reads_count_the_chunks_they_touch() {
        let file = FileCounts::new("a.rpf".into(), 3 * CHUNK + 10, changes());
        assert_eq!(file.chunks.len(), 4);
        file.record(0, 1);
        file.record(CHUNK - 1, 2); // the last byte of 0 and the first of 1
        file.record(CHUNK, CHUNK); // all of 1 and nothing of 2
        file.record(3 * CHUNK + 9, 100); // the last byte; the rest is past the end
        assert_eq!(read(&file), [2, 2, 0, 1]);
        // Nothing read, a read past the end, and one whose end overflows.
        file.record(5, 0);
        file.record(3 * CHUNK + 10, 1);
        file.record(u64::MAX, 5);
        assert_eq!(read(&file), [2, 2, 0, 1]);
        file.record(2 * CHUNK + 5, u64::MAX);
        assert_eq!(read(&file), [2, 2, 1, 2]);
        assert_eq!(file.changes.load(Ordering::Relaxed), 5);
        // An empty file has no chunks and takes any read.
        let empty = FileCounts::new("e".into(), 0, changes());
        empty.record(0, 10);
        assert!(empty.chunks.is_empty());
    }

    #[test]
    fn a_file_is_one_whatever_its_case_and_separators() {
        let session = Session::default();
        let changes = changes();
        let a = session.open(r"data\ttdb\A.ttpf", 10, changes);
        let b = session.open("DATA/ttdb/a.TTPF", 10, changes);
        assert!(std::ptr::eq(a, b));
        assert_eq!(a.path, "data/ttdb/A.ttpf");
        assert_eq!(a.opens.load(Ordering::Relaxed), 2);
        assert_eq!(session.all().len(), 1);
    }

    #[test]
    fn sessions_add_up_with_how_often_and_how_lately() {
        let changes = changes();
        let first = Session::default();
        let a = first.open("a.rpf", 4 * CHUNK, changes);
        a.record(0, 10);
        a.record(0, 10);
        a.record(2 * CHUNK, 10);
        first.open("only-opened.txt", 5, changes);
        let one = Stored::default().with(&first.all(), 100);
        assert_eq!((one.sessions, &one.session_times[..]), (1, &[100][..]));
        let stored = &one.files["a.rpf"];
        assert_eq!((stored.size, stored.opens, stored.last_session), (4 * CHUNK, 1, 1));
        assert_eq!(stored.chunks[&0], Chunk { reads: 2, sessions: 1, last_session: 1 });
        assert_eq!(stored.chunks[&2], Chunk { reads: 1, sessions: 1, last_session: 1 });
        assert_eq!(stored.chunks.len(), 2);
        assert!(one.files["only-opened.txt"].chunks.is_empty());

        // The same session written again adds nothing twice.
        assert_eq!(Stored::default().with(&first.all(), 100), one);

        let second = Session::default();
        let a = second.open("A.rpf", 4 * CHUNK, changes);
        a.open_again();
        a.record(0, 10);
        a.record(3 * CHUNK, 10);
        let two = one.with(&second.all(), 200);
        assert_eq!((two.sessions, &two.session_times[..]), (2, &[100, 200][..]));
        let stored = &two.files["a.rpf"];
        assert_eq!((stored.path.as_str(), stored.opens, stored.last_session), ("A.rpf", 3, 2));
        assert_eq!(stored.chunks[&0], Chunk { reads: 3, sessions: 2, last_session: 2 });
        assert_eq!(stored.chunks[&2], Chunk { reads: 1, sessions: 1, last_session: 1 });
        assert_eq!(stored.chunks[&3], Chunk { reads: 1, sessions: 1, last_session: 2 });
        // A file the second session never opened stays as it was.
        assert_eq!(two.files["only-opened.txt"], one.files["only-opened.txt"]);
    }

    impl FileCounts {
        fn open_again(&self) {
            self.opened();
        }
    }

    #[test]
    fn a_file_of_another_size_starts_over() {
        let changes = changes();
        let first = Session::default();
        first.open("a.rpf", 4 * CHUNK, changes).record(3 * CHUNK, 1);
        let one = Stored::default().with(&first.all(), 1);
        let second = Session::default();
        second.open("a.rpf", 2 * CHUNK, changes).record(0, 1);
        let two = one.with(&second.all(), 2);
        let stored = &two.files["a.rpf"];
        assert_eq!((stored.size, stored.opens), (2 * CHUNK, 1));
        assert_eq!(stored.chunks.keys().copied().collect::<Vec<_>>(), [0]);
    }

    #[test]
    fn the_map_reads_back_as_written() {
        let changes = changes();
        let session = Session::default();
        let a = session.open(r#"pack\we "ird\ü.rpf"#, 70 * CHUNK, changes);
        a.record(0, 3 * CHUNK);
        a.record(69 * CHUNK, 1);
        session.open("empty.txt", 0, changes);
        let stored = Stored::default().with(&session.all(), 1_795_000_000).with(&session.all(), 1_795_000_100);
        let text = stored.to_json();
        assert_eq!(Stored::from_json(&text).unwrap(), stored);
        assert!(text.contains(r#""pack/we \"ird/ü.rpf":{"size":73400320,"opens":2,"last_session":2,"chunks":[[0,2,2,2],[1,2,2,2],[2,2,2,2],[69,2,2,2]]}"#), "{text}");
        // Other spacing, and fields this build does not know, read too.
        let spaced = r#" { "format" : "victory-read-map", "version": 1, "chunk_bytes": 1048576, "new": [true, null, "x\n\u0041"],
            "sessions": 2, "session_times": [5, 6], "files": { "A/b.rpf": { "size": 9, "opens": 1, "last_session": 2, "chunks": [ [ 0, 4, 2, 2 ] ] } } } "#;
        let read = Stored::from_json(spaced).unwrap();
        assert_eq!(read.files["a/b.rpf"].chunks[&0], Chunk { reads: 4, sessions: 2, last_session: 2 });
        assert_eq!(read.files["a/b.rpf"].path, "A/b.rpf");
        for bad in ["", "{}", "[1,2", r#"{"format":"victory-read-map","version":2}"#, r#"{"format":"victory-read-map","version":1,"chunk_bytes":4096,"sessions":0,"files":{}}"#, "{\"format\":\"victory-read-map\",\"version\":1.5}"] {
            assert!(Stored::from_json(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_file_is_written_whole_and_added_to_over_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("map.json");

        let first = Recorder::start(path.clone());
        // Nothing counted: nothing written.
        assert!(!first.write().unwrap());
        assert!(!path.exists());
        let a = first.open("levels.rpf", 8 * CHUNK);
        a.record(CHUNK, 10);
        assert!(first.write().unwrap());
        assert!(!first.write().unwrap(), "nothing new since");
        assert!(!beside(&path, "tmp").exists());
        // Later in the session: the file holds the session once, not twice.
        a.record(CHUNK, 10);
        a.record(5 * CHUNK, 10);
        assert!(first.write().unwrap());
        let stored = Stored::from_json(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(stored.sessions, 1);
        assert_eq!(stored.files["levels.rpf"].chunks[&1], Chunk { reads: 2, sessions: 1, last_session: 1 });

        let second = Recorder::start(path.clone());
        second.open("levels.rpf", 8 * CHUNK).record(5 * CHUNK, 10);
        second.open("data/ttdb/x.ttpf", CHUNK).record(0, 10);
        assert!(second.write().unwrap());
        let stored = Stored::from_json(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(stored.sessions, 2);
        let levels = &stored.files["levels.rpf"];
        assert_eq!(levels.opens, 2);
        assert_eq!(levels.chunks[&1], Chunk { reads: 2, sessions: 1, last_session: 1 });
        assert_eq!(levels.chunks[&5], Chunk { reads: 2, sessions: 2, last_session: 2 });
        assert_eq!(stored.files["data/ttdb/x.ttpf"].chunks[&0], Chunk { reads: 1, sessions: 1, last_session: 2 });

        // A map that cannot be read is kept aside, and a new one started.
        std::fs::write(&path, "{\"format\":\"victory-read-map\",").unwrap();
        let third = Recorder::start(path.clone());
        assert!(beside(&path, "corrupt").exists());
        third.open("a", 1).record(0, 1);
        assert!(third.write().unwrap());
        assert_eq!(Stored::from_json(&std::fs::read_to_string(&path).unwrap()).unwrap().sessions, 1);
    }

    /// The hooks themselves, on real handles and streams: run alone with the
    /// switch set, since the switch is read once for the process:
    /// `VICTORY_READ_MAP=<a new file> cargo test -p victory-hle --lib
    /// -- --ignored the_titles_reads_reach_the_map`.
    #[cfg(windows)]
    #[test]
    #[ignore = "needs VICTORY_READ_MAP"]
    fn the_titles_reads_reach_the_map() {
        use super::super::files;
        use std::ffi::c_void;
        let out = PathBuf::from(std::env::var_os("VICTORY_READ_MAP").expect("VICTORY_READ_MAP"));
        let _ = std::fs::remove_file(&out);
        let package = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        std::fs::create_dir(package.path().join("data")).unwrap();
        std::fs::write(package.path().join("data").join("Big.rpf"), vec![7u8; 5 * CHUNK as usize + 3]).unwrap();
        std::fs::write(package.path().join("small.txt"), b"hello").unwrap();
        std::fs::write(other.path().join("save.bin"), vec![1u8; 2 * CHUNK as usize]).unwrap();
        files::mount_at('G', package.path());
        files::mount_at('T', other.path());
        assert!(on());
        let w = |s: &str| s.encode_utf16().chain([0]).collect::<Vec<u16>>();
        let open = |path: &str| {
            let handle = files::CreateFileW(w(path).as_ptr(), 0x8000_0000, 1, std::ptr::null(), 3, 0, std::ptr::null_mut());
            assert_ne!(handle as isize, -1, "{path}");
            handle
        };
        let close = files::host!("kernel32.dll", "CloseHandle": extern "system" fn(*mut c_void) -> i32);
        let seek = |handle, to: i64| {
            // SAFETY: a file handle we opened.
            assert_ne!(unsafe { windows_sys::Win32::Storage::FileSystem::SetFilePointerEx(handle, to, std::ptr::null_mut(), 0) }, 0);
        };
        let mut buffer = vec![0u8; 2 * CHUNK as usize];
        let mut n = 0u32;

        // At the file pointer: chunk 0, then (after a seek) 3 and 4.
        let big = open(r"G:\data\big.RPF");
        assert_ne!(files::ReadFile(big, buffer.as_mut_ptr(), 100, &mut n, std::ptr::null_mut()), 0);
        assert_eq!(n, 100);
        seek(big, 4 * CHUNK as i64 - 1);
        assert_ne!(files::ReadFile(big, buffer.as_mut_ptr(), 2, &mut n, std::ptr::null_mut()), 0);
        // At an `OVERLAPPED` offset, on a handle that is not overlapped: chunk 2.
        let mut overlapped = [0u64; 4];
        overlapped[2] = 2 * CHUNK + 5;
        assert_ne!(files::ReadFile(big, buffer.as_mut_ptr(), 10, &mut n, overlapped.as_mut_ptr().cast()), 0);
        assert_eq!((n, buffer[0]), (10, 7));
        // A file outside the package is not recorded, nor is a closed
        // handle's value when the host gives it out again.
        handle_closed(big);
        close(big);
        let save = open(r"T:\save.bin");
        assert_ne!(files::ReadFile(save, buffer.as_mut_ptr(), 2 * CHUNK as u32, &mut n, std::ptr::null_mut()), 0);
        assert_eq!(n, 2 * CHUNK as u32);
        close(save);

        // As DirectStorage does: the handle's counters go with the file object.
        let big = open("G:/data/Big.rpf");
        let counts = take_handle(big).expect("a file of the package");
        counts.record(5 * CHUNK, 100);
        assert_ne!(files::ReadFile(big, buffer.as_mut_ptr(), 10, &mut n, std::ptr::null_mut()), 0, "not counted twice");
        close(big);

        // Streams: `fread` at the stream's position; a stream only opened.
        let fseek = files::host!("ucrtbase.dll", "fseek": extern "C" fn(*mut c_void, i32, i32) -> i32);
        let fclose = files::host!("ucrtbase.dll", "fclose": extern "C" fn(*mut c_void) -> i32);
        let stream = files::fopen(c"G:/data/Big.rpf".as_ptr(), c"rb".as_ptr());
        assert!(!stream.is_null());
        assert_eq!(fseek(stream, CHUNK as i32 + 10, 0), 0);
        assert_eq!(files::fread(buffer.as_mut_ptr(), 1, 16, stream), 16);
        stream_closed(stream);
        fclose(stream);
        let small = files::_wfopen(w(r"G:\small.txt").as_ptr(), w("rb").as_ptr());
        assert!(!small.is_null());
        stream_closed(small);
        fclose(small);

        flush();
        let stored = Stored::from_json(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(stored.sessions, 1);
        assert_eq!(stored.files.keys().collect::<Vec<_>>(), ["data/big.rpf", "small.txt"]);
        let big = &stored.files["data/big.rpf"];
        assert_eq!((big.path.as_str(), big.size, big.opens), ("data/Big.rpf", 5 * CHUNK + 3, 3));
        let reads: Vec<(u32, u64)> = big.chunks.iter().map(|(i, c)| (*i, c.reads)).collect();
        assert_eq!(reads, [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]);
        let small = &stored.files["small.txt"];
        assert_eq!((small.size, small.opens, small.chunks.len()), (5, 1, 0));
    }

    #[test]
    fn only_files_under_the_package_folder_are_ours() {
        let root = r"\\?\l:\games\the game\";
        assert_eq!(under_root(root, r"\\?\L:\Games\The Game\data\ttdb\a.ttpf"), Some(r"data\ttdb\a.ttpf"));
        assert_eq!(under_root(root, r"\\?\L:\Games\The Game\"), None);
        assert_eq!(under_root(root, r"\\?\L:\Games\The Game 2\levels.rpf"), None);
        assert_eq!(under_root(root, r"\\?\C:\x"), None);
    }
}
