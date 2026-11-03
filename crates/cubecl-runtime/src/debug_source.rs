//! The compile directory of the kernel debug data, found when a kernel compiles.
//!
//! `file!()` gives a path relative to the rustc working directory, usually the workspace root.
//! A debugger or a profiler finds such a file only when it runs in that directory. The binary
//! keeps the relative path, so `--remap-path-prefix` and `trim-paths` apply to it. When a kernel
//! compiles, cubecl looks for the file, and gives the directory that contains it to the debug
//! data. Thus only the debug data of the kernel has the absolute path, on the computer that has
//! the source.
//!
//! If no directory has the files, and the kernel has their text (`Full`), cubecl can write the
//! text into the source cache (`CUBECL_SOURCE_CACHE`) and give that directory.
//!
//! The search occurs one time for each file in each process, not for each kernel.

use alloc::{
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::{fmt::Write, hash::BuildHasher};
use cubecl_ir::{debug::DebugState, settings::DebugInfo};
use md5::{Digest, Md5};
use std::{
    collections::HashMap,
    path::{Component, Path, PathBuf},
    sync::{Mutex, OnceLock, PoisonError},
};

/// The variable that names the root, for a binary that does not run in its source tree.
pub const SOURCE_ROOT_VAR: &str = "CUBECL_SOURCE_ROOT";

/// The MD5 of the source text of each file of a kernel at `level`, by the path of the file. Empty
/// below [`DebugInfo::Full`], as the macro records the texts only at `Full`.
#[must_use]
pub fn source_md5s(debug: &DebugState, level: DebugInfo) -> HashMap<&str, Arc<str>> {
    match level {
        DebugInfo::Full => debug
            .sources()
            .iter()
            .map(|(path, text)| (path.as_str(), source_md5(text)))
            .collect(),
        _ => HashMap::new(),
    }
}

/// The directory for the relative source paths of a kernel with the debug data `debug`. `md5s`
/// are the MD5s of the texts that the kernel has ([`source_md5s`]). A file with an MD5 matches
/// only a file with the same text.
///
/// The search roots are `CUBECL_SOURCE_ROOT`, then the working directory and its parents. If no
/// root has a file, the directory is the [source cache](crate::config::compilation::CompilationConfig::source_cache),
/// when it is set and the kernel has texts.
/// Returns `None` when no directory has the files, or the directory is not UTF-8.
#[must_use]
pub fn kernel_source_root<S: BuildHasher>(
    debug: &DebugState,
    md5s: &HashMap<&str, Arc<str>, S>,
) -> Option<String> {
    let files = debug
        .files()
        .iter()
        .map(|path| (path.as_str(), md5s.get(path.as_str()).map(AsRef::as_ref)));
    source_root(files)
        .or_else(|| {
            let texts = debug
                .sources()
                .iter()
                .filter_map(|(path, text)| Some((path.as_str(), *text, md5s.get(path.as_str())?)));
            cached_root(source_cache()?, texts)
        })
        .and_then(|root| root.to_str().map(str::to_string))
}

/// The root of the first file in `files` that exists under a search root. Each item is a path
/// and, if the kernel has the text of the file, the MD5 of that text.
fn source_root<'a>(files: impl IntoIterator<Item = (&'a str, Option<&'a str>)>) -> Option<PathBuf> {
    /// The root of each path and MD5, or `None` if no root has the file.
    type Found = HashMap<(String, Option<String>), Option<PathBuf>>;
    static FOUND: OnceLock<Mutex<Found>> = OnceLock::new();
    let found = FOUND.get_or_init(Mutex::default);

    files
        .into_iter()
        .filter(|(path, _)| Path::new(path).is_relative())
        .find_map(|(path, md5)| {
            let key = (path.to_string(), md5.map(str::to_string));
            found
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(key)
                .or_insert_with(|| find_root(path, md5, search_roots()))
                .clone()
        })
}

/// `CUBECL_SOURCE_ROOT`, then the working directory and its parents. They are read one time.
fn search_roots() -> &'static [PathBuf] {
    static ROOTS: OnceLock<Vec<PathBuf>> = OnceLock::new();
    ROOTS.get_or_init(|| {
        let var = std::env::var_os(SOURCE_ROOT_VAR).map(PathBuf::from);
        let cwd = std::env::current_dir().ok();
        let parents = cwd
            .iter()
            .flat_map(|cwd| cwd.ancestors().map(Path::to_path_buf));
        var.into_iter().chain(parents).collect()
    })
}

/// The first of `roots` that contains the file `path`. With `md5`, the file must have that MD5.
fn find_root<'a>(
    path: &str,
    md5: Option<&str>,
    roots: impl IntoIterator<Item = &'a PathBuf>,
) -> Option<PathBuf> {
    roots
        .into_iter()
        .find(|root| {
            let file = root.join(path);
            match md5 {
                None => file.is_file(),
                Some(md5) => std::fs::read(&file).is_ok_and(|bytes| md5_hex(&bytes) == md5),
            }
        })
        .cloned()
}

/// The source cache of the configuration, read one time.
fn source_cache() -> Option<&'static Path> {
    use crate::config::{CubeClRuntimeConfig, RuntimeConfig};

    static CACHE: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHE
        .get_or_init(|| CubeClRuntimeConfig::get().compilation.source_cache.clone())
        .as_deref()
}

/// The directory in `cache` that has the files `texts`, as `<cache>/<tree>/<path>`. Each item is
/// a relative path, its text and the MD5 of the text. `<tree>` is the MD5 of all paths and MD5s,
/// so a different text gets a different directory, and the same files share one.
///
/// A path that is not under its directory, as `../k.rs`, is not written. Returns `None` when no
/// file is in the directory.
fn cached_root<'a>(
    cache: &Path,
    texts: impl IntoIterator<Item = (&'a str, &'a str, &'a Arc<str>)>,
) -> Option<PathBuf> {
    /// The directory of each cache and tree, or `None` if no file could be written.
    type Trees = HashMap<(PathBuf, String), Option<PathBuf>>;
    static TREES: OnceLock<Mutex<Trees>> = OnceLock::new();

    let mut texts: Vec<_> = texts
        .into_iter()
        .filter(|(path, ..)| is_under(Path::new(path)))
        .collect();
    if texts.is_empty() {
        return None;
    }
    texts.sort_unstable_by_key(|(path, ..)| *path);
    let tree = md5_hex(texts.iter().fold(String::new(), |mut key, (path, _, md5)| {
        let _ = writeln!(key, "{path}\0{md5}");
        key
    }));
    TREES
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry((cache.to_path_buf(), tree))
        .or_insert_with_key(|(cache, tree)| {
            let root = cache.join(tree);
            let written = texts
                .iter()
                .filter(|(path, text, _)| write_once(&root.join(path), text))
                .count();
            (written > 0).then_some(root)
        })
        .clone()
}

/// True when the relative path `path` names a file under its directory.
fn is_under(path: &Path) -> bool {
    path.components().all(|c| matches!(c, Component::Normal(_)))
        && path.components().next().is_some()
}

/// Writes `text` to `file`, unless the file exists. Another process can write the same file, so
/// the text goes to a temporary file first. Returns whether the file exists after the call.
fn write_once(file: &Path, text: &str) -> bool {
    if file.is_file() {
        return true;
    }
    let mut temporary = std::ffi::OsString::from(file);
    temporary.push(alloc::format!(".{}.tmp", std::process::id()));
    let result = file
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(&temporary, text))
        .and_then(|()| std::fs::rename(&temporary, file));
    if let Err(err) = result {
        let _ = std::fs::remove_file(&temporary);
        log::warn!("The source cache cannot write {}: {err}", file.display());
    }
    file.is_file()
}

/// The MD5 of the source text `text`, calculated one time for each text in each process.
///
/// The text is a `&'static str` from `include_str!`, so its address identifies it. One MD5 takes
/// approximately 25 µs for a text of 20 KB. If the first compile is too slow, `cubecl-macros` can
/// calculate the MD5 at build time and give it to `debug_source_expand` with the text.
fn source_md5(text: &'static str) -> Arc<str> {
    /// The MD5 of each text, by its address and length.
    type Md5s = HashMap<(usize, usize), Arc<str>>;
    static MD5S: OnceLock<Mutex<Md5s>> = OnceLock::new();
    MD5S.get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry((text.as_ptr().addr(), text.len()))
        .or_insert_with(|| md5_hex(text).into())
        .clone()
}

/// The MD5 of `text`, in lowercase hexadecimal.
pub fn md5_hex(text: impl AsRef<[u8]>) -> String {
    Md5::digest(text)
        .iter()
        .fold(String::with_capacity(32), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A new directory `name` under the temporary directory.
    fn temporary(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(alloc::format!(
            "cubecl-debug-source-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A new directory with the file `src/k.rs`, which has the text `text`.
    fn tree(name: &str, text: &str) -> PathBuf {
        let root = temporary(name);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/k.rs"), text).unwrap();
        root
    }

    #[test]
    fn the_first_root_that_has_the_file() {
        let root = tree("first", "fn k() {}");
        let roots = [root.join("src"), root.clone(), PathBuf::from("/")];
        assert_eq!(find_root("src/k.rs", None, &roots), Some(root.clone()));
        assert_eq!(find_root("src/none.rs", None, &roots), None);
        std::fs::remove_dir_all(root).unwrap();
    }

    /// With the MD5 of the compiled text, a file that was changed is not the source.
    #[test]
    fn a_file_with_a_different_text_is_not_the_source() {
        let old = tree("old", "fn k() {}");
        let new = tree("new", "fn k() { changed }");
        let md5 = md5_hex("fn k() {}");
        let roots = [new.clone(), old.clone()];
        assert_eq!(find_root("src/k.rs", Some(&md5), &roots), Some(old.clone()));
        std::fs::remove_dir_all(old).unwrap();
        std::fs::remove_dir_all(new).unwrap();
    }

    /// An absolute path needs no directory. The compiler tests check that the search finds the
    /// workspace root for a relative path.
    #[test]
    fn an_absolute_path_has_no_root() {
        assert_eq!(source_root([("/abs/k.rs", None)]), None);
    }

    /// The cache gets each text under one directory for the files, and the same files, in any
    /// order, get the same directory. A different text gets a different directory.
    #[test]
    fn the_cache_has_the_texts_under_one_directory() {
        let cache = temporary("cache");
        let (k, f) = ("fn k() {}", "fn f() {}");
        let (k_md5, f_md5): (Arc<str>, Arc<str>) = (md5_hex(k).into(), md5_hex(f).into());
        let files = [("src/k.rs", k, &k_md5), ("lib/f.rs", f, &f_md5)];

        let root = cached_root(&cache, files).expect("a directory");
        assert!(root.starts_with(&cache), "{}", root.display());
        assert_eq!(std::fs::read_to_string(root.join("src/k.rs")).unwrap(), k);
        assert_eq!(std::fs::read_to_string(root.join("lib/f.rs")).unwrap(), f);
        let [k_file, f_file] = files;
        assert_eq!(cached_root(&cache, [f_file, k_file]), Some(root.clone()));

        let changed = "fn k() { changed }";
        let changed_md5: Arc<str> = md5_hex(changed).into();
        let other = cached_root(&cache, [("src/k.rs", changed, &changed_md5)]);
        assert!(other.is_some_and(|other| other != root));
        std::fs::remove_dir_all(cache).unwrap();
    }

    /// A path out of its directory is not written, so the cache writes only under itself.
    #[test]
    fn the_cache_writes_only_under_itself() {
        let cache = temporary("escape");
        let text = "fn k() {}";
        let md5: Arc<str> = md5_hex(text).into();
        for path in ["../k.rs", "/k.rs", "./k.rs", ""] {
            assert_eq!(cached_root(&cache.join("c"), [(path, text, &md5)]), None);
        }
        assert!(!cache.join("k.rs").exists());
        assert!(!cache.join("c").exists());
        std::fs::remove_dir_all(cache).unwrap();
    }
}
