/// Fork of `buildid`'s Windows logic to fix incremental builds. If upstream gets a fix, this can
/// be removed. Linux should work correctly because it uses deterministic build IDs and doesn't
/// keep them on incremental builds from all the info I could find.
#[cfg(target_os = "windows")]
pub mod windows;

pub fn build_id() -> Option<&'static [u8]> {
    cfg_select! {
        windows => windows::build_id(),
        _ => buildid::build_id()
    }
}
