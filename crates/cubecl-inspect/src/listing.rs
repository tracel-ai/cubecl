use crate::report::{Listing, Unreadable};
use crate::{EnvironmentDirectory, Inspector};

/// Every environment file in `directories`, as one list, newest first.
///
/// A directory that does not exist holds none — a machine that never built
/// has no environments — and one that cannot be read is listed as
/// unreadable, beside what the others hold.
pub fn list(directories: &[EnvironmentDirectory]) -> Listing {
    let mut listing = Listing {
        directories: directories
            .iter()
            .map(|directory| directory.path.clone())
            .collect(),
        environments: Vec::new(),
        unreadable: Vec::new(),
    };
    for directory in directories {
        let files = match directory.files() {
            Ok(files) => files,
            Err(err) => {
                listing.unreadable.push(Unreadable {
                    path: directory.path.clone(),
                    reason: err.to_string(),
                });
                continue;
            }
        };
        for path in files {
            match Inspector::open(&path) {
                Ok(inspector) => listing.environments.push(inspector.summary()),
                Err(err) => listing.unreadable.push(Unreadable {
                    path,
                    reason: err.to_string(),
                }),
            }
        }
    }
    listing.environments.sort_by(|a, b| {
        b.created_unix_secs()
            .cmp(&a.created_unix_secs())
            .then_with(|| a.path.cmp(&b.path))
    });
    listing.unreadable.sort_by(|a, b| a.path.cmp(&b.path));
    listing
}
