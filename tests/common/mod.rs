//! Shared helpers for integration tests.

use std::sync::atomic::{AtomicUsize, Ordering};

/// A fresh, unique directory under the OS temp dir for a runner's CSV output,
/// so tests never overwrite the tracked files in `results/`.
pub fn scratch_results_dir() -> String {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir()
        .join(format!("strataexec-test-results-{}-{n}", std::process::id()))
        .to_string_lossy()
        .into_owned()
}
