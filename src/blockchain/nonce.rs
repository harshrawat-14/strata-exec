use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe nonce manager for sequential transaction submission.
pub struct NonceManager {
    current: AtomicU64,
}

impl NonceManager {
    pub fn new(initial: u64) -> Self {
        Self {
            current: AtomicU64::new(initial),
        }
    }

    /// Atomically fetch the next nonce.
    pub fn next(&self) -> u64 {
        self.current.fetch_add(1, Ordering::SeqCst)
    }
}
