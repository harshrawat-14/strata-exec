/// Dedicated logging thread that drains the debug event channel.
///
/// The logger runs on its own OS thread and continuously reads events,
/// printing them in a human-readable format.  When the sender side is
/// dropped the channel closes and the thread exits cleanly.
use std::thread::{self, JoinHandle};

use crate::events::event::{Event as DebugEvent, EventReceiver, EventSender};

/// Channel capacity.  Bounded to prevent unbounded memory growth if the
/// logger thread falls behind.  10 000 events is generous enough that
/// senders will almost never block in practice.
const CHANNEL_CAPACITY: usize = 10_000;

/// Create a bounded debug channel and spawn the logger thread.
///
/// Returns the sender half (clone it to share across components) and
/// the thread handle (join it after the simulation to flush all events).
///
/// # Example
///
/// ```ignore
/// let (tx, handle) = init_debug_channel();
/// // … pass tx.clone() into simulation components …
/// drop(tx); // close the channel
/// handle.join().unwrap(); // wait for the logger to finish
/// ```
pub fn init_debug_channel() -> (EventSender, JoinHandle<()>) {
    let (tx, rx) = crossbeam_channel::bounded::<DebugEvent>(CHANNEL_CAPACITY);
    let handle = spawn_logger(rx);
    (tx, handle)
}

/// Spawn the logger thread that continuously drains events from `rx`.
fn spawn_logger(rx: EventReceiver) -> JoinHandle<()> {
    thread::spawn(move || {
        for event in rx {
            eprintln!("{event}");
        }
    })
}
