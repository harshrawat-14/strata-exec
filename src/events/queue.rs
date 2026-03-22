use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::event::SimEvent;

// ---------------------------------------------------------------------------
// Timestamped event wrapper
// ---------------------------------------------------------------------------

/// An event tagged with a simulation timestamp for priority-queue ordering.
///
/// Lower timestamps are processed first (min-heap behaviour achieved via
/// a reversed `Ord` implementation).
#[derive(Debug, Clone)]
pub struct TimestampedEvent {
    /// Simulation time at which this event should be processed.
    pub time: f64,
    /// Tie-breaking sequence number (lower = earlier).
    pub seq: u64,
    /// The event payload.
    pub event: SimEvent,
}

impl PartialEq for TimestampedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.seq == other.seq
    }
}

impl Eq for TimestampedEvent {}

/// Ordering: we want a **min-heap** (earliest event first), but
/// `BinaryHeap` is a max-heap.  So we reverse the comparison.
impl Ord for TimestampedEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: smaller time = Greater priority.
        other
            .time
            .partial_cmp(&self.time)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for TimestampedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// Event queue
// ---------------------------------------------------------------------------

/// Priority queue of simulation events ordered by timestamp.
pub struct EventQueue {
    heap: BinaryHeap<TimestampedEvent>,
    next_seq: u64,
}

impl EventQueue {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_seq: 0,
        }
    }

    /// Schedule an event at the given simulation time.
    pub fn schedule(&mut self, time: f64, event: SimEvent) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.heap.push(TimestampedEvent { time, seq, event });
    }

    /// Pop the next event (earliest timestamp).
    pub fn pop(&mut self) -> Option<TimestampedEvent> {
        self.heap.pop()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Number of pending events.
    pub fn len(&self) -> usize {
        self.heap.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_processed_in_chronological_order() {
        let mut q = EventQueue::new();
        q.schedule(3.0, SimEvent::TimeTick { step: 3 });
        q.schedule(1.0, SimEvent::TimeTick { step: 1 });
        q.schedule(2.0, SimEvent::TimeTick { step: 2 });

        let first = q.pop().unwrap();
        assert_eq!(first.time, 1.0);

        let second = q.pop().unwrap();
        assert_eq!(second.time, 2.0);

        let third = q.pop().unwrap();
        assert_eq!(third.time, 3.0);

        assert!(q.is_empty());
    }

    #[test]
    fn ties_broken_by_insertion_order() {
        let mut q = EventQueue::new();
        q.schedule(1.0, SimEvent::TimeTick { step: 10 });
        q.schedule(1.0, SimEvent::TimeTick { step: 20 });
        q.schedule(1.0, SimEvent::TimeTick { step: 30 });

        // Same timestamp → FIFO order (lower seq first).
        if let SimEvent::TimeTick { step } = q.pop().unwrap().event {
            assert_eq!(step, 10);
        }
        if let SimEvent::TimeTick { step } = q.pop().unwrap().event {
            assert_eq!(step, 20);
        }
        if let SimEvent::TimeTick { step } = q.pop().unwrap().event {
            assert_eq!(step, 30);
        }
    }
}
