//! Undo / redo for the adjustment params.
//!
//! The GUI hands the current params to [`History::observe`] every frame. A
//! slider drag changes the params on every frame it moves, so a step is only
//! recorded once the edit *settles* (the caller says when: no pointer button
//! held) — one drag is one undo step, not sixty.

/// Steps kept before the oldest is dropped. Params are ~150 bytes.
pub const MAX_STEPS: usize = 200;

#[derive(Debug)]
pub struct History<T> {
    undo: Vec<T>,
    redo: Vec<T>,
    /// The last settled state: what undo returns to while an edit is still in
    /// progress, and what gets pushed when that edit settles.
    committed: T,
}

impl<T: Copy + PartialEq> History<T> {
    /// A fresh history starting at `initial` (a newly opened photo).
    pub fn new(initial: T) -> Self {
        Self { undo: Vec::new(), redo: Vec::new(), committed: initial }
    }

    /// Record `current` as a step once it has settled.
    pub fn observe(&mut self, current: T, settled: bool) {
        if settled && current != self.committed {
            self.undo.push(self.committed);
            if self.undo.len() > MAX_STEPS {
                self.undo.remove(0);
            }
            self.redo.clear();
            self.committed = current;
        }
    }

    /// The state to go back to, or `None` when there is nothing to undo. An
    /// unsettled edit in progress counts as the step being undone.
    pub fn undo(&mut self, current: T) -> Option<T> {
        let target = if current != self.committed { self.committed } else { self.undo.pop()? };
        self.redo.push(current);
        self.committed = target;
        Some(target)
    }

    pub fn redo(&mut self, current: T) -> Option<T> {
        // Anything changed since the undo forks history; there is no redo.
        if current != self.committed {
            return None;
        }
        let target = self.redo.pop()?;
        self.undo.push(current);
        self.committed = target;
        Some(target)
    }

    pub fn can_undo(&self, current: T) -> bool {
        current != self.committed || !self.undo.is_empty()
    }

    pub fn can_redo(&self, current: T) -> bool {
        current == self.committed && !self.redo.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_drag_is_one_step() {
        let mut h = History::new(0);
        for v in 1..=10 {
            h.observe(v, false); // pointer still down
        }
        h.observe(10, true);
        assert_eq!(h.undo(10), Some(0));
        assert_eq!(h.undo(0), None);
    }

    #[test]
    fn undo_then_redo_round_trips() {
        let mut h = History::new(0);
        h.observe(1, true);
        h.observe(2, true);
        assert_eq!(h.undo(2), Some(1));
        assert_eq!(h.undo(1), Some(0));
        assert!(!h.can_undo(0));
        assert_eq!(h.redo(0), Some(1));
        assert_eq!(h.redo(1), Some(2));
        assert_eq!(h.redo(2), None);
    }

    #[test]
    fn a_new_edit_after_undo_drops_redo() {
        let mut h = History::new(0);
        h.observe(1, true);
        assert_eq!(h.undo(1), Some(0));
        h.observe(5, true);
        assert!(!h.can_redo(5));
        assert_eq!(h.undo(5), Some(0));
    }

    #[test]
    fn undo_mid_drag_returns_to_before_the_drag() {
        let mut h = History::new(0);
        h.observe(1, true);
        h.observe(7, false);
        assert_eq!(h.undo(7), Some(1));
        // The abandoned drag is redoable.
        assert_eq!(h.redo(1), Some(7));
    }

    #[test]
    fn redo_is_refused_after_an_unsettled_change() {
        let mut h = History::new(0);
        h.observe(1, true);
        h.undo(1);
        assert!(!h.can_redo(3));
        assert_eq!(h.redo(3), None);
    }

    #[test]
    fn history_is_capped() {
        let mut h = History::new(0);
        for v in 1..=(MAX_STEPS as i32 + 50) {
            h.observe(v, true);
        }
        let mut cur = MAX_STEPS as i32 + 50;
        let mut steps = 0;
        while let Some(prev) = h.undo(cur) {
            cur = prev;
            steps += 1;
        }
        assert_eq!(steps, MAX_STEPS);
        assert_eq!(cur, 50);
    }
}
