/// Represents the lifecycle state of the execution engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineState {
    Idle,
    Running,
    AwaitingConfirmation,
    Aborted,
}
