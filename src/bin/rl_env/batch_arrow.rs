//! Arrow RecordBatch encode/decode for the batched transport.

use crate::batched::BatchStepResult;
use arrow::array::{ArrayRef, BooleanArray, Float32Array, Int32Array, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc as StdArc;

/// Column names for state dims: "s0", "s1", ..., "s{d-1}".
fn state_col_name(i: usize) -> String {
    format!("s{i}")
}

pub(crate) fn result_schema(d: usize) -> Schema {
    let mut fields: Vec<Field> = (0..d)
        .map(|i| Field::new(state_col_name(i), DataType::Float32, false))
        .collect();
    fields.push(Field::new("reward", DataType::Float32, false));
    fields.push(Field::new("done", DataType::Boolean, false));
    Schema::new(fields)
}

/// Encode a `BatchStepResult` (row-major N×D states + reward/done) into an
/// Arrow `RecordBatch` (N rows, D+2 columns: s0..s{D-1}, reward, done).
/// This is the "zero-copy on decode" side of ADR-006: the Python side reads
/// each column as a `pyarrow` buffer-view via `to_numpy(zero_copy_only=True)`,
/// no per-element parsing.
pub(crate) fn encode_result(r: &BatchStepResult) -> RecordBatch {
    let schema = result_schema(r.d);
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(r.d + 2);
    for col in 0..r.d {
        let values: Vec<f32> = (0..r.n).map(|row| r.states[row * r.d + col]).collect();
        columns.push(StdArc::new(Float32Array::from(values)));
    }
    columns.push(StdArc::new(Float32Array::from(r.rewards.clone())));
    columns.push(StdArc::new(BooleanArray::from(r.dones.clone())));
    RecordBatch::try_new(StdArc::new(schema), columns)
        .expect("encode_result: column lengths must match schema by construction")
}

// The real client is Python (via pyarrow, see rl/batched_environment.py),
// which builds request RecordBatches itself -- these Rust-side encoders
// exist only so Rust tests can act as a client without a Python
// dependency. #[cfg(test)] reflects that honestly rather than suppressing
// an unused-code warning on functions no production path calls.
#[cfg(test)]
pub(crate) fn request_schema_step() -> Schema {
    Schema::new(vec![Field::new("action", DataType::Int32, false)])
}

#[cfg(test)]
pub(crate) fn request_schema_reset() -> Schema {
    Schema::new(vec![Field::new("seed", DataType::UInt64, false)])
}

#[cfg(test)]
pub(crate) fn encode_step_request(actions: &[usize]) -> RecordBatch {
    let values: Vec<i32> = actions.iter().map(|&a| a as i32).collect();
    RecordBatch::try_new(
        StdArc::new(request_schema_step()),
        vec![StdArc::new(Int32Array::from(values))],
    )
    .expect("encode_step_request: schema mismatch")
}

#[cfg(test)]
pub(crate) fn encode_reset_request(seeds: &[u64]) -> RecordBatch {
    RecordBatch::try_new(
        StdArc::new(request_schema_reset()),
        vec![StdArc::new(UInt64Array::from(seeds.to_vec()))],
    )
    .expect("encode_reset_request: schema mismatch")
}

pub(crate) fn decode_step_request(batch: &RecordBatch) -> Vec<usize> {
    let col = batch
        .column_by_name("action")
        .expect("step request batch missing 'action' column")
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("'action' column must be Int32");
    (0..col.len()).map(|i| col.value(i) as usize).collect()
}

pub(crate) fn decode_reset_request(batch: &RecordBatch) -> Vec<u64> {
    let col = batch
        .column_by_name("seed")
        .expect("reset request batch missing 'seed' column")
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("'seed' column must be UInt64");
    (0..col.len()).map(|i| col.value(i)).collect()
}

/// Indexed-reset request schema: `{index: uint32, seed: uint64}`, one row
/// per instance to reset. Distinguished from a full reset_batch request
/// (which has a "seed" column but no "index" column) by the presence of
/// "index" -- checked first by the server dispatch.
#[cfg(test)]
pub(crate) fn request_schema_reset_indices() -> Schema {
    Schema::new(vec![
        Field::new("index", DataType::UInt32, false),
        Field::new("seed", DataType::UInt64, false),
    ])
}

#[cfg(test)]
pub(crate) fn encode_reset_indices_request(indices: &[usize], seeds: &[u64]) -> RecordBatch {
    let idx_values: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    RecordBatch::try_new(
        StdArc::new(request_schema_reset_indices()),
        vec![
            StdArc::new(UInt32Array::from(idx_values)),
            StdArc::new(UInt64Array::from(seeds.to_vec())),
        ],
    )
    .expect("encode_reset_indices_request: schema mismatch")
}

pub(crate) fn decode_reset_indices_request(batch: &RecordBatch) -> (Vec<usize>, Vec<u64>) {
    let idx_col = batch
        .column_by_name("index")
        .expect("reset_indices request batch missing 'index' column")
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("'index' column must be UInt32");
    let seed_col = batch
        .column_by_name("seed")
        .expect("reset_indices request batch missing 'seed' column")
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("'seed' column must be UInt64");
    let indices = (0..idx_col.len()).map(|i| idx_col.value(i) as usize).collect();
    let seeds = (0..seed_col.len()).map(|i| seed_col.value(i)).collect();
    (indices, seeds)
}
