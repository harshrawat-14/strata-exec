//! Wire formats: line-delimited JSON over stdio, and the framed Arrow
//! batch server over a Unix domain socket.

use std::io::{self, Write};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::batch_arrow;
use crate::batched::*;
use crate::config::*;
use crate::env::*;

// ---------------------------------------------------------------------------
// JSON wire types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct InMsg {
    #[serde(default)]
    pub(crate) reset: bool,
    #[serde(default)]
    pub(crate) seed: Option<u64>,
    /// Protocol v3 only: where in the day's valid start window to begin
    /// (0 = earliest, 1 = latest). Absent = draw from the seed.
    #[serde(default)]
    pub(crate) start_frac: Option<f64>,
    #[serde(default)]
    pub(crate) action: Option<usize>,
}

#[derive(Serialize)]
pub(crate) struct OutMsg {
    pub(crate) state: Vec<f64>,
    pub(crate) reward: f64,
    pub(crate) done: bool,
    pub(crate) info: Value,
}

// ---------------------------------------------------------------------------
// stdin/stdout loop
// ---------------------------------------------------------------------------

pub(crate) fn write_msg(out: &mut impl Write, msg: &OutMsg) -> io::Result<()> {
    // Write straight to the output stream instead of materialising an
    // intermediate String via serde_json::to_string -- one fewer allocation
    // per step on the hot path (measured via dhat-heap,
    // docs/performance/baseline-profiling.md).
    serde_json::to_writer(&mut *out, msg).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    out.write_all(b"\n")?;
    out.flush()
}

pub(crate) fn step_result_to_msg(r: StepResult) -> OutMsg {
    OutMsg {
        state: r.state,
        reward: r.reward,
        done: r.done,
        info: r.info,
    }
}

/// Scans argv for `--batch-size N --socket-path PATH`, the two batch-mode
/// flags that live outside `EnvConfig` (they're process-level, not
/// per-instance). Absent -> the default single-instance JSON-over-stdio path
/// runs unchanged.
pub(crate) fn parse_batch_args() -> Option<(usize, String)> {
    let args: Vec<String> = std::env::args().collect();
    let mut batch_size: Option<usize> = None;
    let mut socket_path: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--batch-size" => {
                if let Some(v) = args.get(i + 1) {
                    batch_size = v.parse::<usize>().ok();
                    i += 1;
                }
            }
            "--socket-path" => {
                if let Some(v) = args.get(i + 1) {
                    socket_path = Some(v.clone());
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    match (batch_size, socket_path) {
        (Some(n), Some(path)) if n > 0 => Some((n, path)),
        _ => None,
    }
}

/// Serve batched reset/step requests over a Unix domain socket using Arrow
/// RecordBatch IPC (ADR-006). One request = one embedded Arrow IPC stream
/// (schema + one batch + EOS marker) distinguished by column name ("seed" =
/// reset, "action" = step); one response = the same framing, encoding
/// `BatchStepResult`. Accepts exactly one client connection, matching how a
/// single training process drives one batched env.
/// Read one length-prefixed message: a 4-byte little-endian length followed
/// by exactly that many bytes. Returns `None` on EOF/error (client
/// disconnected). See the module doc above `read_framed_message` callers for
/// why this framing exists instead of writing Arrow IPC stream bytes
/// directly to the socket.
pub(crate) fn read_framed_message(stream: &mut impl io::Read) -> Option<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).ok()?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).ok()?;
    Some(buf)
}

pub(crate) fn write_framed_message(stream: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()
}

/// Serialize one RecordBatch to a self-contained Arrow IPC stream buffer
/// (schema + one batch + EOS marker) in memory.
pub(crate) fn encode_batch_to_bytes(batch: &arrow::record_batch::RecordBatch) -> Vec<u8> {
    use arrow::ipc::writer::StreamWriter;
    let mut buf = Vec::new();
    {
        let mut writer =
            StreamWriter::try_new(&mut buf, &batch.schema()).expect("StreamWriter::try_new");
        writer.write(batch).expect("StreamWriter::write");
        writer.finish().expect("StreamWriter::finish");
    }
    buf
}

pub(crate) fn run_batch_server(cfg_template: EnvConfig, batch_size: usize, socket_path: &str) {
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;
    use std::os::unix::net::UnixListener;

    let _ = std::fs::remove_file(socket_path);
    let listener = UnixListener::bind(socket_path)
        .unwrap_or_else(|e| panic!("failed to bind batch socket '{socket_path}': {e}"));
    eprintln!("[rl-env batch] listening on {socket_path}, batch_size={batch_size}");

    let mut batched = BatchedRlEnv::new(cfg_template, batch_size);

    let (mut stream, _addr) = match listener.accept() {
        Ok(x) => x,
        Err(e) => {
            eprintln!("[rl-env batch] accept failed: {e}");
            return;
        }
    };

    // NOTE ON FRAMING: each message is length-prefixed (4-byte LE length +
    // exactly that many raw bytes of a self-contained Arrow IPC stream
    // buffer), rather than writing Arrow IPC stream bytes directly to the
    // live socket. pyarrow's high-level stream reader/writer over a shared
    // persistent connection was found (via the Python client's integration
    // test) to read ahead past one message's EOS marker into the next
    // message's bytes, corrupting the second and subsequent exchanges --
    // Arrow's stream format is designed for "the whole file/stream is one
    // Arrow stream," not for interleaving independent request/response
    // messages on one continuous byte channel. Framing each message
    // explicitly and parsing it from an in-memory buffer (bounded, no
    // over-read possible) is the standard fix.
    loop {
        let Some(msg_bytes) = read_framed_message(&mut stream) else {
            break; // client disconnected
        };
        let reader = match StreamReader::try_new(Cursor::new(msg_bytes), None) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[rl-env batch] malformed request: {e}");
                break;
            }
        };
        let mut got_batch = None;
        for batch in reader {
            match batch {
                Ok(b) => got_batch = Some(b),
                Err(e) => {
                    eprintln!("[rl-env batch] malformed request: {e}");
                    break;
                }
            }
        }
        let Some(request) = got_batch else { break };

        let result = if request.column_by_name("index").is_some()
            && request.column_by_name("seed").is_some()
        {
            let (indices, seeds) = batch_arrow::decode_reset_indices_request(&request);
            batched.reset_indices(&indices, &seeds)
        } else if request.column_by_name("seed").is_some() {
            let seeds = batch_arrow::decode_reset_request(&request);
            batched.reset_batch(&seeds)
        } else if request.column_by_name("action").is_some() {
            let actions = batch_arrow::decode_step_request(&request);
            batched.step_batch(&actions)
        } else {
            eprintln!("[rl-env batch] request has neither 'seed'/'index' nor 'action' column");
            break;
        };

        let response = batch_arrow::encode_result(&result);
        let response_bytes = encode_batch_to_bytes(&response);
        if write_framed_message(&mut stream, &response_bytes).is_err() {
            break;
        }
    }
    let _ = std::fs::remove_file(socket_path);
}
