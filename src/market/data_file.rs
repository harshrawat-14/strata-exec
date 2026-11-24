//! Opening market data files, plain or gzip-compressed.
//!
//! Era-2 data is stored as `.csv.gz` (about 5x smaller), so a year of daily
//! files fits on disk. The decision is made by file name only: a path ending in
//! `.gz` is decompressed while it is read, anything else is read as is.

use std::fs::File;
use std::io::{BufReader, Read};

use flate2::read::MultiGzDecoder;

/// A CSV reader over `path`, decompressing on the fly when it ends in `.gz`.
pub fn csv_reader(path: &str) -> std::io::Result<csv::Reader<Box<dyn Read>>> {
    let file = BufReader::new(File::open(path)?);
    let source: Box<dyn Read> = if path.ends_with(".gz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(csv::Reader::from_reader(source))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn gz_and_plain_files_read_the_same_records() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let plain = dir.join(format!("data_file_test_{pid}.csv"));
        let gz = dir.join(format!("data_file_test_{pid}.csv.gz"));
        let body = "timestamp,price\n1,100.5\n2,101.0\n";
        std::fs::write(&plain, body).unwrap();
        let mut enc = flate2::write::GzEncoder::new(
            File::create(&gz).unwrap(),
            flate2::Compression::default(),
        );
        enc.write_all(body.as_bytes()).unwrap();
        enc.finish().unwrap();

        let rows = |p: &std::path::Path| -> Vec<Vec<String>> {
            csv_reader(p.to_str().unwrap())
                .unwrap()
                .records()
                .map(|r| r.unwrap().iter().map(str::to_string).collect())
                .collect()
        };
        let (a, b) = (rows(&plain), rows(&gz));
        std::fs::remove_file(&plain).ok();
        std::fs::remove_file(&gz).ok();
        assert_eq!(a, vec![vec!["1", "100.5"], vec!["2", "101.0"]]);
        assert_eq!(a, b);
    }
}
