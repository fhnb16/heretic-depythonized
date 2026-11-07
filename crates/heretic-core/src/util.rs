// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

use anyhow::{Result, bail};

/// Recursively merges two TOML tables. Values from `override_` take precedence,
/// nested tables are merged recursively.
pub fn deep_merge_tables(base: toml::Table, override_: toml::Table) -> toml::Table {
    let mut merged = base;
    for (key, value) in override_ {
        match (merged.remove(&key), value) {
            (Some(toml::Value::Table(base_table)), toml::Value::Table(override_table)) => {
                merged.insert(key, toml::Value::Table(deep_merge_tables(base_table, override_table)));
            }
            (_, value) => {
                merged.insert(key, value);
            }
        }
    }
    merged
}

/// Parses sizes like Transformers' `convert_file_size_to_int`:
/// "5GB" = 5 * 10^9 bytes, "5GiB" = 5 * 2^30 bytes, plain integers are bytes.
pub fn parse_size(text: &str) -> Result<u64> {
    let text = text.trim();
    if let Ok(bytes) = text.parse::<u64>() {
        return Ok(bytes);
    }

    let upper = text.to_ascii_uppercase();
    let units: &[(&str, f64)] = &[
        ("TIB", 1024f64.powi(4)),
        ("GIB", 1024f64.powi(3)),
        ("MIB", 1024f64.powi(2)),
        ("KIB", 1024.0),
        ("TB", 1e12),
        ("GB", 1e9),
        ("MB", 1e6),
        ("KB", 1e3),
        ("B", 1.0),
    ];

    for (suffix, factor) in units {
        if let Some(number) = upper.strip_suffix(suffix) {
            let number: f64 = number
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid size: {text}"))?;
            if number < 0.0 {
                bail!("invalid size: {text}");
            }
            let mut bytes = number * factor;
            // Transformers: "if size ends with 'b', the value is in bits".
            if text.ends_with('b') && !text.ends_with("ib") && *suffix != "B" {
                bytes /= 8.0;
            }
            return Ok(bytes as u64);
        }
    }

    bail!("invalid size: {text} (expected e.g. \"5GB\" or \"500MiB\")")
}

/// Formats a duration the way Heretic always has ("1h 5m", "3m 20s", "42s").
pub fn format_duration(seconds: f64) -> String {
    let seconds = seconds.round().max(0.0) as u64;
    let (hours, rest) = (seconds / 3600, seconds % 3600);
    let (minutes, seconds) = (rest / 60, rest % 60);
    if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    }
}

/// Formats a byte count as gigabytes with two decimals.
pub fn format_gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1024f64.powi(3))
}

/// The longest common prefix of a list of strings (like `os.path.commonprefix`,
/// which performs a naive character-wise comparison).
pub fn common_prefix(strings: &[String]) -> String {
    let Some(first) = strings.first() else {
        return String::new();
    };

    let mut end = first.len();
    for other in &strings[1..] {
        end = first
            .char_indices()
            .zip(other.chars())
            .take_while(|((_, a), b)| a == b)
            .last()
            .map_or(0, |((index, c), _)| index + c.len_utf8())
            .min(end);
        if end == 0 {
            break;
        }
    }

    first[..end].to_string()
}

/// Splits a list into consecutive chunks of at most `batch_size` items.
pub fn batchify<T: Clone>(items: &[T], batch_size: usize) -> Vec<Vec<T>> {
    let batch_size = batch_size.max(1);
    items.chunks(batch_size).map(|chunk| chunk.to_vec()).collect()
}

/// Formats an error and its causes on one line.
pub fn format_error(error: &anyhow::Error) -> String {
    format!("{error:#}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_is_deep() {
        let base: toml::Table = toml::from_str("a = 1\n[t]\nx = 1\ny = 2").unwrap();
        let over: toml::Table = toml::from_str("b = 2\n[t]\ny = 3").unwrap();
        let merged = deep_merge_tables(base, over);
        let expected: toml::Table = toml::from_str("a = 1\nb = 2\n[t]\nx = 1\ny = 3").unwrap();
        assert_eq!(merged, expected);
    }

    #[test]
    fn sizes() {
        assert_eq!(parse_size("5GB").unwrap(), 5_000_000_000);
        assert_eq!(parse_size("2GiB").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("500MB").unwrap(), 500_000_000);
        assert_eq!(parse_size("1234").unwrap(), 1234);
        assert!(parse_size("lots").is_err());
    }

    #[test]
    fn durations() {
        assert_eq!(format_duration(42.4), "42s");
        assert_eq!(format_duration(200.0), "3m 20s");
        assert_eq!(format_duration(3900.0), "1h 5m");
    }

    #[test]
    fn prefixes() {
        let strings = ["<think>\nabc".to_string(), "<think>\nabd".to_string()];
        assert_eq!(common_prefix(&strings), "<think>\nab");
        assert_eq!(common_prefix(&["ä1".into(), "ä2".into()]), "ä");
        assert_eq!(common_prefix(&["x".into(), "y".into()]), "");
        assert_eq!(common_prefix(&[]), "");
    }

    #[test]
    fn batches() {
        assert_eq!(batchify(&[1, 2, 3, 4, 5], 2), vec![vec![1, 2], vec![3, 4], vec![5]]);
    }
}
