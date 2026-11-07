// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Streaming safetensors writer: the header is computed from the planned tensor shapes,
//! then tensor data is produced and written one tensor at a time.

use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use safetensors::Dtype;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub struct PlannedTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl PlannedTensor {
    pub fn byte_size(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.bitsize() / 8
    }
}

fn dtype_name(dtype: Dtype) -> String {
    // The serde representation is the canonical name used in headers (e.g. "BF16").
    serde_json::to_value(dtype)
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| format!("{dtype:?}"))
}

/// Writes a safetensors file. `produce(i)` must return the raw bytes of `tensors[i]`.
/// Data is laid out in the given order.
pub fn write_safetensors(
    path: &Path,
    tensors: &[PlannedTensor],
    mut produce: impl FnMut(usize) -> Result<Vec<u8>>,
) -> Result<()> {
    let mut header: BTreeMap<String, Value> = BTreeMap::new();
    header.insert("__metadata__".into(), json!({"format": "pt"}));

    let mut offset = 0usize;
    for tensor in tensors {
        let size = tensor.byte_size();
        header.insert(
            tensor.name.clone(),
            json!({
                "dtype": dtype_name(tensor.dtype),
                "shape": tensor.shape,
                "data_offsets": [offset, offset + size],
            }),
        );
        offset += size;
    }

    let mut header_bytes = serde_json::to_vec(&header)?;
    // The header is padded with spaces so that the data starts at a multiple of 8 bytes.
    while (8 + header_bytes.len()) % 8 != 0 {
        header_bytes.push(b' ');
    }

    let partial = path.with_extension("safetensors.partial");
    let file = std::fs::File::create(&partial).with_context(|| format!("Failed to create {}", partial.display()))?;
    let mut writer = BufWriter::with_capacity(8 << 20, file);
    writer.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    writer.write_all(&header_bytes)?;

    for (index, tensor) in tensors.iter().enumerate() {
        let data = produce(index).with_context(|| format!("Failed to produce tensor {}", tensor.name))?;
        if data.len() != tensor.byte_size() {
            bail!(
                "Tensor {} has {} bytes, expected {}",
                tensor.name,
                data.len(),
                tensor.byte_size()
            );
        }
        writer.write_all(&data)?;
    }

    writer.flush()?;
    drop(writer);
    std::fs::rename(&partial, path).with_context(|| format!("Failed to move {} into place", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn written_files_are_valid_safetensors() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("test.safetensors");
        let tensors = vec![
            PlannedTensor {
                name: "b".into(),
                dtype: Dtype::F32,
                shape: vec![2],
            },
            PlannedTensor {
                name: "a".into(),
                dtype: Dtype::BF16,
                shape: vec![3],
            },
        ];
        let data = [
            [1f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            [half::bf16::from_f32(1.5); 3]
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        ];
        write_safetensors(&path, &tensors, |i| Ok(data[i].clone()))?;

        let bytes = std::fs::read(&path)?;
        let loaded = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let b = loaded.tensor("b").unwrap();
        assert_eq!(b.dtype(), Dtype::F32);
        assert_eq!(b.data(), data[0].as_slice());
        let a = loaded.tensor("a").unwrap();
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.data(), data[1].as_slice());
        Ok(())
    }
}
