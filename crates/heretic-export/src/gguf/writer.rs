// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Writing and reading GGUF v3 files.
//!
//! Mirrors `gguf-py/gguf/gguf_writer.py` from llama.cpp: a header, the key/value metadata,
//! the tensor infos (dimensions in ggml order, i.e. reversed from PyTorch), padding to the
//! alignment, then the tensor data, each tensor padded to the alignment.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{Context as _, Result, bail, ensure};

/// `GGUF_MAGIC` ("GGUF" in little endian).
pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// `GGUF_VERSION`.
pub const GGUF_VERSION: u32 = 3;
/// `GGUF_DEFAULT_ALIGNMENT`.
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
/// `GGML_QUANT_VERSION`.
pub const GGML_QUANT_VERSION: u32 = 2;

/// `GGMLQuantizationType` (only the unquantized types are written by Heretic itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    /// Any other type, read from an existing file.
    Other(u32),
}

impl GgmlType {
    pub fn id(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::BF16 => 30,
            GgmlType::Other(id) => id,
        }
    }

    pub fn from_id(id: u32) -> GgmlType {
        match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            30 => GgmlType::BF16,
            other => GgmlType::Other(other),
        }
    }

    /// Bytes per element of the unquantized types.
    pub fn element_size(self) -> Option<usize> {
        match self {
            GgmlType::F32 => Some(4),
            GgmlType::F16 | GgmlType::BF16 => Some(2),
            GgmlType::Other(_) => None,
        }
    }

    pub fn name(self) -> String {
        match self {
            GgmlType::F32 => "F32".into(),
            GgmlType::F16 => "F16".into(),
            GgmlType::BF16 => "BF16".into(),
            GgmlType::Other(id) => format!("type {id}"),
        }
    }
}

/// A metadata value (`GGUFValueType`).
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    fn type_id(&self) -> u32 {
        match self {
            GgufValue::U8(_) => 0,
            GgufValue::I8(_) => 1,
            GgufValue::U16(_) => 2,
            GgufValue::I16(_) => 3,
            GgufValue::U32(_) => 4,
            GgufValue::I32(_) => 5,
            GgufValue::F32(_) => 6,
            GgufValue::Bool(_) => 7,
            GgufValue::String(_) => 8,
            GgufValue::Array(_) => 9,
            GgufValue::U64(_) => 10,
            GgufValue::I64(_) => 11,
            GgufValue::F64(_) => 12,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            GgufValue::U8(v) => *v as u64,
            GgufValue::U16(v) => *v as u64,
            GgufValue::U32(v) => *v as u64,
            GgufValue::U64(v) => *v,
            GgufValue::I8(v) if *v >= 0 => *v as u64,
            GgufValue::I16(v) if *v >= 0 => *v as u64,
            GgufValue::I32(v) if *v >= 0 => *v as u64,
            GgufValue::I64(v) if *v >= 0 => *v as u64,
            _ => return None,
        })
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::F32(v) => Some(*v as f64),
            GgufValue::F64(v) => Some(*v),
            other => other.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Arrays of strings (token lists, merges).
    pub fn strings(values: impl IntoIterator<Item = String>) -> GgufValue {
        GgufValue::Array(values.into_iter().map(GgufValue::String).collect())
    }

    pub fn i32s(values: impl IntoIterator<Item = i32>) -> GgufValue {
        GgufValue::Array(values.into_iter().map(GgufValue::I32).collect())
    }

    pub fn f32s(values: impl IntoIterator<Item = f32>) -> GgufValue {
        GgufValue::Array(values.into_iter().map(GgufValue::F32).collect())
    }

    pub fn bools(values: impl IntoIterator<Item = bool>) -> GgufValue {
        GgufValue::Array(values.into_iter().map(GgufValue::Bool).collect())
    }
}

/// Ordered metadata. Adding an existing key replaces its value in place, like
/// `GGUFWriter.add_key_value` does after warning about the duplicate.
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    entries: Vec<(String, GgufValue)>,
}

impl Metadata {
    pub fn set(&mut self, key: impl Into<String>, value: GgufValue) {
        let key = key.into();
        match self.entries.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = value,
            None => self.entries.push((key, value)),
        }
    }

    pub fn remove(&mut self, key: &str) {
        self.entries.retain(|(k, _)| k != key);
    }

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn entries(&self) -> &[(String, GgufValue)] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A tensor to be written. `shape` is in PyTorch order (outermost dimension first).
#[derive(Debug, Clone, PartialEq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub ggml_type: GgmlType,
    pub shape: Vec<u64>,
}

impl GgufTensorInfo {
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Size of the data in bytes (unquantized types only).
    pub fn byte_size(&self) -> Result<u64> {
        let size = self
            .ggml_type
            .element_size()
            .with_context(|| format!("Unknown size of {} for {}", self.ggml_type.name(), self.name))?;
        Ok(self.element_count() * size as u64)
    }
}

fn padding(offset: u64, alignment: u64) -> u64 {
    (alignment - offset % alignment) % alignment
}

fn write_string(writer: &mut impl Write, value: &str) -> Result<()> {
    writer.write_all(&(value.len() as u64).to_le_bytes())?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn write_value(writer: &mut impl Write, value: &GgufValue) -> Result<()> {
    match value {
        GgufValue::U8(v) => writer.write_all(&v.to_le_bytes())?,
        GgufValue::I8(v) => writer.write_all(&v.to_le_bytes())?,
        GgufValue::U16(v) => writer.write_all(&v.to_le_bytes())?,
        GgufValue::I16(v) => writer.write_all(&v.to_le_bytes())?,
        GgufValue::U32(v) => writer.write_all(&v.to_le_bytes())?,
        GgufValue::I32(v) => writer.write_all(&v.to_le_bytes())?,
        GgufValue::F32(v) => writer.write_all(&v.to_le_bytes())?,
        GgufValue::Bool(v) => writer.write_all(&[*v as u8])?,
        GgufValue::String(v) => write_string(writer, v)?,
        GgufValue::U64(v) => writer.write_all(&v.to_le_bytes())?,
        GgufValue::I64(v) => writer.write_all(&v.to_le_bytes())?,
        GgufValue::F64(v) => writer.write_all(&v.to_le_bytes())?,
        GgufValue::Array(items) => {
            // Arrays are homogeneous; an empty array is written as an array of u8 like gguf-py.
            let item_type = items.first().map_or(0, GgufValue::type_id);
            if items.iter().any(|item| item.type_id() != item_type) {
                bail!("GGUF arrays must be homogeneous");
            }
            if item_type == 9 {
                bail!("Nested GGUF arrays are not supported");
            }
            writer.write_all(&item_type.to_le_bytes())?;
            writer.write_all(&(items.len() as u64).to_le_bytes())?;
            for item in items {
                write_value(writer, item)?;
            }
        }
    }
    Ok(())
}

/// Writes a GGUF file. `produce(i)` returns the raw little-endian data of tensor `i`, which
/// must have exactly the planned size; tensors are produced one at a time and in order, so
/// memory use is bounded by the largest tensor.
pub fn write_gguf(
    path: &Path,
    metadata: &Metadata,
    tensors: &[GgufTensorInfo],
    mut produce: impl FnMut(usize) -> Result<Vec<u8>>,
) -> Result<()> {
    let alignment = match metadata.get("general.alignment").and_then(GgufValue::as_u64) {
        Some(alignment) if alignment > 0 && alignment.is_power_of_two() => alignment,
        Some(other) => bail!("Invalid general.alignment {other}"),
        None => GGUF_DEFAULT_ALIGNMENT,
    };

    let file = File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let mut writer = BufWriter::with_capacity(8 << 20, file);

    writer.write_all(&GGUF_MAGIC.to_le_bytes())?;
    writer.write_all(&GGUF_VERSION.to_le_bytes())?;
    writer.write_all(&(tensors.len() as u64).to_le_bytes())?;
    writer.write_all(&(metadata.len() as u64).to_le_bytes())?;

    for (key, value) in metadata.entries() {
        write_string(&mut writer, key)?;
        writer.write_all(&value.type_id().to_le_bytes())?;
        write_value(&mut writer, value)?;
    }

    let mut offset = 0u64;
    let mut sizes = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        write_string(&mut writer, &tensor.name)?;
        writer.write_all(&(tensor.shape.len() as u32).to_le_bytes())?;
        for dim in tensor.shape.iter().rev() {
            writer.write_all(&dim.to_le_bytes())?;
        }
        writer.write_all(&tensor.ggml_type.id().to_le_bytes())?;
        writer.write_all(&offset.to_le_bytes())?;
        let size = tensor.byte_size()?;
        sizes.push(size);
        offset += size + padding(size, alignment);
    }

    writer.flush()?;
    let position = writer.stream_position()?;
    writer.write_all(&vec![0u8; padding(position, alignment) as usize])?;

    for (index, tensor) in tensors.iter().enumerate() {
        let data = produce(index)?;
        ensure!(
            data.len() as u64 == sizes[index],
            "Tensor {} has {} bytes, expected {}",
            tensor.name,
            data.len(),
            sizes[index]
        );
        writer.write_all(&data)?;
        writer.write_all(&vec![0u8; padding(sizes[index], alignment) as usize])?;
    }

    writer.flush()?;
    Ok(())
}

/// Header of a GGUF file: metadata, tensor infos and the absolute offset of each tensor's data.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub metadata: Metadata,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_offsets: Vec<u64>,
}

impl GgufHeader {
    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(GgufValue::as_str)
    }
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut buffer = [0u8; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_le_bytes(buffer))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut buffer = [0u8; 8];
    reader.read_exact(&mut buffer)?;
    Ok(u64::from_le_bytes(buffer))
}

fn read_string(reader: &mut impl Read) -> Result<String> {
    let length = read_u64(reader)?;
    ensure!(length < 1 << 32, "Implausible string length {length}");
    let mut buffer = vec![0u8; length as usize];
    reader.read_exact(&mut buffer)?;
    String::from_utf8(buffer).context("Invalid UTF-8 in GGUF string")
}

fn read_value(reader: &mut impl Read, type_id: u32) -> Result<GgufValue> {
    let mut b1 = [0u8; 1];
    let mut b2 = [0u8; 2];
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];
    Ok(match type_id {
        0 => {
            reader.read_exact(&mut b1)?;
            GgufValue::U8(b1[0])
        }
        1 => {
            reader.read_exact(&mut b1)?;
            GgufValue::I8(b1[0] as i8)
        }
        2 => {
            reader.read_exact(&mut b2)?;
            GgufValue::U16(u16::from_le_bytes(b2))
        }
        3 => {
            reader.read_exact(&mut b2)?;
            GgufValue::I16(i16::from_le_bytes(b2))
        }
        4 => {
            reader.read_exact(&mut b4)?;
            GgufValue::U32(u32::from_le_bytes(b4))
        }
        5 => {
            reader.read_exact(&mut b4)?;
            GgufValue::I32(i32::from_le_bytes(b4))
        }
        6 => {
            reader.read_exact(&mut b4)?;
            GgufValue::F32(f32::from_le_bytes(b4))
        }
        7 => {
            reader.read_exact(&mut b1)?;
            GgufValue::Bool(b1[0] != 0)
        }
        8 => GgufValue::String(read_string(reader)?),
        9 => {
            let item_type = read_u32(reader)?;
            let count = read_u64(reader)?;
            ensure!(count < 1 << 32, "Implausible array length {count}");
            let mut items = Vec::with_capacity(count as usize);
            for _ in 0..count {
                items.push(read_value(reader, item_type)?);
            }
            GgufValue::Array(items)
        }
        10 => {
            reader.read_exact(&mut b8)?;
            GgufValue::U64(u64::from_le_bytes(b8))
        }
        11 => {
            reader.read_exact(&mut b8)?;
            GgufValue::I64(i64::from_le_bytes(b8))
        }
        12 => {
            reader.read_exact(&mut b8)?;
            GgufValue::F64(f64::from_le_bytes(b8))
        }
        other => bail!("Unknown GGUF value type {other}"),
    })
}

/// Reads the header of a GGUF file (not the tensor data).
pub fn read_gguf_header(path: &Path) -> Result<GgufHeader> {
    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);

    ensure!(
        read_u32(&mut reader)? == GGUF_MAGIC,
        "{} is not a GGUF file",
        path.display()
    );
    let version = read_u32(&mut reader)?;
    ensure!(version >= 2, "Unsupported GGUF version {version}");
    let tensor_count = read_u64(&mut reader)?;
    let kv_count = read_u64(&mut reader)?;

    let mut metadata = Metadata::default();
    for _ in 0..kv_count {
        let key = read_string(&mut reader)?;
        let type_id = read_u32(&mut reader)?;
        let value = read_value(&mut reader, type_id).with_context(|| format!("Invalid value for {key}"))?;
        metadata.set(key, value);
    }

    let mut tensors = Vec::new();
    let mut relative_offsets = Vec::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut reader)?;
        let n_dims = read_u32(&mut reader)?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(read_u64(&mut reader)?);
        }
        shape.reverse();
        let ggml_type = GgmlType::from_id(read_u32(&mut reader)?);
        relative_offsets.push(read_u64(&mut reader)?);
        tensors.push(GgufTensorInfo { name, ggml_type, shape });
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(GgufValue::as_u64)
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
    let position = reader.stream_position()?;
    let data_start = position + padding(position, alignment);
    let data_offsets = relative_offsets.iter().map(|offset| data_start + offset).collect();

    Ok(GgufHeader {
        version,
        metadata,
        tensors,
        data_offsets,
    })
}

/// Reads the raw data of one tensor (unquantized types only).
pub fn read_gguf_tensor(path: &Path, header: &GgufHeader, name: &str) -> Result<Vec<u8>> {
    let index = header
        .tensors
        .iter()
        .position(|t| t.name == name)
        .with_context(|| format!("No tensor {name} in {}", path.display()))?;
    let size = header.tensors[index].byte_size()?;
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(header.data_offsets[index]))?;
    let mut data = vec![0u8; size as usize];
    file.read_exact(&mut data)?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_header_metadata_and_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.gguf");

        let mut metadata = Metadata::default();
        metadata.set("general.architecture", GgufValue::String("llama".into()));
        metadata.set("general.alignment", GgufValue::U32(32));
        metadata.set("llama.block_count", GgufValue::U32(2));
        metadata.set("llama.rope.freq_base", GgufValue::F32(10000.0));
        metadata.set("tokenizer.ggml.tokens", GgufValue::strings(["a".into(), "bc".into()]));
        metadata.set("tokenizer.ggml.token_type", GgufValue::i32s([1, 3]));
        metadata.set("llama.attention.recurrent_layers", GgufValue::bools([true, false]));
        metadata.set("tokenizer.ggml.add_bos_token", GgufValue::Bool(true));
        // Replacing keeps the position and the count.
        metadata.set("llama.block_count", GgufValue::U32(3));
        assert_eq!(metadata.len(), 8);

        let tensors = vec![
            GgufTensorInfo {
                name: "token_embd.weight".into(),
                ggml_type: GgmlType::F16,
                shape: vec![3, 5],
            },
            GgufTensorInfo {
                name: "output_norm.weight".into(),
                ggml_type: GgmlType::F32,
                shape: vec![5],
            },
        ];
        let data: Vec<Vec<u8>> = vec![(0..30u8).collect(), (100..120u8).collect()];
        write_gguf(&path, &metadata, &tensors, |i| Ok(data[i].clone())).unwrap();

        let header = read_gguf_header(&path).unwrap();
        assert_eq!(header.version, 3);
        assert_eq!(header.architecture(), Some("llama"));
        assert_eq!(header.metadata.get("llama.block_count"), Some(&GgufValue::U32(3)));
        assert_eq!(
            header
                .metadata
                .get("tokenizer.ggml.tokens")
                .unwrap()
                .as_array()
                .unwrap()[1],
            GgufValue::String("bc".into())
        );
        assert_eq!(header.tensors, tensors);
        assert_eq!(header.data_offsets[0] % 32, 0);
        assert_eq!(header.data_offsets[1] % 32, 0);
        assert_eq!(read_gguf_tensor(&path, &header, "token_embd.weight").unwrap(), data[0]);
        assert_eq!(read_gguf_tensor(&path, &header, "output_norm.weight").unwrap(), data[1]);

        // Dimensions are stored in ggml order (innermost first).
        let bytes = std::fs::read(&path).unwrap();
        let name = b"token_embd.weight";
        let position = bytes.windows(name.len()).position(|w| w == name).unwrap() + name.len();
        assert_eq!(u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap()), 2);
        assert_eq!(
            u64::from_le_bytes(bytes[position + 4..position + 12].try_into().unwrap()),
            5
        );
    }

    #[test]
    fn rejects_wrong_tensor_size() {
        let dir = tempfile::tempdir().unwrap();
        let tensors = vec![GgufTensorInfo {
            name: "x".into(),
            ggml_type: GgmlType::F32,
            shape: vec![2],
        }];
        let error = write_gguf(&dir.path().join("x.gguf"), &Metadata::default(), &tensors, |_| {
            Ok(vec![0; 4])
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("expected 8"), "{error}");
    }
}
