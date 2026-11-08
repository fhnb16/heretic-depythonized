// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Tokenizer metadata for GGUF files.
//!
//! Ports of `TextModel.get_vocab_base`, `_set_vocab_gpt2`, `_set_vocab_sentencepiece`,
//! `_set_vocab_llama_hf` (`conversion/base.py`) and `SpecialVocab` / `LlamaHfVocab`
//! (`gguf-py/gguf/vocab.py`) from llama.cpp, reading `tokenizer.json`, `tokenizer.model`,
//! `tokenizer_config.json` and `config.json` directly instead of through `transformers`.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context as _, Result, anyhow, bail};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::pre_tokenizer::{CHECK_TEXT, PRE_TOKENIZER_HASHES};
use super::writer::{GgufValue, Metadata};

/// `gguf.TokenType`.
pub mod token_type {
    pub const NORMAL: i32 = 1;
    pub const UNKNOWN: i32 = 2;
    pub const CONTROL: i32 = 3;
    pub const USER_DEFINED: i32 = 4;
    pub const UNUSED: i32 = 5;
    pub const BYTE: i32 = 6;
}

/// Access to the non-weight files of a model (tokenizer, configs, chat templates).
pub trait ModelFiles {
    /// Local path of a file, if the model has it.
    fn path(&self, name: &str) -> Option<PathBuf>;
    /// Names of all files (relative, with `/` separators).
    fn names(&self) -> Vec<String>;
}

/// The files of a model directory on disk.
pub struct DirectoryFiles(pub PathBuf);

impl ModelFiles for DirectoryFiles {
    fn path(&self, name: &str) -> Option<PathBuf> {
        let path = self.0.join(name);
        path.is_file().then_some(path)
    }

    fn names(&self) -> Vec<String> {
        let mut names = Vec::new();
        let mut stack = vec![(self.0.clone(), String::new())];
        while let Some((dir, prefix)) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for entry in entries.flatten() {
                let name = format!("{prefix}{}", entry.file_name().to_string_lossy());
                if entry.path().is_dir() {
                    stack.push((entry.path(), format!("{name}/")));
                } else {
                    names.push(name);
                }
            }
        }
        names.sort();
        names
    }
}

impl ModelFiles for heretic_core::hub::ResolvedRepo {
    fn path(&self, name: &str) -> Option<PathBuf> {
        if self.has_file(name) {
            self.get(name, None).ok()
        } else {
            None
        }
    }

    fn names(&self) -> Vec<String> {
        self.files.clone()
    }
}

fn read_json(files: &dyn ModelFiles, name: &str) -> Result<Option<Value>> {
    match files.path(name) {
        Some(path) => {
            let text = std::fs::read_to_string(&path).with_context(|| format!("Failed to read {}", path.display()))?;
            Ok(Some(
                serde_json::from_str(&text).with_context(|| format!("Invalid JSON in {name}"))?,
            ))
        }
        None => Ok(None),
    }
}

/// Python truthiness of an optional JSON value.
fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
    }
}

/// `TextModel.does_token_look_special`.
pub fn does_token_look_special(token: &str) -> bool {
    matches!(token, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || (token.starts_with("<|") && token.ends_with("|>"))
        || (token.starts_with("<｜") && token.ends_with("｜>"))
        || (token.starts_with("<unused") && token.ends_with('>'))
}

/// The pre-tokenizer hash of `get_vocab_base_pre`: SHA-256 of the Python `str()` of the token
/// IDs that the tokenizer produces for [`CHECK_TEXT`] (with special tokens, like `encode`).
pub fn pre_tokenizer_hash(tokenizer_json: &std::path::Path) -> Result<String> {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_json)
        .map_err(|e| anyhow!("Failed to load {}: {e}", tokenizer_json.display()))?;
    let encoding = tokenizer
        .encode(CHECK_TEXT, true)
        .map_err(|e| anyhow!("Failed to tokenize the check text: {e}"))?;
    let ids: Vec<String> = encoding.get_ids().iter().map(u32::to_string).collect();
    let text = format!("[{}]", ids.join(", "));
    Ok(Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// `tokenizer.ggml.pre` for a hash, if llama.cpp knows the pre-tokenizer.
pub fn pre_tokenizer_name(hash: &str) -> Option<&'static str> {
    PRE_TOKENIZER_HASHES
        .iter()
        .rev()
        .find(|(h, _)| *h == hash)
        .map(|(_, name)| *name)
}

/// A token list with types (and scores for SentencePiece vocabularies).
pub struct Vocab {
    pub model: &'static str,
    pub pre: String,
    pub tokens: Vec<String>,
    pub types: Vec<i32>,
    pub scores: Option<Vec<f32>>,
}

struct AddedToken {
    id: u32,
    content: String,
    special: bool,
}

fn added_tokens(tokenizer: &Value) -> Vec<AddedToken> {
    tokenizer
        .get("added_tokens")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .filter_map(|t| {
                    Some(AddedToken {
                        id: t.get("id")?.as_u64()? as u32,
                        content: t.get("content")?.as_str()?.to_string(),
                        special: t.get("special").and_then(Value::as_bool).unwrap_or(false),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn hparam_vocab_size(hparams: &Value) -> Option<usize> {
    hparams.get("vocab_size").and_then(Value::as_u64).map(|v| v as usize)
}

/// `TextModel.get_vocab_base` for GPT-2 style BPE vocabularies from `tokenizer.json`.
pub fn gpt2_vocab(files: &dyn ModelFiles, hparams: &Value) -> Result<Vocab> {
    let path = files
        .path("tokenizer.json")
        .ok_or_else(|| anyhow!("tokenizer.json not found; GGUF export needs a fast (tokenizer.json) tokenizer"))?;
    let tokenizer = read_json(files, "tokenizer.json")?.unwrap_or_default();
    let model = tokenizer.get("model").cloned().unwrap_or_default();
    let base = model.get("vocab").and_then(Value::as_object).ok_or_else(|| {
        anyhow!(
            "tokenizer.json has no BPE vocabulary (model type {:?})",
            model.get("type")
        )
    })?;

    let added = added_tokens(&tokenizer);
    let added_by_content: HashMap<&str, &AddedToken> = added.iter().map(|t| (t.content.as_str(), t)).collect();

    // `tokenizer.vocab` of a fast tokenizer: the model vocabulary updated with added tokens.
    let mut reverse: HashMap<u32, String> = HashMap::new();
    for (token, id) in base {
        if let Some(id) = id.as_u64() {
            reverse.insert(id as u32, token.clone());
        }
    }
    for token in &added {
        reverse.insert(token.id, token.content.clone());
    }

    let max_id = reverse.keys().copied().max().unwrap_or(0) as usize;
    let vocab_size = hparam_vocab_size(hparams).unwrap_or(reverse.len());
    if max_id >= vocab_size {
        bail!("Token ID {max_id} is outside the vocabulary size {vocab_size} from config.json");
    }

    let hash = pre_tokenizer_hash(&path)?;
    let pre = pre_tokenizer_name(&hash).ok_or_else(|| {
        anyhow!(
            "llama.cpp does not recognize the BPE pre-tokenizer of this model (chkhsh {hash}). \
             Only tokenizers known to llama.cpp's convert_hf_to_gguf.py can be exported to GGUF."
        )
    })?;

    let mut tokens = Vec::with_capacity(vocab_size);
    let mut types = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size as u32 {
        match reverse.get(&i) {
            None => {
                tokens.push(format!("[PAD{i}]"));
                types.push(token_type::UNUSED);
            }
            Some(token) => {
                // Upstream also re-normalizes added tokens marked `normalized: false` with
                // decode(encode(token)), which is the identity for the tokenizers it supports.
                let mut token = token.clone();
                if let Some(entry) = added_by_content.get(token.as_str()) {
                    if entry.special || does_token_look_special(&token) {
                        types.push(token_type::CONTROL);
                    } else {
                        token = token.replace('\u{2581}', " ");
                        types.push(token_type::USER_DEFINED);
                    }
                } else {
                    types.push(token_type::NORMAL);
                }
                tokens.push(token);
            }
        }
    }

    Ok(Vocab {
        model: "gpt2",
        pre: pre.to_string(),
        tokens,
        types,
        scores: None,
    })
}

/// One piece of a SentencePiece model (`sentencepiece_model.proto`).
#[derive(Debug, Clone, PartialEq)]
pub struct SentencePiece {
    pub piece: String,
    pub score: f32,
    /// NORMAL = 1, UNKNOWN = 2, CONTROL = 3, USER_DEFINED = 4, UNUSED = 5, BYTE = 6.
    pub kind: i32,
}

fn read_varint(data: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let byte = *data.get(*pos).ok_or_else(|| anyhow!("Truncated protobuf"))?;
        *pos += 1;
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift > 63 {
            bail!("Invalid protobuf varint");
        }
    }
}

/// A protobuf field: (field number, wire type, length-delimited or fixed payload, varint value).
type ProtobufField<'a> = (u64, u8, &'a [u8], u64);

/// Splits a protobuf message into its fields.
fn protobuf_fields(data: &[u8]) -> Result<Vec<ProtobufField<'_>>> {
    let mut fields = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let key = read_varint(data, &mut pos)?;
        let (field, wire) = (key >> 3, (key & 7) as u8);
        match wire {
            0 => {
                let value = read_varint(data, &mut pos)?;
                fields.push((field, wire, &data[0..0], value));
            }
            1 => {
                let end = pos + 8;
                let bytes = data.get(pos..end).ok_or_else(|| anyhow!("Truncated protobuf"))?;
                fields.push((field, wire, bytes, 0));
                pos = end;
            }
            2 => {
                let length = read_varint(data, &mut pos)? as usize;
                let end = pos + length;
                let bytes = data.get(pos..end).ok_or_else(|| anyhow!("Truncated protobuf"))?;
                fields.push((field, wire, bytes, 0));
                pos = end;
            }
            5 => {
                let end = pos + 4;
                let bytes = data.get(pos..end).ok_or_else(|| anyhow!("Truncated protobuf"))?;
                fields.push((field, wire, bytes, 0));
                pos = end;
            }
            other => bail!("Unsupported protobuf wire type {other}"),
        }
    }
    Ok(fields)
}

/// Reads the pieces of a SentencePiece `tokenizer.model` (`ModelProto.pieces`, field 1).
pub fn parse_sentencepiece_model(data: &[u8]) -> Result<Vec<SentencePiece>> {
    let mut pieces = Vec::new();
    for (field, wire, payload, _) in protobuf_fields(data)? {
        if field != 1 || wire != 2 {
            continue;
        }
        let mut piece = SentencePiece {
            piece: String::new(),
            score: 0.0,
            kind: 1,
        };
        for (field, wire, payload, value) in protobuf_fields(payload)? {
            match (field, wire) {
                (1, 2) => piece.piece = String::from_utf8_lossy(payload).into_owned(),
                (2, 5) => piece.score = f32::from_le_bytes(payload.try_into()?),
                (3, 0) => piece.kind = value as i32,
                _ => {}
            }
        }
        pieces.push(piece);
    }
    if pieces.is_empty() {
        bail!("tokenizer.model contains no SentencePiece pieces");
    }
    Ok(pieces)
}

/// `TextModel._create_vocab_sentencepiece`. `None` if the model has no `tokenizer.model`.
pub fn sentencepiece_vocab(files: &dyn ModelFiles, hparams: &Value) -> Result<Option<Vocab>> {
    let Some(path) = files.path("tokenizer.model") else {
        return Ok(None);
    };
    let pieces = parse_sentencepiece_model(&std::fs::read(&path)?)
        .with_context(|| format!("Failed to read {}", path.display()))?;

    let vocab_size = ["vocab_size_per_layer_input", "vocab_size"]
        .iter()
        .find_map(|key| hparams.get(*key).and_then(Value::as_u64))
        .map(|v| v as usize)
        .unwrap_or(pieces.len());

    let mut tokens: Vec<String> = (0..vocab_size).map(|i| format!("[PAD{i}]")).collect();
    let mut scores = vec![-10000.0f32; vocab_size];
    let mut types = vec![token_type::UNUSED; vocab_size];

    for (id, piece) in pieces.iter().enumerate() {
        if id >= vocab_size {
            break;
        }
        // IsUnknown / IsControl / IsUnused / IsByte; everything else (including user defined
        // pieces) is NORMAL, exactly like upstream.
        types[id] = match piece.kind {
            2 => token_type::UNKNOWN,
            3 => token_type::CONTROL,
            5 => token_type::UNUSED,
            6 => token_type::BYTE,
            _ => token_type::NORMAL,
        };
        tokens[id] = piece.piece.clone();
        scores[id] = piece.score;
    }

    if let Some(Value::Object(added)) = read_json(files, "added_tokens.json")? {
        for (key, id) in added {
            let Some(id) = id.as_u64().map(|id| id as usize) else {
                continue;
            };
            if id >= vocab_size {
                continue;
            }
            tokens[id] = key;
            scores[id] = -1000.0;
            types[id] = token_type::USER_DEFINED;
        }
    }

    if let Some(config) = read_json(files, "tokenizer_config.json")?
        && let Some(decoder) = config.get("added_tokens_decoder").and_then(Value::as_object)
    {
        for (id, data) in decoder {
            let (Ok(id), Some(content)) = (id.parse::<usize>(), data.get("content").and_then(Value::as_str)) else {
                continue;
            };
            if id >= vocab_size {
                continue;
            }
            let special = data.get("special").and_then(Value::as_bool).unwrap_or(false);
            let mut token = content.to_string();
            if special || does_token_look_special(&token) {
                types[id] = token_type::CONTROL;
            } else {
                token = token.replace('\u{2581}', " ");
                types[id] = token_type::USER_DEFINED;
            }
            scores[id] = -1000.0;
            tokens[id] = token;
        }
    }

    Ok(Some(Vocab {
        model: "llama",
        pre: "default".into(),
        tokens,
        types,
        scores: Some(scores),
    }))
}

/// `LlamaHfVocab`: SentencePiece-like BPE (byte fallback) vocabularies from `tokenizer.json`.
/// `None` if the tokenizer is not of that kind (upstream raises FileNotFoundError / TypeError).
pub fn llama_hf_vocab(files: &dyn ModelFiles, hparams: &Value) -> Result<Option<Vocab>> {
    let Some(tokenizer) = read_json(files, "tokenizer.json")? else {
        return Ok(None);
    };
    let model = tokenizer.get("model").cloned().unwrap_or_default();
    let is_bpe = model.get("type").and_then(Value::as_str) == Some("BPE");
    let is_llama3 = is_bpe
        && model.get("ignore_merges").and_then(Value::as_bool).unwrap_or(false)
        && !model.get("byte_fallback").and_then(Value::as_bool).unwrap_or(true);
    if is_llama3
        || !is_bpe
        || !model.get("byte_fallback").and_then(Value::as_bool).unwrap_or(false)
        || tokenizer.pointer("/decoder/type").and_then(Value::as_str) != Some("Sequence")
    {
        return Ok(None);
    }

    let base = model
        .get("vocab")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let base_size = base
        .values()
        .filter_map(Value::as_u64)
        .max()
        .map_or(0, |m| m as usize + 1);
    let mut reverse: HashMap<usize, String> = HashMap::new();
    for (token, id) in &base {
        if let Some(id) = id.as_u64() {
            reverse.insert(id as usize, token.clone());
        }
    }

    let added = added_tokens(&tokenizer);
    // `all_special_ids` of the tokenizer: the added tokens marked special.
    let special_ids: Vec<usize> = added.iter().filter(|t| t.special).map(|t| t.id as usize).collect();
    let mut extra: Vec<&AddedToken> = added.iter().filter(|t| t.id as usize >= base_size).collect();
    extra.sort_by_key(|t| t.id);

    let mut tokens = Vec::new();
    let mut types = Vec::new();
    for id in 0..base_size {
        let token = reverse.get(&id).cloned().unwrap_or_default();
        let is_byte = token.len() == 6
            && token.starts_with("<0x")
            && token.ends_with('>')
            && token[3..5].chars().all(|c| c.is_ascii_hexdigit());
        types.push(if is_byte {
            token_type::BYTE
        } else if special_ids.contains(&id) {
            token_type::CONTROL
        } else {
            token_type::NORMAL
        });
        tokens.push(token);
    }
    for token in extra {
        types.push(if token.special {
            token_type::CONTROL
        } else {
            token_type::USER_DEFINED
        });
        tokens.push(token.content.clone());
    }
    let _ = hparams;

    let scores = vec![-1000.0; tokens.len()];
    Ok(Some(Vocab {
        model: "llama",
        pre: "default".into(),
        tokens,
        types,
        scores: Some(scores),
    }))
}

/// A chat template: a single string or named templates (`[{name, template}]`).
#[derive(Debug, Clone, PartialEq)]
pub enum ChatTemplate {
    Single(String),
    Named(Vec<(String, String)>),
}

/// `gguf.SpecialVocab`.
#[derive(Debug, Default)]
pub struct SpecialVocab {
    pub merges: Vec<String>,
    pub add_special_token: Vec<(String, bool)>,
    pub special_token_ids: Vec<(String, u32)>,
    pub chat_template: Option<ChatTemplate>,
    pub normalizer_nfc: Option<bool>,
    pub normalizer_lowercase: Option<bool>,
    pub normalizer_strip_accents: Option<bool>,
    pub clean_spaces: Option<bool>,
    pub metaspace_prepend_scheme: Option<String>,
    load_merges: bool,
    n_vocab: Option<usize>,
    special_token_types: Vec<String>,
}

impl SpecialVocab {
    pub fn load(files: &dyn ModelFiles, load_merges: bool, n_vocab: Option<usize>) -> Result<SpecialVocab> {
        let mut vocab = SpecialVocab {
            load_merges,
            n_vocab,
            special_token_types: ["bos", "eos", "unk", "sep", "pad", "cls", "mask"]
                .map(String::from)
                .to_vec(),
            ..SpecialVocab::default()
        };
        vocab.try_load_from_tokenizer_json(files)?;
        vocab.try_load_from_config_json(files)?;
        if vocab.load_merges && vocab.merges.is_empty() {
            vocab.try_load_merges_txt(files)?;
        }
        Ok(vocab)
    }

    fn set_add(&mut self, typ: &str, value: bool) {
        match self.add_special_token.iter_mut().find(|(t, _)| t == typ) {
            Some(entry) => entry.1 = value,
            None => self.add_special_token.push((typ.to_string(), value)),
        }
    }

    fn get_add(&self, typ: &str) -> Option<bool> {
        self.add_special_token.iter().find(|(t, _)| t == typ).map(|(_, v)| *v)
    }

    fn set_add_default(&mut self, typ: &str, value: bool) {
        if self.get_add(typ).is_none() {
            self.set_add(typ, value);
        }
    }

    fn set_special_token(&mut self, typ: &str, id: Option<&Value>) {
        let Some(id) = id.and_then(Value::as_u64) else { return };
        if self.n_vocab.is_none_or(|n| (id as usize) < n) {
            if self.special_token_ids.iter().any(|(t, _)| t == typ) {
                return;
            }
            self.special_token_ids.push((typ.to_string(), id as u32));
        }
    }

    pub fn set_special_token_id(&mut self, typ: &str, id: u32) {
        self.set_special_token(typ, Some(&Value::from(id)));
    }

    fn try_load_merges_txt(&mut self, files: &dyn ModelFiles) -> Result<()> {
        let Some(path) = files.path("merges.txt") else {
            return Ok(());
        };
        let text = std::fs::read_to_string(path)?;
        let mut lines = text.lines().peekable();
        if lines.peek().is_some_and(|line| line.trim().starts_with('#')) {
            lines.next();
        }
        self.merges = lines
            .filter_map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                (parts.len() == 2).then(|| format!("{} {}", parts[0], parts[1]))
            })
            .collect();
        Ok(())
    }

    fn parse_normalizer(&mut self, normalizer: &Value) {
        match normalizer.get("type").and_then(Value::as_str) {
            Some("Lowercase") => self.normalizer_lowercase = Some(true),
            Some("NFC") => self.normalizer_nfc = Some(true),
            Some("StripAccents") => self.normalizer_strip_accents = Some(true),
            Some("BertNormalizer") => {
                if let Some(v) = normalizer.get("lowercase").and_then(Value::as_bool) {
                    self.normalizer_lowercase = Some(v);
                }
                if let Some(v) = normalizer.get("strip_accents").and_then(Value::as_bool) {
                    self.normalizer_strip_accents = Some(v);
                }
            }
            Some("Sequence") => {
                for child in normalizer
                    .get("normalizers")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    self.parse_normalizer(child);
                }
            }
            _ => {}
        }
    }

    fn parse_pre_tokenizer(&mut self, pre: &Value) {
        match pre.get("type").and_then(Value::as_str) {
            Some("Metaspace") => {
                if let Some(scheme) = pre.get("prepend_scheme").and_then(Value::as_str) {
                    self.metaspace_prepend_scheme = Some(scheme.to_string());
                }
            }
            Some("Sequence") => {
                for child in pre.get("pretokenizers").and_then(Value::as_array).into_iter().flatten() {
                    self.parse_pre_tokenizer(child);
                }
            }
            _ => {}
        }
    }

    fn try_load_from_tokenizer_json(&mut self, files: &dyn ModelFiles) -> Result<()> {
        let tokenizer = read_json(files, "tokenizer.json")?;
        let mut added: Vec<Value> = Vec::new();
        if let Some(tokenizer) = &tokenizer {
            if self.load_merges
                && let Some(merges) = tokenizer.pointer("/model/merges").and_then(Value::as_array)
                && !merges.is_empty()
            {
                self.merges = if merges[0].is_string() {
                    merges.iter().filter_map(|m| m.as_str().map(String::from)).collect()
                } else {
                    // New format since transformers 4.45: pairs; spaces are encoded as U+0120.
                    merges
                        .iter()
                        .map(|pair| {
                            let parts = pair
                                .as_array()
                                .filter(|p| p.len() == 2)
                                .ok_or_else(|| anyhow!("Unknown tokenizer merges format"))?;
                            let encode = |part: &Value| -> Result<String> {
                                Ok(part
                                    .as_str()
                                    .ok_or_else(|| anyhow!("Unknown tokenizer merges format"))?
                                    .chars()
                                    .map(|c| if c == ' ' { '\u{120}' } else { c })
                                    .collect())
                            };
                            Ok(format!("{} {}", encode(&parts[0])?, encode(&parts[1])?))
                        })
                        .collect::<Result<_>>()?
                };
            }
            if let Some(normalizer) = tokenizer.get("normalizer").filter(|n| truthy(Some(n))) {
                self.parse_normalizer(normalizer);
            }
            if let Some(pre) = tokenizer.get("pre_tokenizer").filter(|n| truthy(Some(n))) {
                self.parse_pre_tokenizer(pre);
            }
            added = tokenizer
                .get("added_tokens")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
        }

        let mut config: Option<Map<String, Value>> = read_json(files, "tokenizer_config.json")?.and_then(|c| match c {
            Value::Object(map) => Some(map),
            _ => None,
        });
        let config_truthy = |config: &Option<Map<String, Value>>| config.as_ref().is_some_and(|c| !c.is_empty());

        if let Some(tokenizer) = &tokenizer {
            let get =
                |config: &Option<Map<String, Value>>, key: &str| config.as_ref().and_then(|c| c.get(key)).cloned();
            let mut special_bos = get(&config, "bos_token");
            let mut special_cls = get(&config, "cls_token");
            let mut special_eos = get(&config, "eos_token");
            let mut special_sep = get(&config, "sep_token");
            if !truthy(special_bos.as_ref()) && truthy(special_cls.as_ref()) && config_truthy(&config) {
                special_bos = special_cls.clone();
                config
                    .as_mut()
                    .unwrap()
                    .insert("bos_token".into(), special_bos.clone().unwrap());
            }
            if !truthy(special_eos.as_ref()) && truthy(special_sep.as_ref()) && config_truthy(&config) {
                special_eos = special_sep.clone();
                config
                    .as_mut()
                    .unwrap()
                    .insert("eos_token".into(), special_eos.clone().unwrap());
            }

            if let Some(post) = tokenizer.get("post_processor").filter(|p| truthy(Some(p))) {
                let processors = post
                    .get("processors")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_else(|| vec![post.clone()]);
                for processor in &processors {
                    match processor.get("type").and_then(Value::as_str) {
                        Some("ByteLevel") => {
                            self.set_add_default("bos", false);
                            self.set_add_default("eos", false);
                        }
                        Some("RobertaProcessing") => {
                            self.set_add("bos", true);
                            self.set_add("eos", true);
                            self.set_add("sep", true);
                            if !truthy(special_cls.as_ref()) && config_truthy(&config) {
                                special_cls = processor.pointer("/cls/0").cloned().or_else(|| special_bos.clone());
                                if let Some(cls) = &special_cls {
                                    config.as_mut().unwrap().insert("cls_token".into(), cls.clone());
                                }
                            }
                            if !truthy(special_sep.as_ref()) && config_truthy(&config) {
                                special_sep = processor.pointer("/sep/0").cloned().or_else(|| special_eos.clone());
                                if let Some(sep) = &special_sep {
                                    config.as_mut().unwrap().insert("sep_token".into(), sep.clone());
                                }
                            }
                        }
                        Some("TemplateProcessing") => {
                            // Crude parsing of TemplateProcessing, like upstream.
                            let single = processor
                                .get("single")
                                .and_then(Value::as_array)
                                .cloned()
                                .unwrap_or_default();
                            let pair = processor
                                .get("pair")
                                .and_then(Value::as_array)
                                .cloned()
                                .unwrap_or_default();
                            let special_id = |item: &Value| -> Option<Value> {
                                item.pointer("/SpecialToken/id").filter(|v| truthy(Some(v))).cloned()
                            };
                            let mut special_first = None;
                            let mut special_last = None;
                            if single.len() > 1 {
                                special_first = special_id(&single[0]);
                                if let Some(first) = &special_first {
                                    let matches_bos = |bos: &Option<Value>, cls: &Option<Value>| {
                                        bos.as_ref() == Some(first) || cls.as_ref() == Some(first)
                                    };
                                    if !config_truthy(&config) {
                                        special_bos = Some(first.clone());
                                    } else if !matches_bos(&special_bos, &special_cls) {
                                        if !truthy(special_bos.as_ref()) {
                                            special_bos = Some(first.clone());
                                            config.as_mut().unwrap().insert("bos_token".into(), first.clone());
                                        }
                                        if !truthy(special_cls.as_ref()) {
                                            special_cls = Some(first.clone());
                                            config.as_mut().unwrap().insert("cls_token".into(), first.clone());
                                        }
                                    }
                                    let add = matches_bos(&special_bos, &special_cls);
                                    self.set_add("bos", add);
                                }
                                special_last = special_id(single.last().unwrap());
                                if let Some(last) = &special_last {
                                    if !config_truthy(&config) {
                                        special_eos = Some(last.clone());
                                    } else if special_eos.as_ref() != Some(last) {
                                        let previous = special_eos.clone().unwrap_or(Value::Null);
                                        if !self.special_token_types.iter().any(|t| t == "eot") {
                                            self.special_token_types.push("eot".into());
                                            config.as_mut().unwrap().insert("eot_token".into(), previous);
                                        } else if !self.special_token_types.iter().any(|t| t == "eom") {
                                            self.special_token_types.push("eom".into());
                                            config.as_mut().unwrap().insert("eom_token".into(), previous);
                                        }
                                        special_eos = Some(last.clone());
                                        config.as_mut().unwrap().insert("eos_token".into(), last.clone());
                                    }
                                    let add = special_eos.as_ref() == Some(last);
                                    self.set_add("eos", add);
                                }
                            }
                            if !pair.is_empty() {
                                let start =
                                    usize::from(special_first.is_some() && special_id(&pair[0]) == special_first);
                                let stop = if special_last.is_some() && special_id(pair.last().unwrap()) == special_last
                                {
                                    pair.len() - 1
                                } else {
                                    pair.len()
                                };
                                let inner: Vec<Value> = if start < stop {
                                    pair[start..stop].to_vec()
                                } else {
                                    Vec::new()
                                };
                                if !inner.is_empty() {
                                    let seq_id = |item: &Value| {
                                        item.pointer("/Sequence/id").and_then(Value::as_str).map(String::from)
                                    };
                                    let (a, b) = (seq_id(&inner[0]), seq_id(inner.last().unwrap()));
                                    if a.as_deref() == Some("A") && b.as_deref() == Some("B") && inner.len() > 2 {
                                        let middle = &inner[1..inner.len() - 1];
                                        let is_separator = |id: &Value| {
                                            special_sep.as_ref() == Some(id) || special_eos.as_ref() == Some(id)
                                        };
                                        let mut add_sep = false;
                                        if let Some(entry) = special_id(&middle[0])
                                            && is_separator(&entry)
                                            && special_last.is_none()
                                        {
                                            add_sep = true;
                                        }
                                        if middle.len() == 2
                                            && let Some(entry) = special_id(&middle[1])
                                            && is_separator(&entry)
                                        {
                                            add_sep = true;
                                        }
                                        self.set_add("sep", add_sep);
                                        if add_sep && !truthy(special_sep.as_ref()) && config_truthy(&config) {
                                            config
                                                .as_mut()
                                                .unwrap()
                                                .insert("sep_token".into(), special_eos.clone().unwrap_or(Value::Null));
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        let Some(config) = config.filter(|c| !c.is_empty()) else {
            return Ok(());
        };

        let mut chat_template = config.get("chat_template").cloned();
        if let Some(path) = files.path("chat_template.jinja") {
            let template = std::fs::read_to_string(path)?;
            let mut additional: Vec<String> = files
                .names()
                .into_iter()
                .filter(|n| n.starts_with("additional_chat_templates/") && n.ends_with(".jinja"))
                .collect();
            additional.sort();
            if additional.is_empty() {
                chat_template = Some(Value::String(template));
            } else {
                let mut list = vec![serde_json::json!({"name": "default", "template": template})];
                for name in additional {
                    if let Some(path) = files.path(&name) {
                        let stem = name
                            .trim_start_matches("additional_chat_templates/")
                            .trim_end_matches(".jinja")
                            .to_string();
                        list.push(serde_json::json!({"name": stem, "template": std::fs::read_to_string(path)?}));
                    }
                }
                chat_template = Some(Value::Array(list));
            }
        } else if let Some(json) = read_json(files, "chat_template.json")? {
            chat_template = json.get("chat_template").cloned();
        }
        self.chat_template = match chat_template {
            Some(Value::String(template)) => Some(ChatTemplate::Single(template)),
            Some(Value::Array(list)) => Some(ChatTemplate::Named(
                list.iter()
                    .filter_map(|item| {
                        Some((
                            item.get("name")?.as_str()?.to_string(),
                            item.get("template")?.as_str()?.to_string(),
                        ))
                    })
                    .collect(),
            )),
            _ => None,
        };

        if let Some(clean) = config.get("clean_up_tokenization_spaces").and_then(Value::as_bool) {
            self.clean_spaces = Some(clean);
        }

        for typ in self.special_token_types.clone() {
            if let Some(add) = config.get(&format!("add_{typ}_token")).and_then(Value::as_bool)
                && self.get_add(&typ).is_none()
            {
                self.set_add(&typ, add);
            }
            let content = match config.get(&format!("{typ}_token")) {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Object(o)) => match o.get("content").and_then(Value::as_str) {
                    Some(s) => s.to_string(),
                    None => continue,
                },
                _ => continue,
            };
            let id = added
                .iter()
                .find(|t| t.get("content").and_then(Value::as_str) == Some(content.as_str()))
                .and_then(|t| t.get("id"))
                .cloned();
            self.set_special_token(&typ, id.as_ref());
        }
        Ok(())
    }

    fn try_load_from_config_json(&mut self, files: &dyn ModelFiles) -> Result<()> {
        let Some(config) = read_json(files, "config.json")? else {
            return Ok(());
        };
        for typ in self.special_token_types.clone() {
            let key = format!("{typ}_token_id");
            let id = config
                .get(&key)
                .filter(|v| !v.is_null())
                .or_else(|| config.get("text_config").and_then(|t| t.get(&key)))
                .cloned();
            self.set_special_token(&typ, id.as_ref());
        }
        Ok(())
    }

    /// `SpecialVocab.add_to_gguf`.
    pub fn add_to(&self, metadata: &mut Metadata) {
        if !self.merges.is_empty() {
            metadata.set("tokenizer.ggml.merges", GgufValue::strings(self.merges.iter().cloned()));
        }
        for (typ, id) in &self.special_token_ids {
            let key = match typ.as_str() {
                "bos" => "tokenizer.ggml.bos_token_id",
                "eos" => "tokenizer.ggml.eos_token_id",
                "unk" => "tokenizer.ggml.unknown_token_id",
                "sep" => "tokenizer.ggml.seperator_token_id",
                "pad" => "tokenizer.ggml.padding_token_id",
                "mask" => "tokenizer.ggml.mask_token_id",
                "eot" => "tokenizer.ggml.eot_token_id",
                "eom" => "tokenizer.ggml.eom_token_id",
                "prefix" => "tokenizer.ggml.prefix_token_id",
                "suffix" => "tokenizer.ggml.suffix_token_id",
                "middle" => "tokenizer.ggml.middle_token_id",
                // No handler upstream (e.g. cls): skipped with a warning.
                _ => continue,
            };
            metadata.set(key, GgufValue::U32(*id));
        }
        for (typ, value) in &self.add_special_token {
            let key = match typ.as_str() {
                "bos" => "tokenizer.ggml.add_bos_token",
                "eos" => "tokenizer.ggml.add_eos_token",
                "sep" => "tokenizer.ggml.add_sep_token",
                _ => continue,
            };
            metadata.set(key, GgufValue::Bool(*value));
        }
        match &self.chat_template {
            Some(ChatTemplate::Single(template)) => {
                metadata.set("tokenizer.chat_template", GgufValue::String(template.clone()));
            }
            Some(ChatTemplate::Named(list)) => {
                let mut names = Vec::new();
                let mut default = None;
                for (name, template) in list {
                    let name: String = name
                        .chars()
                        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                        .collect();
                    if name.is_empty() {
                        continue;
                    }
                    if name == "default" {
                        default = Some(template.clone());
                    } else {
                        metadata.set(
                            format!("tokenizer.chat_template.{name}"),
                            GgufValue::String(template.clone()),
                        );
                        names.push(name);
                    }
                }
                if !names.is_empty() {
                    metadata.set("tokenizer.chat_templates", GgufValue::strings(names));
                }
                if let Some(default) = default {
                    metadata.set("tokenizer.chat_template", GgufValue::String(default));
                }
            }
            None => {}
        }
        if let Some(v) = self.normalizer_nfc {
            metadata.set("tokenizer.ggml.normalizer.nfc", GgufValue::Bool(v));
        }
        if let Some(v) = self.normalizer_lowercase {
            metadata.set("tokenizer.ggml.normalizer.lowercase", GgufValue::Bool(v));
        }
        if let Some(v) = self.normalizer_strip_accents {
            metadata.set("tokenizer.ggml.normalizer.strip_accents", GgufValue::Bool(v));
        }
        if let Some(v) = self.clean_spaces {
            metadata.set("tokenizer.ggml.clean_spaces", GgufValue::Bool(v));
        }
        if let Some(v) = &self.metaspace_prepend_scheme {
            metadata.set("tokenizer.ggml.metaspace.prepend_scheme", GgufValue::String(v.clone()));
        }
    }
}

/// Writes the token list (and scores) of a vocabulary.
pub fn add_vocab(metadata: &mut Metadata, vocab: &Vocab) {
    metadata.set("tokenizer.ggml.model", GgufValue::String(vocab.model.into()));
    metadata.set("tokenizer.ggml.pre", GgufValue::String(vocab.pre.clone()));
    metadata.set(
        "tokenizer.ggml.tokens",
        GgufValue::strings(vocab.tokens.iter().cloned()),
    );
    if let Some(scores) = &vocab.scores {
        metadata.set("tokenizer.ggml.scores", GgufValue::f32s(scores.iter().copied()));
    }
    metadata.set(
        "tokenizer.ggml.token_type",
        GgufValue::i32s(vocab.types.iter().copied()),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_varint(mut value: u64, out: &mut Vec<u8>) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    fn piece(text: &str, score: f32, kind: Option<u64>) -> Vec<u8> {
        let mut message = Vec::new();
        message.push(0x0a);
        encode_varint(text.len() as u64, &mut message);
        message.extend_from_slice(text.as_bytes());
        message.push(0x15);
        message.extend_from_slice(&score.to_le_bytes());
        if let Some(kind) = kind {
            message.push(0x18);
            encode_varint(kind, &mut message);
        }
        let mut field = vec![0x0a];
        encode_varint(message.len() as u64, &mut field);
        field.extend(message);
        field
    }

    #[test]
    fn parses_sentencepiece_pieces_and_skips_other_fields() {
        let mut model = Vec::new();
        model.extend(piece("<unk>", 0.0, Some(2)));
        model.extend(piece("<s>", 0.0, Some(3)));
        model.extend(piece("\u{2581}the", -1.5, None));
        model.extend(piece("<0x0A>", 0.0, Some(6)));
        // trainer_spec (field 2) with a nested varint
        model.extend([0x12, 0x02, 0x08, 0x01]);
        let pieces = parse_sentencepiece_model(&model).unwrap();
        assert_eq!(pieces.len(), 4);
        assert_eq!(pieces[0].kind, 2);
        assert_eq!(pieces[2].piece, "\u{2581}the");
        assert_eq!(pieces[2].score, -1.5);
        assert_eq!(pieces[2].kind, 1);
        assert_eq!(pieces[3].kind, 6);
    }

    #[test]
    fn special_token_heuristics() {
        assert!(does_token_look_special("<|im_start|>"));
        assert!(does_token_look_special("<unused12>"));
        assert!(does_token_look_special("<pad>"));
        assert!(!does_token_look_special("<think>"));
    }

    #[test]
    fn special_vocab_from_llama3_style_files() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = serde_json::json!({
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": [["a", "b"], ["a b", "c"]]},
            "added_tokens": [
                {"id": 3, "content": "<|begin_of_text|>", "special": true},
                {"id": 4, "content": "<|eot_id|>", "special": true}
            ],
            "post_processor": {"type": "Sequence", "processors": [
                {"type": "ByteLevel"},
                {"type": "TemplateProcessing",
                 "single": [{"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
                 "pair": [{"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                          {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 1}}, {"Sequence": {"id": "B", "type_id": 1}}]}
            ]}
        });
        std::fs::write(dir.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();
        let config = serde_json::json!({
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
            "chat_template": "{{ messages }}",
            "clean_up_tokenization_spaces": true
        });
        std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string()).unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"bos_token_id": 3, "eos_token_id": [4, 5], "pad_token_id": 2}"#,
        )
        .unwrap();

        let files = DirectoryFiles(dir.path().to_path_buf());
        let special = SpecialVocab::load(&files, true, None).unwrap();
        assert_eq!(special.merges, vec!["a b".to_string(), "a\u{120}b c".to_string()]);
        assert_eq!(special.get_add("bos"), Some(true));
        assert_eq!(special.get_add("eos"), Some(false));
        assert_eq!(special.get_add("sep"), Some(false));
        assert_eq!(
            special.special_token_ids,
            vec![("bos".to_string(), 3), ("eos".to_string(), 4), ("pad".to_string(), 2)]
        );
        assert_eq!(
            special.chat_template,
            Some(ChatTemplate::Single("{{ messages }}".into()))
        );

        let mut metadata = Metadata::default();
        special.add_to(&mut metadata);
        assert_eq!(metadata.get("tokenizer.ggml.eos_token_id"), Some(&GgufValue::U32(4)));
        assert_eq!(
            metadata.get("tokenizer.ggml.add_bos_token"),
            Some(&GgufValue::Bool(true))
        );
        assert_eq!(
            metadata.get("tokenizer.ggml.clean_spaces"),
            Some(&GgufValue::Bool(true))
        );
    }

    #[test]
    fn known_pre_tokenizer_hashes() {
        assert_eq!(
            pre_tokenizer_name("0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5"),
            Some("llama-bpe")
        );
        // Duplicated hashes resolve to the last entry, like the upstream `if` chain.
        assert_eq!(
            pre_tokenizer_name("b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166"),
            Some("olmo")
        );
        assert_eq!(pre_tokenizer_name("0000"), None);
    }
}
