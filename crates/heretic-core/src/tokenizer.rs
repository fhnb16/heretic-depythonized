// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Tokenizer with chat template, loaded from a model repository like
//! Transformers' `AutoTokenizer`.

use std::path::Path;

use anyhow::{Context as _, Result, anyhow, bail};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

use crate::chat::{ChatTemplate, SpecialTokens, TemplateDate, apply_system_role};
use crate::hub::ResolvedRepo;
use crate::prompt::{Message, Prompt};

/// The files of a model repository that make up its tokenizer and chat template.
pub const TOKENIZER_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
    "config.json",
];

/// A batch of left-padded token sequences.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedBatch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
}

impl EncodedBatch {
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }

    pub fn seq_len(&self) -> usize {
        self.input_ids.first().map_or(0, Vec::len)
    }
}

pub struct ModelTokenizer {
    inner: Tokenizer,
    template: Option<ChatTemplate>,
    pub special_tokens: SpecialTokens,
    pub pad_token_id: u32,
    /// Token IDs that end generation (from the tokenizer and `generation_config.json`).
    pub eos_token_ids: Vec<u32>,
    pub bos_token_id: Option<u32>,
    clean_up_tokenization_spaces: bool,
    supports_system_role: bool,
}

impl std::fmt::Debug for ModelTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelTokenizer")
            .field("special_tokens", &self.special_tokens)
            .field("pad_token_id", &self.pad_token_id)
            .field("eos_token_ids", &self.eos_token_ids)
            .finish()
    }
}

fn read_json(path: Option<&Path>) -> Result<serde_json::Value> {
    match path {
        None => Ok(serde_json::Value::Null),
        Some(path) => {
            let text = std::fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
            serde_json::from_str(&text).with_context(|| format!("Invalid JSON in {}", path.display()))
        }
    }
}

/// Special token values are either strings or AddedToken objects (`{"content": ...}`).
fn token_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(map) => map.get("content").and_then(|c| c.as_str()).map(String::from),
        _ => None,
    }
}

fn token_ids(value: &serde_json::Value) -> Vec<u32> {
    match value {
        serde_json::Value::Number(n) => n.as_u64().map(|id| vec![id as u32]).unwrap_or_default(),
        serde_json::Value::Array(items) => items.iter().filter_map(|i| i.as_u64()).map(|id| id as u32).collect(),
        _ => Vec::new(),
    }
}

/// Selects the chat template source, like Transformers does: `chat_template.jinja`,
/// then `tokenizer_config.json`, then `chat_template.json` (used by processors).
fn chat_template_source(repo: &ResolvedRepo, tokenizer_config: &serde_json::Value) -> Result<Option<String>> {
    if let Some(path) = repo.get_optional("chat_template.jinja")? {
        return Ok(Some(std::fs::read_to_string(path)?));
    }

    match tokenizer_config.get("chat_template") {
        Some(serde_json::Value::String(source)) => return Ok(Some(source.clone())),
        Some(serde_json::Value::Array(templates)) => {
            let named = |name: &str| {
                templates
                    .iter()
                    .find(|t| t.get("name").and_then(|n| n.as_str()) == Some(name))
                    .and_then(|t| t.get("template").and_then(|s| s.as_str()))
                    .map(String::from)
            };
            if let Some(source) = named("default").or_else(|| {
                templates
                    .first()
                    .and_then(|t| t.get("template").and_then(|s| s.as_str()).map(String::from))
            }) {
                return Ok(Some(source));
            }
        }
        _ => {}
    }

    if let Some(path) = repo.get_optional("chat_template.json")? {
        let value = read_json(Some(&path))?;
        if let Some(source) = value.get("chat_template").and_then(|s| s.as_str()) {
            return Ok(Some(source.to_string()));
        }
    }

    Ok(None)
}

impl ModelTokenizer {
    /// Loads the tokenizer, special tokens and chat template of a model repository.
    pub fn from_repo(repo: &ResolvedRepo) -> Result<Self> {
        let tokenizer_path = repo.get_optional("tokenizer.json")?.ok_or_else(|| {
            anyhow!(
                "{} has no tokenizer.json (SentencePiece-only tokenizers are not supported)",
                repo.id
            )
        })?;
        let inner = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load {}: {e}", tokenizer_path.display()))?;

        let tokenizer_config = read_json(repo.get_optional("tokenizer_config.json")?.as_deref())?;
        let special_tokens_map = read_json(repo.get_optional("special_tokens_map.json")?.as_deref())?;
        let generation_config = read_json(repo.get_optional("generation_config.json")?.as_deref())?;
        let model_config = read_json(repo.get_optional("config.json")?.as_deref())?;

        let mut special_tokens = SpecialTokens::new();
        for key in [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
        ] {
            // Like Transformers, special_tokens_map.json takes precedence over tokenizer_config.json.
            let value = special_tokens_map
                .get(key)
                .and_then(token_string)
                .or_else(|| tokenizer_config.get(key).and_then(token_string));
            if let Some(value) = value {
                special_tokens.insert(key.to_string(), value);
            }
        }

        let template = match chat_template_source(repo, &tokenizer_config)? {
            Some(source) => Some(
                ChatTemplate::new(&source, special_tokens.clone())
                    .with_context(|| format!("Invalid chat template of {}", repo.id))?,
            ),
            None => None,
        };

        let id_of = |token: &str| inner.token_to_id(token);

        // Fallback for tokenizers that don't declare a special pad token.
        let pad_token = special_tokens
            .get("pad_token")
            .or_else(|| special_tokens.get("eos_token"))
            .cloned();
        let pad_token_id = pad_token
            .as_deref()
            .and_then(id_of)
            .or_else(|| {
                generation_config
                    .get("pad_token_id")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32)
            })
            .or_else(|| token_ids(&generation_config["eos_token_id"]).first().copied())
            .or_else(|| token_ids(&model_config["eos_token_id"]).first().copied())
            .ok_or_else(|| anyhow!("Could not determine a pad token for {}", repo.id))?;
        if let Some(pad) = &pad_token {
            special_tokens.entry("pad_token".into()).or_insert_with(|| pad.clone());
        }

        let mut eos_token_ids = token_ids(&generation_config["eos_token_id"]);
        if eos_token_ids.is_empty() {
            eos_token_ids = token_ids(&model_config["eos_token_id"]);
        }
        if let Some(id) = special_tokens.get("eos_token").and_then(|t| id_of(t))
            && !eos_token_ids.contains(&id)
        {
            eos_token_ids.push(id);
        }

        let bos_token_id = special_tokens.get("bos_token").and_then(|t| id_of(t));

        let clean_up_tokenization_spaces = tokenizer_config
            .get("clean_up_tokenization_spaces")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let supports_system_role = template.as_ref().is_some_and(ChatTemplate::supports_system_role);

        let mut tokenizer = Self {
            inner,
            template,
            special_tokens,
            pad_token_id,
            eos_token_ids,
            bos_token_id,
            clean_up_tokenization_spaces,
            supports_system_role,
        };
        tokenizer.configure_padding();
        Ok(tokenizer)
    }

    fn configure_padding(&mut self) {
        let pad_token = self.inner.id_to_token(self.pad_token_id).unwrap_or_default();
        // CRITICAL: Always use left-padding for decoder-only models during generation.
        // Right-padding causes empty outputs because the model sees PAD tokens after
        // the prompt and thinks the sequence is complete.
        self.inner.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Left,
            pad_to_multiple_of: None,
            pad_id: self.pad_token_id,
            pad_type_id: 0,
            pad_token,
        }));
        let _ = self.inner.with_truncation(None);
    }

    pub fn has_chat_template(&self) -> bool {
        self.template.is_some()
    }

    pub fn chat_template(&self) -> Option<&ChatTemplate> {
        self.template.as_ref()
    }

    /// Whether the chat template accepts a system role (see [`ChatTemplate::supports_system_role`]).
    pub fn supports_system_role(&self) -> bool {
        self.supports_system_role
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Renders a chat with the chat template, folding system messages if necessary.
    pub fn apply_chat_template(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
        date: TemplateDate,
    ) -> Result<String> {
        let template = self
            .template
            .as_ref()
            .ok_or_else(|| anyhow!("The model has no chat template"))?;
        let messages = apply_system_role(messages, self.supports_system_role);
        template.render(&messages, add_generation_prompt, date)
    }

    /// Renders a prompt (system + user message) with a generation prompt and the fixed date.
    pub fn render_prompt(&self, prompt: &Prompt) -> Result<String> {
        let messages: Vec<Message> = prompt.into();
        self.apply_chat_template(&messages, true, TemplateDate::Fixed)
    }

    /// Tokenizes texts with left padding. Like `tokenizer(texts, padding=True)` in
    /// Transformers, special tokens are added according to the tokenizer's post-processor.
    pub fn encode_batch(&self, texts: &[String]) -> Result<EncodedBatch> {
        if texts.is_empty() {
            bail!("Cannot encode an empty batch");
        }
        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("Tokenization failed: {e}"))?;
        Ok(EncodedBatch {
            input_ids: encodings.iter().map(|e| e.get_ids().to_vec()).collect(),
            attention_mask: encodings.iter().map(|e| e.get_attention_mask().to_vec()).collect(),
        })
    }

    /// Tokenizes a single text without padding.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode_fast(text, add_special_tokens)
            .map_err(|e| anyhow!("Tokenization failed: {e}"))?;
        Ok(encoding
            .get_ids()
            .iter()
            .zip(encoding.get_attention_mask())
            .filter(|(_, mask)| **mask == 1)
            .map(|(id, _)| *id)
            .collect())
    }

    /// Decodes token IDs like `tokenizer.decode` in Transformers.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let text = self
            .inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow!("Decoding failed: {e}"))?;
        Ok(if self.clean_up_tokenization_spaces {
            clean_up_tokenization(&text)
        } else {
            text
        })
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }
}

/// Transformers' `clean_up_tokenization`.
pub fn clean_up_tokenization(text: &str) -> String {
    text.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_repo(dir: &Path, with_pad: bool) {
        let tokenizer = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 2, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "<eos>": 1, "<pad>": 2, "hello": 3, "world": 4, "a": 5}, "unk_token": "[UNK]"}
        });
        std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();

        let mut config = serde_json::json!({
            "eos_token": "<eos>",
            "chat_template": "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
        });
        if with_pad {
            config["pad_token"] = serde_json::json!({"content": "<pad>"});
        }
        std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
        std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": [1, 5]}"#).unwrap();
    }

    fn load(with_pad: bool) -> ModelTokenizer {
        let dir = tempfile::tempdir().unwrap();
        write_test_repo(dir.path(), with_pad);
        let repo = ResolvedRepo::open(crate::hub::RepoType::Model, dir.path().to_str().unwrap(), None).unwrap();
        ModelTokenizer::from_repo(&repo).unwrap()
    }

    #[test]
    fn left_padding() {
        let tokenizer = load(true);
        assert_eq!(tokenizer.pad_token_id, 2);
        let batch = tokenizer
            .encode_batch(&["hello world a".into(), "hello".into()])
            .unwrap();
        assert_eq!(batch.input_ids, vec![vec![3, 4, 5], vec![2, 2, 3]]);
        assert_eq!(batch.attention_mask, vec![vec![1, 1, 1], vec![0, 0, 1]]);
    }

    #[test]
    fn pad_falls_back_to_eos() {
        let tokenizer = load(false);
        assert_eq!(tokenizer.pad_token_id, 1);
        assert_eq!(tokenizer.special_tokens["pad_token"], "<eos>");
    }

    #[test]
    fn eos_ids_come_from_generation_config() {
        let tokenizer = load(true);
        assert_eq!(tokenizer.eos_token_ids, vec![1, 5]);
    }

    #[test]
    fn renders_prompt_with_system_role() {
        let tokenizer = load(true);
        assert!(tokenizer.supports_system_role());
        let rendered = tokenizer.render_prompt(&Prompt::new("S", "U")).unwrap();
        assert_eq!(rendered, "system: S\nuser: U\nassistant:");
    }

    #[test]
    fn decoding_skips_special_tokens() {
        let tokenizer = load(true);
        assert_eq!(tokenizer.decode(&[3, 4, 1], true).unwrap(), "hello world");
    }

    #[test]
    fn cleanup() {
        assert_eq!(
            clean_up_tokenization("I do n't know , really ."),
            "I don't know, really."
        );
    }
}
