// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Chat template rendering, compatible with Transformers' `apply_chat_template`.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use anyhow::{Context as _, Result, anyhow};
use minijinja::value::Serde;
use minijinja::{Environment, Error, ErrorKind, Value, syntax::SyntaxConfig};
use regex::Regex;

use crate::prompt::Message;

/// Which date `strftime_now` returns inside templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateDate {
    /// A fixed date (2026-01-01 UTC). Some chat templates (e.g. gpt-oss) insert the
    /// current date into the prompt, which would make the residuals, and therefore the
    /// modified model, depend on the day Heretic is run.
    Fixed,
    /// The actual current date (used for interactive chat).
    Now,
}

/// Special token strings exposed to templates (`bos_token`, `eos_token`, ...).
pub type SpecialTokens = BTreeMap<String, String>;

/// A compiled chat template.
pub struct ChatTemplate {
    source: String,
    special_tokens: SpecialTokens,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("special_tokens", &self.special_tokens)
            .finish()
    }
}

/// Removes Transformers' `{% generation %}` extension tags, which only mark
/// assistant spans and don't affect the rendered text.
fn strip_generation_tags(source: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"\{%-?\s*(?:end)?generation\s*-?%\}").unwrap());
    re.replace_all(source, "").into_owned()
}

fn python_strftime(format: &str, date: chrono::DateTime<chrono::Utc>) -> String {
    // chrono's strftime syntax is a superset of the commonly used Python directives.
    // Invalid directives are rendered verbatim instead of panicking.
    use std::fmt::Write;
    let mut out = String::new();
    match write!(out, "{}", date.format(format)) {
        Ok(()) => out,
        Err(_) => format.to_string(),
    }
}

impl ChatTemplate {
    pub fn new(source: &str, special_tokens: SpecialTokens) -> Result<Self> {
        let template = Self {
            source: strip_generation_tags(source),
            special_tokens,
        };
        // Compile once up front so syntax errors surface at load time.
        template.environment(TemplateDate::Fixed)?;
        Ok(template)
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    fn environment(&self, date: TemplateDate) -> Result<Environment<'static>> {
        let mut env = Environment::new();
        // Transformers renders templates with trim_blocks and lstrip_blocks enabled.
        env.set_syntax(
            SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|e| anyhow!("{e}"))?,
        );
        // Python string/dict methods such as .strip(), .items(), .startswith().
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        minijinja_contrib::add_to_environment(&mut env);

        env.add_function("raise_exception", |message: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        });

        let now = match date {
            TemplateDate::Fixed => chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
                .expect("valid date")
                .with_timezone(&chrono::Utc),
            TemplateDate::Now => chrono::Utc::now(),
        };
        env.add_function("strftime_now", move |format: String| python_strftime(&format, now));

        env.add_template_owned("chat", self.source.clone())
            .context("Invalid chat template")?;
        Ok(env)
    }

    /// Renders a chat. Equivalent to `tokenizer.apply_chat_template(messages, tokenize=False, ...)`.
    pub fn render(&self, messages: &[Message], add_generation_prompt: bool, date: TemplateDate) -> Result<String> {
        let env = self.environment(date)?;
        let template = env.get_template("chat")?;

        let mut context: BTreeMap<String, Value> = self
            .special_tokens
            .iter()
            .map(|(k, v)| (k.clone(), Value::from(v.clone())))
            .collect();
        context.insert("messages".into(), Value::from(Serde(messages)));
        context.insert("add_generation_prompt".into(), Value::from(add_generation_prompt));

        template
            .render(Value::from(Serde(&context)))
            .map_err(|error| anyhow!("Failed to render chat template: {error}"))
    }

    /// Whether the template accepts a system role.
    ///
    /// Some templates (e.g. Gemma 2 and its fine-tunes) reject the system role outright,
    /// raising when one is present; others silently drop it. The only reliable way to tell
    /// is to render a probe chat and check whether a sentinel system message survives.
    /// This is also how llama.cpp's template engine (minja) detects system-role support.
    pub fn supports_system_role(&self) -> bool {
        const SENTINEL: &str = "HERETIC_SYSTEM_ROLE_PROBE";
        match self.render(
            &[Message::system(SENTINEL), Message::user("Hi")],
            true,
            TemplateDate::Now,
        ) {
            Ok(rendered) => rendered.contains(SENTINEL),
            Err(_) => false,
        }
    }
}

/// Prepares messages for templates that don't support the system role.
///
/// Templates that support the system role get the messages unchanged. For templates
/// that reject it (e.g. Gemma 2), each system message is folded into the next user
/// message (what Gemma's own later templates do), so a system role anywhere in the chat
/// is removed before rendering, not just a leading one. Empty system prompts are dropped;
/// a system message with no following user message becomes a user turn.
pub fn apply_system_role(messages: &[Message], supports_system_role: bool) -> Vec<Message> {
    if supports_system_role {
        return messages.to_vec();
    }

    let mut result = Vec::new();
    let mut pending: Vec<String> = Vec::new();

    for message in messages {
        if message.role == "system" {
            if !message.content.is_empty() {
                pending.push(message.content.clone());
            }
            continue;
        }
        let mut message = message.clone();
        if !pending.is_empty() && message.role == "user" {
            pending.push(message.content);
            message.content = pending.join("\n\n");
            pending.clear();
        }
        result.push(message);
    }

    if !pending.is_empty() {
        result.push(Message::user(pending.join("\n\n")));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A template that rejects the system role the way Gemma 2's does.
    const GEMMA_STYLE_TEMPLATE: &str = "{% for message in messages %}\
        {% if message['role'] == 'system' %}\
        {{ raise_exception('System role not supported') }}\
        {% endif %}\
        <start_of_turn>{{ message['role'] }}\n{{ message['content'] }}<end_of_turn>\n\
        {% endfor %}\
        {% if add_generation_prompt %}<start_of_turn>model\n{% endif %}";

    /// A template that renders the system role normally.
    const SYSTEM_TEMPLATE: &str = "{% for message in messages %}\
        <|{{ message['role'] }}|>\n{{ message['content'] }}\n\
        {% endfor %}\
        {% if add_generation_prompt %}<|assistant|>\n{% endif %}";

    fn template(source: &str) -> ChatTemplate {
        ChatTemplate::new(source, SpecialTokens::new()).unwrap()
    }

    #[test]
    fn detects_unsupported_system_role() {
        assert!(!template(GEMMA_STYLE_TEMPLATE).supports_system_role());
    }

    #[test]
    fn detects_supported_system_role() {
        assert!(template(SYSTEM_TEMPLATE).supports_system_role());
    }

    #[test]
    fn passthrough_when_supported() {
        let messages = vec![Message::system("You are helpful."), Message::user("hi")];
        assert_eq!(apply_system_role(&messages, true), messages);
    }

    #[test]
    fn folds_into_first_user_message() {
        let result = apply_system_role(&[Message::system("You are helpful."), Message::user("hi")], false);
        assert_eq!(result, vec![Message::user("You are helpful.\n\nhi")]);
    }

    #[test]
    fn drops_empty_system_prompt() {
        let result = apply_system_role(&[Message::system(""), Message::user("hi")], false);
        assert_eq!(result, vec![Message::user("hi")]);
    }

    #[test]
    fn no_system_message_is_unchanged() {
        let messages = vec![Message::user("hi")];
        assert_eq!(apply_system_role(&messages, false), messages);
    }

    #[test]
    fn folds_system_role_appearing_later() {
        let result = apply_system_role(
            &[
                Message::user("hi"),
                Message::assistant("hello"),
                Message::system("Be terse."),
                Message::user("and now?"),
            ],
            false,
        );
        assert_eq!(
            result,
            vec![
                Message::user("hi"),
                Message::assistant("hello"),
                Message::user("Be terse.\n\nand now?"),
            ]
        );
    }

    #[test]
    fn trailing_system_message_becomes_user_turn() {
        let result = apply_system_role(&[Message::user("hi"), Message::system("Be terse.")], false);
        assert_eq!(result, vec![Message::user("hi"), Message::user("Be terse.")]);
    }

    #[test]
    fn folded_later_system_role_renders_without_raising() {
        let chat = apply_system_role(
            &[
                Message::user("hi"),
                Message::system("Be terse."),
                Message::user("and now?"),
            ],
            false,
        );
        let rendered = template(GEMMA_STYLE_TEMPLATE)
            .render(&chat, true, TemplateDate::Fixed)
            .unwrap();
        assert!(rendered.contains("Be terse."));
    }

    #[test]
    fn renders_like_transformers() {
        let rendered = template(SYSTEM_TEMPLATE)
            .render(&[Message::system("S"), Message::user("U")], true, TemplateDate::Fixed)
            .unwrap();
        assert_eq!(rendered, "<|system|>\nS\n<|user|>\nU\n<|assistant|>\n");
    }

    #[test]
    fn trim_and_lstrip_blocks_are_enabled() {
        let source = "{% for m in messages %}\n    {% if true %}\n[{{ m.content }}]\n    {% endif %}\n{% endfor %}";
        let rendered = template(source)
            .render(&[Message::user("a"), Message::user("b")], false, TemplateDate::Fixed)
            .unwrap();
        assert_eq!(rendered, "[a]\n[b]\n");
    }

    #[test]
    fn fixed_date_and_python_methods() {
        let source = "{{ strftime_now('%d %B %Y') }}|{{ messages[0].content.strip().upper() }}|{{ bos_token }}\
            {% generation %}x{% endgeneration %}";
        let mut special = SpecialTokens::new();
        special.insert("bos_token".into(), "<s>".into());
        let rendered = ChatTemplate::new(source, special)
            .unwrap()
            .render(&[Message::user("  hi  ")], false, TemplateDate::Fixed)
            .unwrap();
        assert_eq!(rendered, "01 January 2026|HI|<s>x");
    }

    #[test]
    fn namespace_and_loop_controls() {
        let source = "{% set ns = namespace(n=0) %}{% for m in messages %}{% if loop.index > 2 %}{% break %}{% endif %}\
            {% set ns.n = ns.n + 1 %}{% endfor %}{{ ns.n }}";
        let rendered = template(source)
            .render(
                &[Message::user("a"), Message::user("b"), Message::user("c")],
                false,
                TemplateDate::Fixed,
            )
            .unwrap();
        assert_eq!(rendered, "2");
    }
}
