// SPDX-License-Identifier: AGPL-3.0-or-later
// User interface strings (Russian and English).
// Copy rules (see AGENTS.md): no exclamation marks, no trailing period on labels, buttons,
// headings and one-sentence hints; paragraphs of several sentences keep their periods.

const en = {
  navHome: "Home",
  navNewRun: "New run",
  navRun: "Run",
  navResults: "Results",
  navChat: "Chat",
  navSettings: "Settings",

  homeTitle: "Remove refusals and slop from a local model",
  homeSubtitle: "Heretic searches for the edit with the fewest refusals and the least drift from the original model, on your own GPU and without training",
  homeStart: "New run",
  homeProjects: "Runs",
  homeNoProjects: "No runs yet. Every run saves its progress and can be continued later.",
  projectFinished: "Finished",
  projectInterrupted: "Interrupted",
  projectTrials: "trials",
  projectContinue: "Continue",
  projectResults: "Open results",
  projectDelete: "Delete run",
  projectDeleteConfirm: "Delete this run? All its trials will be lost.",

  stepModel: "Model",
  stepGoal: "Goal",
  stepParams: "Parameters",
  modelSearchPlaceholder: "Search Hugging Face",
  modelLocal: "Local folder",
  modelChecking: "Checking model…",
  modelSupported: "Supported",
  modelUnsupported: "Not supported",
  modelArchitecture: "Architecture",
  modelLayers: "Layers",
  modelWeights: "Weights",
  modelVram: "Memory needed",
  modelMultimodal: "Multimodal, only the text model is modified",
  modelDetails: "Details",
  problem_gguf: "This is a GGUF file",
  problem_gguf_hint:
    "Heretic edits the original full-precision weights (Transformers format, safetensors). Select the folder of the original model with config.json, *.safetensors and tokenizer.json, or enter its Hugging Face ID, e.g. Qwen/Qwen3.5-0.8B. You can turn the result into GGUF afterwards: Export, GGUF.",
  problem_gguf_only: "Only GGUF weights here",
  problem_gguf_only_hint:
    "Heretic needs the original model in Transformers format (config.json and *.safetensors). Use the repository the GGUF was made from, usually named without “GGUF”, e.g. Qwen/Qwen3.5-0.8B.",
  problem_file: "Select a folder, not a file",
  problem_file_hint: "Choose the folder that contains config.json, the *.safetensors weights and tokenizer.json.",
  problem_missing_path: "Folder not found",
  problem_missing_path_hint: "Check the path. Hugging Face IDs look like org/model, without slashes at the start.",
  problem_no_config: "No config.json",
  problem_no_config_hint:
    "This is not a Transformers model folder. Download the full repository from Hugging Face (config.json, *.safetensors, tokenizer files) or enter the model ID and Heretic will download it.",
  problem_bad_config: "config.json is damaged",
  problem_bad_config_hint: "The file is not valid JSON. Download it again.",
  problem_hub_not_found: "Model not found on Hugging Face",
  problem_hub_not_found_hint: "Check the ID, it has the form org/model, e.g. google/gemma-3-1b-it. IDs are case sensitive.",
  problem_hub_gated: "Access to this model is restricted",
  problem_hub_gated_hint:
    "Open the model page on huggingface.co, accept its license, then add your access token in Settings. Ungated mirrors (e.g. unsloth/…) work without a token.",
  problem_hub_error: "Hugging Face is unavailable",
  problem_hub_error_hint: "Check the internet connection or a proxy, then try again. Models already downloaded work offline.",
  problem_no_weights: "No weights in safetensors format",
  problem_no_weights_hint: "Heretic reads *.safetensors only. Use a repository that has them.",
  problem_bin_only: "Only PyTorch .bin weights",
  problem_bin_only_hint:
    "Heretic reads *.safetensors only (they are safer and can be memory-mapped). Look for a version of the model with safetensors; most repositories have one.",
  problem_no_chat_template: "No chat template",
  problem_no_chat_template_hint:
    "This looks like a base model. Heretic needs an instruction-tuned model that answers chat messages, usually named “-Instruct” or “-it”.",
  problem_unsupported: "This architecture isn't supported yet",
  problem_unsupported_hint:
    "Supported: Llama 2/3, Mistral, Qwen2/2.5, Qwen3, Qwen3.5 (dense), Gemma, Gemma 2, Gemma 3, Gemma 4 (dense). Mixture-of-experts models will come later.",
  next: "Next",
  back: "Back",
  start: "Start",
  cancel: "Cancel",
  close: "Close",

  goalTitle: "What should change",
  goalCustom: "Custom config",
  goalCustomHint: "Import a config.toml file",

  paramsTrials: "Trials",
  paramsTrialsHint: "More trials find better trade-offs and take longer. 100 is a good default.",
  paramsMethod: "Method",
  paramsDevice: "Device",
  paramsDtype: "Precision",
  paramsAdvanced: "Advanced settings",
  paramsToml: "Configuration (TOML)",
  paramsApplyToml: "Apply",
  paramsResponseLength: "Response length, tokens",
  paramsSystemPrompt: "System prompt",
  paramsInvalid: "The configuration has problems:",

  checkpointTitle: "This model was processed before",
  checkpointFinished: "The previous run is finished. You can open its results or start from scratch.",
  checkpointInterrupted: "The previous run stopped after {n} trials. You can continue it or start from scratch.",
  checkpointContinue: "Continue",
  checkpointRestart: "Start from scratch",

  runPreparing: "Preparing",
  runRunning: "Optimizing",
  runReady: "Finished",
  runFailed: "Failed",
  runTrial: "Trial",
  runElapsed: "Elapsed",
  runRemaining: "Left",
  runStop: "Stop after this trial",
  runCancel: "Abort",
  runShowResults: "Show results",
  runLog: "Log",
  runLatest: "Latest trials",
  runCurrentParameters: "Parameters being evaluated",
  runNoTrials: "Trials appear here as they are evaluated",
  phaseLoadingModel: "Loading model",
  phaseBatchSize: "Batch size",
  phaseResponsePrefix: "Response prefix",
  phaseScorers: "Baseline",
  phaseModifier: "Analysis",
  phaseOptimizing: "Optimization",
  phaseFinished: "Done",

  resultsTitle: "Best trade-offs",
  resultsHint: "Each of these trials is Pareto optimal: no other trial beats it on every metric. Pick one to chat with it, save it or upload it.",
  resultsBaseline: "original",
  resultsSelected: "Selected",
  resultsApplying: "Applying…",
  resultsParameters: "Parameters",
  resultsMore: "Run more trials",
  resultsRun: "Run",
  resultsAllTrials: "All trials",
  resultsChartHint: "Lower left is better",
  resultsParetoMark: "On the Pareto front",
  chartOriginal: "original",
  actionChat: "Chat",
  actionSave: "Save",
  actionUpload: "Upload to Hugging Face",

  chatPlaceholder: "Message",
  chatHintSend: "send",
  chatHintNewline: "new line",
  chatSend: "Send",
  chatStop: "Stop",
  chatModified: "Modified",
  chatOriginal: "Original",
  chatCompare: "Compare side by side",
  chatSystem: "System prompt",
  chatClear: "Clear",
  chatNoModel: "Select a trial on the results page first",

  exportLocal: "Save locally",
  exportHub: "Hugging Face",
  exportFolder: "Folder",
  exportChoose: "Choose",
  exportMerge: "Full model",
  exportMergeHint: "LoRA merged into the weights, ready to use anywhere",
  exportAdapter: "LoRA adapter only",
  exportAdapterHint: "Small download, applied on top of the original model",
  exportSave: "Save",
  exportSaved: "Saved to {path}",
  exportToken: "Access token",
  exportTokenHint: "Kept in memory for this session only",
  exportCheck: "Check",
  exportLoggedIn: "Signed in as {name}",
  exportRepo: "Repository",
  exportPrivate: "Private repository",
  exportCard: "Model card preview",
  exportUpload: "Upload",
  exportUploaded: "Uploaded to {repo}",
  exportGguf: "GGUF for llama.cpp",
  exportGgufHint: "One file for llama.cpp, LM Studio and Ollama, quantized to the size you need",
  exportGgufType: "Type",
  exportGgufFolderHint: "A folder (the file is named after the model and type) or a path ending in .gguf",
  exportGgufNeedsLlama: "Quantized types need llama.cpp. Choose its folder in Settings, or pick F16/BF16.",
  exportGgufLlama: "llama.cpp: {path}",
  exportGgufSaved: "Saved {path} ({size})",
  ggufQ8: "8-bit, practically lossless",
  ggufQ6: "6-bit, very close to the original",
  ggufQ5m: "5-bit, high quality",
  ggufQ5s: "5-bit, slightly smaller",
  ggufQ4m: "4-bit, the usual balance of size and quality",
  ggufQ4s: "4-bit, smaller",
  ggufQ40: "4-bit legacy format, fast on some hardware",
  ggufQ3: "3-bit, noticeable quality loss",
  ggufQ2: "2-bit, large quality loss",
  ggufF16: "16-bit, unquantized, no llama.cpp needed",
  ggufBF16: "bfloat16, unquantized, no llama.cpp needed",

  settingsLlama: "llama.cpp",
  settingsLlamaFolder: "llama.cpp folder",
  settingsLlamaHint:
    "Used to quantize models saved as GGUF. Download a release for your system from github.com/ggml-org/llama.cpp/releases, unpack it and choose the folder with llama-quantize.",
  settingsLlamaCheck: "Check",
  settingsLlamaFound: "Found",
  settingsLlamaVersion: "Found, version {version}",
  settingsLlamaMissing: "Not found",
  settingsLlamaNotSet: "Not set",
  settingsLlamaReleases: "Open releases",

  settingsAppearance: "Appearance",
  settingsLanguage: "Language",
  settingsMode: "Mode",
  settingsTheme: "Theme",
  settingsThemeHint: "Themes change colors, type and corner radii. Layout stays the same.",
  themeSystem: "System",
  themeLight: "Light",
  themeDark: "Dark",
  settingsStorage: "Storage",
  settingsCheckpoints: "Run checkpoints folder",
  settingsCache: "Model cache",
  settingsToken: "Forget token",
  settingsTokenHint: "Hugging Face access token of this session",
  settingsAbout: "About",
  settingsCuda: "CUDA support",
  yes: "yes",
  no: "no",
  updateSection: "Updates",
  updateAvailable: "Update available",
  updateInstall: "Update",
  updateDownload: "Download",
  updateLater: "Later",
  updateNotes: "Release notes",
  updateDownloading: "Downloading",
  updateVerifying: "Checking the file",
  updateInstalling: "Starting the installer",
  updateAuto: "Check for updates on start",
  updatePrereleases: "Include builds between versions",
  updateCheck: "Check for updates",
  updateChecking: "Checking",
  updateUpToDate: "Up to date",
  updateLastCheck: "Checked {time}",
  updateNever: "Not checked yet",
  updateFailed: "Update check failed",
  updateManual: "This build is updated by hand: the release page opens in the browser",
  updatePrerelease: "build between versions",

  presetDefault: "Remove censorship",
  presetDefaultDesc: "Suppress refusals and keep the model as intact as possible",
  presetNoslop: "Less slop",
  presetNoslopDesc: "Fewer clichés and less purple prose in creative writing",
  presetNohumor: "No humor",
  presetNohumorDesc: "Serious, matter-of-fact answers",
  presetNosyco: "Less sycophancy",
  presetNosycoDesc: "Fewer empty compliments and less agreeing for its own sake",
  presetEmotions: "Emotional tone",
  presetEmotionsDesc: "Shift the emotional coloring of responses",
  presetTerse: "Terse",
  presetTerseDesc: "Short answers that stay on topic",
  presetDefaultMore:
    "Measured by: how many of 100 harmful requests the model refuses (“I can't help with that”). Learns from harmless vs harmful requests.",
  presetNoslopMore:
    "Measured by: how many short stories contain stock phrases like “tapestry” or “a testament to”. Learns from plain vs flowery writing.",
  presetNosycoMore:
    "Measured by: how often the model praises the question or the user instead of answering. Learns from neutral vs flattering answers.",
  presetNohumorMore: "Measured by: how many answers turn into jokes. Learns from ordinary requests vs requests for jokes.",
  presetTerseMore:
    "Measured by: the average length of answers. Learns from a terse assistant vs a talkative one. Needs the benchmark scorer, which isn't available in this version yet.",
  presetEmotionsMore:
    "Measured by: a small emotion-recognition model that rates each answer. Needs the classifier scorer, which isn't available in this version yet.",
  goalCustomMore: "For experts: a config.toml with your own prompts, keywords and settings.",
  goalMultiHint: "You can pick several goals. They are combined into one edit.",
  goalCombined:
    "Combined goals: {goals}. Heretic looks for one edit that achieves all of them at once and measures each goal separately. More goals need more trials, so give it at least 150.",

  helpHowTitle: "How does Heretic change a model? (simple explanation)",
  helpHowBody:
    "Imagine the model as a huge machine with millions of tiny knobs. Somewhere inside, some knobs together make a habit, for example “refuse when the question sounds dangerous”.\n\n" +
    "**1. Finding the habit.** Heretic shows the model two piles of questions: ordinary ones (“how do I bake bread”) and ones that trigger the habit (“how do I pick a lock”). It looks at what happens inside the model in both cases and finds the difference. That difference is the habit.\n\n" +
    "**2. Turning it down.** Heretic adds a small correction that weakens exactly that difference. It doesn't retrain the model and doesn't touch anything else on purpose.\n\n" +
    "**3. Trying many strengths.** Too weak and the habit stays; too strong and the model gets dumber. So Heretic tries many variants (“trials”), and for each one it measures two things: how much of the habit is left, and how much the model changed in general.\n\n" +
    "**4. You choose.** At the end you see the best variants and pick the one you like, chat with it, and save it.",
  helpCombineTitle: "What happens when I pick several goals?",
  helpCombineBody:
    "Heretic merges the goals into one run:\n\n" +
    "- The “bad” examples of all goals go into one pile, the “good” ones into another. The correction removes what all unwanted habits have in common.\n" +
    "- Each goal keeps its own measurement (refusals, slop, flattery…), and the search looks for variants that are good on all of them at once.\n" +
    "- How much the model changed is measured on the ordinary questions of all goals together.\n\n" +
    "Combining works best for related habits (refusals + flattery). Goals that pull in different directions (terse + emotional) may get in each other's way; then run them one after another: save the result of the first run and use it as the model for the second.",
  helpModelTitle: "Which model fits?",
  helpModelBody:
    "- **A chat model.** Its name usually contains “Instruct”, “-it” or “Chat”. Base models without a chat template don't work.\n" +
    "- **The original format.** A folder with config.json and *.safetensors files, or a Hugging Face ID like Qwen/Qwen3.5-0.8B. GGUF files don't work as input, but you can save the result as GGUF.\n" +
    "- **Fits into memory.** “Memory needed” should be smaller than your video memory (VRAM). An 8 GB card fits models up to about 3–4B parameters in bfloat16. If it doesn't fit, Heretic can run on the CPU, but much slower.\n" +
    "- **Supported families:** Llama, Mistral, Qwen2/2.5, Qwen3, Qwen3.5, Gemma 1–4. To try things out quickly, take a small model (0.5–1.5B).",
  helpTrialsTitle: "What is a trial and how many do I need?",
  helpTrialsBody:
    "A trial is one attempt: Heretic picks a strength and a place for the correction, applies it, and tests the model on a set of questions. That's like one shot at a target.\n\n" +
    "- The first ~30 trials are random, to get a feel for the space.\n" +
    "- After that each trial learns from the previous ones and aims better.\n" +
    "- **30–50** for a quick look, **100** for a good result, **200+** when you combine goals or want the very best variant.\n\n" +
    "One trial takes from a few seconds (small model, GPU) to a few minutes (large model, CPU). The run page shows how long is left. You can stop at any time and continue later; nothing is lost.",
  paramsMethodHint: "ARA usually damages the model less. Abliteration is simpler and faster.",
  helpMethodTitle: "ARA or Abliteration, what's the difference?",
  helpMethodBody:
    "Both methods remove the same habit, they just do it differently.\n\n" +
    "**Abliteration** is like an eraser. Heretic computes one “direction” inside the model, an arrow that points from ordinary answers toward the unwanted ones, and erases that arrow from the model's weights. It's quick, simple and well known. But the arrow is an average, so it can also erase a bit of what the model needed elsewhere.\n\n" +
    "**ARA** (Arbitrary-Rank Ablation) is like a careful sculptor. Instead of one arrow it learns a small correction that moves every unwanted answer toward the nearest ordinary answers, while keeping everything else in place. It takes longer to prepare (a few minutes before the trials start), but the model usually stays smarter at the same level of habit removal.\n\n" +
    "**What to choose:** ARA by default. Abliteration if ARA is too slow on your computer, or to compare the two.",
  paramsResponseLengthHint: "How long answers are when the model is tested. One token is about ¾ of an English word.",
  helpHardwareTitle: "Device and precision in simple words",
  helpHardwareBody:
    "**Device.** The graphics card (CUDA, NVIDIA only) is 10–50 times faster than the processor (CPU). “Auto” uses the card if this build supports it and there is one.\n\n" +
    "**Precision** is how many digits the model uses for each number:\n\n" +
    "- **bfloat16**: 2 bytes per number. The best choice for RTX 30xx/40xx/50xx cards.\n" +
    "- **float16**: also 2 bytes. For older cards (GTX 10xx/16xx, RTX 20xx). Some models overflow in it.\n" +
    "- **float32**: 4 bytes, twice the memory, slower on the card. For the CPU or when a model misbehaves in 16-bit.\n" +
    "- **Auto**: tries the precision the model was saved in, then the others, and takes the first that works.",
  paramsSystemPromptHint: "A note the model reads before every question. Leave the default if unsure.",
  helpSystemTitle: "What is a system prompt?",
  helpSystemBody:
    "A system prompt is an instruction the model receives before every conversation: who it is and how to behave. The user doesn't see it, but the model always follows it.\n\n" +
    "**In a run** the system prompt is used for all test questions. Keep it neutral (“You are a helpful assistant.”) so the edit works for any later use. A special prompt here makes the edit fit that prompt only.\n\n" +
    "**In the chat** you can change it freely to see how the edited model behaves in different roles. Some goals set their own system prompt for their examples (for example “Less slop” uses a writer's prompt), and that stays as is.",
  helpTomlTitle: "What is this text?",
  helpTomlBody:
    "This is the full configuration of the run in TOML format: everything the form above sets plus dozens of expert settings (datasets, keywords, search ranges). You can edit it and press “Apply”, or save it as config.toml for the command-line version. If you don't know what a setting does, leave it alone; the defaults are sensible.",
  spExamples: "Examples, click to use:",
  spEmpty: "None",
  spEmptyValue: "",
  spAssistant: "Helpful assistant",
  spAssistantValue: "You are a helpful assistant.",
  spWriter: "Writer",
  spWriterValue: "You are a professional writer.",
  spTerse: "Brief",
  spTerseValue: "You are a concise assistant. Answer in as few words as possible, without introductions.",
  spCoder: "Programmer",
  spCoderValue: "You are an expert programmer. Answer with working code and a short explanation.",
  spTeacher: "Teacher",
  spTeacherValue: "You are a patient teacher. Explain step by step in simple words, with an example.",
  spRussian: "Answer in Russian",
  spRussianValue: "You are a helpful assistant. Always answer in Russian.",
  spRoleplay: "Character",
  spRoleplayValue:
    "You are Captain Mara Voss, a gruff but kind starship captain. Stay in character and speak in the first person.",
  chatSystemHint: "Changes take effect with the next message. Clear the chat to start the conversation over.",
  helpRunTitle: "What is happening now?",
  helpRunBody:
    "- **Loading model**: reading the weights from disk into memory.\n" +
    "- **Batch size**: finding how many questions fit into memory at once; more means faster.\n" +
    "- **Response prefix**: checking whether answers always start the same way (for example with a thinking block) so it can be skipped.\n" +
    "- **Baseline**: measuring the original model so every trial can be compared with it.\n" +
    "- **Analysis**: finding the habit inside the model (for ARA this also prepares the correction).\n" +
    "- **Optimization**: the trials themselves. Each dot on the chart is one trial: further left means less of the habit, lower means less damage.\n\n" +
    "“Stop after this trial” finishes cleanly and keeps everything; you can continue the run later from the home page.",
  helpResultsTitle: "How do I read these numbers and which trial do I pick?",
  helpResultsBody:
    "Every trial has two kinds of numbers:\n\n" +
    "- **The goal** (for example “Refusals 3/100”): how often the unwanted habit still shows up. Lower is better. The original model is shown for comparison.\n" +
    "- **KL divergence**: how much the model changed on ordinary questions. 0 means not at all. Below 0.1 you won't notice a difference, 0.1–0.5 is small, above 1 the model may get noticeably worse.\n\n" +
    "The list shows only the **best trade-offs** (the Pareto front): for each of them, no other trial is better on both numbers at once. The staircase line on the chart connects them.\n\n" +
    "**How to choose:** start with the trial that has a low goal number and KL below about 0.3. Click “Chat” and talk to it. If the habit is still there, take a trial further left; if answers got worse, take one further down. Nothing is saved until you press “Save”.",
  helpChatTitle: "How to test the edited model",
  helpChatBody:
    "- **Modified** talks to the selected trial, **Original** to the model before the edit, **Compare side by side** sends the same message to both side by side.\n" +
    "- Ask the kind of question the goal was about (for refusals: something the original refuses) and a few everyday ones (maths, a recipe, code) to check that the model didn't get dumber.\n" +
    "- Answers are generated on your computer; nothing is sent anywhere.",
  helpExportTitle: "Which format should I save in?",
  helpExportBody:
    "- **Full model**: a complete copy of the model with the edit built in, in the same format as the original. Works everywhere (Transformers, vLLM, text-generation-webui). Takes as much disk space as the original.\n" +
    "- **LoRA adapter only**: just the correction, a few megabytes. To use it you need the original model and a program that can apply LoRA. Good for sharing or keeping several variants.\n" +
    "- **GGUF**: one file for llama.cpp, LM Studio, Ollama and koboldcpp. Can be compressed (“quantized”): Q4_K_M is about 3.5 times smaller with little quality loss and is the usual choice; Q8_0 is almost lossless; F16/BF16 are not compressed. Compression needs llama.cpp (choose its folder in Settings).\n\n" +
    "**Hugging Face** uploads the full model or the adapter to your account. You need an access token with write permission: huggingface.co/settings/tokens.",
  helpStorageTitle: "What is stored where?",
  helpStorageBody:
    "- **Run checkpoints**: every run saves all its trials in a small .jsonl file here. That's how you can stop and continue later or reopen results. Deleting a run deletes this file only.\n" +
    "- **Model cache**: models and datasets downloaded from Hugging Face. They can take many gigabytes; delete the folders of models you no longer need to free space.",
};

const ru = {
  navHome: "Главная",
  navNewRun: "Новый запуск",
  navRun: "Запуск",
  navResults: "Результаты",
  navChat: "Чат",
  navSettings: "Настройки",

  homeTitle: "Уберите отказы и шаблонность из локальной модели",
  homeSubtitle: "Heretic ищет правку с наименьшим числом отказов и минимальным отклонением от оригинала прямо на вашем GPU и без обучения",
  homeStart: "Новый запуск",
  homeProjects: "Запуски",
  homeNoProjects: "Запусков пока нет. Каждый запуск сохраняет прогресс, его можно продолжить позже.",
  projectFinished: "Завершён",
  projectInterrupted: "Прерван",
  projectTrials: "попыток",
  projectContinue: "Продолжить",
  projectResults: "Открыть результаты",
  projectDelete: "Удалить запуск",
  projectDeleteConfirm: "Удалить этот запуск? Все его попытки будут потеряны.",

  stepModel: "Модель",
  stepGoal: "Цель",
  stepParams: "Параметры",
  modelSearchPlaceholder: "Поиск на Hugging Face",
  modelLocal: "Папка на диске",
  modelChecking: "Проверяю модель…",
  modelSupported: "Поддерживается",
  modelUnsupported: "Не поддерживается",
  modelArchitecture: "Архитектура",
  modelLayers: "Слоёв",
  modelWeights: "Веса",
  modelVram: "Нужно памяти",
  modelMultimodal: "Мультимодальная, меняется только текстовая часть",
  modelDetails: "Подробности",
  problem_gguf: "Это файл GGUF",
  problem_gguf_hint:
    "Heretic меняет исходные веса модели в полной точности (формат Transformers, safetensors). Выберите папку оригинальной модели с config.json, *.safetensors и tokenizer.json или введите её ID на Hugging Face, например Qwen/Qwen3.5-0.8B. GGUF можно получить из результата: Экспорт, GGUF.",
  problem_gguf_only: "Здесь только веса GGUF",
  problem_gguf_only_hint:
    "Нужна исходная модель в формате Transformers (config.json и *.safetensors). Возьмите репозиторий, из которого сделан GGUF, обычно без «GGUF» в названии, например Qwen/Qwen3.5-0.8B.",
  problem_file: "Выберите папку, а не файл",
  problem_file_hint: "Нужна папка, в которой лежат config.json, веса *.safetensors и tokenizer.json.",
  problem_missing_path: "Папка не найдена",
  problem_missing_path_hint: "Проверьте путь. ID на Hugging Face выглядит как org/model, без косой черты в начале.",
  problem_no_config: "Нет config.json",
  problem_no_config_hint:
    "Это не папка модели Transformers. Скачайте репозиторий с Hugging Face целиком (config.json, *.safetensors, файлы токенизатора) или введите ID модели, и Heretic скачает её сам.",
  problem_bad_config: "config.json повреждён",
  problem_bad_config_hint: "Файл не является корректным JSON. Скачайте его заново.",
  problem_hub_not_found: "Модель не найдена на Hugging Face",
  problem_hub_not_found_hint: "Проверьте ID, он имеет вид org/model, например google/gemma-3-1b-it. Регистр букв важен.",
  problem_hub_gated: "Доступ к модели ограничен",
  problem_hub_gated_hint:
    "Откройте страницу модели на huggingface.co, примите лицензию и добавьте токен доступа в настройках. Открытые зеркала (например, unsloth/…) работают без токена.",
  problem_hub_error: "Hugging Face недоступен",
  problem_hub_error_hint: "Проверьте подключение к интернету или прокси и повторите. Уже скачанные модели работают без сети.",
  problem_no_weights: "Нет весов в формате safetensors",
  problem_no_weights_hint: "Heretic читает только *.safetensors. Возьмите репозиторий, где они есть.",
  problem_bin_only: "Только веса PyTorch .bin",
  problem_bin_only_hint:
    "Heretic читает только *.safetensors (они безопаснее и отображаются в память). Поищите версию модели с safetensors, у большинства репозиториев она есть.",
  problem_no_chat_template: "Нет шаблона чата",
  problem_no_chat_template_hint:
    "Похоже на базовую модель. Нужна модель, обученная отвечать на сообщения чата, обычно с «-Instruct» или «-it» в названии.",
  problem_unsupported: "Эта архитектура пока не поддерживается",
  problem_unsupported_hint:
    "Поддерживаются: Llama 2/3, Mistral, Qwen2/2.5, Qwen3, Qwen3.5 (плотные), Gemma, Gemma 2, Gemma 3, Gemma 4 (плотные). Модели со смесью экспертов (MoE) появятся позже.",
  next: "Далее",
  back: "Назад",
  start: "Старт",
  cancel: "Отмена",
  close: "Закрыть",

  goalTitle: "Что нужно изменить",
  goalCustom: "Свой конфиг",
  goalCustomHint: "Импорт файла config.toml",

  paramsTrials: "Попыток",
  paramsTrialsHint: "Больше попыток — лучше компромисс, но дольше. 100 — хороший выбор по умолчанию.",
  paramsMethod: "Метод",
  paramsDevice: "Устройство",
  paramsDtype: "Точность",
  paramsAdvanced: "Расширенные настройки",
  paramsToml: "Конфигурация (TOML)",
  paramsApplyToml: "Применить",
  paramsResponseLength: "Длина ответа, токенов",
  paramsSystemPrompt: "Системный промпт",
  paramsInvalid: "В конфигурации есть ошибки:",

  checkpointTitle: "Эта модель уже обрабатывалась",
  checkpointFinished: "Предыдущий запуск завершён. Можно открыть его результаты или начать заново.",
  checkpointInterrupted: "Предыдущий запуск остановился после {n} попыток. Можно продолжить его или начать заново.",
  checkpointContinue: "Продолжить",
  checkpointRestart: "Начать заново",

  runPreparing: "Подготовка",
  runRunning: "Оптимизация",
  runReady: "Готово",
  runFailed: "Ошибка",
  runTrial: "Попытка",
  runElapsed: "Прошло",
  runRemaining: "Осталось",
  runStop: "Остановить после попытки",
  runCancel: "Прервать",
  runShowResults: "Показать результаты",
  runLog: "Журнал",
  runLatest: "Последние попытки",
  runCurrentParameters: "Параметры, которые сейчас оцениваются",
  runNoTrials: "Попытки появятся здесь по мере оценки",
  phaseLoadingModel: "Загрузка модели",
  phaseBatchSize: "Размер батча",
  phaseResponsePrefix: "Префикс ответа",
  phaseScorers: "Базовая оценка",
  phaseModifier: "Анализ",
  phaseOptimizing: "Оптимизация",
  phaseFinished: "Готово",

  resultsTitle: "Лучшие компромиссы",
  resultsHint: "Каждая из этих попыток Парето-оптимальна: никакая другая не лучше её сразу по всем метрикам. Выберите одну, чтобы пообщаться с моделью, сохранить или загрузить её.",
  resultsBaseline: "оригинал",
  resultsSelected: "Выбрана",
  resultsApplying: "Применяю…",
  resultsParameters: "Параметры",
  resultsMore: "Ещё попытки",
  resultsRun: "Запустить",
  resultsAllTrials: "Все попытки",
  resultsChartHint: "Чем ниже и левее, тем лучше",
  resultsParetoMark: "На Парето-фронте",
  chartOriginal: "оригинал",
  actionChat: "Чат",
  actionSave: "Сохранить",
  actionUpload: "Загрузить на Hugging Face",

  chatPlaceholder: "Сообщение",
  chatHintSend: "отправить",
  chatHintNewline: "новая строка",
  chatSend: "Отправить",
  chatStop: "Стоп",
  chatModified: "Изменённая",
  chatOriginal: "Оригинал",
  chatCompare: "Сравнить бок о бок",
  chatSystem: "Системный промпт",
  chatClear: "Очистить",
  chatNoModel: "Сначала выберите попытку на странице результатов",

  exportLocal: "Сохранить локально",
  exportHub: "Hugging Face",
  exportFolder: "Папка",
  exportChoose: "Выбрать",
  exportMerge: "Полная модель",
  exportMergeHint: "LoRA вмержена в веса, модель готова к использованию где угодно",
  exportAdapter: "Только LoRA-адаптер",
  exportAdapterHint: "Маленький размер, применяется поверх оригинальной модели",
  exportSave: "Сохранить",
  exportSaved: "Сохранено в {path}",
  exportToken: "Токен доступа",
  exportTokenHint: "Хранится в памяти только до конца сессии",
  exportCheck: "Проверить",
  exportLoggedIn: "Вход выполнен: {name}",
  exportRepo: "Репозиторий",
  exportPrivate: "Приватный репозиторий",
  exportCard: "Предпросмотр карточки модели",
  exportUpload: "Загрузить",
  exportUploaded: "Загружено в {repo}",
  exportGguf: "GGUF для llama.cpp",
  exportGgufHint: "Один файл для llama.cpp, LM Studio и Ollama, сжатый до нужного размера",
  exportGgufType: "Тип",
  exportGgufFolderHint: "Папка (файл будет назван по модели и типу) или путь с расширением .gguf",
  exportGgufNeedsLlama: "Для квантованных типов нужен llama.cpp. Укажите его папку в настройках или выберите F16/BF16.",
  exportGgufLlama: "llama.cpp: {path}",
  exportGgufSaved: "Сохранено: {path} ({size})",
  ggufQ8: "8 бит, практически без потерь",
  ggufQ6: "6 бит, очень близко к оригиналу",
  ggufQ5m: "5 бит, высокое качество",
  ggufQ5s: "5 бит, чуть меньше",
  ggufQ4m: "4 бита, обычный баланс размера и качества",
  ggufQ4s: "4 бита, меньше",
  ggufQ40: "4 бита, старый формат, быстрый на части оборудования",
  ggufQ3: "3 бита, заметная потеря качества",
  ggufQ2: "2 бита, сильная потеря качества",
  ggufF16: "16 бит без квантования, llama.cpp не нужен",
  ggufBF16: "bfloat16 без квантования, llama.cpp не нужен",

  settingsLlama: "llama.cpp",
  settingsLlamaFolder: "Папка llama.cpp",
  settingsLlamaHint:
    "Нужна для квантования моделей, сохранённых в GGUF. Скачайте релиз для своей системы на github.com/ggml-org/llama.cpp/releases, распакуйте и выберите папку с llama-quantize.",
  settingsLlamaCheck: "Проверить",
  settingsLlamaFound: "Найден",
  settingsLlamaVersion: "Найден, версия {version}",
  settingsLlamaMissing: "Не найден",
  settingsLlamaNotSet: "Не указан",
  settingsLlamaReleases: "Открыть релизы",

  settingsAppearance: "Оформление",
  settingsLanguage: "Язык",
  settingsMode: "Режим",
  settingsTheme: "Тема",
  settingsThemeHint: "Тема меняет цвета, шрифт и скругления. Раскладка остаётся прежней.",
  themeSystem: "Системный",
  themeLight: "Светлый",
  themeDark: "Тёмный",
  settingsStorage: "Хранение",
  settingsCheckpoints: "Папка для чекпоинтов запусков",
  settingsCache: "Кэш моделей",
  settingsToken: "Забыть токен",
  settingsTokenHint: "Токен доступа Hugging Face этой сессии",
  settingsAbout: "О программе",
  settingsCuda: "Поддержка CUDA",
  yes: "да",
  no: "нет",
  updateSection: "Обновления",
  updateAvailable: "Доступно обновление",
  updateInstall: "Обновить",
  updateDownload: "Скачать",
  updateLater: "Позже",
  updateNotes: "Что нового",
  updateDownloading: "Загрузка",
  updateVerifying: "Проверка файла",
  updateInstalling: "Запуск установщика",
  updateAuto: "Проверять обновления при запуске",
  updatePrereleases: "Учитывать промежуточные сборки",
  updateCheck: "Проверить обновления",
  updateChecking: "Проверка",
  updateUpToDate: "Установлена последняя версия",
  updateLastCheck: "Проверено {time}",
  updateNever: "Ещё не проверялось",
  updateFailed: "Не удалось проверить обновления",
  updateManual: "Эта сборка обновляется вручную: страница релиза откроется в браузере",
  updatePrerelease: "промежуточная сборка",

  presetDefault: "Снять цензуру",
  presetDefaultDesc: "Убрать отказы и сохранить модель максимально нетронутой",
  presetNoslop: "Меньше шаблонности",
  presetNoslopDesc: "Меньше клише и «пурпурной прозы» в творческих текстах",
  presetNohumor: "Без юмора",
  presetNohumorDesc: "Серьёзные, деловые ответы",
  presetNosyco: "Меньше подхалимства",
  presetNosycoDesc: "Меньше пустых комплиментов и согласия ради согласия",
  presetEmotions: "Эмоциональный тон",
  presetEmotionsDesc: "Сдвинуть эмоциональную окраску ответов",
  presetTerse: "Лаконичность",
  presetTerseDesc: "Короткие ответы по делу",
  presetDefaultMore:
    "Как измеряется: сколько из 100 опасных просьб модель отклоняет («Я не могу с этим помочь»). Учится на разнице между обычными и опасными просьбами.",
  presetNoslopMore:
    "Как измеряется: в скольких коротких рассказах встречаются заезженные обороты вроде «гобелен судеб» или «свидетельство тому». Учится на разнице между простым и вычурным текстом.",
  presetNosycoMore:
    "Как измеряется: как часто модель хвалит вопрос или собеседника вместо ответа. Учится на разнице между нейтральными и льстивыми ответами.",
  presetNohumorMore: "Как измеряется: сколько ответов превращаются в шутки. Учится на разнице между обычными просьбами и просьбами пошутить.",
  presetTerseMore:
    "Как измеряется: средняя длина ответов. Учится на разнице между лаконичным и болтливым ассистентом. Нужен оценщик-бенчмарк, которого в этой версии пока нет.",
  presetEmotionsMore:
    "Как измеряется: небольшая модель распознавания эмоций оценивает каждый ответ. Нужен оценщик-классификатор, которого в этой версии пока нет.",
  goalCustomMore: "Для опытных: config.toml со своими промптами, ключевыми словами и настройками.",
  goalMultiHint: "Можно выбрать несколько целей сразу, они объединятся в одну правку.",
  goalCombined:
    "Цели вместе: {goals}. Heretic ищет одну правку, которая решает их все сразу, и измеряет каждую цель отдельно. Чем больше целей, тем больше нужно попыток, поставьте хотя бы 150.",

  helpHowTitle: "Как Heretic меняет модель? (простыми словами)",
  helpHowBody:
    "Представьте модель как огромную машину с миллионами крошечных ручек. Где-то внутри несколько ручек вместе образуют привычку, например «отказывать, если вопрос звучит опасно».\n\n" +
    "**1. Найти привычку.** Heretic показывает модели две стопки вопросов: обычные («как испечь хлеб») и те, что вызывают привычку («как вскрыть замок»). Он смотрит, что происходит внутри модели в обоих случаях, и находит разницу. Эта разница и есть привычка.\n\n" +
    "**2. Приглушить её.** Heretic добавляет маленькую поправку, которая ослабляет именно эту разницу. Он не переобучает модель и специально ничего другого не трогает.\n\n" +
    "**3. Попробовать разную силу.** Слишком слабо, и привычка останется; слишком сильно, и модель поглупеет. Поэтому Heretic перебирает много вариантов («попыток») и для каждого измеряет две вещи: сколько осталось от привычки и насколько вообще изменилась модель.\n\n" +
    "**4. Вы выбираете.** В конце вы видите лучшие варианты, выбираете понравившийся, общаетесь с ним и сохраняете.",
  helpCombineTitle: "Что будет, если выбрать несколько целей?",
  helpCombineBody:
    "Heretic объединит цели в один запуск:\n\n" +
    "- «Плохие» примеры всех целей складываются в одну стопку, «хорошие» в другую. Поправка убирает то, что общее у всех нежелательных привычек.\n" +
    "- У каждой цели остаётся своя мерка (отказы, шаблонность, лесть…), и поиск ищет варианты, хорошие сразу по всем.\n" +
    "- Насколько изменилась модель, измеряется на обычных вопросах всех целей вместе.\n\n" +
    "Лучше всего сочетаются родственные привычки (отказы + лесть). Цели, которые тянут в разные стороны (лаконичность + эмоциональность), могут мешать друг другу. Тогда запускайте их по очереди: сохраните результат первого запуска и выберите его как модель для второго.",
  helpModelTitle: "Какая модель подойдёт?",
  helpModelBody:
    "- **Модель для чата.** В названии обычно есть «Instruct», «-it» или «Chat». Базовые модели без шаблона чата не подойдут.\n" +
    "- **Исходный формат.** Папка с config.json и файлами *.safetensors или ID на Hugging Face вроде Qwen/Qwen3.5-0.8B. GGUF на вход не подходит, но результат можно сохранить в GGUF.\n" +
    "- **Помещается в память.** «Нужно памяти» должно быть меньше видеопамяти (VRAM). На карту 8 ГБ помещаются модели примерно до 3–4 млрд параметров в bfloat16. Если не помещается, Heretic может работать на процессоре, но намного медленнее.\n" +
    "- **Поддерживаемые семейства:** Llama, Mistral, Qwen2/2.5, Qwen3, Qwen3.5, Gemma 1–4. Чтобы быстро попробовать, возьмите маленькую модель (0,5–1,5 млрд).",
  helpTrialsTitle: "Что такое попытка и сколько их нужно?",
  helpTrialsBody:
    "Попытка — это один вариант: Heretic выбирает силу и место поправки, применяет её и проверяет модель на наборе вопросов. Как один выстрел по мишени.\n\n" +
    "- Первые 30 попыток случайные, чтобы осмотреться.\n" +
    "- Дальше каждая попытка учится на предыдущих и целится точнее.\n" +
    "- **30–50**, чтобы быстро посмотреть, **100** для хорошего результата, **200+**, если цели объединены или нужен самый лучший вариант.\n\n" +
    "Одна попытка занимает от нескольких секунд (маленькая модель, видеокарта) до нескольких минут (большая модель, процессор). На странице запуска видно, сколько осталось. Остановить можно в любой момент и продолжить позже, ничего не потеряется.",
  paramsMethodHint: "ARA обычно меньше портит модель. Abliteration проще и быстрее.",
  helpMethodTitle: "ARA или Abliteration, в чём разница?",
  helpMethodBody:
    "Оба метода убирают одну и ту же привычку, просто по-разному.\n\n" +
    "**Abliteration** похож на ластик. Heretic вычисляет одно «направление» внутри модели — стрелку от обычных ответов к нежелательным — и стирает эту стрелку из весов модели. Быстро, просто и хорошо изучено. Но стрелка усреднённая, поэтому заодно может стереть немного того, что модели нужно в другом месте.\n\n" +
    "**ARA** (Arbitrary-Rank Ablation) похож на аккуратного скульптора. Вместо одной стрелки он подбирает небольшую поправку, которая сдвигает каждый нежелательный ответ к ближайшим обычным и при этом старается оставить всё остальное на месте. Подготовка дольше (несколько минут перед попытками), зато при той же силе эффекта модель обычно остаётся умнее.\n\n" +
    "**Что выбрать:** ARA по умолчанию. Abliteration, если ARA слишком медленный на вашем компьютере или хочется сравнить.",
  paramsResponseLengthHint: "Длина ответов при проверке модели. Один токен — примерно половина русского слова.",
  helpHardwareTitle: "Устройство и точность простыми словами",
  helpHardwareBody:
    "**Устройство.** Видеокарта (CUDA, только NVIDIA) в 10–50 раз быстрее процессора (CPU). «Auto» выберет видеокарту, если она есть и эта сборка её поддерживает.\n\n" +
    "**Точность** — сколько цифр модель хранит для каждого числа:\n\n" +
    "- **bfloat16**: 2 байта на число. Лучший выбор для карт RTX 30xx/40xx/50xx.\n" +
    "- **float16**: тоже 2 байта. Для старых карт (GTX 10xx/16xx, RTX 20xx). Некоторые модели в нём «переполняются».\n" +
    "- **float32**: 4 байта, в два раза больше памяти и медленнее на видеокарте. Для процессора или если модель ведёт себя странно в 16 битах.\n" +
    "- **Auto**: сначала пробует точность, в которой сохранена модель, потом остальные, и берёт первую, что сработала.",
  paramsSystemPromptHint: "Записка, которую модель читает перед каждым вопросом. Если не уверены, оставьте как есть.",
  helpSystemTitle: "Что такое системный промпт?",
  helpSystemBody:
    "Системный промпт — это инструкция, которую модель получает перед каждым разговором: кто она и как себя вести. Пользователь её не видит, но модель всегда ей следует.\n\n" +
    "**В запуске** системный промпт используется для всех проверочных вопросов. Оставьте нейтральный («You are a helpful assistant.»), чтобы правка работала при любом дальнейшем использовании. Особый промпт здесь подгонит правку только под него.\n\n" +
    "**В чате** его можно свободно менять и смотреть, как изменённая модель ведёт себя в разных ролях. У некоторых целей свой системный промпт для их примеров (например, у «Меньше шаблонности» — промпт писателя), он сохраняется.",
  helpTomlTitle: "Что это за текст?",
  helpTomlBody:
    "Это полная конфигурация запуска в формате TOML: всё, что задаёт форма выше, плюс десятки настроек для опытных (наборы данных, ключевые слова, диапазоны поиска). Можно отредактировать и нажать «Применить» или сохранить как config.toml для версии командной строки. Если не знаете, что делает настройка, не трогайте её: значения по умолчанию разумные.",
  spExamples: "Примеры, нажмите, чтобы подставить:",
  spEmpty: "Без промпта",
  spEmptyValue: "",
  spAssistant: "Помощник",
  spAssistantValue: "You are a helpful assistant.",
  spWriter: "Писатель",
  spWriterValue: "Ты профессиональный писатель.",
  spTerse: "Кратко",
  spTerseValue: "Ты лаконичный ассистент. Отвечай как можно короче, без вступлений.",
  spCoder: "Программист",
  spCoderValue: "Ты опытный программист. Отвечай рабочим кодом и коротким пояснением.",
  spTeacher: "Учитель",
  spTeacherValue: "Ты терпеливый учитель. Объясняй по шагам, простыми словами и с примером.",
  spRussian: "Отвечать по-русски",
  spRussianValue: "Ты полезный ассистент. Всегда отвечай на русском языке.",
  spRoleplay: "Персонаж",
  spRoleplayValue:
    "Ты капитан Мара Восс, суровый, но добрый капитан звездолёта. Не выходи из роли и говори от первого лица.",
  chatSystemHint: "Изменения действуют со следующего сообщения. Очистите чат, чтобы начать разговор заново.",
  helpRunTitle: "Что сейчас происходит?",
  helpRunBody:
    "- **Загрузка модели**: веса читаются с диска в память.\n" +
    "- **Размер батча**: сколько вопросов помещается в память за раз; больше — быстрее.\n" +
    "- **Префикс ответа**: проверка, не начинаются ли ответы всегда одинаково (например, с блока размышлений), чтобы его пропускать.\n" +
    "- **Базовая оценка**: измеряется исходная модель, чтобы с ней можно было сравнить каждую попытку.\n" +
    "- **Анализ**: поиск привычки внутри модели (для ARA тут же готовится поправка).\n" +
    "- **Оптимизация**: сами попытки. Каждая точка на графике — одна попытка: левее — меньше привычки, ниже — меньше вреда.\n\n" +
    "«Остановить после попытки» аккуратно завершит запуск и всё сохранит; продолжить можно позже с главной страницы.",
  helpResultsTitle: "Как читать эти числа и какую попытку выбрать?",
  helpResultsBody:
    "У каждой попытки два вида чисел:\n\n" +
    "- **Цель** (например, «Отказы 3/100»): как часто нежелательная привычка ещё проявляется. Меньше — лучше. Для сравнения показана исходная модель.\n" +
    "- **KL-дивергенция**: насколько модель изменилась на обычных вопросах. 0 — совсем не изменилась. Меньше 0,1 разницы не заметно, 0,1–0,5 — небольшая, больше 1 — модель может заметно ухудшиться.\n\n" +
    "В списке только **лучшие компромиссы** (фронт Парето): ни одна другая попытка не лучше каждой из них сразу по обоим числам. Их соединяет ступенчатая линия на графике.\n\n" +
    "**Как выбрать:** начните с попытки, у которой мало по цели и KL примерно меньше 0,3. Нажмите «Чат» и поговорите с ней. Если привычка осталась, возьмите попытку левее; если ответы стали хуже — ниже. Пока вы не нажмёте «Сохранить», ничего не записывается.",
  helpChatTitle: "Как проверить изменённую модель",
  helpChatBody:
    "- **Изменённая** отвечает выбранной попыткой, **Оригинал** — моделью до правки, **Сравнить бок о бок** отправляет одно сообщение обеим и показывает ответы рядом.\n" +
    "- Задайте вопросы по теме цели (для отказов — то, что исходная модель отклоняет) и несколько обычных (математика, рецепт, код), чтобы убедиться, что модель не поглупела.\n" +
    "- Ответы генерируются на вашем компьютере, ничего никуда не отправляется.",
  helpExportTitle: "В каком формате сохранить?",
  helpExportBody:
    "- **Полная модель**: целая копия модели со встроенной правкой в том же формате, что и оригинал. Работает везде (Transformers, vLLM, text-generation-webui). Занимает столько же места, сколько оригинал.\n" +
    "- **Только LoRA-адаптер**: одна поправка, несколько мегабайт. Для использования нужны исходная модель и программа, которая умеет применять LoRA. Удобно, чтобы делиться или хранить несколько вариантов.\n" +
    "- **GGUF**: один файл для llama.cpp, LM Studio, Ollama и koboldcpp. Можно сжать («квантовать»): Q4_K_M примерно в 3,5 раза меньше почти без потери качества, это обычный выбор; Q8_0 почти без потерь; F16/BF16 без сжатия. Для сжатия нужен llama.cpp (укажите его папку в настройках).\n\n" +
    "**Hugging Face** загружает полную модель или адаптер в ваш аккаунт. Нужен токен доступа с правом записи: huggingface.co/settings/tokens.",
  helpStorageTitle: "Что где хранится?",
  helpStorageBody:
    "- **Сохранения запусков**: каждый запуск записывает все свои попытки в небольшой файл .jsonl в этой папке. Поэтому запуск можно остановить и продолжить или снова открыть результаты. Удаление запуска удаляет только этот файл.\n" +
    "- **Кэш моделей**: модели и наборы данных, скачанные с Hugging Face. Могут занимать много гигабайт; удалите папки ненужных моделей, чтобы освободить место.",
};

const dictionaries = { en, ru };

export function translate(lang, key, params = {}) {
  let text = dictionaries[lang]?.[key] ?? en[key] ?? key;
  for (const [name, value] of Object.entries(params)) {
    text = text.replaceAll(`{${name}}`, String(value));
  }
  return text;
}

export function detectLanguage() {
  return navigator.language?.toLowerCase().startsWith("ru") ? "ru" : "en";
}
