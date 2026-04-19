# agent.md — Autonomous Agent Behaviour Rules for SensorSpeak

This file governs how Claude Code behaves when operating autonomously (sub-agents,
tool-use loops, multi-step tasks) inside this project directory.

---

## Project Scope Guard

Before executing any autonomous task, confirm it falls within one of these categories:

| Allowed | Not Allowed |
|---|---|
| Edit or extend `SensorSpeak.ipynb` | Create a new notebook from scratch without asking |
| Tune constants in the named-constants block | Change the pipeline architecture silently |
| Add a new event type with rule + seed + colour | Remove an existing event type |
| Improve docstrings or markdown cells | Add cloud API calls or remove offline fallbacks |
| Fix bugs in existing functions | Introduce new ML libraries (torch, sklearn, tensorflow) |
| Add visualisation panels to the existing figure | Create a separate visualisation file |
| Write new helper functions inside the correct section | Move logic between sections |
| Update README.md / QUICKSTART.md to match code changes | Overwrite README with a shorter version |

If a request falls outside the allowed column, **ask before acting**.

---

## Decision Rules for Autonomous Tasks

### When editing the notebook

1. Read the current cell before editing — never overwrite without reading
2. Keep all named constants at the **top of their cell**, not buried in function bodies
3. Do not change a constant's default value unless explicitly instructed
4. Do not add `import` statements inside functions — imports go in Section 0 or Section 1
5. After any edit, verify JSON validity: the notebook must parse as valid `.ipynb`
6. Preserve `_segment_label`, `_raw_accel_x/y/z` column names — downstream cells depend on them

### When adding a new event type

Update all five of these locations atomically:

1. `_classify_sample()` — add the rule
2. `_SEED_MAP` — add the plain-English seed
3. `EVENT_COLORS` — add a hex colour
4. README.md **Detected Event Types** table
5. CLAUDE.md **Event Types** table

### When changing a threshold constant

1. Change the constant definition only — never the threshold name
2. Re-run the summary of affected downstream constants in a code comment
3. Update the README.md **Tunable Parameters** table if the default changes

### When the user asks for a "new feature"

Check the Roadmap in CLAUDE.md first:
- Phase 1 features: implement directly
- Phase 2/3 features: confirm with user before starting

---

## Output Hygiene

- Always save outputs to the `outputs/` directory — never to the project root
- Never delete existing output files without explicit instruction
- The PNG file is regenerated on each full run — this is expected behaviour
- CSV files are overwritten on each full run — this is expected behaviour

---

## LLM / Ollama Rules

- Always keep the keyword fallback path functional — never make Ollama a hard dependency
- If Ollama is unavailable, the pipeline must complete with degraded (keyword) answers, not crash
- Never swap the LLM model silently — always surface the current `OLLAMA_MODEL` value in output
- The fallback message must tell the user how to enable Ollama (tip line in `build_index()`)

---

## Dependency Rules

- Do not add any package not already in `requirements.txt` without asking
- Do not pin a package to an exact version (use `>=` lower bounds as in the existing file)
- Do not add `torch`, `tensorflow`, `keras`, `scikit-learn`, `xgboost`, or any ML training library
- Do not add `openai`, `anthropic`, `boto3`, `google-cloud-*`, or any cloud SDK

---

## Code Style (non-negotiable)

- **Function names**: `snake_case`, verb-first (`detect_events`, `build_index`, `query_events`)
- **Constant names**: `UPPER_SNAKE_CASE`
- **Private helpers**: prefix with `_` (`_classify_sample`, `_severity_label`)
- **Dataclass fields**: `snake_case`, typed
- **No inline lambdas** in production logic — named functions only
- **No walrus operator** (`:=`) — Python 3.10 minimum compatibility
- **No f-string debug prints left in final code** — use proper `print()` calls with labels

---

## What to Do When Stuck

1. Read `CLAUDE.md` for project constraints
2. Read `README.md` for architecture context
3. Read the relevant cell in `SensorSpeak.ipynb` for current implementation
4. If still unclear, **ask the user** — do not guess and silently change behaviour

---

## Commit / Save Behaviour

- Never commit without explicit user instruction
- When saving edits to the notebook, always write valid JSON
- Prefer `Edit` tool for small changes, `Write` tool only for full-cell rewrites
- After any file change, confirm the change is visible in the file (do not assume success)

---

## Memory Sync

After any session where project constants, event types, or pipeline structure change,
update both `CLAUDE.md` and `agent.md` to reflect the new state.
These files are the single source of truth for future sessions.
