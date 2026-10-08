# Antigravity Workspace Rules for YUNSEUL-LMM

## Rule 1: PERMANENT PRESERVATION OF TRAINING DATA
- Under NO circumstance should any AI agent, script, or operation delete, reset, or remove files in `trained_vocal_models/` (`.pth`, `ref_vocal.wav`, `work/slices`, `audio_cache.json`, `meta.json`).
- If cleaning temporary files or cache, explicitly EXCLUDE `trained_vocal_models/` and any user music/audio directories.
- All training data must be treated as immutable, valuable assets that accumulate incrementally.
- Ensure `.permanent_vault/` snapshot mirroring remains intact.
