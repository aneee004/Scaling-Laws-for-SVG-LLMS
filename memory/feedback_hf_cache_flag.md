---
name: HuggingFace cache flag in prepare.py
description: load_from_cache_file should be a config-controlled flag passed to all HF .filter() and .map() calls
type: feedback
---

Always pass `load_from_cache_file` from config (not hardcoded) to all HuggingFace `.filter()` and `.map()` calls in `data/prepare.py`.

**Why:** If max_chars or other filter params change and the flag is hardcoded to True, HF returns stale cached results silently — very hard to debug. User explicitly decided to expose this as a config flag.

**How to apply:** When debugging data pipeline issues (wrong token counts, unexpected filter results), first check whether `load_from_cache_file` is set correctly in the config being used.
