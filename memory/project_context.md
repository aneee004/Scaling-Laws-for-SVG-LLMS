---
name: Project context and role constraints
description: NYU CS-GY 6923 ML project — scaling laws for SVG transformers. Claude's role is design/debug only, no code writing.
type: project
---

NYU Tandon CS-GY 6923 (Spring 2026), due 2026-05-01. Scaling laws study for decoder-only Transformers trained on SVG code.

**Why:** Aniruth is writing all code himself. Claude's role is design suggestions and debugging only — no code, no implementations, even if asked directly.

**How to apply:** Decline code writing requests and redirect to design/debugging discussion. Guide stage by stage.

Dataset chosen: `starvector/svg-stack` (primary). Rejected svg-stack-simple (27MB outliers, narrow distribution), svg-fonts (too homogeneous), svg-icons (too small).

Config lives in `configs/base.yaml`. Model-specific overrides in `configs/1m.yaml` etc.
