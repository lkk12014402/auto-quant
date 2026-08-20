# Quant recipes (pin-as-artifact)

Per-model quantization recipes pinned here by the agent-optimize flow live as
`<org>__<model>.json`. They are the single source of truth: the agent proposes a
recipe once, it is validated and pinned, and later runs read it instead of
re-deriving. See `skills/quantize/SKILL.md`.
