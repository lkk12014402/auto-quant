"""Native evaluation kernel for autoquant-agent.

Structure adapted from NVIDIA-NeMo/Evaluator (Apache-2.0): declarative benchmark
recipes + a scorer library + an eval loop, exposed here through a pluggable
``Harness`` interface so lm-eval / evalscope can sit behind the same contract.
"""
