# proofsearch

Experiments with Llemma and hypertree proof search ([HTPS](https://arxiv.org/abs/2205.11491)) for Lean4.

Relies on
* [LeanDojo](https://github.com/lean-dojo/LeanDojo)
* [Llemma](https://github.com/EleutherAI/math-lm)
* [VLLM](https://github.com/vllm-project/vllm)

# Baseline results
Run `bash scripts/eval_llemma7b.sh` to run Llemma 7b (with awq quantization)
on the minif2f validation data.
Compute the accuracy with `python compute_metrics.py`.
This should get around 26% accuracy.
