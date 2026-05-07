"""
vlm_eval — MME-style evaluation for Qwen2-VL with random weights.

This package implements the MME (MultiModal Evaluation) benchmark pipeline
referencing VLMEvalKit (https://github.com/open-compass/VLMEvalKit), using
a tiny Qwen2VLForConditionalGeneration model with random weights since
pre-trained checkpoints are unavailable in this environment.

Modules
-------
dataset   : Toy MME dataset generator (PIL-based, no file downloads).
model     : Random-weight Qwen2VL model + DummyProcessor.
score     : MME ACC / ACC+ / task-score computation.
evaluate  : Full evaluation pipeline + VLMEval comparison helpers.

Quick start
-----------
    from vlm_eval.evaluate import run_evaluation
    result, records = run_evaluation(seed=42)
    print(result)
"""
