"""Evaluation & reward layer.

sandbox         hardened code execution (timeout + rlimit + no-net)
error_classify  syntax / runtime / logic / timeout taxonomy
rewards         binary / partial-credit / IRT-weighted reward functions
epr             Effective Prompt Ratio (gradient-starvation metric)
run_eval        pass@1, pass@k, teacher-student win matrix
"""
