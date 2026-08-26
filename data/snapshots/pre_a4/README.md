# Exact pre-A4 training inputs

These snapshots are the exact inputs used by the A1 continuation, A3-prime
pilot, and matched A3 confirmatory runs. They are checked in because the
historical 254-row SFT generation did not record a vLLM sampling seed; a hash
alone would not let another researcher reproduce the executed training input.

| file | rows | SHA-256 |
|---|---:|---|
| `prompt_pool.clean.jsonl` | 392 | `03cad62c6988d674ae05f301a0fa8d6bdf1affd966e75ccf0bdc46f17ad91287` |
| `prompt_pool.a3prime.jsonl` | 310 | `66c08352c0799ed6677a28e422c6ab3dce1a48c4baa4296a452232a83d027cfd` |
| `prompt_pool.a3matched.jsonl` | 310 | `4edfb977f4b96abff8e40e575710ed6910bc2d89939fed6e1cd493f628129cc3` |
| `sft_base.jsonl` | 254 | `dd218cb418aaa90b3df03f0d2d38b46cd87a05a005fdccb572a2d34ef04321d3` |

`prompt_pool.a3prime.jsonl` and `prompt_pool.a3matched.jsonl` can be rebuilt
deterministically from the clean snapshot:

```bash
python -m data.build_a3prime_pool \
  --pool data/snapshots/pre_a4/prompt_pool.clean.jsonl \
  --out /tmp/prompt_pool.a3prime.jsonl \
  --control-out /tmp/prompt_pool.a3matched.jsonl \
  --visible-count 1 --seed 20260818
```

The A3-prime and A3-matched files have identical prompt IDs. A3-prime exposes
one fixed assertion to the reward and retains the others for held-out scoring;
A3-matched exposes every assertion. Training pads both 310-row datasets to 312
rows, producing 78 updates with the shared effective batch size of 32.

The project writeup originally described the seed-0 SFT set as 255 rows. These
continuation runs used the archived 254-row file above, so the two inputs are
not treated as identical.
