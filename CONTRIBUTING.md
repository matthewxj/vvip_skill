# Contributing

Keep the installable skill self-contained in `skills/vvip`. Use the standard library for its planning and HTTP tools. Do not vendor vLLM or bypass version/configuration checks to make a test pass.

## Check a change

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q skills/vvip/scripts
```

The upstream-method check is optional locally and skips without a source tree. To run it against the exact audited implementation:

```sh
VVIP_UPSTREAM_SOURCE=/path/to/vllm/package \
  python3 -m unittest discover -s tests -v
```

CPU checks are not GPU acceptance. Runtime or source-profile changes require new target GPU evidence; preserve the previous model revision, configuration, and runtime hash. Report ordinary-request costs and failed or divergent outputs alongside VIP gains. Do not edit recorded measurements to match new code.

## Build a portable skill

Commit the intended files, then package only the tracked skill:

```sh
mkdir -p dist
git archive --format=zip --prefix=vvip/ --output=dist/vvip-skill.zip HEAD:skills/vvip
python3 -m pip wheel --no-deps . --wheel-dir dist
```

The skill ZIP includes its MIT license. It excludes the repository banner, benchmark dataset, Python caches, and local artifacts. The optional wheel exposes `vvip` and contains the scheduler profile; it is not the agent skill package.

## Reproduce the report

```sh
python3 skills/vvip/scripts/analyze_benchmark.py \
  docs/evidence/qwen38/benchmark-reports.jsonl.gz \
  --out artifacts/reproduced-summary.json
```

Figure generation uses Matplotlib in a separate reporting environment; it is not a runtime dependency:

```sh
python3 scripts/render_docs.py
python3 docs/evidence/plot_performance.py docs/evidence/qwen38/performance-summary.json \
  --out artifacts/performance.svg
```

Keep the root and standalone-skill license copies identical. Before sharing new evidence, remove credentials, private hostnames, local paths, and customer prompts. Use synthetic workloads in published datasets.
