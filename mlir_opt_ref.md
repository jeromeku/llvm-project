# mlir-opt Cheat Sheet (Dialect‑agnostic)

> Scope: **driver/IO**, **pass manager**, **printing/inspection**, **verification**, **tracing/debugging/visualization**, **timing/stats**, **reproducers**, **logging**, and a few **safety toggles**. Goal: understand/inspect IR, trace pipelines, and debug, without leaning on any one dialect.

---

## Driver / IO / Environment

* `--help`, `--help-list`, `--version`

  Prints options, pass lists, or tool version.

  **Use when**: sanity check what your build supports.

* `--list-passes`

  Lists all registered passes (by name).

  **Use when**: composing `--pass-pipeline`.

* `--show-dialects`

  Lists registered dialects in the current process.

  **Use when**: confirming plugins/registrations.

* `--load-dialect-plugin=<so>` / `--load-pass-plugin=<so>`

  `dlopen` plugins that register dialects/passes.

  **Use when**: testing out-of-tree dialects/passes.

* `--irdl-file=<file.irdl>`

  Registers dialects/types from an IRDL spec.

  **Use when**: experimenting with IRDL-defined dialects.

* `--allow-unregistered-dialect`

  Parse/print IR even if dialects aren’t registered.

  **Use when**: you just need to **inspect** a file, not run passes.

* `--no-implicit-module`

  Disables auto-insertion of a top-level `module`.

  **Use when**: testing parsing behaviors or round-tripping raw ops.

* `--split-input-file[=<marker>]`

  Treat a single file as multiple chunks separated by a marker.

  **Use when**: big test files; isolate failing fragments.

* `-o <filename>`

  Output path (use `-o -` to print to stdout).

  **Use when**: piping into other tools or diffing.

* `--emit-bytecode` / `--emit-bytecode-version=<N>`

  Emit MLIR bytecode instead of text; pick version.

  **Use when**: size/perf experiments; compatibility tests.

* `--elide-resource-data-from-bytecode`

  Drop resource payloads when emitting bytecode.

  **Use when**: shrinking reproducers for sharing.

* `--mlir-disable-threading`

  Force single-threaded pass execution.

  **Use when**: debugging nondeterminism.

* `--color`

  Colorize diagnostics (auto/force).

  **Use when**: local TTY readability.

---

## Pass Manager / Pipeline Control

* `--pass-pipeline='<p1,p2,...>'`

  Run a textual pipeline.

  **Use when**: scripting transformations in CI.

* `--print-pipeline-passes`

  Print a `-passes`-compatible string for the configured pipeline.

  **Use when**: serializing complex PM setups.

* `--dump-pass-pipeline`

  Pretty-print the pipeline (hierarchical).

  **Use when**: auditing nested or op-scoped pipelines.

* `--verify-each`

  Run the verifier after **every** pass.

  **Use when**: bisecting pass-induced IR breakage.

* `--verify-diagnostics[=only-expected|all]`

  Check `// expected-...` directives in the input vs emitted diagnostics.

  **Use when**: authoring FileCheck-style diagnostic tests.

* `--verify-roundtrip`

  Parse → print → parse again; ensure success.

  **Use when**: testing printers/parsers for your dialects or generic forms.

* `--verify-region-info`

  Expensive checker for region invariants.

  **Use when**: deep correctness hunts; expect extra runtime.

---

## Printing / Inspection (Structure, Syntax, Semantics)

**Core printer knobs (affect textual IR output, not semantics):**

* `--mlir-pretty-debuginfo` / `--mlir-print-debuginfo`

  Include readable debug locations; `pretty` adds formatting.

  **Use when**: mapping IR back to source.

* `--mlir-print-local-scope`

  Inline aliases for attrs/types/locs; fewer `#attr` indirections.

  **Use when**: reading unknown IR quickly.

* `--mlir-use-nameloc-as-prefix`

  Use `NameLoc` as SSA value prefixes.

  **Use when**: correlate values to source names.

* `--mlir-print-unique-ssa-ids`

  Stable numeric IDs for SSA values/args.

  **Use when**: diffing IR across runs.

* `--mlir-print-value-users`

  Print user info as comments next to defs.

  **Use when**: local def-use reading without external tools.

* `--mlir-print-ir-module-scope`

  Always print from top module for before/after dumps.

  **Use when**: consistent context in logs.

* `--mlir-print-skip-regions`

  Collapse region bodies (for brevity).

  **Use when**: high-level scans of huge functions.

* `--mlir-elide-elementsattrs-if-larger=<N>`

  Replace big DenseElements with `...`.

  **Use when**: cut noise from giant constants.

* `--mlir-print-elementsattrs-with-hex-if-larger=<N>`

  Print large tensors as hex strings.

  **Use when**: binary payloads, reproducible diffs.

* `--mlir-elide-resource-strings-if-larger=<N>`

  Elide oversized resource strings.

  **Use when**: compact logs.

**Structural summaries & graphs:**

* `--print-op-stats [--json]`

  Op counts per kind (optionally JSON).

  **Use when**: “what’s in this IR?”; gauge dialect mix.

* `--view-op-graph [--print-attrs --print-control-flow-edges --print-data-flow-edges --print-result-types --max-label-len=N]`

  Emit a Graphviz graph of the op graph.

  **Use when**: eyeballing control/data flow at a high level.

* `--dot-cfg-mssa=<file.dot>`

  Emit CFG (with MemorySSA flavor) as dot.

  **Use when**: CFG visualization, alias/memory flow context.

**Quick logging helpers:**

* `--print-ir [--label=TAG]`

  Dump the current IR to the debug stream with an optional label.

  **Use when**: ad-hoc checkpoints inside long pipelines.

**Registry awareness:**

* `--show-dialects` (again)

  **Use when**: confirming availability before parsing.

---

## Tracing / Debugging / Visualization

**Before/After dumps:**

* `--mlir-print-ir-before-all`

  Print IR before **every** pass.

  **Use when**: full trace; pair with filters below to cut noise.

* `--mlir-print-ir-after-all`

  Print IR after **every** pass.

  **Use when**: full trace; combine with `--mlir-print-ir-after-change`.

* `--mlir-print-ir-before=<pass>` / `--mlir-print-ir-after=<pass>`

  Restrict dumps to specific passes.

  **Use when**: focusing on suspicious transforms.

* `--mlir-print-ir-after-change`

  Only dump if IR changed.

  **Use when**: reduce spam in large pipelines.

* `--mlir-print-ir-after-failure`

  Dump when a pass fails.

  **Use when**: catching the failing state for repro.

* `--mlir-print-ir-tree-dir=<dir>`

  Write a file tree (per op scope, before/after) into a directory.

  **Use when**: post-mortem browsing; version control diffs.

**Debugger/Instrumentation hooks:**

* `--mlir-enable-debugger-hook`

  Enable debugger hooks around “MLIR actions.”

  **Use when**: stepping with external debuggers.

* `--mlir-debug-counter=<spec>` / `--mlir-print-debug-counter`

  Deterministically skip/limit rewrites using counters; print results.

  **Use when**: bisecting a single problematic pattern rewrite.

**Visualization (again):**

* `--view-op-graph ...`, `--dot-cfg-mssa=...`

  **Use when**: generate DOT, then render to PDF/SVG.

---

## Timing / Stats

* `--mlir-timing [--mlir-output-format={text,json}]`

  Wall-clock timing per pass.

  **Use when**: find slow passes; compare pipeline variants.

* `--mlir-timing-display={list,tree}`

  How timing is rendered (flat vs hierarchical).

  **Use when**: nested pipelines benefit from `tree`.

* `--mlir-pass-statistics`

  Per-pass counters (custom stats registered by passes).

  **Use when**: understanding what a pass actually did.

* `--mlir-pass-statistics-display={list,pipeline}`

  Control stats presentation.

  **Use when**: choose the view that matches your mental model.

---

## Reproducers / Crash Capture

* `--mlir-pass-pipeline-crash-reproducer=<path.mlir>`

  On crash/failure, dump a self-contained repro.

  **Use when**: filing bugs, pinning nondeterministic issues.

* `--mlir-pass-pipeline-local-reproducer`

  Emit the **smallest** possible reproducer (best-effort).

  **Use when**: you want minimal inputs for review.

* `--mlir-generate-reproducer=<path.mlir>`

  Always produce a reproducer (no crash required).

  **Use when**: capturing environment/IR for CI artifacts.

* `--run-reproducer`

  Execute the pipeline stored inside a reproducer file.

  **Use when**: consistent CI and bug reproduction.

---

## Logging / Profiling (tool-level)

* `--log-actions-to=<file|->`

  Log “action” execution (parsing, passes, etc.).

  **Use when**: high-level audit trail.

* `--log-mlir-actions-filter=<loc1,loc2,...>`

  Restrict logging to certain locations.

  **Use when**: reduce noise to relevant IR sections.

* `--profile-actions-to=<file|->`

  Profile action timings (distinct from pass stats).

  **Use when**: end-to-end tool profiling beyond passes.

* `--mlir-diagnostic-verbosity-level={errors,warnings,remarks}`

  Increase diagnostic detail (remarks==info).

  **Use when**: pass authorship, deep dives.

* `--mlir-disable-diagnostic-notes`

  Suppress side-notes.

  **Use when**: you only want the primary messages.

---

## Safety / Power-tools (use carefully)

* `--mlir-very-unsafe-disable-verifier-on-parsing`

  Skip verifier on parse.

  **Use when**: **never in CI**; only for forensic reads of broken IR.

* `--allow-unregistered-dialect` (also in Driver/IO)

  Let you *inspect* unknown IR without passes.

  **Use when**: triaging alien dumps without the full plugin stack.

---

## High-signal Recipes

**1) Clean, readable printout (no transforms):**

```bash
mlir-opt input.mlir \
  --mlir-pretty-debuginfo \
  --mlir-print-local-scope \
  --mlir-use-nameloc-as-prefix \
  --mlir-print-unique-ssa-ids \
  -o -
```

**2) “What’s in here?” quick census:**

```bash
mlir-opt input.mlir --print-op-stats --show-dialects -o /dev/null
```

**3) Trace a pipeline, but only when IR changes:**

```bash
mlir-opt input.mlir \
  --pass-pipeline='canonicalize,cse,inline' \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-change \
  --mlir-timing \
  --mlir-pass-statistics \
  -o out.mlir
```

**4) Full before/after into a directory tree:**

```bash
mlir-opt input.mlir \
  --pass-pipeline='...' \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-all \
  --mlir-print-ir-tree-dir=trace_dir \
  -o /dev/null
```

**5) Crash repro on failure (minimal if possible):**

```bash
mlir-opt input.mlir \
  --pass-pipeline='...' \
  --mlir-pass-pipeline-crash-reproducer=/tmp/repro.mlir \
  --mlir-pass-pipeline-local-reproducer \
  -o /dev/null
```

**6) Visualize structure (Graphviz):**

```bash
mlir-opt demo2-entering-mlir/demo2.mlir --view-op-graph='print-data-flow-edges' --dump-pass-pipeline
mlir-opt demo2-entering-mlir/demo2.mlir --view-op-graph='print-data-flow-edges print-attrs print-result-types print-control-flow-edges' -o /dev/null 2>graph.dot && dot -Tpdf graph.dot > graph.pdf
# Or:
mlir-opt input.mlir --dot-cfg-mssa=/tmp/cfg.dot -o /dev/null
```

**7) Deterministic rewrite bisection (pattern skip/limit):**

```bash
mlir-opt input.mlir \
  --pass-pipeline='canonicalize' \
  --mlir-debug-counter='rewrite-pattern=skip(100),count(1)' \
  --mlir-print-debug-counter \
  -o /dev/null
```

---

## Minimal “inspection only” pipeline (safe default)

Add this as a VSCode task or script:

```bash
mlir-opt "$1" \
  --mlir-pretty-debuginfo \
  --mlir-print-local-scope \
  --mlir-use-nameloc-as-prefix \
  --mlir-print-unique-ssa-ids \
  --print-op-stats \
  --verify-roundtrip \
  -o -
```

This never mutates IR, keeps prints legible, verifies round-trip, and summarizes ops.

---

# mlir-opt Tracing & Debugging Sheet (Workflows)

Short, task-oriented snippets for common debugging scenarios. Copy, run, and iterate.

---

## 1) Full IR trace with minimal spam

Goal: see every pass, but only dump *after changes*.

```bash
mlir-opt input.mlir \
  --pass-pipeline='canonicalize,cse,inline' \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-change \
  --mlir-timing --mlir-pass-statistics \
  -o out.mlir
```

**Tip**: Add `--mlir-print-ir-after-failure` to catch failing states.

---

## 2) Focus on one suspicious pass

```bash
mlir-opt input.mlir \
  --pass-pipeline='...' \
  --mlir-print-ir-before='canonicalize' \
  --mlir-print-ir-after='canonicalize' \
  -o /dev/null
```

**Why**: Cuts noise; isolates deltas attributed to that pass.

---

## 3) File-tree dumps for code review

```bash
mlir-opt input.mlir \
  --pass-pipeline='...' \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-all \
  --mlir-print-ir-tree-dir=trace_dir \
  -o /dev/null
```

Open `trace_dir/` and diff per-op, per-phase.

---

## 4) Deterministic rewrite bisection with debug counters

```bash
mlir-opt input.mlir \
  --pass-pipeline='canonicalize' \
  --mlir-debug-counter='rewrite-pattern=skip(0),count(1)' \
  --mlir-print-debug-counter \
  --mlir-print-ir-after-change \
  -o /dev/null
```

Iterate `skip(k)` to find the exact rewrite that flips behavior.

---

## 5) Minimal crash reproducer on failure

```bash
mlir-opt input.mlir \
  --pass-pipeline='...' \
  --mlir-pass-pipeline-crash-reproducer=/tmp/repro.mlir \
  --mlir-pass-pipeline-local-reproducer \
  -o /dev/null
```

Share `/tmp/repro.mlir` in bugs/PRs.

---

## 6) Visualize control/data flow quickly

```bash
mlir-opt input.mlir \
  --view-op-graph --print-data-flow-edges --print-result-types \
  -o /dev/null

mlir-opt input.mlir --dot-cfg-mssa=/tmp/cfg.dot -o /dev/null
# then: dot -Tpdf /tmp/cfg.dot > /tmp/cfg.pdf
```

Great for onboarding and diagramming.

---

## 7) Stable diffs across runs

```bash
mlir-opt input.mlir \
  --mlir-print-unique-ssa-ids \
  --mlir-use-nameloc-as-prefix \
  --mlir-elide-elementsattrs-if-larger=128 \
  -o -
```

Prevents noisy SSA renumbering and giant tensors from dominating diffs.

---

## 8) CI-friendly inspection (no transforms, verify)

```bash
mlir-opt input.mlir \
  --verify-roundtrip \
  --print-op-stats --mlir-output-format=text \
  -o /dev/null
```

Catches printer/parser regressions and reports op census.

---

## 9) Audit a nested pipeline definition

```bash
mlir-opt input.mlir \
  --pass-pipeline='...' \
  --dump-pass-pipeline \
  --print-pipeline-passes \
  -o /dev/null
```

Use both: pretty tree view + a serializable `-passes` string.

---

## 10) Nondeterminism hunts (disable threads)

```bash
mlir-opt input.mlir \
  --mlir-disable-threading \
  --pass-pipeline='...' \
  --mlir-timing \
  -o /dev/null
```

Eliminate scheduling races; combine with debug counters if needed.
