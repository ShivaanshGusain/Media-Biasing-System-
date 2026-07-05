"""
run_pipeline.py — BiasLens Master Pipeline Orchestrator

Runs all 17 pipeline steps in the correct order.
Each step is a separate Python script in src/.

Usage:
  python run_pipeline.py              # Run full pipeline
  python run_pipeline.py --from 5     # Resume from step 5
  python run_pipeline.py --only 3     # Run only step 3
  python run_pipeline.py --skip-llm   # Skip LLM-dependent steps (3, 12, 13, 14)

For GitHub Actions: python run_pipeline.py
"""

import subprocess
import sys
import os
import time
from datetime import datetime

# Resolve project root (this script lives at the project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

# ---------------------------------------------------------------------------
# PIPELINE DEFINITION — order matters!
# ---------------------------------------------------------------------------
PIPELINE = [
    # (step_num, script_name, description, is_llm_dependent)
    (1,  "collection_bin.py",          "Scrape new articles from RSS/sitemaps",       False),
    (2,  "prep_articles.py",           "Clean & normalize articles",                  False),
    (3,  "embed_articles.py",          "Generate embeddings (Ollama)",             True),
    (4,  "cluster_events.py",          "Cluster articles into events",                False),
    (5,  "cleaning_clusters.py",       "Clean clusters (dedup, merge, wire removal)", False),
    (6,  "event_quality_pipeline.py",  "Article validation & quality scoring",        False),
    (7,  "canonical_event_audit.py",   "Audit canonical events for coherence",        False),
    (8,  "merge_main.py",             "Merge article + event master table",           False),
    (9,  "event_filter.py",           "Assign analysis tiers",                        False),
    (10, "split_corpora.py",          "Split into shared/exclusive corpora",           False),
    (11, "segment_passage.py",        "Segment articles into passages",               False),
    (12, "entity_coref.py",           "Extract entities + coreference (Ollama)",   True),
    (13, "score_passages.py",         "Score passage sentiment & framing (Ollama)", True),
    (14, "triples_explain.py",        "Generate SVO explanation triples (Ollama)", True),
    (15, "cross_outlet_bias.py",      "Cross-outlet bias analysis",                   False),
    (16, "coverage_bias.py",          "Coverage bias metrics",                        False),
    (17, "Visualization.py",          "Generate static charts",                       False),
]


def run_step(step_num, script, description, is_llm, skip_llm=False):
    """Run a single pipeline step and return (success, elapsed_seconds)."""
    if skip_llm and is_llm:
        print(f"  ⏩ SKIPPED (--skip-llm)")
        return True, 0

    script_path = os.path.join(SRC_DIR, script)
    if not os.path.exists(script_path):
        print(f"  ❌ SCRIPT NOT FOUND: {script_path}")
        return False, 0

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,       # Run from project root so Data/ paths work
            capture_output=False,
            text=True,
            timeout=None,           
            env={**os.environ, "PYTHONPATH": SRC_DIR},  # Allow imports from src/
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"  ❌ FAILED (exit code {result.returncode}, {elapsed:.0f}s)")
            if result.stderr:
                # Print last 20 lines of stderr
                err_lines = result.stderr.strip().split("\n")
                for line in err_lines[-20:]:
                    print(f"     {line}")
            return False, elapsed
        else:
            print(f"  ✅ OK ({elapsed:.0f}s)")
            # Print key output lines (lines with numbers or SUCCESS)
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if any(kw in line.upper() for kw in ["SUCCESS", "SAVED", "TOTAL", "COMPLETE", "GENERATED", "FOUND"]):
                        print(f"     {line.strip()}")
            return True, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  ⏰ TIMEOUT after {elapsed:.0f}s")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"  💥 ERROR: {e}")
        return False, elapsed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BiasLens Pipeline Orchestrator")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this step number (default: 1)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only this step number")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM-dependent steps (3, 12, 13, 14)")
    args = parser.parse_args()

    print("=" * 65)
    print("  BiasLens Pipeline — Master Orchestrator")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  Data:    {DATA_DIR}")
    print(f"  Python:  {sys.executable}")
    if args.skip_llm:
        print(f"  Mode:    SKIP LLM steps")
    if args.only:
        print(f"  Mode:    RUN ONLY step {args.only}")
    elif args.from_step > 1:
        print(f"  Mode:    RESUME from step {args.from_step}")
    print("=" * 65)

    # Ensure Data directory exists

    results = []
    total_start = time.time()
    had_failure = False

    for step_num, script, desc, is_llm in PIPELINE:
        # Filter steps
        if args.only and step_num != args.only:
            continue
        if step_num < args.from_step:
            continue

        llm_tag = " 🤖" if is_llm else ""
        print(f"\n[Step {step_num:2d}/17] {desc}{llm_tag}")
        print(f"  Script: src/{script}")

        success, elapsed = run_step(step_num, script, desc, is_llm, args.skip_llm)
        results.append((step_num, script, desc, success, elapsed))

        if not success and not (args.skip_llm and is_llm):
            had_failure = True
            # Determine if downstream steps depend on this one
            # Steps 1-5 are strictly sequential; 6-10 are sequential; 11-14 are sequential; 15-17 are independent
            critical_chains = [
                range(1, 6),     # scrape → prep → embed → cluster → clean
                range(6, 11),    # quality → audit → merge → filter → split
                range(11, 15),   # segment → entity → score → triples
            ]
            should_stop = False
            for chain in critical_chains:
                if step_num in chain:
                    # Check if the next step is also in this chain
                    next_step = step_num + 1
                    if next_step in chain:
                        print(f"\n  ⛔ Step {step_num} failed. Stopping chain (step {next_step} depends on it).")
                        should_stop = True
                    break

            if should_stop:
                # Skip remaining steps in this chain, but continue with independent ones
                continue

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 65)
    print("  PIPELINE SUMMARY")
    print("=" * 65)
    for step_num, script, desc, success, elapsed in results:
        status = "✅" if success else "❌"
        print(f"  {status} Step {step_num:2d}: {desc} ({elapsed:.0f}s)")
    print(f"\n  Total time: {total_elapsed/60:.1f} minutes")
    print(f"  Completed:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    passed = sum(1 for _, _, _, s, _ in results if s)
    failed = sum(1 for _, _, _, s, _ in results if not s)
    print(f"  Results:    {passed} passed, {failed} failed out of {len(results)} steps")
    print("=" * 65)

    sys.exit(1 if had_failure else 0)


if __name__ == "__main__":
    main()
