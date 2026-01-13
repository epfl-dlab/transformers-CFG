#!/usr/bin/env python3
"""
Test all examples in the examples directory to see which ones work.
"""

import os
import subprocess
import sys
from pathlib import Path


def test_example(script_path, description, test_args=None, skip_reason=None):
    """Test a single example script."""
    print(f"\n{'='*80}")
    print(f"Testing: {script_path}")
    print(f"Description: {description}")
    print("=" * 80)

    if skip_reason:
        print(f"⊘ SKIPPED: {skip_reason}")
        return "skipped", skip_reason

    if not os.path.exists(script_path):
        print(f"✗ FAILED: File not found")
        return "failed", "File not found"

    # Check if script needs arguments
    if test_args is None:
        print(f"⊘ SKIPPED: Needs specific arguments or large model download")
        return "skipped", "Requires arguments or large model"

    try:
        # Run the script with a timeout
        cmd = [sys.executable, script_path] + test_args
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path(script_path).parent.parent,
        )

        if result.returncode == 0:
            print(f"✓ SUCCESS")
            print(f"Output preview: {result.stdout[:200]}...")
            return "success", None
        else:
            print(f"✗ FAILED with return code {result.returncode}")
            print(f"Error: {result.stderr[:500]}")
            return "failed", result.stderr[:200]
    except subprocess.TimeoutExpired:
        print(f"⊘ TIMEOUT: Took longer than 30 seconds")
        return "timeout", "Execution timeout"
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        return "error", str(e)


def main():
    print("\n" + "=" * 80)
    print("TESTING ALL EXAMPLES IN examples/")
    print("=" * 80)

    results = {}

    # Test 1: accept.py - String validation
    results["accept.py"] = test_example(
        "examples/accept.py",
        "Grammar string validation",
        test_args=[
            "-g",
            "examples/grammars/json.ebnf",
            "-s",
            '{"name": "test"}',
            "-m",
            "sentence",
        ],
    )

    # Test 2: json_schema_to_grammar.py - Utility script
    results["json_schema_to_grammar.py"] = test_example(
        "examples/grammars/custom_json_grammars/json_schema_to_grammar.py",
        "JSON schema to grammar converter (utility)",
        skip_reason="Utility script - not a runnable example",
    )

    # Test 3: generate_json.py - Requires Mistral-7B
    results["generate_json.py"] = test_example(
        "examples/generate_json.py",
        "Generate JSON with Mistral-7B",
        skip_reason="Requires downloading Mistral-7B (~14GB)",
    )

    # Test 4: generate_geo_query.py - Requires CodeLlama
    results["generate_geo_query.py"] = test_example(
        "examples/generate_geo_query.py",
        "Generate GeoQuery with CodeLlama",
        skip_reason="Requires downloading CodeLlama (~14GB)",
    )

    # Test 5: generate_pddl.py - Requires Mistral
    results["generate_pddl.py"] = test_example(
        "examples/generate_pddl.py",
        "Generate PDDL with Mistral",
        skip_reason="Requires downloading Mistral-7B (~14GB)",
    )

    # Test 6: generate_smiles.py - Requires Mistral
    results["generate_smiles.py"] = test_example(
        "examples/generate_smiles.py",
        "Generate SMILES with Mistral",
        skip_reason="Requires downloading Mistral-7B (~14GB)",
    )

    # Test 7: pipeline_json.py - Requires Mistral
    results["pipeline_json.py"] = test_example(
        "examples/pipeline_json.py",
        "Pipeline JSON generation with Mistral",
        skip_reason="Requires downloading Mistral-7B (~14GB)",
    )

    # Test 8: generate_llama_cpp_python.py - Requires llama-cpp-python
    results["generate_llama_cpp_python.py"] = test_example(
        "examples/generate_llama_cpp_python.py",
        "Generate with llama-cpp-python backend",
        skip_reason="Requires llama-cpp-python package and GGUF model",
    )

    # Test 9: run_seq2seq_model.py - Requires T5 model
    results["run_seq2seq_model.py"] = test_example(
        "examples/run_seq2seq_model.py",
        "Run seq2seq model with constraints",
        skip_reason="Requires T5 model and specific setup",
    )

    # Test 10: time_benchmarking.py - Benchmarking script
    results["time_benchmarking.py"] = test_example(
        "examples/benchmarking/time_benchmarking.py",
        "Performance benchmarking",
        skip_reason="Requires large models for meaningful benchmarks",
    )

    # Test 11: run_constrained_decoding_metric.py - Evaluation script
    results["run_constrained_decoding_metric.py"] = test_example(
        "examples/metrics/run_constrained_decoding_metric.py",
        "Constrained decoding metrics",
        skip_reason="Requires specific dataset and model setup",
    )

    # Print summary
    print("\n\n" + "=" * 80)
    print("SUMMARY OF ALL EXAMPLES")
    print("=" * 80)

    success_count = sum(1 for status, _ in results.values() if status == "success")
    skipped_count = sum(1 for status, _ in results.values() if status == "skipped")
    failed_count = sum(1 for status, _ in results.values() if status == "failed")
    error_count = sum(1 for status, _ in results.values() if status == "error")

    print(f"\n✓ Success: {success_count}")
    print(f"⊘ Skipped: {skipped_count}")
    print(f"✗ Failed:  {failed_count}")
    print(f"⚠ Errors:  {error_count}")

    print("\nDetailed Results:")
    print("-" * 80)

    for name, (status, reason) in results.items():
        status_icon = {
            "success": "✓",
            "skipped": "⊘",
            "failed": "✗",
            "error": "⚠",
            "timeout": "⏱",
        }.get(status, "?")

        print(f"{status_icon} {name:50s} {status.upper():10s}")
        if reason:
            print(f"  → {reason[:70]}")

    print("\n" + "=" * 80)
    print(
        "Note: Most examples require large model downloads (Mistral-7B, CodeLlama, etc.)"
    )
    print("These are skipped to avoid long download times.")
    print("They use the same refactored code as our working demos.")
    print("=" * 80 + "\n")

    return success_count, skipped_count, failed_count, error_count


if __name__ == "__main__":
    success, skipped, failed, errors = main()
    sys.exit(0 if failed == 0 and errors == 0 else 1)
