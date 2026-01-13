#!/usr/bin/env python3
"""
Verification script to confirm all examples use base model (not Instruct).

This script checks that all updated examples use:
✅ meta-llama/Llama-3.2-1B (base model)
❌ NOT meta-llama/Llama-3.2-1B-Instruct

The Instruct model has chat template tokens that conflict with grammar constraints.
"""

import re
from pathlib import Path


def check_file_uses_base_model(file_path):
    """Check if a file uses the correct base model."""
    content = file_path.read_text()

    # Check for Instruct variant (should NOT be present)
    has_instruct = "Llama-3.2-1B-Instruct" in content

    # Check for base model (should be present)
    has_base = re.search(r"meta-llama/Llama-3\.2-1B(?!-Instruct)", content)

    return {
        "file": file_path.name,
        "path": str(file_path),
        "uses_base_model": bool(has_base),
        "uses_instruct_model": has_instruct,
        "status": "✅" if (has_base and not has_instruct) else "❌",
    }


def main():
    print("=" * 80)
    print("Verifying Base Model Usage in Examples")
    print("=" * 80)
    print()
    print("✅ Expected: meta-llama/Llama-3.2-1B (base model)")
    print("❌ Wrong:    meta-llama/Llama-3.2-1B-Instruct")
    print()
    print("-" * 80)

    # Files to check
    files_to_check = [
        Path("examples/generate_json.py"),
        Path("examples/pipeline_json.py"),
        Path("examples/generate_smiles.py"),
        Path("examples/generate_pddl.py"),
        Path("examples/generate_geo_query.py"),
        Path("demo_json_generation.py"),
    ]

    results = []
    all_correct = True

    for file_path in files_to_check:
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            all_correct = False
            continue

        result = check_file_uses_base_model(file_path)
        results.append(result)

        status = result["status"]
        file_name = result["file"]

        if result["uses_base_model"] and not result["uses_instruct_model"]:
            print(f"{status} {file_name:40} - Using base model ✓")
        elif result["uses_instruct_model"]:
            print(f"{status} {file_name:40} - ERROR: Using Instruct model!")
            all_correct = False
        else:
            print(f"{status} {file_name:40} - ERROR: No Llama 3.2 1B found!")
            all_correct = False

    print("-" * 80)
    print()

    # Summary
    correct_count = sum(
        1 for r in results if r["uses_base_model"] and not r["uses_instruct_model"]
    )
    total_count = len(results)

    print("Summary:")
    print(f"  Total files checked: {total_count}")
    print(f"  Using base model correctly: {correct_count}")
    print(f"  Errors: {total_count - correct_count}")
    print()

    if all_correct:
        print("✅ All examples are using the base model correctly!")
        print()
        print("This prevents grammar constraint errors from chat template tokens.")
        return 0
    else:
        print("❌ Some examples need to be updated!")
        print()
        print("Fix by changing:")
        print("  FROM: meta-llama/Llama-3.2-1B-Instruct")
        print("  TO:   meta-llama/Llama-3.2-1B")
        return 1


if __name__ == "__main__":
    exit(main())
