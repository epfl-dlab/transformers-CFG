#!/usr/bin/env python3
"""
Verify that all examples have been updated to use Llama 3.2 1B.
"""

import re
from pathlib import Path


def check_file_for_model(filepath):
    """Check if a file uses Llama 3.2 1B."""
    with open(filepath, "r") as f:
        content = f.read()

    # Look for model_id definitions
    llama_32_pattern = r"meta-llama/Llama-3\.2-1B"
    old_mistral_pattern = r"mistralai/Mistral-7B"
    old_local_path = r"/dlabdata1/llm_hub"

    has_llama_32 = bool(re.search(llama_32_pattern, content))
    has_old_mistral = bool(re.search(old_mistral_pattern, content))
    has_old_path = bool(re.search(old_local_path, content))

    return has_llama_32, has_old_mistral, has_old_path, content


def main():
    print("=" * 80)
    print("VERIFYING EXAMPLES UPDATED TO LLAMA 3.2 1B")
    print("=" * 80)

    examples_to_check = [
        "examples/generate_json.py",
        "examples/pipeline_json.py",
        "examples/generate_smiles.py",
        "examples/generate_pddl.py",
        "examples/generate_geo_query.py",
        "demo_json_generation.py",
    ]

    results = []

    for example in examples_to_check:
        print(f"\n{'='*80}")
        print(f"Checking: {example}")
        print("=" * 80)

        if not Path(example).exists():
            print(f"❌ File not found: {example}")
            results.append((example, "missing"))
            continue

        has_llama, has_old_mistral, has_old_path, content = check_file_for_model(
            example
        )

        if has_llama and not has_old_mistral and not has_old_path:
            print(f"✅ UPDATED: Uses Llama 3.2 1B")

            # Show the relevant line
            for line in content.split("\n"):
                if "meta-llama/Llama-3.2" in line:
                    print(f"   Found: {line.strip()}")
                    break
            results.append((example, "updated"))

        elif has_old_mistral or has_old_path:
            print(f"❌ NOT UPDATED: Still uses old model")
            if has_old_mistral:
                print("   Found: mistralai/Mistral-7B")
            if has_old_path:
                print("   Found: /dlabdata1/llm_hub")
            results.append((example, "old"))

        else:
            print(f"⚠️  UNCLEAR: No model ID pattern found")
            results.append((example, "unclear"))

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    updated = sum(1 for _, status in results if status == "updated")
    old = sum(1 for _, status in results if status == "old")
    unclear = sum(1 for _, status in results if status == "unclear")
    missing = sum(1 for _, status in results if status == "missing")

    print(f"\n✅ Updated to Llama 3.2 1B: {updated}/{len(examples_to_check)}")
    print(f"❌ Still using old models:   {old}/{len(examples_to_check)}")
    print(f"⚠️  Unclear:                 {unclear}/{len(examples_to_check)}")
    print(f"❓ Missing files:            {missing}/{len(examples_to_check)}")

    print("\nDetailed Results:")
    print("-" * 80)
    for example, status in results:
        icon = {"updated": "✅", "old": "❌", "unclear": "⚠️", "missing": "❓"}.get(
            status, "?"
        )
        print(f"{icon} {example:50s} {status.upper()}")

    if updated == len(examples_to_check):
        print("\n" + "=" * 80)
        print("🎉 SUCCESS! All examples updated to Llama 3.2 1B")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print(f"⚠️  {len(examples_to_check) - updated} example(s) need updating")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
