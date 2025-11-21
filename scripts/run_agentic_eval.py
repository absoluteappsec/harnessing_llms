#!/usr/bin/env python3
"""
Simple test runner for the agentic evaluation system.
Usage: python run_agentic_eval.py
"""

import asyncio
import sys
import os

# Add the current directory to the path so we can import the evaluation module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from importlib import import_module

    eval_module = import_module("16-agentic-eval")
    run_model_comparison = eval_module.run_model_comparison
    AVAILABLE_MODELS = eval_module.AVAILABLE_MODELS
except ImportError as e:
    print(f"❌ Failed to import evaluation module: {e}")
    print("Make sure 16-agentic-eval.py is in the same directory")
    sys.exit(1)


async def run_quick_test():
    """Run a quick test with default models"""
    print("🧪 Running Quick Model Comparison Test")
    print("-" * 40)
    print("Testing: Claude Haiku vs Claude Sonnet (first test case only)")

    try:
        # Import the battle system and run one test
        from importlib import import_module

        eval_module = import_module("16-agentic-eval")

        model1_config = eval_module.AVAILABLE_MODELS["claude_haiku"]
        model2_config = eval_module.AVAILABLE_MODELS["claude_sonnet"]
        battle_system = eval_module.AgenticBattleSystem(model1_config, model2_config)

        test_cases = battle_system.get_test_cases()
        if test_cases:
            battle_result = await battle_system.run_battle(test_cases[0])
            print(f"\n✅ Quick test completed!")
            print(f"Winner: {battle_result.winner}")
            return True
        else:
            print("❌ No test cases found")
            return False
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


async def run_full_tournament():
    """Run the full tournament with model selection"""
    print("🏟️ Running Full Model Comparison Tournament")
    print("-" * 50)

    # Show available models
    print("\nAvailable models:")
    for key, config in AVAILABLE_MODELS.items():
        print(f"  {key}: {config.name}")

    # Get user selection
    print("\nSelect models to compare:")
    model1_key = (
        input(f"Model 1 ({'/'.join(AVAILABLE_MODELS.keys())}): ").strip().lower()
    )
    model2_key = (
        input(f"Model 2 ({'/'.join(AVAILABLE_MODELS.keys())}): ").strip().lower()
    )

    # Validate selections
    if model1_key not in AVAILABLE_MODELS:
        print(f"❌ Invalid model 1. Using default: claude_haiku")
        model1_key = "claude_haiku"

    if model2_key not in AVAILABLE_MODELS:
        print(f"❌ Invalid model 2. Using default: claude_sonnet")
        model2_key = "claude_sonnet"

    if model1_key == model2_key:
        print(f"❌ Cannot compare model to itself. Using claude_haiku vs claude_sonnet")
        model1_key, model2_key = "claude_haiku", "claude_sonnet"

    # Run comparison
    stats = await run_model_comparison(model1_key, model2_key)
    return stats


def main():
    """Main function with user choice"""
    print("🤖 LLM Security Analysis Comparison")
    print("=" * 40)
    print("Choose evaluation mode:")
    print("1. Quick test (single battle, default models)")
    print("2. Full tournament (all battles, choose models)")
    print("3. Default comparison (Claude Haiku vs Sonnet)")
    print("4. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == "1":
                print("\nStarting quick test...")
                success = asyncio.run(run_quick_test())
                if success:
                    print("\n🎉 Quick test completed successfully!")
                break

            elif choice == "2":
                print("\nStarting full tournament...")
                stats = asyncio.run(run_full_tournament())
                print("\n🎉 Full tournament completed successfully!")
                break

            elif choice == "3":
                print("\nRunning default comparison...")
                stats = asyncio.run(
                    run_model_comparison("claude_haiku", "claude_sonnet")
                )
                print("\n🎉 Default comparison completed successfully!")
                break

            elif choice == "4":
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            break


if __name__ == "__main__":
    main()
