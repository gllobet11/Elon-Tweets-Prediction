import os
import pickle
import subprocess
import sys

# --- Constants ---
OPTIMIZER_SCRIPT_PATH = "src/strategy/financial_optimizer.py"
OUTPUT_ARTIFACT = "risk_params.pkl"
RUN1_OUTPUT = "risk_params_run1.pkl"
RUN2_OUTPUT = "risk_params_run2.pkl"


def run_optimizer():
    """Executes the financial optimizer script and returns its exit code."""
    try:
        python_executable = sys.executable
        result = subprocess.run(
            [python_executable, OPTIMIZER_SCRIPT_PATH],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",  # Ensure consistent encoding
        )
        print("Optimizer script ran successfully.")
        # Optional: print stdout only if needed for debugging, can be verbose
        # print(result.stdout)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running optimizer script: {e}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return e.returncode


def load_params(filepath):
    """Loads and returns the 'alpha' and 'kelly' parameters from a .pkl file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        data = pickle.load(f)
        return {"alpha": data.get("alpha"), "kelly": data.get("kelly")}


def main():
    """
    Runs the determinism test by comparing the core parameters ('alpha', 'kelly')
    from two consecutive runs of the optimizer script.
    """
    print("--- Starting Determinism Test for financial_optimizer.py ---")

    # --- First Run ---
    print("\n--- First Execution ---")
    if os.path.exists(OUTPUT_ARTIFACT):
        os.remove(OUTPUT_ARTIFACT)

    exit_code1 = run_optimizer()
    if exit_code1 != 0 or not os.path.exists(OUTPUT_ARTIFACT):
        print(
            "❌ Test failed: First run of the optimizer script did not succeed or produce the artifact.",
        )
        sys.exit(1)

    params1 = load_params(OUTPUT_ARTIFACT)
    os.remove(OUTPUT_ARTIFACT)  # Clean up immediately
    print(f"✅ First run complete. Parameters found: {params1}")

    # --- Second Run ---
    print("\n--- Second Execution ---")
    exit_code2 = run_optimizer()
    if exit_code2 != 0 or not os.path.exists(OUTPUT_ARTIFACT):
        print(
            "❌ Test failed: Second run of the optimizer script did not succeed or produce the artifact.",
        )
        sys.exit(1)

    params2 = load_params(OUTPUT_ARTIFACT)
    os.remove(OUTPUT_ARTIFACT)  # Clean up immediately
    print(f"✅ Second run complete. Parameters found: {params2}")

    # --- Comparison ---
    print("\n--- Comparing Core Parameters ---")

    # Use a tolerance for floating point comparisons
    is_alpha_same = (
        params1["alpha"] is not None
        and params2["alpha"] is not None
        and abs(params1["alpha"] - params2["alpha"]) < 1e-9
    )

    is_kelly_same = (
        params1["kelly"] is not None
        and params2["kelly"] is not None
        and abs(params1["kelly"] - params2["kelly"]) < 1e-9
    )

    if is_alpha_same and is_kelly_same:
        print(
            "\n✅ PASSED: The core parameters (alpha, kelly) are identical across both runs. The logic is deterministic.",
        )
    else:
        print("\n❌ FAILED: The core parameters are different.")
        print(f"    Run 1: alpha={params1.get('alpha')}, kelly={params1.get('kelly')}")
        print(f"    Run 2: alpha={params2.get('alpha')}, kelly={params2.get('kelly')}")
        sys.exit(1)

    print("\n--- Determinism Test Finished ---")


if __name__ == "__main__":
    main()
