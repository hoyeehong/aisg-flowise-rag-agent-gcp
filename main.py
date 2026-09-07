"""Digital Economy Research & Report Agent — Main Entrypoint."""

import subprocess
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("eval", "evaluate", "evals"):
        eval_args = [sys.executable, "evals/run_evaluations.py"] + sys.argv[2:]
        sys.exit(subprocess.call(eval_args))

    print("=" * 65)
    print(" Digital Economy Multi-Agent RAG System & Evaluation Suite")
    print("=" * 65)
    print("\nQuickstart Commands:")
    print("  • Run automated evaluations:  uv run python evals/run_evaluations.py")
    print("  • Launch evaluation notebook: uv run jupyter lab evals/evaluations_notebook.ipynb")
    print("  • Deploy to Google Cloud Run: bash deploy/deploy_gcp.sh")
    print("=" * 65)


if __name__ == "__main__":
    main()
