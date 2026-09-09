"""Digital Economy Research & Report Agent — Main Entrypoint."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
V1_DIR = REPO_ROOT / "low-code-rapid-prototype-v1"
V1_EVAL_RUNNER = V1_DIR / "evals" / "run_evaluations.py"


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("eval", "evaluate", "evals"):
        eval_args = [sys.executable, str(V1_EVAL_RUNNER)] + sys.argv[2:]
        sys.exit(subprocess.call(eval_args))

    print("=" * 65)
    print(" Digital Economy Multi-Agent RAG System & Evaluation Suite")
    print("=" * 65)
    print("\nv1 — Low-code rapid prototype (Flowise Agentflow v2):")
    print("  • Run automated evaluations:  uv run python main.py eval")
    print("  • Launch evaluation notebook: uv run jupyter lab \\")
    print("      low-code-rapid-prototype-v1/evals/evaluations_notebook.ipynb")
    print("  • Deploy to Google Cloud Run: bash \\")
    print("      low-code-rapid-prototype-v1/deploy/deploy_gcp.sh")
    print("=" * 65)


if __name__ == "__main__":
    main()
