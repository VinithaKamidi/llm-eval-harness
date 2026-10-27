"""LLM Prompt & Model Evaluation Harness.

Asks several LLMs the same test questions using different prompt wordings,
checks every answer against simple rules, and reports pass rates.

Usage:
    python harness.py                       # compares flan-t5-base and flan-t5-large
    python harness.py --models google/flan-t5-base
    python harness.py --mock                # instant demo with a fake model (not a real result)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent

# Three ways of asking the same question. {q} is replaced by each test question.
VARIANTS = {
    "plain": "{q}",
    "concise": "Answer in one word or short phrase. {q}",
    "expert": "You are a precise assistant. Give only the correct answer. {q}",
}

DEFAULT_MODELS = ["google/flan-t5-base", "google/flan-t5-large"]


# ---------- Checking ----------
def check(answer: str, case: dict) -> dict:
    """Check one answer against one test case. Every rule is optional except must_include."""
    text = answer.lower()
    result = {
        "has_required": all(w.lower() in text for w in case.get("must_include", [])),
        "no_forbidden": not any(w.lower() in text for w in case.get("must_not_include", [])),
        "within_length": len(answer.split()) <= case.get("max_words", 10**6),
    }
    result["passed"] = all(result.values())
    return result


# ---------- Models ----------
class MockLLM:
    """A fake model for trying the pipeline instantly. Its results are NOT real."""

    ANSWERS = {"capital of france": "Paris", "largest": "Jupiter", "12 multiplied by 5": "60"}

    def __call__(self, prompt: str) -> str:
        low = prompt.lower()
        for key, answer in self.ANSWERS.items():
            if key in low:
                return answer
        return "I do not know"


def load_local_model(name: str):
    """Load a Hugging Face seq2seq model and return a function: prompt -> answer text."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)

    def ask(prompt: str) -> str:
        ids = tokenizer(prompt, return_tensors="pt")
        out = model.generate(**ids, max_new_tokens=40)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    return ask


# ---------- Running ----------
def run(cases: list, variants: dict, models: dict) -> pd.DataFrame:
    """models maps a model name to a function (prompt -> answer)."""
    rows = []
    for model_name, ask in models.items():
        for case in cases:
            for variant, template in variants.items():
                answer = ask(template.format(q=case["question"]))
                rows.append(
                    {
                        "model": model_name,
                        "case": case["id"],
                        "variant": variant,
                        "answer": answer,
                        **check(answer, case),
                    }
                )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Pass rate (%) for every model and prompt variant."""
    table = df.groupby(["model", "variant"])["passed"].mean().mul(100).round(0).astype(int)
    return table.unstack("variant")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM answers across prompts and models.")
    parser.add_argument("--cases", default=str(HERE / "test_cases.json"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--mock", action="store_true", help="use a fake model (demo only)")
    parser.add_argument("--out", default=str(HERE / "results.csv"))
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text())

    if args.mock:
        models = {"mock": MockLLM()}
    else:
        models = {}
        for name in args.models:
            print(f"Loading {name} ...")
            models[name.split("/")[-1]] = load_local_model(name)

    print(f"\n{len(cases)} test cases x {len(VARIANTS)} prompt variants x {len(models)} model(s)\n")
    df = run(cases, VARIANTS, models)
    df.to_csv(args.out, index=False)

    print("=== Pass rate (%) by model and prompt variant ===")
    print(summarize(df).to_string())

    failures = df[~df["passed"]]
    print(f"\n=== {len(failures)} failures out of {len(df)} answers (first 15) ===")
    print(failures[["model", "case", "variant", "answer"]].head(15).to_string(index=False))
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
