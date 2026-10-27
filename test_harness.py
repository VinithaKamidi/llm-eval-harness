import pandas as pd

from harness import VARIANTS, MockLLM, check, run, summarize


def test_passes_when_required_word_present():
    assert check("The capital is Paris.", {"must_include": ["Paris"]})["passed"]


def test_matching_ignores_case():
    assert check("PARIS", {"must_include": ["paris"]})["passed"]


def test_fails_when_required_word_missing():
    result = check("london", {"must_include": ["Paris"]})
    assert not result["has_required"] and not result["passed"]


def test_fails_on_forbidden_word():
    assert not check("Maybe Lyon", {"must_include": [], "must_not_include": ["Lyon"]})["no_forbidden"]


def test_fails_when_too_long():
    assert not check("this answer is far too long", {"must_include": [], "max_words": 3})["within_length"]


def test_run_makes_one_row_per_model_case_and_variant():
    cases = [
        {"id": "a", "question": "What is the capital of France?", "must_include": ["Paris"]},
        {"id": "b", "question": "What is 12 multiplied by 5?", "must_include": ["60"]},
    ]
    df = run(cases, VARIANTS, {"mock": MockLLM(), "mock2": MockLLM()})
    assert len(df) == 2 * 2 * len(VARIANTS)
    assert df["passed"].all()


def test_summarize_gives_pass_rate_percentages():
    df = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "variant": ["plain", "plain", "expert", "expert"],
            "passed": [True, False, True, True],
        }
    )
    table = summarize(df)
    assert table.loc["m", "plain"] == 50
    assert table.loc["m", "expert"] == 100
