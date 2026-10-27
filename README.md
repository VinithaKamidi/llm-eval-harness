# LLM Prompt & Model Evaluation Harness

A small QA framework for language models. It asks each model the same test questions in several prompt wordings, checks every answer against a rule, and reports pass rates, so you can see which model and which prompt perform better, and where they fail.

It applies audit-style quality checking (rules, pass/fail, failure review) to AI output.

## How it works
1. `test_cases.json` holds questions, each with the word(s) the answer must contain (optionally words it must not contain and a maximum length).
2. Three prompt variants ask the same question differently: `plain`, `concise` and `expert`.
3. Each model answers every question with every variant; `check()` marks each answer pass or fail.
4. The script prints pass rates per model and variant, lists failures, and saves everything to `results.csv`.

Models tested: `google/flan-t5-base` (about 250M parameters) and `google/flan-t5-large` (about 780M parameters), both open models run locally with Hugging Face Transformers. No API key is needed.

## Run it (Google Colab recommended)
```bash
!git clone https://github.com/VinithaKamidi/llm-eval-harness.git
%cd llm-eval-harness
!pip install -q transformers sentencepiece pandas pytest
!python -m pytest -q          # unit tests
!python harness.py            # real evaluation (downloads about 4 GB of models the first time)
!python harness.py --mock     # instant demo with a fake model (not a real result)
```

## Results
Measured in Google Colab (CPU). 10 questions x 3 prompt variants = 30 answers per model.

| Model | plain | concise | expert |
|-------|-------|---------|--------|
| flan-t5-base  | 20% | 20% | 20% |
| flan-t5-large | 50% | 50% | 40% |

## Findings
- The larger model passed clearly more often than the smaller one (about 47% vs 20% on average).
- Prompt wording made little difference in this test. With only 10 questions, one question equals 10 percentage points, so the 50% vs 40% gap for flan-t5-large is a single question and is not meaningful.
- The failures are genuine model errors, not checker mistakes. For example, flan-t5-base answered "london" for the capital of France, "chennai" for the capital of India and "venus" for the largest planet. For the boiling-point question it gave "212", which is the Fahrenheit value, so it correctly failed the Celsius check.

## Limitations
- Small test set (10 questions); results are indicative, not statistically reliable.
- Keyword checking can fail a correct answer written differently (for example "one hundred" instead of "100").
- The test questions are easy general-knowledge facts; real use needs domain-specific tests.

## Ideas to extend
More test cases (50+), an LLM-as-judge scorer, semantic-similarity scoring, comparison with API-hosted models, and a results chart.

## Files
- `harness.py`: runner, checker and summary
- `test_cases.json`: test questions and rules
- `test_harness.py`: unit tests (`pytest -q`)
