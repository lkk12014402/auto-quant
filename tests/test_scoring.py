from aqa.eval.scoring import (
    ScorerInput,
    answer_line,
    exact_match,
    extract_last_number,
    fuzzy_match,
    multichoice_regex,
    normalize_text,
    numeric_match,
)


def test_normalize_text():
    assert normalize_text("The, Quick  Brown!") == "quick brown"


def test_exact_match():
    assert exact_match(ScorerInput("The cat.", "a cat"))["score"] == 1.0
    assert exact_match(ScorerInput("dog", "cat"))["score"] == 0.0


def test_extract_last_number():
    assert extract_last_number("so the answer is 42.") == "42"
    assert extract_last_number("$1,234.50 total") == "1234.50"
    assert extract_last_number("no digits here") is None


def test_numeric_match():
    assert numeric_match(ScorerInput("The answer is 18", "#### 18"))["score"] == 1.0
    assert numeric_match(ScorerInput("result: 7", "8"))["score"] == 0.0
    # numeric equality ignores formatting
    assert numeric_match(ScorerInput("1,000", "1000"))["score"] == 1.0


def test_answer_line():
    assert answer_line(ScorerInput("blah\n#### 72", "72"))["score"] == 1.0
    assert answer_line(ScorerInput("The answer is: cat", "cat"))["score"] == 1.0


def test_multichoice_regex():
    assert multichoice_regex(ScorerInput("The answer is B.", "B"))["score"] == 1.0
    r = multichoice_regex(ScorerInput("I pick A", "Paris", choices=["Paris", "London"]))
    assert r["target_letter"] == "A" and r["score"] == 1.0
    assert multichoice_regex(ScorerInput("C", "B"))["score"] == 0.0


def test_fuzzy_match():
    full = fuzzy_match(ScorerInput("the quick brown fox", "quick brown fox"))
    assert full["f1"] > 0.8
    assert fuzzy_match(ScorerInput("apple", "orange"))["f1"] == 0.0
