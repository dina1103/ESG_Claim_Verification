import json
import re
import requests
from pathlib import Path

INPUT_FILE  = r"C:\Users\dina_\Desktop\esg_verification_draft\src\ingestion\ml_promise_dataset\llm_extraction_test_data.json"
OUTPUT_FILE = r"C:\Users\dina_\Desktop\esg_verification_draft\src\ingestion\ml_promise_dataset\llm_extraction_test_results.json"

# Ollama API endpoint and model
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

# valid label sets — used to normalize and validate LLM output
VALID_LABELS = {
    "promise_status":        {"Yes", "No"},
    "verification_timeline": {"Already", "Less than 2 years", "2 to 5 years", "More than 5 years", "N/A"},
    "evidence_status":       {"Yes", "No", "N/A"},
    "evidence_quality":      {"Clear", "Not Clear", "Misleading", "N/A"},
}


# the prompt template — paragraph gets substituted in
PROMPT_TEMPLATE = '''You are an expert in extracting ESG-related promises and their corresponding evidence from corporate ESG reports. Follow the instructions below carefully.

TASK:
Given a paragraph from an ESG report, decide:
1. promise_status — does the paragraph contain a promise?
2. verification_timeline — when can the promise be verified?
3. evidence_status — does the paragraph contain actionable evidence supporting the promise?
4. evidence_quality — how clear is the evidence in relation to the promise?

DEFINITIONS:

A promise is a statement related to ESG criteria that expresses a company principle (e.g., diversity and inclusion), a commitment (e.g., reducing emissions), or a strategy (e.g., partnership development, protocol).

Evidence is information that supports the promise being kept, including concrete examples, company measures, numbers, or data tables.

LABEL OPTIONS (use exactly these strings):

promise_status: "Yes" or "No"

verification_timeline:
- "Already" — results are already verifiable, OR the promise has already been implemented
- "Less than 2 years" — results expected within 2 years
- "2 to 5 years" — results expected within 2 to 5 years
- "More than 5 years" — results expected after more than 5 years
- "N/A" — use when promise_status is "No"

evidence_status: "Yes" or "No". Use "N/A" only when promise_status is "No".

evidence_quality:
- "Clear" — sufficient information; intelligible and logical
- "Not Clear" — information is missing or only partially intelligible
- "Misleading" — evidence has no clear connection with the promise
- "N/A" — use when promise_status is "No" or when evidence_status is "No"

CRITICAL RULES:
- If promise_status is "No", then verification_timeline, evidence_status, and evidence_quality must all be "N/A".
- If evidence_status is "No", then evidence_quality must be "N/A".
- Output ONLY a valid JSON object with these four fields. No explanation, no commentary.

EXAMPLES:

Paragraph: "Land Use• Recognizing existing or potential future recreational, traditional, cultural and other land uses, our environmental team consults with land users to identify potential conflicts and to mitigate or eliminate impacts."
Answer: {{"promise_status": "No", "verification_timeline": "N/A", "evidence_status": "No", "evidence_quality": "N/A"}}

Paragraph: "Managing risk We believe in habitually stopping and engaging 100 per cent of the conscious mind to identify the risks before continuing the task."
Answer: {{"promise_status": "Yes", "verification_timeline": "Already", "evidence_status": "No", "evidence_quality": "N/A"}}

Paragraph: "To date, in line with our Science Based Targets, we have reduced our scope 1 and 2 emissions by 84% (compared to a 2017 baseline) by improving the efficiency of our retail and corporate offices and by transitioning to renewable energy."
Answer: {{"promise_status": "Yes", "verification_timeline": "Already", "evidence_status": "Yes", "evidence_quality": "Clear"}}

Paragraph: "Combating financial crime 2022 highlights We successfully enhanced transaction monitoring across our payment platforms, with continued investment in our anti-money laundering systems."
Answer: {{"promise_status": "Yes", "verification_timeline": "Already", "evidence_status": "Yes", "evidence_quality": "Not Clear"}}

Now annotate this paragraph:

Paragraph: "{paragraph}"
Answer:'''


def call_llm(paragraph):
    # send the prompt to Ollama and return the raw response text
    prompt = PROMPT_TEMPLATE.format(paragraph=paragraph.replace('"', "'"))
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,    # deterministic for reproducibility
                "num_predict": 200,    # JSON answer is short, cap output
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def parse_llm_response(raw_text):
    # try to extract a valid JSON object from the LLM response
    # returns (parsed_dict, error_message) — error_message is None if successful

    # strip markdown code fences if present
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # try direct parse
    try:
        parsed = json.loads(text)
        return parsed, None
    except json.JSONDecodeError:
        pass

    # fallback: extract first {...} block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed, None
        except json.JSONDecodeError as e:
            return None, f"JSON decode error: {e}"

    return None, "No JSON object found in response"


def normalize_labels(parsed):
    # normalize each label: strip whitespace, fix common case issues
    # returns (normalized_dict, list_of_warnings)
    if not parsed:
        return None, ["empty parse result"]

    warnings = []
    normalized = {}

    for field, valid_set in VALID_LABELS.items():
        raw = parsed.get(field, "")
        if not isinstance(raw, str):
            warnings.append(f"{field}: non-string value {raw!r}")
            normalized[field] = None
            continue

        # strip whitespace and trailing punctuation
        cleaned = raw.strip().rstrip(".,;")

        # case-correct match against valid set
        match = None
        for valid in valid_set:
            if cleaned.lower() == valid.lower():
                match = valid
                break

        if match:
            normalized[field] = match
        else:
            warnings.append(f"{field}: invalid value {raw!r}")
            normalized[field] = None

    return normalized, warnings


def compare(predicted, gold):
    # returns dict of per-field correct/incorrect
    return {
        field: (predicted.get(field) == gold.get(field))
        for field in VALID_LABELS
    }


def main():
    # load sanity-check input
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} sanity-check records\n")

    # check Ollama is reachable
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: cannot reach Ollama at localhost:11434 — {e}")
        return

    results = []
    parse_failures = 0
    correct_counts = {field: 0 for field in VALID_LABELS}

    for rec in records:
        rid = rec["id"]
        paragraph = rec["data"]
        gold = rec["gold_labels"]
        category = rec["category"]

        print(f"[{rid:2d}] {category} (idx={rec['source_index']})... ", end="", flush=True)

        # call the LLM
        try:
            raw = call_llm(paragraph)
        except Exception as e:
            print(f"LLM ERROR: {e}")
            results.append({
                "id": rid, "category": category, "source_index": rec["source_index"],
                "gold_labels": gold, "raw_response": None, "predicted": None,
                "warnings": [f"LLM call failed: {e}"], "comparison": None,
            })
            continue

        # parse the response
        parsed, parse_error = parse_llm_response(raw)

        if parse_error:
            print(f"PARSE FAIL: {parse_error}")
            parse_failures += 1
            results.append({
                "id": rid, "category": category, "source_index": rec["source_index"],
                "gold_labels": gold, "raw_response": raw, "predicted": None,
                "warnings": [parse_error], "comparison": None,
            })
            continue

        # normalize labels
        predicted, warnings = normalize_labels(parsed)

        # compare against gold
        comp = compare(predicted, gold)
        for field, ok in comp.items():
            if ok:
                correct_counts[field] += 1

        # quick result line
        all_correct = all(comp.values())
        marker = "✓" if all_correct else "✗"
        print(f"{marker}  PI:{predicted.get('promise_status')} TV:{predicted.get('verification_timeline')} AE:{predicted.get('evidence_status')} CPEP:{predicted.get('evidence_quality')}")

        results.append({
            "id": rid, "category": category, "source_index": rec["source_index"],
            "gold_labels": gold, "raw_response": raw, "predicted": predicted,
            "warnings": warnings, "comparison": comp,
        })

    # save results
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # summary table
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    print(f"{'#':<3} {'category':<25} {'task':<6} {'predicted':<22} {'gold':<22}")
    print("-" * 80)
    for r in results:
        if r["predicted"] is None:
            print(f"{r['id']:<3} {r['category']:<25} (parse failure)")
            continue
        for field in VALID_LABELS:
            short = {"promise_status": "PI", "verification_timeline": "TV",
                     "evidence_status": "AE", "evidence_quality": "CPEP"}[field]
            ok = "✓" if r["comparison"][field] else "✗"
            print(f"{r['id']:<3} {r['category']:<25} {short:<6} {str(r['predicted'][field])+' '+ok:<22} {r['gold_labels'][field]:<22}")
        print()

    # summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    n = len(records)
    print(f"JSON parse success: {n - parse_failures}/{n}")
    print(f"\nPer-task accuracy:")
    for field, count in correct_counts.items():
        print(f"  {field:<24} {count}/{n}")
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()