import json
import re
import time
import random
import requests
import pandas as pd
from pathlib import Path


INPUT_PARQUET = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\llm_paragraphs"
PROMPT_FILE   = r"C:\Users\dina_\Desktop\esg_verification_draft\src\ingestion\llm_extraction_prompt.txt"
OUTPUT_FILE   = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\llm_extraction_test_results"


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
SEED = 42


def pick_stratified_paragraphs(df, seed=42):
    # for each company, pick 1 shorter (25-50th percentile) and 1 longer (75-90th percentile) paragraph
    random.seed(seed)
    picked = []

    for company in sorted(df["company_name"].unique()):
        sub = df[df["company_name"] == company].copy()
        sub["text_length"] = sub["text"].str.len()

        # determine length bins for this company
        p25 = sub["text_length"].quantile(0.25)
        p50 = sub["text_length"].quantile(0.50)
        p75 = sub["text_length"].quantile(0.75)
        p90 = sub["text_length"].quantile(0.90)

        # shorter pool: between p25 and p50
        shorter_pool = sub[(sub["text_length"] >= p25) & (sub["text_length"] <= p50)]
        # longer pool: between p75 and p90
        longer_pool = sub[(sub["text_length"] >= p75) & (sub["text_length"] <= p90)]

        # randomly sample one from each pool
        if len(shorter_pool) > 0:
            picked.append(shorter_pool.sample(n=1, random_state=seed).iloc[0])
        if len(longer_pool) > 0:
            picked.append(longer_pool.sample(n=1, random_state=seed).iloc[0])

    return pd.DataFrame(picked).reset_index(drop=True)


def call_llm(prompt_template, paragraph_text):
    # substitute paragraph into the prompt
    full_prompt = prompt_template.replace("{paragraph}", paragraph_text.replace('"', "'"))

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 2048,    # larger output for multi-claim
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"]


def parse_llm_response(raw_text):
    # try to extract a valid JSON object from the LLM response
    text = raw_text.strip()

    # strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # try direct parse
    try:
        parsed = json.loads(text)
        return parsed, None
    except json.JSONDecodeError:
        pass

    # fallback: try to extract first {...} block (greedy match for nested braces)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed, None
        except json.JSONDecodeError as e:
            return None, f"JSON decode error: {e}"

    return None, "No JSON object found in response"


def main():
    # load prompt template
    print(f"Loading prompt from {PROMPT_FILE}...")
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    print(f"  prompt: {len(prompt_template)} chars")

    # load paragraphs
    print(f"\nLoading paragraphs from {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  loaded {len(df):,} paragraphs")

    # check Ollama is reachable
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"\nERROR: cannot reach Ollama at localhost:11434 — {e}")
        return

    # pick 10 paragraphs
    print(f"\nPicking 10 paragraphs (2 per company, stratified by length, seed={SEED})...")
    sample = pick_stratified_paragraphs(df, seed=SEED)
    print(f"  picked {len(sample)} paragraphs")
    for _, row in sample.iterrows():
        print(f"    [{row['company_name']:30s} {row['year']}] {len(row['text']):>6} chars | block_id: {row['block_id'][:70]}")

    # run LLM on each
    print(f"\nRunning LLM on each paragraph (this will take 5-15 minutes)...\n")
    results = []
    parse_failures = 0

    for i, row in sample.iterrows():
        print(f"[{i+1:2d}/{len(sample)}] {row['company_name']} {row['year']} ({len(row['text'])} chars)... ", end="", flush=True)

        t0 = time.time()
        try:
            raw = call_llm(prompt_template, row["text"])
            elapsed = time.time() - t0
        except Exception as e:
            elapsed = time.time() - t0
            print(f"LLM ERROR ({elapsed:.0f}s): {e}")
            results.append({
                "block_id": row["block_id"],
                "company_name": row["company_name"],
                "year": int(row["year"]),
                "report_type": row["report_type"],
                "page_number_min": int(row["page_number_min"]),
                "page_number_max": int(row["page_number_max"]),
                "text_length": len(row["text"]),
                "n_sentences": int(row["n_sentences"]),
                "n_esg_sentences": int(row["n_esg_sentences"]),
                "input_text": row["text"],
                "raw_response": None,
                "parsed_claims": None,
                "parse_error": f"LLM call failed: {e}",
                "elapsed_seconds": elapsed,
            })
            continue

        parsed, parse_error = parse_llm_response(raw)

        if parse_error:
            print(f"PARSE FAIL ({elapsed:.0f}s): {parse_error}")
            parse_failures += 1
            parsed_claims = None
        else:
            claims = parsed.get("claims", []) if isinstance(parsed, dict) else []
            parsed_claims = claims
            print(f"OK ({elapsed:.0f}s) — {len(claims)} claim(s) extracted")

        results.append({
            "block_id": row["block_id"],
            "company_name": row["company_name"],
            "year": int(row["year"]),
            "report_type": row["report_type"],
            "page_number_min": int(row["page_number_min"]),
            "page_number_max": int(row["page_number_max"]),
            "text_length": len(row["text"]),
            "n_sentences": int(row["n_sentences"]),
            "n_esg_sentences": int(row["n_esg_sentences"]),
            "input_text": row["text"],
            "raw_response": raw,
            "parsed_claims": parsed_claims,
            "parse_error": parse_error,
            "elapsed_seconds": elapsed,
        })

    # save results
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    n = len(results)
    parsed_ok = sum(1 for r in results if r["parsed_claims"] is not None)
    total_claims = sum(len(r["parsed_claims"]) if r["parsed_claims"] else 0 for r in results)
    total_time = sum(r["elapsed_seconds"] for r in results)
    print(f"Successful parses:  {parsed_ok}/{n}")
    print(f"Parse failures:     {parse_failures}/{n}")
    print(f"Total claims:       {total_claims}")
    print(f"Avg claims/block:   {total_claims/parsed_ok:.1f}" if parsed_ok else "  (no successful parses)")
    print(f"Total time:         {total_time:.0f}s")
    print(f"Avg time/paragraph: {total_time/n:.1f}s")
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()