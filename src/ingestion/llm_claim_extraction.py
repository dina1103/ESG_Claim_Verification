import json
import re
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

INPUT_PARQUET = r"C:\Users\dell\ESG_Claim_Verification\data\processed\llm_paragraphs"
PROMPT_FILE   = r"C:\Users\dell\ESG_Claim_Verification\src\ingestion\llm_extraction_prompt.txt"
OUTPUT_JSONL  = r"C:\Users\dell\ESG_Claim_Verification\data\processed\llm_claim_extraction_result.jsonl"
SUMMARY_FILE  = r"C:\Users\dell\ESG_Claim_Verification\data\processed\llm_claim_extractionsummary.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

# print progress every N paragraphs
PROGRESS_EVERY = 1

# smoke test: limit run to N paragraphs (set to None for full corpus)
MAX_PARAGRAPHS = 5

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
                "num_predict": 2048,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"]


def parse_llm_response(raw_text):
    # try to extract a valid JSON object from the LLM response
    # repair_status: None on clean parse, "repaired_from_truncation" if recovered,
    # or an error string if unrecoverable
    text = raw_text.strip()

    # strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # try direct parse
    try:
        parsed = json.loads(text)
        return parsed, None
    except json.JSONDecodeError as e:
        original_error = str(e)

    # fallback 1: try to extract first {...} block (greedy match for nested braces)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed, None
        except json.JSONDecodeError:
            pass

    # fallback 2: repair truncated output by finding last complete claim and closing json
    last_complete = text.rfind("},")
    if last_complete != -1:
        repaired = text[:last_complete + 1] + "]}"
        try:
            parsed = json.loads(repaired)
            return parsed, "repaired_from_truncation"
        except json.JSONDecodeError:
            pass

    return None, f"JSON decode error: {original_error}"


def load_processed_ids(output_path):
    # read existing jsonl checkpoint, return set of already-processed block_ids
    processed = set()
    if not Path(output_path).exists():
        return processed

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                processed.add(rec["block_id"])
            except (json.JSONDecodeError, KeyError):
                # skip malformed lines
                continue
    return processed


def format_eta(seconds):
    # format seconds as a human-readable duration
    return str(timedelta(seconds=int(seconds)))


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

    # check for existing checkpoint and resume
    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    processed_ids = load_processed_ids(OUTPUT_JSONL)
    if processed_ids:
        print(f"\nFound existing checkpoint: {len(processed_ids):,} paragraphs already processed")
        print(f"  resuming from where we left off")
    else:
        print(f"\nNo checkpoint found — starting fresh")

    # filter to paragraphs not yet processed
    remaining = df[~df["block_id"].isin(processed_ids)].reset_index(drop=True)

    # smoke test limiter
    if MAX_PARAGRAPHS is not None:
        remaining = remaining.head(MAX_PARAGRAPHS)
        print(f"  SMOKE TEST: limited to {MAX_PARAGRAPHS} paragraphs")

    print(f"  remaining to process: {len(remaining):,} paragraphs")

    if len(remaining) == 0:
        print("\nAll paragraphs already processed. Nothing to do.")
        return

    # run LLM on each remaining paragraph
    print(f"\nRunning LLM on each paragraph (appending to {OUTPUT_JSONL})...\n")
    run_start = time.time()
    n_processed_this_run = 0
    n_parsed_ok = 0
    n_repaired = 0
    n_failed = 0
    n_llm_errors = 0
    total_claims_this_run = 0

    # open jsonl in append mode so we never overwrite existing data
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for i, row in remaining.iterrows():
            t0 = time.time()
            try:
                raw = call_llm(prompt_template, row["text"])
                elapsed = time.time() - t0
                llm_error = None
            except Exception as e:
                elapsed = time.time() - t0
                raw = None
                llm_error = f"LLM call failed: {e}"
                n_llm_errors += 1

            if llm_error:
                parsed_claims = None
                parse_error = llm_error
                repair_status = None
            else:
                parsed, repair_status = parse_llm_response(raw)
                if parsed is not None:
                    parsed_claims = parsed.get("claims", []) if isinstance(parsed, dict) else []
                    parse_error = None
                    n_parsed_ok += 1
                    if repair_status == "repaired_from_truncation":
                        n_repaired += 1
                    total_claims_this_run += len(parsed_claims)
                else:
                    parsed_claims = None
                    parse_error = repair_status
                    n_failed += 1

            # write record to jsonl checkpoint
            record = {
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
                "repair_status": repair_status,
                "elapsed_seconds": elapsed,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()  # ensure write is committed in case of crash

            n_processed_this_run += 1

            # progress reporting
            if n_processed_this_run % PROGRESS_EVERY == 0 or n_processed_this_run == len(remaining):
                elapsed_run = time.time() - run_start
                avg_time = elapsed_run / n_processed_this_run
                remaining_count = len(remaining) - n_processed_this_run
                eta_seconds = avg_time * remaining_count

                total_done = len(processed_ids) + n_processed_this_run
                pct_total = 100 * total_done / len(df)

                print(f"  [{n_processed_this_run:>5}/{len(remaining):,}] "
                      f"total {total_done:,}/{len(df):,} ({pct_total:.1f}%) | "
                      f"ok={n_parsed_ok} repaired={n_repaired} failed={n_failed} llm_err={n_llm_errors} | "
                      f"claims={total_claims_this_run} | "
                      f"avg={avg_time:.1f}s/p | "
                      f"eta={format_eta(eta_seconds)}")

    # final summary
    run_elapsed = time.time() - run_start
    print("\n" + "=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)
    print(f"Processed this run:    {n_processed_this_run:,}")
    print(f"  Successful parses:   {n_parsed_ok:,}")
    print(f"    of which repaired: {n_repaired:,}")
    print(f"  Parse failures:      {n_failed:,}")
    print(f"  LLM call errors:     {n_llm_errors:,}")
    print(f"Total claims this run: {total_claims_this_run:,}")
    print(f"Total time:            {format_eta(run_elapsed)}")
    if n_processed_this_run > 0:
        print(f"Avg time/paragraph:    {run_elapsed/n_processed_this_run:.1f}s")
    print(f"\nOutput: {OUTPUT_JSONL}")

    # save run summary
    summary = {
        "run_completed_at": datetime.now().isoformat(),
        "n_processed_this_run": n_processed_this_run,
        "n_parsed_ok": n_parsed_ok,
        "n_repaired": n_repaired,
        "n_failed": n_failed,
        "n_llm_errors": n_llm_errors,
        "total_claims_this_run": total_claims_this_run,
        "run_elapsed_seconds": run_elapsed,
        "avg_seconds_per_paragraph": run_elapsed / n_processed_this_run if n_processed_this_run else None,
        "total_corpus_size": len(df),
        "total_processed_overall": len(processed_ids) + n_processed_this_run,
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()