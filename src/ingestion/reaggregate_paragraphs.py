import pandas as pd
from pathlib import Path

INPUT_PATH  = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\segmentation_esg_sdg"
OUTPUT_PATH = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\llm_paragraphs"

ESG_LABELS = {"Environmental", "Social", "Governance"}


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  loaded {len(df):,} sentence rows")

    # sort by sentence_id so concatenation preserves reading order within each paragraph
    df = df.sort_values(["source_document", "paragraph_id", "sentence_id"]).reset_index(drop=True)

    # build paragraph-level records
    print("\nAggregating into paragraphs...")
    grouped = df.groupby(["source_document", "paragraph_id"], sort=False)

    records = []
    for (src_doc, para_id), group in grouped:
        # count esg sentences in this paragraph
        n_esg = group["esg_label"].isin(ESG_LABELS).sum()

        # esg label distribution as a dict
        esg_counts = group["esg_label"].value_counts().to_dict()
        # convert numpy int to python int for cleaner output
        esg_counts = {k: int(v) for k, v in esg_counts.items()}

        # distinct sdg labels (drop nulls and None-string)
        sdg_labels = group["sdg_label"].dropna().unique().tolist()
        sdg_labels = [s for s in sdg_labels if s and s != "None"]

        # take first row for paragraph-level metadata that should be constant
        first = group.iloc[0]

        records.append({
            "block_id": f"{src_doc}__{para_id}",
            "source_document": src_doc,
            "paragraph_id": para_id,
            "company_name": first["company_name"],
            "year": int(first["year"]),
            "report_type": first["report_type"],
            "framework": first["framework"],
            "page_number_min": int(group["page_number"].min()),
            "page_number_max": int(group["page_number"].max()),
            "text": " ".join(group["text"].astype(str).tolist()),
            "n_sentences": len(group),
            "n_esg_sentences": int(n_esg),
            "esg_label_distribution": str(esg_counts),
            "sdg_labels": sdg_labels,
            "sentence_ids": group["sentence_id"].tolist(),
        })

    paragraphs = pd.DataFrame(records)
    print(f"  built {len(paragraphs):,} paragraph rows (all)")

    # filter to paragraphs with at least one esg-labeled sentence
    before = len(paragraphs)
    paragraphs = paragraphs[paragraphs["n_esg_sentences"] > 0].reset_index(drop=True)
    print(f"  filtered to ESG-containing: {len(paragraphs):,} ({before - len(paragraphs):,} dropped)")

    # diagnostic stats
    print("\nDiagnostics:")
    print(f"  unique documents: {paragraphs['source_document'].nunique()}")
    print(f"  unique companies: {paragraphs['company_name'].nunique()}")
    print(f"  year range: {paragraphs['year'].min()}-{paragraphs['year'].max()}")
    print(f"\n  text length (chars):")
    text_lens = paragraphs["text"].str.len()
    print(f"    median: {text_lens.median():.0f}")
    print(f"    mean:   {text_lens.mean():.0f}")
    print(f"    P90:    {text_lens.quantile(0.9):.0f}")
    print(f"    P99:    {text_lens.quantile(0.99):.0f}")
    print(f"    max:    {text_lens.max()}")
    print(f"\n  sentences per paragraph:")
    print(f"    median: {paragraphs['n_sentences'].median():.0f}")
    print(f"    mean:   {paragraphs['n_sentences'].mean():.1f}")
    print(f"    max:    {paragraphs['n_sentences'].max()}")
    print(f"\n  ESG sentences per paragraph (after filter):")
    print(f"    median: {paragraphs['n_esg_sentences'].median():.0f}")
    print(f"    mean:   {paragraphs['n_esg_sentences'].mean():.1f}")
    print(f"    max:    {paragraphs['n_esg_sentences'].max()}")

    # save
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    paragraphs.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()