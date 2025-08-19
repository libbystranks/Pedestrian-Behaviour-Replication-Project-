# inspect_timeline.py
import os, csv, torch
import numpy as np

OUT_PT  = "ptFiles/pipeline_output.pt"
OUT_CSV = "ptFiles/pipeline_timeline.csv"

def print_switches():
    if not os.path.exists(OUT_CSV):
        print("No pipeline_timeline.csv found. Run the pipeline first.")
        return
    print("\n=== Switch summary ===")
    with open(OUT_CSV, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            print(f"@ base frame {row['switch_at_base_frame']}: {row['selected_file']} "
                  f"(center {row['selected_center_frame']}), fade {row['fade_len']}, "
                  f"snippet {row['snippet_len_used']}")

def print_source_spans():
    d = torch.load(OUT_PT, map_location="cpu")
    source = d.get("source", None)
    if source is None:
        print("No 'source' tensor in output. Re-run pipeline after patching.")
        return
    src = source.numpy().tolist()
    # compress runs to spans
    spans = []
    start = 0
    for i in range(1, len(src)+1):
        if i == len(src) or src[i] != src[start]:
            spans.append((src[start], start, i-1))
            start = i
    print("\n=== Source spans (0=base, 1=fade, 2=snippet) ===")
    for code, a, b in spans[:200]:
        print(f"[{a:4d}..{b:4d}]  -> {code}")

if __name__ == "__main__":
    print_switches()
    print_source_spans()

