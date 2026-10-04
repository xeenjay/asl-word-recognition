#!/usr/bin/env python3
"""Build class-aligned auxiliary targets from pinned public training metadata.

Only the official train CSV defines joins and attribute vocabularies. No video
labels are needed at inference; this artifact supplies training targets only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import pickle

WORKSPACE = Path(__file__).resolve().parents[1]
LEX_SHA = "080ecc3de4b307a04dd9b2c2583c22bc623a5c731ab869d6e3905d4a3540fbd3"
TRAIN_SHA = "87a105649b98ea577182ebfce49d59fe9ca834a2489384f6fe7bc03f14966329"
LABEL_SHA = "cd29470b041e1a029e62bc66f47313b08744f669b283b58cf647d532482268e6"
GROUPS = {"handshape": "Handshape", "location": "MajorLocation", "movement": "Movement"}
MISSING = {"", "NA", "N/A"}  # The lexical Movement category "None" is real.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def segment_values(row, prefix):
    """Union the explicitly declared morphemes; mask incomplete supervision."""
    count_text = row.get("NumberOfMorphemes.2.0", "").strip()
    if count_text not in {str(i) for i in range(1, 7)}:
        raise ValueError(f"Invalid morpheme count for {row.get('Code')}: {count_text!r}")
    count = int(count_text)
    columns = [prefix + ("" if i == 1 else f"M{i}") + ".2.0" for i in range(1, 7)]
    values = [row.get(column, "").strip() for column in columns]
    # A real pinned row (J_03_065/CARDS) declares one morpheme but has M3/M4
    # values. Do not guess which field is correct or discard its gloss CE.
    valid = (all(value not in MISSING for value in values[:count])
             and not any(value not in MISSING for value in values[count:]))
    return sorted({value for value in values[:count] if value not in MISSING}), valid


def build_targets(train_rows, lex_rows, id_to_label):
    if set(id_to_label) != set(range(len(id_to_label))) or len(set(id_to_label.values())) != len(id_to_label):
        raise ValueError("Class IDs must be contiguous with unique gloss labels")
    lex = {}
    for row in lex_rows:
        code = row["Code"].strip()
        if code in MISSING or code in lex:
            raise ValueError(f"Missing or duplicate lexical code: {code!r}")
        lex[code] = row
    codes, counts, videos = {}, {}, set()
    for row in train_rows:
        gloss, code, video = row["Gloss"], row["ASL-LEX Code"].strip(), row["Video file"]
        if video in videos:
            raise ValueError(f"Duplicate training video: {video}")
        videos.add(video)
        if gloss in codes and codes[gloss] != code:
            raise ValueError(f"Ambiguous lexical code for class {gloss}")
        codes[gloss] = code
        counts[gloss] = counts.get(gloss, 0) + 1
    if set(codes) != set(id_to_label.values()):
        raise ValueError("Training classes differ from the official classifier vocabulary")
    parsed = {}
    for gloss, code in codes.items():
        if code not in MISSING and code not in lex:
            raise ValueError(f"Nonmissing code absent from the pinned dictionary: {gloss}/{code}")
        parsed[gloss] = {group: segment_values(lex[code], prefix) if code in lex else ([], False)
                         for group, prefix in GROUPS.items()}
    vocabularies = {group: sorted({value for by_group in parsed.values()
                                  for value in by_group[group][0]}) for group in GROUPS}
    records = []
    for class_id in range(len(id_to_label)):
        gloss = id_to_label[class_id]
        targets, masks = {}, {}
        for group, vocabulary in vocabularies.items():
            values, valid = parsed[gloss][group]
            targets[group] = [int(value in values) if valid else 0 for value in vocabulary]
            masks[group] = [valid] * len(vocabulary)
        records.append({"class_id": class_id, "gloss": gloss,
                        "asllex_code": codes[gloss] if codes[gloss] not in MISSING else None,
                        "train_video_count": counts[gloss], "targets": targets, "masks": masks})
    statistics = {}
    for group, vocabulary in vocabularies.items():
        statistics[group] = {
            "valid_classes": sum(bool(row["masks"][group]) and all(row["masks"][group]) for row in records),
            "valid_train_videos": sum(row["train_video_count"] for row in records
                                      if row["masks"][group] and all(row["masks"][group])),
            "positive_train_video_counts": [sum(row["train_video_count"] * row["targets"][group][j]
                                                  for row in records) for j in range(len(vocabulary))],
        }
    return {"schema_version": 1, "policy": "asllex-exact-code-declared-morpheme-union-v1",
            "target_scope": "lexical reference attributes, not per-video variant annotations",
            "fit_split": "train", "vocabularies": vocabularies,
            "head_dimensions": {group: len(values) for group, values in vocabularies.items()},
            "missing_policy": "Keep gloss CE; mask the whole auxiliary group if any declared morpheme is missing or fields exist outside the declared count. Movement None is valid.",
            "records": records, "train_statistics": statistics,
            "unmatched_glosses": sorted(gloss for gloss, code in codes.items() if code in MISSING),
            "matched_but_masked_glosses": {group: sorted(gloss for gloss, code in codes.items()
                                                       if code in lex and not parsed[gloss][group][1]) for group in GROUPS},
            "loss_suggestion": "Unweighted BCE averaged over valid elements within each group, then fixed mean over all heads; a fully missing head contributes zero. Shared encoder receives gradients. No inference labels."}


class PlainMappingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Only a plain official int/string mapping is allowed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lex-csv", type=Path, default=WORKSPACE / "assets/asllex/signdata.csv")
    parser.add_argument("--train-csv", type=Path, default=WORKSPACE / "data/asl_citizen/splits/train.csv")
    parser.add_argument("--label-map", type=Path, default=WORKSPACE / "assets/checkpoints/d_LId2Label.pkl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for name, path, expected in (("lex_csv", args.lex_csv, LEX_SHA), ("train_csv", args.train_csv, TRAIN_SHA),
                                 ("id_to_label", args.label_map, LABEL_SHA)):
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Pinned source hash mismatch: {name}")
        sources[name] = {"path": str(path.resolve()), "sha256": actual}
    with args.lex_csv.open(encoding="latin1", newline="") as stream:
        lex = list(csv.DictReader(stream))
    with args.train_csv.open(encoding="utf-8-sig", newline="") as stream:
        train = list(csv.DictReader(stream))
    labels = PlainMappingUnpickler(io.BytesIO(args.label_map.read_bytes())).load()
    if len(labels) != 2731 or len(train) != 40154 or len(lex) != 2723:
        raise ValueError("Pinned dataset inventory mismatch")
    result = build_targets(train, lex, labels)
    result["sources"] = sources
    result["generator_sha256"] = digest(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "head_dimensions": result["head_dimensions"], "unmatched_glosses": result["unmatched_glosses"],
                      "matched_but_masked_glosses": result["matched_but_masked_glosses"],
                      "valid_train_videos": {k: v["valid_train_videos"] for k, v in result["train_statistics"].items()}}))


if __name__ == "__main__":
    main()
