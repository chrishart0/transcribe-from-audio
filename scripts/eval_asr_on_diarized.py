"""Score an ASR backend against already-diarized gold transcripts."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from whisper_diarize.qwen_asr import transcribe_array

_PUNCT_RE = re.compile(r"[^\w\s']+", re.UNICODE)


def normalize(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).lower()
    folded = _PUNCT_RE.sub(" ", folded)
    return folded.split()


def word_error_rate(reference: str, hypothesis: str) -> tuple[float, int, int]:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return (0.0 if not hyp else 1.0), 0, len(hyp)
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    distance = dp[-1][-1]
    return distance / len(ref), distance, len(ref)


def load_gold(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of utterances in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-duration", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N utterances")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    audio, sr = sf.read(str(args.audio), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    gold = load_gold(args.gold)
    if args.limit:
        gold = gold[: args.limit]

    rows = []
    total_edits = 0
    total_ref = 0
    for i, utt in enumerate(gold, start=1):
        start = float(utt["start"])
        end = float(utt["end"])
        ref_text = str(utt.get("text") or "")
        duration = end - start
        if duration < args.min_duration or not normalize(ref_text):
            continue
        start_i = max(0, int(start * sr))
        end_i = min(len(audio), int(end * sr))
        clip = np.asarray(audio[start_i:end_i], dtype=np.float32)
        hyp = transcribe_array(clip, sr, model_id=args.model, device=args.device, language="en")
        wer, edits, n_ref = word_error_rate(ref_text, hyp)
        total_edits += edits
        total_ref += n_ref
        rows.append(
            {
                "index": i,
                "speaker": utt.get("speaker"),
                "start": start,
                "end": end,
                "wer": round(wer, 4),
                "ref": ref_text,
                "hyp": hyp,
            }
        )
        print(f"[{i}/{len(gold)}] {utt.get('speaker')} {start:.1f}-{end:.1f}s WER={wer:.3f}")
        print(f"  REF: {ref_text[:160]}")
        print(f"  HYP: {hyp[:160]}")

    overall = (total_edits / total_ref) if total_ref else 0.0
    summary = {
        "audio": str(args.audio),
        "gold": str(args.gold),
        "model": args.model,
        "utterances_scored": len(rows),
        "overall_wer": round(overall, 4),
        "edits": total_edits,
        "ref_words": total_ref,
        "rows": rows,
    }
    print(f"\nOverall WER: {overall:.3f}  ({total_edits}/{total_ref} on {len(rows)} utterances)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
