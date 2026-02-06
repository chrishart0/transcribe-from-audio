"""Command-line interface for diarized transcription."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from whisper_diarize import pipeline
from whisper_diarize.pipeline import PipelineConfig

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diarized transcription using Whisper + pyannote.audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("input", type=str, help="Path to input audio/video file")
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for pyannote models (defaults to HF_TOKEN env var or .env)",
    )

    # Transcription options
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference"
    )
    parser.add_argument("--model", type=str, default="large-v3", help="Whisper model size")
    parser.add_argument(
        "--compute-type", type=str, default="int8_float16", help="CTranslate2 compute type"
    )
    parser.add_argument(
        "--language", type=str, default=None, help="Language code (auto-detect if not set)"
    )

    # Audio cleaning
    parser.add_argument("--no-clean", action="store_true", help="Skip audio cleanup")
    parser.add_argument(
        "--save-cleaned", action="store_true", help="Save cleaned audio as .cleaned.wav"
    )
    parser.add_argument(
        "--highpass", type=int, default=80, help="High-pass filter frequency (0 to disable)"
    )
    parser.add_argument(
        "--noise-reduction", type=float, default=0.8, help="Noise reduction strength (0-1)"
    )

    # Diarization options
    parser.add_argument(
        "--num-speakers", type=int, default=None, help="Exact number of speakers (if known)"
    )
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum expected speakers")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum expected speakers")

    # Execution
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel diarization/transcription (use if GPU memory is limited)",
    )

    args = parser.parse_args()

    load_dotenv()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        console.print(
            "[red]Error:[/red] No Hugging Face token found. "
            "Provide --hf-token, set HF_TOKEN env var, or add it to .env"
        )
        raise SystemExit(1)

    input_path = Path(args.input).expanduser().resolve()

    if not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {input_path}")
        raise SystemExit(1)

    config = PipelineConfig(
        clean_audio=not args.no_clean,
        highpass_freq=args.highpass,
        noise_reduction_strength=args.noise_reduction,
        whisper_model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[bold]{task.percentage:>5.1f}%[/bold]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Starting...", total=100)

        def on_progress(stage: str, overall_fraction: float) -> None:
            progress.update(
                task_id,
                description=stage,
                completed=overall_fraction * 100,
            )

        result = pipeline.run(
            input_path,
            hf_token=hf_token,
            config=config,
            save_cleaned=args.save_cleaned,
            on_progress=on_progress,
            parallel=not args.no_parallel,
        )

        progress.update(task_id, description="Done", completed=100)

    utterance_count = len(result.utterances)
    turn_count = len(result.turns)
    console.print(
        f"\n[green]✓[/green] Processed {utterance_count} utterances from {turn_count} turns"
    )
    console.print(f"  Outputs: {', '.join(str(p) for p in result.output_paths.values())}")


if __name__ == "__main__":
    main()
