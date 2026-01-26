"""Command-line interface for diarized transcription."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from sales_agent import pipeline
from sales_agent.pipeline import PipelineConfig

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diarized transcription using Whisper + pyannote.audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("input", type=str, help="Path to input audio/video file")
    parser.add_argument("--hf-token", type=str, required=True, help="Hugging Face token for pyannote models")

    # Transcription options
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference")
    parser.add_argument("--model", type=str, default="large-v3", help="Whisper model size")
    parser.add_argument("--compute-type", type=str, default="int8_float16", help="CTranslate2 compute type")
    parser.add_argument("--language", type=str, default=None, help="Language code (auto-detect if not set)")

    # Audio cleaning
    parser.add_argument("--no-clean", action="store_true", help="Skip audio cleanup")
    parser.add_argument("--save-cleaned", action="store_true", help="Save cleaned audio as .cleaned.wav")
    parser.add_argument("--highpass", type=int, default=80, help="High-pass filter frequency (0 to disable)")
    parser.add_argument("--noise-reduction", type=float, default=0.8, help="Noise reduction strength (0-1)")

    # Diarization options
    parser.add_argument("--num-speakers", type=int, default=None, help="Exact number of speakers (if known)")
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum expected speakers")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum expected speakers")

    args = parser.parse_args()

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
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description="Processing...", total=None)
        result = pipeline.run(
            input_path,
            hf_token=args.hf_token,
            config=config,
            save_cleaned=args.save_cleaned,
        )

    console.print(f"\n[green]✓[/green] Processed {len(result.utterances)} utterances from {len(result.turns)} speaker turns")
    console.print(f"  Outputs: {', '.join(str(p) for p in result.output_paths.values())}")


if __name__ == "__main__":
    main()
