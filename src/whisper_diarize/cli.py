"""Command-line interface for diarized transcription."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from whisper_diarize import audio, pipeline
from whisper_diarize.pipeline import PipelineConfig

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diarized transcription using Whisper + pyannote.audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("input", type=str, help="Path to audio/video file or directory")
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
        "--diarization-model",
        type=str,
        default=None,
        help="Diarization model ID (default: pyannote/speaker-diarization-community-1)",
    )
    parser.add_argument(
        "--num-speakers", type=int, default=None, help="Exact number of speakers (if known)"
    )
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum expected speakers")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum expected speakers")
    parser.add_argument(
        "--clustering-threshold",
        type=float,
        default=None,
        help="Override pyannote agglomerative clustering threshold",
    )
    parser.add_argument(
        "--min-duration-off",
        type=float,
        default=None,
        help="Override pyannote minimum inter-turn silence duration (seconds)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files (default: next to input file)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories when input is a directory",
    )

    # Execution
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel diarization/transcription (use if GPU memory is limited)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of files to process concurrently (for directory input)",
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

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
        console.print(f"[red]Error:[/red] Path not found: {input_path}")
        raise SystemExit(1)

    if input_path.is_dir():
        files = audio.find_media_files(input_path, recursive=args.recursive)
        if not files:
            console.print(f"[red]Error:[/red] No supported audio/video files found in {input_path}")
            raise SystemExit(1)
    else:
        files = [input_path]

    config_kwargs: dict = dict(
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
        clustering_threshold=args.clustering_threshold,
        min_duration_off=args.min_duration_off,
    )
    if args.diarization_model is not None:
        config_kwargs["diarization_model"] = args.diarization_model
    config = PipelineConfig(**config_kwargs)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    workers = min(args.workers, len(files))

    if workers == 1:
        _process_files_sequential(files, hf_token, config, args, output_dir)
    else:
        _process_files_parallel(files, hf_token, config, args, output_dir, workers)


def _process_files_sequential(
    files: list[Path],
    hf_token: str,
    config: PipelineConfig,
    args: argparse.Namespace,
    output_dir: Path | None,
) -> None:
    for i, file_path in enumerate(files, start=1):
        if len(files) > 1:
            console.print(f"\n[bold][{i}/{len(files)}] {file_path.name}[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[bold]{task.percentage:>5.1f}%[/bold]"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Starting...", total=100)

            def on_progress(stage: str, overall_fraction: float, _tid: TaskID = task_id) -> None:
                progress.update(
                    _tid,
                    description=stage,
                    completed=overall_fraction * 100,
                )

            result = pipeline.run(
                file_path,
                hf_token=hf_token,
                config=config,
                save_cleaned=args.save_cleaned,
                on_progress=on_progress,
                parallel=not args.no_parallel,
                output_dir=output_dir,
            )

            progress.update(task_id, description="Done", completed=100)

        utterance_count = len(result.utterances)
        turn_count = len(result.turns)
        console.print(
            f"[green]✓[/green] Processed {utterance_count} utterances from {turn_count} turns"
        )
        console.print(f"  Outputs: {', '.join(str(p) for p in result.output_paths.values())}")


def _process_files_parallel(
    files: list[Path],
    hf_token: str,
    config: PipelineConfig,
    args: argparse.Namespace,
    output_dir: Path | None,
    workers: int,
) -> None:
    successes: list[tuple[Path, pipeline.PipelineResult]] = []
    errors: list[tuple[Path, Exception]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[bold]{task.percentage:>5.1f}%[/bold]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_ids: dict[Path, TaskID] = {}
        for file_path in files:
            task_ids[file_path] = progress.add_task(
                f"[dim]Queued[/dim] {file_path.name}", total=100
            )

        def _make_progress_callback(file_path: Path) -> pipeline.ProgressCallback:
            tid = task_ids[file_path]

            def on_progress(stage: str, overall_fraction: float) -> None:
                progress.update(
                    tid,
                    description=f"{file_path.name}: {stage}",
                    completed=overall_fraction * 100,
                )

            return on_progress

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_path = {}
            for file_path in files:
                future = pool.submit(
                    pipeline.run,
                    file_path,
                    hf_token=hf_token,
                    config=config,
                    save_cleaned=args.save_cleaned,
                    on_progress=_make_progress_callback(file_path),
                    parallel=not args.no_parallel,
                    output_dir=output_dir,
                )
                future_to_path[future] = file_path

            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                tid = task_ids[file_path]
                try:
                    result = future.result()
                    successes.append((file_path, result))
                    progress.update(tid, description=f"{file_path.name}: Done", completed=100)
                except Exception as exc:
                    errors.append((file_path, exc))
                    progress.update(
                        tid, description=f"[red]{file_path.name}: FAILED[/red]", completed=100
                    )

    # Print summary
    for file_path, result in successes:
        utterance_count = len(result.utterances)
        turn_count = len(result.turns)
        console.print(
            f"[green]✓[/green] {file_path.name}: "
            f"{utterance_count} utterances from {turn_count} turns"
        )
        console.print(f"  Outputs: {', '.join(str(p) for p in result.output_paths.values())}")

    for file_path, exc in errors:
        console.print(f"[red]✗[/red] {file_path.name}: {exc}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
