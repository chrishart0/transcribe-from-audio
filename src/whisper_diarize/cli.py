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
    parser.add_argument(
        "--profile",
        type=str,
        default="accuracy",
        choices=["accuracy", "balanced", "speed"],
        help="Transcription profile preset",
    )
    parser.add_argument("--model", type=str, default=None, help="Whisper model size or HF model ID")
    parser.add_argument(
        "--compute-type",
        type=str,
        default=None,
        help="CTranslate2 compute type (profile default when omitted)",
    )
    parser.add_argument(
        "--language", type=str, default=None, help="Language code (auto-detect if not set)"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam search size (profile default when omitted)",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Condition each segment on previous text (can increase repetition on long audio)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Decoding temperature override (omit to keep built-in fallback schedule)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Penalty for repeated token generation (>1.0 discourages loops)",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=None,
        help="Prevent repeating n-grams of this size (0 disables)",
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
    parser.add_argument(
        "--alignment-mode",
        type=str,
        default=None,
        choices=["accurate", "fast"],
        help="Alignment/smoothing behavior (profile default when omitted)",
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
        default=0,
        help="Number of files to process concurrently (0 = auto by available VRAM)",
    )
    parser.add_argument(
        "--no-retry-sequential",
        action="store_true",
        help="Disable sequential retry when parallel run fails due to memory pressure",
    )

    args = parser.parse_args()

    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if args.beam_size is not None and args.beam_size < 1:
        parser.error("--beam-size must be >= 1")
    if args.repetition_penalty is not None and args.repetition_penalty < 1.0:
        parser.error("--repetition-penalty must be >= 1.0")
    if args.no_repeat_ngram_size is not None and args.no_repeat_ngram_size < 0:
        parser.error("--no-repeat-ngram-size must be >= 0")

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
        device=args.device,
        profile=args.profile,
        language=args.language,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        clustering_threshold=args.clustering_threshold,
        min_duration_off=args.min_duration_off,
        retry_sequential_on_failure=not args.no_retry_sequential,
    )
    if args.model is not None:
        config_kwargs["whisper_model"] = args.model
    if args.compute_type is not None:
        config_kwargs["compute_type"] = args.compute_type
    if args.beam_size is not None:
        config_kwargs["beam_size"] = args.beam_size
    if args.condition_on_previous_text is not None:
        config_kwargs["condition_on_previous_text"] = args.condition_on_previous_text
    if args.temperature is not None:
        config_kwargs["temperature"] = args.temperature
    if args.repetition_penalty is not None:
        config_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size is not None:
        config_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    if args.alignment_mode is not None:
        config_kwargs["alignment_mode"] = args.alignment_mode
    if args.diarization_model is not None:
        config_kwargs["diarization_model"] = args.diarization_model
    config = pipeline.resolve_runtime_config(PipelineConfig(**config_kwargs))

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    vram_gb = pipeline.detect_cuda_total_memory_gb()

    if args.workers == 0:
        workers = pipeline.recommend_worker_count(len(files), config, vram_gb)
    else:
        workers = min(args.workers, len(files))

    vram_text = f"{vram_gb:.1f} GB" if vram_gb is not None else "n/a"
    console.print(
        "[dim]Resolved runtime: "
        f"profile={config.profile}, device={config.device}, vram={vram_text}, "
        f"model={config.whisper_model}, compute={config.compute_type}, beam={config.beam_size}, "
        f"cond_prev={config.condition_on_previous_text}, temp={config.temperature}, "
        f"rep_pen={config.repetition_penalty}, no_repeat_ngram={config.no_repeat_ngram_size}, "
        f"alignment={config.alignment_mode}, workers={workers}, parallel={not args.no_parallel}, "
        f"retry_sequential={not args.no_retry_sequential}[/dim]"
    )

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
