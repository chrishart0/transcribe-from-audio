"""Tests for audio module utilities."""

from __future__ import annotations

from pathlib import Path

from whisper_diarize.audio import find_media_files


class TestFindMediaFiles:
    def test_finds_audio_files(self, tmp_path: Path):
        (tmp_path / "song.mp3").touch()
        (tmp_path / "clip.wav").touch()
        (tmp_path / "notes.txt").touch()

        files = find_media_files(tmp_path)

        names = [f.name for f in files]
        assert "song.mp3" in names
        assert "clip.wav" in names
        assert "notes.txt" not in names

    def test_finds_video_files(self, tmp_path: Path):
        (tmp_path / "video.mp4").touch()
        (tmp_path / "movie.mkv").touch()

        files = find_media_files(tmp_path)

        names = [f.name for f in files]
        assert "video.mp4" in names
        assert "movie.mkv" in names

    def test_empty_directory(self, tmp_path: Path):
        assert find_media_files(tmp_path) == []

    def test_no_supported_files(self, tmp_path: Path):
        (tmp_path / "readme.txt").touch()
        (tmp_path / "data.csv").touch()

        assert find_media_files(tmp_path) == []

    def test_flat_ignores_subdirectories(self, tmp_path: Path):
        (tmp_path / "top.mp3").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.wav").touch()

        files = find_media_files(tmp_path, recursive=False)

        names = [f.name for f in files]
        assert "top.mp3" in names
        assert "nested.wav" not in names

    def test_recursive_finds_subdirectories(self, tmp_path: Path):
        (tmp_path / "top.mp3").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.wav").touch()

        files = find_media_files(tmp_path, recursive=True)

        names = [f.name for f in files]
        assert "top.mp3" in names
        assert "nested.wav" in names

    def test_case_insensitive_extensions(self, tmp_path: Path):
        (tmp_path / "loud.MP3").touch()
        (tmp_path / "video.Mp4").touch()

        files = find_media_files(tmp_path)

        names = [f.name for f in files]
        assert "loud.MP3" in names
        assert "video.Mp4" in names

    def test_results_are_sorted(self, tmp_path: Path):
        (tmp_path / "charlie.wav").touch()
        (tmp_path / "alice.mp3").touch()
        (tmp_path / "bob.flac").touch()

        files = find_media_files(tmp_path)

        assert files == sorted(files)
