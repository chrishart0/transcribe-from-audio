# Publishing to PyPI

## Pre-publish Checklist

- [ ] **Update URLs** in `pyproject.toml` with your actual GitHub repo
- [ ] **Verify README renders** - PyPI uses the README as the package description
- [ ] **Run tests** - `uv run pytest tests/ -v`
- [ ] **Test build locally** - `uv build`
- [ ] **Create PyPI account** at https://pypi.org/account/register/
- [ ] **Create API token** at https://pypi.org/manage/account/token/

---

## Publishing Steps

### 1. Build the package

```bash
uv build
```

This creates:
- `dist/whisper_diarize-0.1.0-py3-none-any.whl`
- `dist/whisper_diarize-0.1.0.tar.gz`

### 2. Test on TestPyPI first (recommended)

```bash
# Upload to TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/ --token $TEST_PYPI_TOKEN

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ your-package-name
```

### 3. Publish to PyPI

```bash
uv publish --token $PYPI_TOKEN
```

Or configure `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-your-api-token-here
```

Then just:
```bash
uv publish
```

---

## Version Bumping

Before each release, update the version in both:
1. `pyproject.toml` - `version = "X.Y.Z"`
2. `src/whisper_diarize/__init__.py` - `__version__ = "X.Y.Z"`

Follow [semantic versioning](https://semver.org/):
- **MAJOR** (1.0.0) - breaking API changes
- **MINOR** (0.2.0) - new features, backwards compatible
- **PATCH** (0.1.1) - bug fixes

---

## GitHub Actions (Optional)

Add `.github/workflows/publish.yml` for automated releases:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv build
      - run: uv publish --token ${{ secrets.PYPI_TOKEN }}
```

Add `PYPI_TOKEN` to your repo secrets at Settings → Secrets → Actions.

