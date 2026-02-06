# Whisper-Diarize Project Rules

## Development Philosophy

- **KISS above all**: Simple > clever, readable > fancy, tested > fast
- **Plan before build**: Non-trivial tasks get exploration + questions first
- **Vertical slices**: Small, shippable increments—not big bang
- **Clean up after**: Tests pass, lints clean, no broken windows

## Code Style

- Write self-documenting code; avoid comments that explain what code does
- Only comment when there's a gotcha or non-obvious intent
- Follow existing patterns in the codebase

## Testing

Test functionality, not implementation:
- Don't test helper functions directly—test the feature that uses them
- Don't test internal state changes—test observable behavior
- Test as high-level as feasible

### Test Structure
```
tests/
  unit/           # Fast, isolated, run before commits
  integration/    # Real audio/transcript validation
```

### Running Tests
```bash
# Unit tests (fast, run often)
uv run pytest tests/unit -v

# Integration tests (slower, real audio)
uv run pytest tests/integration -v

# All tests
uv run pytest tests/ -v
```

### Bug Fix Workflow
1. Ask: "Why didn't a test catch this?"
2. Write a failing test that reproduces the bug
3. Fix the bug
4. Test passes

## Tooling

- **Package management**: `uv` for all operations (add packages, run tests, etc.)
- **Linting/Formatting**: Ruff (`uv run ruff check .` and `uv run ruff format .`)
- **Type checking**: Pyright (`uv run pyright`)

## What NOT To Do

- Don't over-engineer
- Don't add features not requested
- Don't create abstractions for one-time use
- Don't design for hypothetical futures
- Don't skip tests
- Don't leave broken windows

