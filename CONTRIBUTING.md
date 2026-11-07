# Contributing

Thanks for helping improve the Smallscale Crystal Diffusion Model.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Workflow

1. Fork the repository.
2. Create a feature branch.
3. Make focused changes with tests.
4. Run the project checks locally.
5. Submit a pull request with a clear description.

## Tests

```bash
python -m pytest -q
python -m ruff check .
```

## Coding expectations

- Prefer small, readable functions.
- Keep scientific code documented and explicit.
- Add tests for new behavior.
- Preserve compatibility with Python 3.10+.
- Keep user-facing examples up to date.
