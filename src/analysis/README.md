# Presentation Layer

Data visualization and analysis layer for the custom MLP project, generating training plots and decision boundaries.

The code lives in the `analysis` package. `main.py` is the entry point.

## Commands

```bash
uv sync                       # install dependencies
uv run main.py --help         # show the CLI help
uv run pytest                 # run tests
uv run ruff check             # lint
uv run ruff format            # format
```

If `uv` fails with `invalid peer certificate: UnknownIssuer` (for example behind a TLS-intercepting proxy or antivirus), add `--system-certs` to use the system certificate store.
