# Style Guide

## Code
- Terse, concise
- Inline comments only where logic isn't self-evident
- Type hints for public interfaces
- Prefer explicit over clever

## Python
- Python 3.12
- Ruff for linting/formatting
- Pydantic for data validation

## Documentation
- Docstrings: one-line summary only, no params/returns unless complex
- README per service: setup instructions only

## Naming
- `snake_case`: files, functions, variables
- `PascalCase`: classes
- `SCREAMING_SNAKE`: constants
- Prefix internal/private with `_`

## Git
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`
- One logical change per commit
