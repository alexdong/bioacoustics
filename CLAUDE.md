# Bird Songs Project Development Guide

## Project Commands

```bash
# Linting
ruff check .

# Type checking (options)
mypy .
pyright

# Run tests
pytest

# Run a specific test file
pytest path/to/test_file.py

# Run a specific test function
pytest path/to/test_file.py::test_function_name
```

## Code Standards

- **Python version**: 3.12
- **Line length**: 88 characters
- **Types**: Required for all functions except `self`/`cls` parameters
- **Imports**: Standard library first, then third-party, then local modules
- **Style**: Two blank lines between functions/classes, one blank line between code blocks
- **Logging**: Use `[LEVEL] message` format for print statements
- **Documentation**: Docstrings for complex functions only
- **Error handling**: Prefer assertions over try/except
- **Testing**: Include `if __name__ == "__main__"` blocks for quick demos and tests
- **Architecture**: Prefer simple functions over classes where possible