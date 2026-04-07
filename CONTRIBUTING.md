# Contributing to TGF

Thank you for your interest in contributing to TGF. This document provides guidelines for contributing to the project.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Madhvansh/TGF.git
cd TGF

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Code Style

- We use [ruff](https://github.com/astral-sh/ruff) for linting
- Type hints are encouraged for public APIs
- Docstrings should explain *why*, not just *what*

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_physics_engine.py -v

# Quick smoke test
pytest tests/test_smoke.py -v
```

## Making Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run the test suite: `pytest tests/ -v`
5. Submit a pull request

## Areas Where Help Is Needed

- Edge deployment optimization (Raspberry Pi, ONNX export)
- Additional anomaly detection benchmarks
- Multi-tower fleet management
- Dashboard UI improvements
- Documentation improvements

## Reporting Issues

Please use GitHub Issues with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
