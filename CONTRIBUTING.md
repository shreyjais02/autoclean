# Contributing to AutoClean Pipeline

Thank you for your interest in contributing to **AutoClean Pipeline**! We welcome contributions from the community.

---

## 🛠️ Development Setup

1. **Fork and Clone** the repository:
   ```bash
   git clone https://github.com/<your-username>/autoclean-pipeline.git
   cd autoclean-pipeline
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS / Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Running Tests

Before submitting a pull request, ensure all tests pass:

```bash
# Run complete test suite
python -m unittest discover -s tests -v

# Run the example demo
python examples/run_example.py
```

---

## 🚀 Pull Request Guidelines

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Write clean, documented code following PEP 8 conventions.
3. Add unit tests for any new features or bug fixes in `tests/`.
4. Commit your changes with descriptive commit messages.
5. Push to your branch and open a Pull Request against `main`.

---

## 📄 License
By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
