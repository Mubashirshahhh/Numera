# Numera

**Transform mathematics into beautiful AI-generated animations.**

Numera is an AI-powered mathematics visualization engine that converts mathematical expressions into animated visual explanations. It combines symbolic mathematics, large language models, and Manim to generate accurate, educational animations from natural language or mathematical equations.

## Features

* AI-powered mathematical reasoning
* Automatic Manim animation generation
* Symbolic mathematics using SymPy
* Mathematical validation before rendering
* Self-healing code generation pipeline
* Docker-based secure rendering
* Intelligent rendering fallback
* SQLite job tracking and caching
* Support for algebra, calculus, graphing, and many high-school mathematics topics

## Example

Generate an animation directly from the terminal:

```bash
numera "Plot the quadratic equation x^2 - 4x + 3"
```

Differentiate a function:

```bash
numera "Differentiate sin(x)"
```

Integrate an expression:

```bash
numera "Integrate x^3"
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Mubashirshahhh/Numera.git
cd Numera
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install Numera:

```bash
pip install -e .
```

Once published to PyPI:

```bash
pip install numera
```

## Requirements

* Python 3.11+
* Docker
* FFmpeg
* LaTeX (recommended for best rendering quality)

## How it Works

1. User submits a mathematical prompt or equation.
2. Numera analyzes the mathematics using SymPy.
3. AI generates optimized Manim code.
4. Mathematical invariants are validated.
5. Invalid generations are automatically repaired.
6. The animation is rendered securely inside Docker.
7. The final video is returned to the user.

## Project Structure

```
Numera/
├── pyproject.toml
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── numera/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   └── pipeline.py
│
└── tests/
```

## Roadmap

* Interactive CLI
* Web interface
* Additional mathematical domains
* Improved AI planning
* Faster rendering pipeline
* Plugin architecture
* PyPI release

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

## Author

**Mubashir Shah**

GitHub: https://github.com/Mubashirshahhh

## Contributing

Contributions, feature requests, and bug reports are welcome. Please open an issue or submit a pull request to help improve Numera.

