""Verified few-shot library (RAG-lite) for Manim code generation.

This is where the "training on 50 equations" actually lives. Instead of
fine-tuning, we curate hand-verified {equation -> working Manim scene} pairs
and inject the nearest ones into the codegen prompt. Every example here MUST
have been rendered successfully with manimcommunity/manim before being added.

To add your equations:
1. Run an equation through the pipeline (or write the scene by hand).
2. Confirm the .mp4 renders and the math is correct.
3. Append an entry to examples.json (preferred) or SEED_EXAMPLES below with
   the correct problem_type tag.

Problem types are decided by SymPy structure, not an LLM, so classification
is deterministic and free.
"""

import os
import json
import logging

import sympy as sp

log = logging.getLogger(__name__)

EXAMPLES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples.json")

PROBLEM_TYPES = (
    "derivative", "integral", "limit", "trigonometry",
    "exponential_log", "polynomial", "linear", "general",
)

# Intent keywords in the raw request override pure structure: "derivative of
# x^2" parses to a polynomial but the USER wants a derivative visualization.
_INTENT_KEYWORDS = {
    "derivative": ("derivative", "differentiate", "d/dx", "slope", "tangent", "rate of change", "\\frac{d}"),
    "integral": ("integral", "integrate", "area under", "antiderivative", "\\int"),
    "limit": ("limit", "approaches", "lim ", "\\lim"),
}


def classify_problem(expression: str | None, raw_input: str = "") -> str:
    """Deterministic problem-type classification via keywords + SymPy structure."""
    lowered = raw_input.lower()
    for ptype, keywords in _INTENT_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return ptype

    if not expression:
        return "general"
    try:
        expr = sp.sympify(expression)
    except Exception:
        return "general"

    if expr.has(sp.Integral):
        return "integral"
    if expr.has(sp.Derivative):
        return "derivative"
    if expr.has(sp.Limit):
        return "limit"
    if expr.has(sp.sin, sp.cos, sp.tan, sp.sec, sp.csc, sp.cot):
        return "trigonometry"
    if expr.has(sp.exp, sp.log):
        return "exponential_log"
    if expr.is_polynomial():
        try:
            degree = sp.degree(expr) if expr.free_symbols else 0
            return "linear" if degree == 1 else "polynomial"
        except Exception:
            return "polynomial"
    return "general"


def _load_examples() -> list[dict]:
    examples = list(SEED_EXAMPLES)
    if os.path.exists(EXAMPLES_JSON):
        try:
            with open(EXAMPLES_JSON) as f:
                examples.extend(json.load(f))
        except Exception as e:
            log.warning(f"Could not load {EXAMPLES_JSON}: {e}")
    return examples


def retrieve_examples(problem_type: str, k: int = 2) -> list[dict]:
    """Return up to k verified examples: exact problem-type matches first,
    then 'general' as filler so the prompt always has at least one
    known-good scene to imitate."""
    examples = _load_examples()
    exact = [ex for ex in examples if ex["problem_type"] == problem_type]
    filler = [ex for ex in examples if ex["problem_type"] != problem_type]
    return (exact + filler)[:k]


def add_verified_example(equation: str, problem_type: str, code: str):
    """Append a newly verified {equation, code} pair to examples.json.
    Call this ONLY after confirming the scene renders and the math is right."""
    if problem_type not in PROBLEM_TYPES:
        raise ValueError(f"Unknown problem_type '{problem_type}'. Use one of {PROBLEM_TYPES}.")
    if "class MathScene" not in code:
        raise ValueError("Code must define 'class MathScene(Scene)'.")
    existing = []
    if os.path.exists(EXAMPLES_JSON):
        with open(EXAMPLES_JSON) as f:
            existing = json.load(f)
    existing.append({"equation": equation, "problem_type": problem_type, "code": code})
    with open(EXAMPLES_JSON, "w") as f:
        json.dump(existing, f, indent=2)
    log.info(f"Added verified example ({problem_type}): {equation}")


# --- Hand-verified seed examples (Manim Community Edition) ---
# These are the starting point; grow this to your 50 curated equations.

_POLYNOMIAL_EXAMPLE = '''\
from manim import *
import numpy as np

class MathScene(Scene):
    def construct(self):
        title = MathTex(r"f(x) = x^2 - 4x + 3", font_size=48).to_edge(UP)
        self.play(Write(title))

        ax = Axes(x_range=[-2, 6, 1], y_range=[-3, 8, 1],
                  axis_config={"include_numbers": True}).scale(0.85).shift(DOWN * 0.5)
        self.play(Create(ax))

        curve = ax.plot(lambda x: x**2 - 4*x + 3, x_range=[-1.2, 5.2], color=BLUE)
        self.play(Create(curve), run_time=2)

        # Highlight the roots x = 1 and x = 3
        for root in (1, 3):
            dot = Dot(ax.c2p(root, 0), color=YELLOW)
            label = MathTex(f"x = {root}", font_size=32).next_to(dot, DOWN)
            self.play(FadeIn(dot, scale=2), Write(label))

        # Vertex at (2, -1)
        vertex = Dot(ax.c2p(2, -1), color=RED)
        vlabel = MathTex(r"(2, -1)", font_size=32).next_to(vertex, DOWN)
        self.play(FadeIn(vertex, scale=2), Write(vlabel))
        self.wait(2)
'''

_DERIVATIVE_EXAMPLE = '''\
from manim import *
import numpy as np

class MathScene(Scene):
    def construct(self):
        title = MathTex(r"f(x) = x^2, \\quad f'(x) = 2x", font_size=44).to_edge(UP)
        self.play(Write(title))

        ax = Axes(x_range=[-3, 3, 1], y_range=[-1, 9, 2],
                  axis_config={"include_numbers": True}).scale(0.8).shift(DOWN * 0.4)
        self.play(Create(ax))

        f = lambda x: x**2
        df = lambda x: 2*x
        curve = ax.plot(f, x_range=[-2.8, 2.8], color=BLUE)
        self.play(Create(curve), run_time=2)

        # Sweep a tangent line along the curve using a ValueTracker
        t = ValueTracker(-2)
        dot = always_redraw(lambda: Dot(ax.c2p(t.get_value(), f(t.get_value())), color=YELLOW))
        tangent = always_redraw(lambda: ax.plot(
            lambda x: f(t.get_value()) + df(t.get_value()) * (x - t.get_value()),
            x_range=[t.get_value() - 1.2, t.get_value() + 1.2], color=RED))
        slope_text = always_redraw(lambda: MathTex(
            f"f'({t.get_value():.1f}) = {df(t.get_value()):.1f}",
            font_size=36).to_corner(UR))

        self.play(FadeIn(dot), Create(tangent), Write(slope_text))
        self.play(t.animate.set_value(2), run_time=5, rate_func=smooth)
        self.wait(2)
'''

_INTEGRAL_EXAMPLE = '''\
from manim import *
import numpy as np

class MathScene(Scene):
    def construct(self):
        title = MathTex(r"\\int_0^2 x^2 \\, dx = \\frac{8}{3}", font_size=44).to_edge(UP)
        self.play(Write(title))

        ax = Axes(x_range=[-1, 3, 1], y_range=[-1, 5, 1],
                  axis_config={"include_numbers": True}).scale(0.8).shift(DOWN * 0.4)
        self.play(Create(ax))

        curve = ax.plot(lambda x: x**2, x_range=[-0.8, 2.6], color=BLUE)
        self.play(Create(curve), run_time=2)

        # Riemann rectangles converging to the smooth area
        rects = ax.get_riemann_rectangles(curve, x_range=[0, 2], dx=0.5,
                                          color=GREEN, fill_opacity=0.6)
        self.play(Create(rects))
        for dx in (0.25, 0.1):
            finer = ax.get_riemann_rectangles(curve, x_range=[0, 2], dx=dx,
                                              color=GREEN, fill_opacity=0.6)
            self.play(Transform(rects, finer), run_time=1.5)

        area = ax.get_area(curve, x_range=[0, 2], color=GREEN, opacity=0.5)
        self.play(FadeOut(rects), FadeIn(area))
        result = MathTex(r"\\text{Area} = \\frac{8}{3}", font_size=40).to_corner(UR)
        self.play(Write(result))
        self.wait(2)
'''

_TRIG_EXAMPLE = '''\
from manim import *
import numpy as np

class MathScene(Scene):
    def construct(self):
        title = MathTex(r"f(x) = \\sin(x)", font_size=48).to_edge(UP)
        self.play(Write(title))

        ax = Axes(x_range=[-2 * PI, 2 * PI, PI], y_range=[-1.5, 1.5, 0.5],
                  axis_config={"include_numbers": False}).scale(0.85).shift(DOWN * 0.4)
        self.play(Create(ax))

        curve = ax.plot(np.sin, x_range=[-2 * PI, 2 * PI], color=BLUE)
        self.play(Create(curve), run_time=3)

        # Trace the unit-circle connection: dot moving on the curve
        t = ValueTracker(-2 * PI)
        dot = always_redraw(lambda: Dot(ax.c2p(t.get_value(), np.sin(t.get_value())), color=YELLOW))
        v_line = always_redraw(lambda: ax.get_vertical_line(
            ax.c2p(t.get_value(), np.sin(t.get_value())), color=GREY))
        self.play(FadeIn(dot), Create(v_line))
        self.play(t.animate.set_value(2 * PI), run_time=5, rate_func=linear)
        self.wait(2)
'''

SEED_EXAMPLES = [
    {"equation": "x^2 - 4x + 3", "problem_type": "polynomial", "code": _POLYNOMIAL_EXAMPLE},
    {"equation": "derivative of x^2", "problem_type": "derivative", "code": _DERIVATIVE_EXAMPLE},
    {"equation": "integral of x^2 from 0 to 2", "problem_type": "integral", "code": _INTEGRAL_EXAMPLE},
    {"equation": "sin(x)", "problem_type": "trigonometry", "code": _TRIG_EXAMPLE},
]
