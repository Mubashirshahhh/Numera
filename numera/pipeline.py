import os
import re
import json
import hashlib
import logging
import sqlite3
import shutil
import subprocess
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load .env FIRST
from dotenv import load_dotenv
load_dotenv()

import multiprocessing

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from openai import OpenAI
from anthropic import Anthropic

try:
    # Deterministic LaTeX parsing (requires antlr4-python3-runtime).
    from sympy.parsing.latex import parse_latex
    LATEX_PARSER_AVAILABLE = True
except Exception:
    LATEX_PARSER_AVAILABLE = False

from example_library import classify_problem, retrieve_examples

# --- 1. OBSERVABILITY & CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# Secrets must come from the environment (.env / secret manager), never literals in source.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Export it before starting the worker.")

openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    max_retries=0,
)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

DB_PATH = os.environ.get("PRODUCTION_DB_PATH", "production_state.db")
MAX_RETRIES = int(os.environ.get("RENDER_MAX_RETRIES", "3"))
# A full Manim scene routinely exceeds 2000 tokens; truncated code either fails
# extraction or (worse) passes it while being subtly broken.
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
# SymPy has no internal timeouts: one pathological integral can stall a worker
# forever. Symbolic ops run in a subprocess with this budget (seconds).
SYMPY_TIMEOUT = int(os.environ.get("SYMPY_TIMEOUT", "10"))

MANIM_RULES = """
Rules for the scene:
- The scene class MUST be named MathScene and subclass Scene (Manim Community Edition).
- Use Axes() with explicit x_range/y_range, ax.plot(lambda x: ...), and MathTex for labels.
- Use ValueTracker + always_redraw for animated sweeps (tangent lines, moving dots).
- Keep total runtime under ~30 seconds of animation (self.wait calls included).
- Never read files, use network access, or import anything beyond manim/numpy/math.
"""


# --- 2. STATE MANAGEMENT & CACHING (Database) ---
@contextmanager
def db_conn():
    """Single shared connection pattern with WAL mode so concurrent threads
    don't collide with 'database is locked' errors, and changes are committed
    or rolled back consistently."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db_conn() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS jobs
                     (job_id TEXT PRIMARY KEY, status TEXT, equation TEXT,
                      code TEXT, video_path TEXT, error TEXT,
                      updated_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS cache
                     (hash_key TEXT PRIMARY KEY, video_path TEXT)''')


def update_job_state(job_id: str, status: str, code: str = "", video_path: str = "", error: str = ""):
    with db_conn() as conn:
        conn.execute('''INSERT INTO jobs (job_id, status, code, video_path, error)
                     VALUES (?, ?, ?, ?, ?)
                     ON CONFLICT(job_id) DO UPDATE SET
                        status=excluded.status,
                        code=CASE WHEN excluded.code != '' THEN excluded.code ELSE jobs.code END,
                        video_path=CASE WHEN excluded.video_path != '' THEN excluded.video_path ELSE jobs.video_path END,
                        error=excluded.error,
                        updated_at=CURRENT_TIMESTAMP''',
                  (job_id, status, code, video_path, error))
    log.info(f"Job {job_id} -> {status}")


def get_cache(hash_key: str) -> Optional[str]:
    with db_conn() as conn:
        row = conn.execute("SELECT video_path FROM cache WHERE hash_key=?", (hash_key,)).fetchone()
    return row[0] if row else None


def set_cache(hash_key: str, video_path: str):
    with db_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO cache (hash_key, video_path) VALUES (?, ?)",
                     (hash_key, video_path))


# --- 3. MODEL FALLBACK ROUTER ---
FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
]

def call_llm(prompt: str, system_role: str, temperature: float = 0.1) -> str:
    """Tries each free OpenRouter model in sequence; falls back to Anthropic
    if configured. Raises only when all providers are exhausted."""
    last_err = None
    for model in FREE_MODELS:
        try:
            log.info(f"Trying model: {model}")
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=LLM_MAX_TOKENS,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Model returned empty/None content.")
            log.info(f"Success with model: {model}")
            return content
        except Exception as e:
            log.warning(f"Model {model} failed: {e}")
            last_err = e
            continue

    if anthropic_client:
        try:
            log.info("Falling back to Anthropic...")
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=LLM_MAX_TOKENS,
                temperature=temperature,
                system=system_role,
                messages=[{"role": "user", "content": prompt}]
            )
            return "".join(block.text for block in response.content if block.type == "text")
        except Exception as fallback_err:
            raise RuntimeError(f"All LLM providers failed. Last: {fallback_err}") from fallback_err

    raise RuntimeError(f"All free models failed. Last error: {last_err}") from last_err


def extract_code_block(raw_output: str) -> str:
    """Pulls a python code block out of LLM output; falls back to the raw
    text only if it looks like it could plausibly be code, otherwise raises
    so a bad LLM response fails loudly instead of being rendered as-is."""
    match = re.search(r"```python\n(.*?)\n```", raw_output, re.DOTALL)
    if match:
        return match.group(1)
    if "class MathScene" in raw_output or "def construct" in raw_output:
        return raw_output
    raise ValueError(f"LLM did not return a recognizable Manim code block:\n{raw_output[:300]}")


# --- 4. DETERMINISTIC MATH ENGINE ---
# sp.sympify() on untrusted strings is eval-based and can execute arbitrary
# code. parse_expr with a locked-down namespace only recognizes math.
_SAFE_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def safe_sympify(expr_str: str):
    """Parse a math string without eval-based code execution risk."""
    return parse_expr(
        expr_str,
        transformations=_SAFE_TRANSFORMS,
        evaluate=True,
    )


def _sympy_op_worker(queue, op_name, expr_srepr, var_name):
    try:
        expr = sp.sympify(expr_srepr)  # srepr round-trip: trusted, we built it
        var = sp.Symbol(var_name)
        if op_name == "diff":
            result = sp.simplify(sp.diff(expr, var))
        else:
            result = sp.simplify(sp.integrate(expr, var))
        queue.put(("ok", sp.srepr(result), str(result)))
    except Exception as e:
        queue.put(("err", None, str(e)))


def sympy_op_with_timeout(op_name: str, expr, var, timeout: int = SYMPY_TIMEOUT) -> Optional[str]:
    """Run sp.diff/sp.integrate + simplify in a subprocess so a pathological
    expression can't hang the worker forever. Returns str(result) or None."""
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_sympy_op_worker, args=(queue, op_name, sp.srepr(expr), str(var))
    )
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        log.warning(f"SymPy {op_name} timed out after {timeout}s -- skipping.")
        return None
    if not queue.empty():
        status, _, text = queue.get()
        if status == "ok":
            return text
        log.warning(f"SymPy {op_name} failed: {text}")
    return None


@dataclass
class MathFacts:
    raw_input: str
    parsed_clean: bool
    expression: Optional[str] = None
    variable: Optional[str] = None
    derivative: Optional[str] = None
    integral: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def try_deterministic_parse(user_input: str):
    """Users type LaTeX -- parse it deterministically before ever touching an
    LLM. Less LLM in the parse step means fewer silent math errors. Returns a
    SymPy expression or None."""
    stripped = user_input.strip().strip("$").strip()

    # 1. Direct LaTeX parse (handles \frac, \int, \sqrt, x^2, \sin, ...)
    if LATEX_PARSER_AVAILABLE:
        try:
            expr = parse_latex(stripped)
            log.info("Parsed input deterministically via parse_latex.")
            return expr
        except Exception:
            pass

    # 2. Direct SymPy parse (handles plain 'x**2 - 4*x + 3', 'x^2' via xor->pow)
    try:
        candidate = stripped.replace("^", "**")
        if "=" in candidate:
            lhs, rhs = candidate.split("=", 1)
            expr = safe_sympify(lhs.strip()) - safe_sympify(rhs.strip())
        else:
            expr = safe_sympify(candidate)
        log.info("Parsed input deterministically via safe_sympify.")
        return expr
    except Exception:
        return None


def extract_clean_expression(user_input: str) -> str:
    """LLM fallback for natural-language phrasing ('derivative of ...') that
    deterministic parsing can't handle. The LLM's only job is translation to
    SymPy-parseable notation -- SymPy remains the source of truth for
    everything mathematical."""
    raw = call_llm(
        f"User request: {user_input}\n\n"
        "Extract the core mathematical expression or equation and rewrite it "
        "in plain SymPy-parseable Python syntax (use ** for powers, * for all "
        "multiplication, standard function names like sin, cos, log, exp). "
        "Respond with ONLY the expression/equation, nothing else.",
        "You are a precise math-notation translator. Output only the expression.",
        temperature=0.0
    )
    return raw.strip().strip("`")


def analyze_mathematical_properties(user_input: str) -> MathFacts:
    log.info("Parsing expression via SymPy...")
    try:
        # Deterministic first (LaTeX / plain notation), LLM only as fallback.
        expr = try_deterministic_parse(user_input)
        if expr is None:
            expr_str = extract_clean_expression(user_input)
            if "=" in expr_str:
                lhs_str, rhs_str = expr_str.split("=", 1)
                expr = safe_sympify(lhs_str.strip()) - safe_sympify(rhs_str.strip())
            else:
                expr = safe_sympify(expr_str)

        free_syms = sorted(expr.free_symbols, key=str)
        primary_var = free_syms[0] if free_syms else None

        facts = MathFacts(
            raw_input=user_input,
            parsed_clean=True,
            expression=str(expr),
            variable=str(primary_var) if primary_var else None,
        )
        if primary_var:
            # Subprocess + timeout: sp.integrate can hang indefinitely.
            facts.derivative = sympy_op_with_timeout("diff", expr, primary_var)
            facts.integral = sympy_op_with_timeout("integrate", expr, primary_var)
        return facts

    except Exception as e:
        log.error(f"Math parsing failed: {e}")
        return MathFacts(raw_input=user_input, parsed_clean=False, error=str(e))


# --- 5. DEEP SEMANTIC VALIDATION ---
def _symbolically_present(expected_expr: str, code: str, variable: str) -> bool:
    """Checks algebraic equivalence rather than string matching, so '2*x',
    'x*2', and '2.0*x' are all correctly recognized as the same derivative.
    Scans candidate sub-expressions pulled from assignment/call lines in the
    generated code."""
    try:
        var = sp.Symbol(variable)
        expected = safe_sympify(expected_expr)
    except Exception:
        return True  # can't evaluate expected side -> don't block the pipeline on it

    raw_candidates = re.findall(r"=\s*([^\n#]+)", code) + \
        re.findall(r"\(([^()\n]*" + re.escape(variable) + r"[^()\n]*)\)", code)

    candidates = []
    for cand in raw_candidates:
        cand = cand.strip().rstrip(",")
        # Strip a leading "lambda <var>:" or "lambda <var> :" so expressions
        # inside lambdas (a common Manim pattern: ax.plot(lambda x: 2*x+3))
        # are still checked rather than silently skipped.
        cand = re.sub(r"^lambda\s+\w+\s*:\s*", "", cand)
        candidates.append(cand)

    for cand in candidates:
        try:
            cand_expr = safe_sympify(cand)
            if sp.simplify(cand_expr - expected) == 0:
                return True
        except Exception:
            continue
    return False


def validate_math_invariants(code: str, facts: MathFacts) -> bool:
    log.info("Running pre-render semantic validation...")
    if not facts.parsed_clean:
        return True  # nothing to validate against
    if facts.derivative and facts.variable:
        if not _symbolically_present(facts.derivative, code, facts.variable):
            log.error(f"Validation failed: derivative '{facts.derivative}' not found (symbolically) in generated code.")
            return False
    return True


# --- 6. AGENTS ---
def generate_code_pipeline(facts: MathFacts, params: dict, feedback: str = "") -> str:
    """3-agent chain: analyst -> visual director -> Manim programmer.
    Injects the nearest verified examples for this problem type (RAG-lite over
    the curated training library) so codegen imitates known-good scenes."""
    problem_type = classify_problem(facts.expression, facts.raw_input)
    examples = retrieve_examples(problem_type, k=2)
    examples_block = "\n\n".join(
        f"### Verified example ({ex['problem_type']}): {ex['equation']}\n```python\n{ex['code']}\n```"
        for ex in examples
    ) or "(no examples available for this problem type)"

    report = call_llm(f"Evaluate: {json.dumps(facts.to_dict())}", "You are a Mathematical Analyst.")
    visuals = call_llm(
        f"Design a 3blue1brown-style scene (problem type: {problem_type}) for:\n{report}",
        "You are an Educational Visual Director.", 0.3
    )

    feedback_block = f"\nPrevious attempt was rejected: {feedback}\n" if feedback else ""
    prompt = (
        f"Script:\n{visuals}\n\n{MANIM_RULES}\n\n"
        f"Verified working examples of similar scenes -- imitate their structure and style:\n{examples_block}\n\n"
        f"Params:\n{json.dumps(params)}\n{feedback_block}\n"
        "Output a complete Manim Community Edition scene named MathScene (subclass of Scene). "
        "Output ONLY python code in a single ```python code block."
    )
    raw_output = call_llm(prompt, "You are an Expert Manim Programmer. Output ONLY python code in a block.")
    return extract_code_block(raw_output)


def heal_code(broken_code: str, error_log: str, attempt_history: list) -> str:
    """Heal with memory: the model sees prior failed attempts so it doesn't
    ping-pong between the same two errors. Manim tracebacks are huge, so the
    error log is truncated to the tail where the actual exception lives."""
    history_block = ""
    if attempt_history:
        history_block = "Previous failed attempts (do NOT repeat these mistakes):\n" + "\n".join(
            f"- Attempt {i + 1} error: {err[:300]}" for i, err in enumerate(attempt_history)
        ) + "\n\n"
    prompt = (
        f"{history_block}Broken code:\n{broken_code}\n\nError (tail):\n{error_log[-3000:]}\n\n"
        "Fix the code so it renders successfully. Output ONLY the corrected python code in a ```python block."
    )
    raw_output = call_llm(prompt, "You are an Automated Code Reviewer fixing Manim errors.")
    return extract_code_block(raw_output)


# --- 7. LOCAL MATPLOTLIB RENDERER ---
def render_locally_matplotlib(facts: MathFacts, job_id: str,
                              params: Optional[dict] = None) -> Optional[str]:
    """FALLBACK renderer only: a generic animated MP4 via matplotlib + ffmpeg
    when Docker/Manim is unavailable. Returns the video path on success."""
    if not facts.parsed_clean or not facts.expression or not facts.variable:
        log.warning("Cannot render locally: math facts are incomplete.")
        return None

    out_dir = os.path.abspath(f"./output_{job_id}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "render.mp4")

    try:
        var = sp.Symbol(facts.variable)
        expr = safe_sympify(facts.expression)
        deriv_expr = safe_sympify(facts.derivative) if facts.derivative else None

        f_num = sp.lambdify(var, expr, "numpy")
        d_num = sp.lambdify(var, deriv_expr, "numpy") if deriv_expr else None

        # Honor the UI-provided x_range instead of hardcoding [-6, 6].
        x_range = (params or {}).get("x_range")
        if x_range and len(x_range) >= 2:
            x_min, x_max = float(x_range[0]), float(x_range[1])
        else:
            x_min, x_max = -6.0, 6.0

        x = np.linspace(x_min, x_max, 400)
        try:
            y_raw = f_num(x)
            # Constant expressions (f(x) = 5) return a scalar -- broadcast it.
            y = np.broadcast_to(np.asarray(y_raw, dtype=float), x.shape).copy()
        except Exception:
            y = np.array([float(f_num(xi)) for xi in x])
        y = np.clip(y, -20, 20)

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#1e1e2e")
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")
        ax.set_xlim(x_min, x_max)
        y_pad = max((np.nanmax(y) - np.nanmin(y)) * 0.2, 1)
        ax.set_ylim(np.nanmin(y) - y_pad, np.nanmax(y) + y_pad)
        ax.axhline(0, color="#585b70", linewidth=0.8, zorder=1)
        ax.axvline(0, color="#585b70", linewidth=0.8, zorder=1)
        ax.grid(True, color="#313244", linewidth=0.5, linestyle="--", alpha=0.6)

        expr_str = str(sp.latex(expr))
        # If the input was an equation, expr is (lhs - rhs); don't mislabel it as f(x).
        if "=" in facts.raw_input:
            title = f"${expr_str} = 0$"
        else:
            title = f"$f({facts.variable}) = {expr_str}$"
        ax.set_title(title, color="#cdd6f4", fontsize=15, pad=12)
        ax.set_xlabel(facts.variable, color="#cdd6f4")
        ax.set_ylabel(f"f({facts.variable})", color="#cdd6f4")

        curve_line, = ax.plot([], [], color="#89b4fa", linewidth=2.5, label=f"f({facts.variable})", zorder=3)

        tangent_line, = ax.plot([], [], color="#f38ba8", linewidth=1.8,
                                linestyle="--", label="tangent f'", zorder=4)
        sweep_fill = [None]

        dot, = ax.plot([], [], "o", color="#a6e3a1", markersize=9, zorder=5)
        info_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                            color="#cdd6f4", fontsize=10, va="top",
                            bbox=dict(boxstyle="round,pad=0.4", facecolor="#313244", alpha=0.8))
        ax.legend(loc="upper right", facecolor="#313244", edgecolor="#45475a",
                  labelcolor="#cdd6f4", fontsize=9)

        FRAMES = 120

        def init():
            curve_line.set_data([], [])
            tangent_line.set_data([], [])
            dot.set_data([], [])
            info_text.set_text("")
            return curve_line, tangent_line, dot, info_text

        def update(frame):
            # Phase 1 (0-40): draw the curve progressively
            if frame <= 40:
                idx = max(1, int(len(x) * frame / 40))
                curve_line.set_data(x[:idx], y[:idx])
                tangent_line.set_data([], [])
                dot.set_data([], [])
                info_text.set_text("Drawing f(x)...")
            # Phase 2 (41-80): sweep a point + tangent along the curve
            elif frame <= 80:
                curve_line.set_data(x, y)
                t = (frame - 41) / 39.0
                xi = x_min + t * (x_max - x_min)
                yi = float(f_num(xi))
                dot.set_data([xi], [yi])
                if d_num is not None:
                    slope = float(d_num(xi))
                    tx = np.array([xi - 1.5, xi + 1.5])
                    ty = yi + slope * (tx - xi)
                    tangent_line.set_data(tx, ty)
                    info_text.set_text(
                        f"x = {xi:.2f}\nf(x) = {yi:.2f}\nf'(x) = {slope:.2f}"
                    )
                else:
                    info_text.set_text(f"x = {xi:.2f}\nf(x) = {yi:.2f}")
            # Phase 3 (81-120): shade area under curve from x_min to 0
            else:
                curve_line.set_data(x, y)
                tangent_line.set_data([], [])
                dot.set_data([], [])
                if sweep_fill[0] is not None:
                    sweep_fill[0].remove()
                t = (frame - 81) / 39.0
                x_end = x_min + t * (0 - x_min)
                mask = x <= x_end
                if mask.any():
                    sweep_fill[0] = ax.fill_between(
                        x[mask], 0, y[mask],
                        alpha=0.35, color="#cba6f7", zorder=2
                    )
                info_text.set_text("Integral region (shaded)")
            return curve_line, tangent_line, dot, info_text

        ani = animation.FuncAnimation(
            fig, update, frames=FRAMES, init_func=init,
            interval=50, blit=False
        )

        writer = animation.FFMpegWriter(fps=24, bitrate=1200,
                                        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
        ani.save(out_path, writer=writer, dpi=120)
        plt.close(fig)
        log.info(f"Local render saved: {out_path}")
        return out_path

    except Exception as e:
        log.error(f"Local matplotlib render failed: {e}", exc_info=True)
        plt.close("all")
        return None


# --- 8. SANDBOXED DOCKER EXECUTION & HEALING ---
def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _find_rendered_video(host_dir: str) -> Optional[str]:
    """Manim's actual output path depends on quality flag and scene name --
    search for it rather than hardcoding a guess."""
    media_dir = os.path.join(host_dir, "media", "videos")
    if not os.path.isdir(media_dir):
        return None
    for root, _, files in os.walk(media_dir):
        for f in files:
            if f.endswith(".mp4"):
                return os.path.join(root, f)
    return None


def _syntax_check(code: str) -> Optional[str]:
    """Cheap pre-flight: catch syntax errors before paying for a Docker render."""
    try:
        compile(code, "<manim_scene>", "exec")
        return None
    except SyntaxError as e:
        return f"SyntaxError before render: line {e.lineno}: {e.msg}"


def execute_and_heal_pipeline(initial_code: str, job_id: str,
                              facts: Optional[MathFacts] = None,
                              params: Optional[dict] = None,
                              max_retries: int = MAX_RETRIES) -> Optional[str]:
    """Manim (Docker) is the PRIMARY renderer -- it's the whole point of the
    pipeline. The generic matplotlib animation is only a last-resort fallback
    when Docker is unavailable or every heal attempt is exhausted."""
    if not _docker_available():
        log.warning("Docker not available -- falling back to generic matplotlib render.")
        if facts is not None:
            video_path = render_locally_matplotlib(facts, job_id, params)
            if video_path:
                update_job_state(job_id, "completed_fallback", video_path=video_path)
                return video_path
        update_job_state(job_id, "failed", error="docker_not_available")
        return None

    current_code = initial_code
    attempt_history: list = []
    host_dir = os.path.abspath(f"./tmp_{job_id}")
    os.makedirs(host_dir, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        log.info(f"Rendering Attempt {attempt}/{max_retries} [Docker Sandbox]...")

        # Don't waste a Docker render on code that can't even compile.
        syntax_err = _syntax_check(current_code)
        if syntax_err:
            log.warning(syntax_err)
            attempt_history.append(syntax_err)
            if attempt == max_retries:
                break
            try:
                current_code = heal_code(current_code, syntax_err, attempt_history[:-1])
            except Exception as e:
                log.error(f"Healing step failed: {e}")
            continue

        file_path = os.path.join(host_dir, "script.py")
        with open(file_path, "w") as f:
            f.write(current_code)

        command = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", "512m",
            "--cpus", "1.0",
            "-v", f"{host_dir}:/manim",
            "manimcommunity/manim",
            "manim", "-ql", "/manim/script.py", "MathScene"
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            log.warning("Render timed out.")
            attempt_history.append("render_timeout (120s)")
            if attempt == max_retries:
                break
            continue

        if result.returncode == 0:
            video_path = _find_rendered_video(host_dir)
            if video_path:
                log.info(f"Render completed: {video_path}")
                update_job_state(job_id, "completed", current_code, video_path)
                return video_path
            log.warning("Render reported success but no .mp4 was found.")

        error_log = result.stderr or result.stdout or "unknown_error"
        log.warning("Sandbox render failed.")
        attempt_history.append(error_log[-500:])
        if attempt == max_retries:
            break

        update_job_state(job_id, f"healing_attempt_{attempt}")
        try:
            current_code = heal_code(current_code, error_log, attempt_history[:-1])
        except Exception as e:
            # A failed heal call should burn this retry, not kill the job.
            log.error(f"Healing step failed (continuing to next attempt): {e}")

    # All Manim attempts exhausted -> generic matplotlib fallback so the user
    # still gets SOMETHING, clearly marked as a fallback render.
    log.warning("All Manim render attempts failed. Trying matplotlib fallback...")
    if facts is not None:
        video_path = render_locally_matplotlib(facts, job_id, params)
        if video_path:
            update_job_state(job_id, "completed_fallback", code=current_code, video_path=video_path)
            return video_path

    last_error = attempt_history[-1] if attempt_history else "unknown"
    update_job_state(job_id, "failed", code=current_code, error=last_error[-2000:])
    return None


# --- 9. ASYNC WORKER (Job Queue Simulation) ---
def process_job(job_id: str, user_math_request: str, params: dict):
    """The function executed by a Celery worker (simulated here via ThreadPool)."""
    try:
        update_job_state(job_id, "analyzing")
        facts = analyze_mathematical_properties(user_math_request)

        # Canonical cache key: hash the SymPy srepr, not the raw string, so
        # "x^2", " x^2 " and "x**2" all hit the same cache entry.
        canonical = facts.expression if facts.parsed_clean else user_math_request.strip()
        cache_key = hashlib.sha256(
            (canonical + json.dumps(params, sort_keys=True)).encode()
        ).hexdigest()
        cached_video = get_cache(cache_key)
        if cached_video and os.path.exists(cached_video):
            log.info(f"Cache hit for {job_id}.")
            update_job_state(job_id, "completed", video_path=cached_video)
            return

        update_job_state(job_id, "coding")
        generated_code = generate_code_pipeline(facts, params)

        # Act on validation instead of just logging it: one regeneration pass
        # with explicit feedback about what was wrong.
        if not validate_math_invariants(generated_code, facts):
            log.warning("Semantic validation failed -- regenerating with feedback...")
            update_job_state(job_id, "regenerating")
            generated_code = generate_code_pipeline(
                facts, params,
                feedback=(f"The scene must actually use the correct derivative "
                          f"'{facts.derivative}' of the function. The previous code did not.")
            )
            if not validate_math_invariants(generated_code, facts):
                log.warning("Regenerated code still fails validation; proceeding to render anyway.")

        log.info(f"\n{'='*60}\nGENERATED MANIM CODE for {job_id}:\n{'='*60}\n{generated_code}\n{'='*60}\n")

        update_job_state(job_id, "rendering")
        video_path = execute_and_heal_pipeline(generated_code, job_id, facts=facts, params=params)

        if video_path:
            set_cache(cache_key, video_path)
            log.info(f"\n{'='*60}\nVIDEO SAVED: {video_path}\n{'='*60}")

    except Exception as e:
        log.error(f"Job {job_id} crashed: {e}", exc_info=True)
        update_job_state(job_id, "system_error", error=str(e))


# --- EXECUTION ---
if __name__ == "__main__":
    init_db()

    requests = [
        {"job_id": "job_001", "eq": "Plot the quadratic equation x^2 - 4x + 3"},
    ]

    ui_params = {"x_range": [-5, 5, 1]}

    log.info("Pushing tasks to worker queue...")
    with ThreadPoolExecutor(max_workers=1) as worker_pool:
        futures = [worker_pool.submit(process_job, req["job_id"], req["eq"], ui_params) for req in requests]
        for f in futures:
            f.result()  # surface exceptions instead of swallowing them silently

    log.info("All workers finished processing queue.")
