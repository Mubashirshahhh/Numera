import os
import re
import json
import hashlib
import logging
import sqlite3
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# Load .env FIRST
from dotenv import load_dotenv
load_dotenv()

import sympy as sp
from openai import OpenAI
from anthropic import Anthropic

# --- 1. OBSERVABILITY & CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# Secrets must come from the environment (.env / secret manager), never literals in source.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Export it before starting the worker.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

DB_PATH = os.environ.get("PRODUCTION_DB_PATH", "production_state.db")
MAX_RETRIES = int(os.environ.get("RENDER_MAX_RETRIES", 3))

MANIM_TEMPLATE_LIBRARY = """
Template 1: GraphScene Base
Use Axes(), ax.plot(), and ValueTracker() for dynamic scaling and plotting.
The scene class must be named MathScene and subclass Scene.
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
def call_llm(prompt: str, system_role: str, temperature: float = 0.1) -> str:
    """Tries OpenAI first; falls back to Anthropic if it's configured and the
    primary call fails. Raises if both are unavailable, so callers never
    silently treat an error string as real model output."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        log.warning(f"Primary LLM (OpenAI) failed: {e}")
        if anthropic_client is None:
            raise RuntimeError(f"Primary LLM failed and no fallback configured: {e}") from e
        try:
            log.info("Falling back to Anthropic...")
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                temperature=temperature,
                system=system_role,
                messages=[{"role": "user", "content": prompt}]
            )
            return "".join(block.text for block in response.content if block.type == "text")
        except Exception as fallback_err:
            raise RuntimeError(f"Both LLM providers failed. Primary: {e}. Fallback: {fallback_err}") from fallback_err


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


def extract_clean_expression(user_input: str) -> str:
    """Natural language varies too much for regex alone (function names like
    sin/cos/log, phrasing like 'derivative of ...'). The LLM's only job here
    is translation to SymPy-parseable notation -- SymPy remains the source of
    truth for everything mathematical."""
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
        expr_str = extract_clean_expression(user_input)

        if "=" in expr_str:
            lhs_str, rhs_str = expr_str.split("=", 1)
            expr = sp.sympify(lhs_str.strip()) - sp.sympify(rhs_str.strip())
        else:
            expr = sp.sympify(expr_str)

        free_syms = sorted(expr.free_symbols, key=str)
        primary_var = free_syms[0] if free_syms else None

        facts = MathFacts(
            raw_input=user_input,
            parsed_clean=True,
            expression=str(expr),
            variable=str(primary_var) if primary_var else None,
        )
        if primary_var:
            facts.derivative = str(sp.simplify(sp.diff(expr, primary_var)))
            facts.integral = str(sp.simplify(sp.integrate(expr, primary_var)))
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
        expected = sp.sympify(expected_expr)
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
            cand_expr = sp.sympify(cand)
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
def generate_code_pipeline(facts: MathFacts, params: dict) -> str:
    report = call_llm(f"Evaluate: {json.dumps(facts.to_dict())}", "You are a Mathematical Analyst.")
    visuals = call_llm(f"Design scene for:\n{report}", "You are an Educational Visual Director.", 0.3)

    prompt = (
        f"Script:\n{visuals}\n\nTemplates:\n{MANIM_TEMPLATE_LIBRARY}\n\nParams:\n{json.dumps(params)}\n\n"
        "Output a complete Manim CommunityEdition scene named MathScene (subclass of Scene). "
        "Output ONLY python code in a single ```python code block."
    )
    raw_output = call_llm(prompt, "You are an Expert Manim Programmer. Output ONLY python code in a block.")
    return extract_code_block(raw_output)


def heal_code(broken_code: str, error_log: str) -> str:
    prompt = f"Broken code:\n{broken_code}\n\nError:\n{error_log}\n\nFix the code so it renders successfully. Output ONLY the corrected python code in a ```python block."
    raw_output = call_llm(prompt, "You are an Automated Code Reviewer fixing Manim errors.")
    return extract_code_block(raw_output)


# --- 7. SANDBOXED EXECUTION & HEALING ---
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


def execute_and_heal_pipeline(initial_code: str, job_id: str, max_retries: int = MAX_RETRIES) -> Optional[str]:
    """Runs code inside an isolated, resource-constrained Docker container.
    Returns the real video path on success, or None on failure."""
    if not _docker_available():
        log.error("Docker is not installed or not on PATH. Cannot render.")
        update_job_state(job_id, "failed", error="docker_not_available")
        return None

    current_code = initial_code
    host_dir = os.path.abspath(f"./tmp_{job_id}")
    os.makedirs(host_dir, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        log.info(f"Rendering Attempt {attempt}/{max_retries} [Docker Sandbox]...")

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
            if attempt == max_retries:
                update_job_state(job_id, "failed", error="render_timeout")
                return None
            continue

        if result.returncode == 0:
            video_path = _find_rendered_video(host_dir)
            if video_path:
                log.info(f"Render completed: {video_path}")
                update_job_state(job_id, "completed", current_code, video_path)
                return video_path
            log.warning("Render reported success but no .mp4 was found.")

        error_log = result.stderr or result.stdout
        log.warning("Sandbox render failed.")
        if attempt == max_retries:
            update_job_state(job_id, "failed", code=current_code, error=error_log[-2000:])
            return None

        update_job_state(job_id, f"healing_attempt_{attempt}")
        try:
            current_code = heal_code(current_code, error_log)
        except Exception as e:
            log.error(f"Healing step itself failed: {e}")
            update_job_state(job_id, "failed", error=f"heal_failed: {e}")
            return None

    return None


# --- 8. ASYNC WORKER (Job Queue Simulation) ---
def process_job(job_id: str, user_math_request: str, params: dict):
    """The function executed by a Celery worker (simulated here via ThreadPool)."""
    try:
        cache_key = hashlib.sha256((user_math_request + json.dumps(params, sort_keys=True)).encode()).hexdigest()
        cached_video = get_cache(cache_key)
        if cached_video and os.path.exists(cached_video):
            log.info(f"Cache hit for {job_id}.")
            update_job_state(job_id, "completed", video_path=cached_video)
            return

        update_job_state(job_id, "analyzing")
        facts = analyze_mathematical_properties(user_math_request)

        update_job_state(job_id, "coding")
        generated_code = generate_code_pipeline(facts, params)

        if not validate_math_invariants(generated_code, facts):
            update_job_state(job_id, "failed_semantic_validation")
            return

        update_job_state(job_id, "rendering")
        video_path = execute_and_heal_pipeline(generated_code, job_id)

        if video_path:
            set_cache(cache_key, video_path)

    except Exception as e:
        log.error(f"Job {job_id} crashed: {e}", exc_info=True)
        update_job_state(job_id, "system_error", error=str(e))


# --- EXECUTION ---
if __name__ == "__main__":
    init_db()

    requests = [
        {"job_id": "job_001", "eq": "Evaluate derivative of x^2 + 3*x - 5"},
        {"job_id": "job_002", "eq": "Plot sin(x)"},
    ]

    ui_params = {"x_range": [-5, 5, 1]}

    log.info("Pushing tasks to worker queue...")
    with ThreadPoolExecutor(max_workers=2) as worker_pool:
        futures = [worker_pool.submit(process_job, req["job_id"], req["eq"], ui_params) for req in requests]
        for f in futures:
            f.result()  # surface exceptions instead of swallowing them silently

    log.info("All workers finished processing queue.")