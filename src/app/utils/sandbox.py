from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


_BLOCKED_MODULES = {
    "builtins",
    "ctypes",
    "importlib",
    "io",
    "os",
    "pathlib",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
}

_BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "help",
    "input",
    "open",
}

_ALLOWED_IMPORTS = {
    "type1": {"z3"},
    "type2": {"math", "sympy"},
}

_PRELUDES = {
    "type1": "from z3 import *",
    "type2": "\n".join(
        [
            "import math",
            "try:",
            "    import sympy as sp",
            "except Exception:",
            "    sp = None",
        ]
    ),
}


def _called_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def validate_python_code(code: str, *, sandbox_type: str) -> str | None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"syntax error: {exc}"

    allowed_imports = _ALLOWED_IMPORTS.get(sandbox_type, set())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed_imports:
                    return f"blocked import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if module not in allowed_imports:
                return f"blocked import: {node.module or ''}"
        elif isinstance(node, ast.Call):
            name = _called_name(node.func)
            if name in _BLOCKED_CALLS:
                return f"blocked call: {name}"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return f"blocked attribute: {node.attr}"

    return None


def run_python_code(code: str, *, sandbox_type: str, timeout_seconds: float) -> SandboxResult:
    """Run model-generated Python in a short-lived subprocess.

    This is a lightweight inference-time sandbox for competition physics solvers.
    It is not a security boundary for untrusted public users; keep the endpoint private-ish
    to the evaluator and avoid exposing arbitrary code execution to the open internet.
    """
    code = (code or "").strip()
    if not code:
        return SandboxResult(False, "", "empty python_code", 1)

    validation_error = validate_python_code(code, sandbox_type=sandbox_type)
    if validation_error:
        return SandboxResult(False, "", validation_error, 1)

    prelude = _PRELUDES.get(sandbox_type)
    if not prelude:
        return SandboxResult(False, "", f"unknown sandbox_type: {sandbox_type}", 1)

    full_code = prelude + "\n\n" + code + "\n"
    with tempfile.TemporaryDirectory(prefix="exact_sandbox_") as td:
        script = os.path.join(td, "solve.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(full_code)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", script],
                cwd=td,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                env={"PYTHONPATH": "", "PATH": os.environ.get("PATH", "")},
            )
            return SandboxResult(
                ok=proc.returncode == 0 and bool(proc.stdout.strip()),
                stdout=proc.stdout.strip(),
                stderr=proc.stderr.strip(),
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(False, exc.stdout or "", exc.stderr or "timeout", 124, timed_out=True)


def last_stdout_value(stdout: str) -> str:
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""
