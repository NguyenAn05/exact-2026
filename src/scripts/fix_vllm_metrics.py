"""
Robust fix for the vLLM/prometheus_fastapi_instrumentator route crash.

The exact vLLM file layout has moved across releases, so this script must:
  1. Patch prometheus_fastapi_instrumentator defensively.
  2. Patch any matching vLLM metrics entrypoint if it exists.
  3. Never fail the image build just because one historical path disappeared.
"""

from pathlib import Path
import sys


def _find_site_packages() -> Path:
    for p in sys.path:
        candidate = Path(p)
        if candidate.name == "site-packages":
            return candidate
    return Path("/usr/local/lib/python3.12/site-packages")


def patch_prometheus_routing(site: Path) -> bool:
    """Replace routing.py with a version that skips non-route objects."""
    target = site / "prometheus_fastapi_instrumentator" / "routing.py"
    if not target.exists():
        print(f"[fix] routing.py not found at {target}, skipping")
        return False

    patched_code = '''"""Patched routing helpers that ignore vLLM _IncludedRouter objects."""
from typing import Any, Dict, Optional, Sequence


def _route_path(route: Any) -> Optional[str]:
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else None


def _get_route_name(scope: Dict[str, Any], routes: Sequence[Any]) -> Optional[str]:
    for route in routes:
        path = _route_path(route)
        if path is None:
            continue
        if path == scope.get("path"):
            return path
        matches = getattr(route, "matches", None)
        if matches is None:
            continue
        try:
            match, child_scope = matches(scope)
        except Exception:
            continue
        matched_route = child_scope.get("route") if isinstance(child_scope, dict) else None
        matched_path = _route_path(matched_route)
        if matched_path:
            return matched_path
    return None


def get_route_name(request: Any) -> Optional[str]:
    app = getattr(request, "app", None)
    if app is None:
        return None
    routes = getattr(app, "routes", None)
    if routes is None:
        return None
    scope = getattr(request, "scope", None)
    if not isinstance(scope, dict):
        return None
    return _get_route_name(scope, routes)
'''
    target.write_text(patched_code, encoding="utf-8")
    print(f"[fix] Patched {target}")
    return True


def _iter_vllm_metric_candidates(site: Path) -> list[Path]:
    base = site / "vllm"
    if not base.exists():
        return []
    explicit = [
        base / "entrypoints" / "serve" / "instrumentator" / "metrics.py",
        base / "entrypoints" / "openai" / "instrumentator.py",
        base / "entrypoints" / "openai" / "metrics.py",
    ]
    discovered: list[Path] = []
    for candidate in explicit:
        if candidate.exists():
            discovered.append(candidate)
    for candidate in base.rglob("*.py"):
        if "instrument" not in str(candidate).lower() and "metrics" not in str(candidate).lower():
            continue
        try:
            content = candidate.read_text("utf-8")
        except Exception:
            continue
        if "get_prometheus_registry" in content and "attach_router" in content:
            if candidate not in discovered:
                discovered.append(candidate)
    return discovered


def patch_vllm_metrics(site: Path) -> bool:
    candidates = _iter_vllm_metric_candidates(site)
    patched = False
    for target in candidates:
        backup = target.with_suffix(target.suffix + ".bak")
        content = target.read_text("utf-8")
        if not backup.exists():
            backup.write_text(content, encoding="utf-8")
        if "get_prometheus_registry" in content and "attach_router" in content:
            new_content = '''import prometheus_client
import regex as re
from fastapi import FastAPI, Response
from prometheus_client import make_asgi_app
from starlette.routing import Mount
from vllm.v1.metrics.prometheus import get_prometheus_registry


class PrometheusResponse(Response):
    media_type = prometheus_client.CONTENT_TYPE_LATEST


def attach_router(app: FastAPI):
    registry = get_prometheus_registry()
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
'''
            target.write_text(new_content, encoding="utf-8")
            print(f"[fix] Patched {target}")
            patched = True
    return patched


def main() -> None:
    site = _find_site_packages()
    print(f"[fix] Site-packages: {site}")

    p1 = patch_prometheus_routing(site)
    p2 = patch_vllm_metrics(site)

    if p1:
        print("[fix] prometheus_fastapi_instrumentator routing.py patched")
    if p2:
        print("[fix] vLLM metrics file patched")
    if not p1 and not p2:
        print("[fix] WARNING: No patches applied; vLLM may still crash on /v1/completions")
    print("[fix] Done")


if __name__ == "__main__":
    main()
