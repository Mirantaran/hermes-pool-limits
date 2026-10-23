#!/usr/bin/env python3
"""Remaining quota of every account in a CLIProxyAPI pool, as one compact JSON line.

Run on the machine that hosts the pool (the Hermes backend). The Desktop chip calls it through
the gateway's ``shell.exec`` RPC, so updating this file needs no backend restart.

STRICTLY READ-ONLY. It reads CLIProxyAPI's auth files and asks each provider's own usage
endpoint with the access token exactly as stored. It never refreshes a token (refresh tokens
are single-use — rotating one here would log the account out of the proxy), never writes a
file, and never prints a token: only label, plan, window percentages and reset times leave it.

API-key subscriptions are reported too, without any auth files: an ``openai-compatibility``
provider of the running proxy whose base-url host is known (Zhipu GLM) is polled at that
provider's own quota endpoint, and the Kimi key is taken from ``KIMI_API_KEY`` in the
Hermes ``.env`` — again exactly as stored, never printed.

stdlib only, so it runs under whatever ``python3`` the host has.

Output (keys are short because ``shell.exec`` returns only the last 4000 chars of stdout):
    {"ok": true, "n": 6, "accounts": [
        {"i": 1, "t": "codex", "l": "user@example.com", "p": "plus", "off": false,
         "w": [["Session", 12.5, 1790000000], ["Weekly", 95.0, 1790400000]], "d": [], "e": null}]}
``w`` rows are [label, used_percent | null, reset_epoch | null]; ``e`` is a short error or null.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

HTTP_TIMEOUT_S = 10
MAX_WORKERS = 8
MAX_ACCOUNTS = 24  # keeps the line under shell.exec's 4000-char stdout tail
SUPPORTED_TYPES = ("codex", "claude")

CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
CODEX_WINDOWS_BY_SECONDS = {18000: "Session", 604800: "Weekly"}
CODEX_WINDOWS_POSITIONAL = (("primary_window", "Session"), ("secondary_window", "Weekly"))

CLAUDE_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
CLAUDE_WINDOWS = (("five_hour", "Session"), ("seven_day", "Weekly"),
                  ("seven_day_opus", "Opus week"), ("seven_day_sonnet", "Sonnet week"))

_PROXY_PROCESS = re.compile(r"cli-?proxy", re.I)
_AUTH_DIR_LINE = re.compile(r"^\s*auth-dir\s*:\s*(.+?)\s*(?:#.*)?$", re.M)
_SKIP_DIRS = {"node_modules", ".git", "__pycache__", "sessions", "logs", "cache", "skills", "venv", ".venv",
              "backups", "image_cache", "audio_cache", "site-packages"}


# ── locating the pool ────────────────────────────────────────────────


def _read(path: Path, limit: int = 1_000_000) -> str:
    try:
        with open(path, "rb") as fh:
            return fh.read(limit).decode("utf-8", "replace")
    except OSError:
        return ""


def _proxy_processes() -> Iterable[Path]:
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for entry in proc.iterdir():
        if entry.name.isdigit() and _PROXY_PROCESS.search(_read(entry / "cmdline", 4096).replace("\0", " ")):
            yield entry


def _proxy_config(entry: Path) -> Optional[Path]:
    """The ``--config`` file of a running CLIProxyAPI (else ``<cwd>/config.yaml``)."""
    argv = [a for a in _read(entry / "cmdline", 4096).split("\0") if a]
    try:
        cwd = Path(os.readlink(entry / "cwd"))
    except OSError:
        cwd = None
    config: Optional[Path] = None
    for i, arg in enumerate(argv):
        if arg in ("-config", "--config") and i + 1 < len(argv):
            config = Path(argv[i + 1])
        elif arg.startswith(("-config=", "--config=")):
            config = Path(arg.split("=", 1)[1])
    if config is None and cwd is not None:
        config = cwd / "config.yaml"
    if config is None:
        return None
    if not config.is_absolute() and cwd is not None:
        config = cwd / config
    return config


def _auth_dir_from_process(entry: Path) -> Optional[Path]:
    """``auth-dir`` of a running CLIProxyAPI: its ``-config`` file (else ``<cwd>/config.yaml``)."""
    config = _proxy_config(entry)
    if config is None:
        return None
    match = _AUTH_DIR_LINE.search(_read(config))
    raw = match.group(1).strip().strip("'\"") if match else "~/.cli-proxy-api"
    if raw.startswith("~"):
        env = dict(kv.split("=", 1) for kv in _read(entry / "environ", 65536).split("\0") if "=" in kv)
        raw = (env.get("HOME") or os.path.expanduser("~")) + raw[1:]
    path = Path(raw)
    if not path.is_absolute():
        path = config.parent / path
    return path


def _looks_like_auth_dir(path: Path) -> bool:
    try:
        return any(_load_auth_file(p) for p in sorted(path.glob("*.json"))[:40])
    except OSError:
        return False


def _search_auth_dir(roots: Iterable[Path], max_depth: int = 4, max_dirs: int = 4000) -> Optional[Path]:
    """Bounded fallback walk for a directory holding CLIProxyAPI auth files."""
    seen = 0
    stack = [(root, 0) for root in roots if root.is_dir()]
    while stack and seen < max_dirs:
        current, depth = stack.pop()
        seen += 1
        if _looks_like_auth_dir(current):
            return current
        if depth >= max_depth:
            continue
        try:
            children = [c for c in current.iterdir() if c.is_dir() and not c.is_symlink() and c.name not in _SKIP_DIRS]
        except OSError:
            continue
        # Likely names first (stack pops from the end).
        children.sort(key=lambda c: bool(re.search(r"proxy|auth|pool|codex", c.name, re.I)))
        stack.extend((c, depth + 1) for c in children)
    return None


def _pinned_auth_dir() -> Optional[str]:
    """First line of ``auth-dir.txt`` beside this script — a per-host override that
    survives ``git pull`` (the file is git-ignored)."""
    lines = _read(Path(__file__).resolve().parent / "auth-dir.txt", 4096).strip().splitlines()
    return lines[0].strip() if lines and lines[0].strip() else None


def find_auth_dir(explicit: Optional[str] = None) -> Optional[Path]:
    for raw in (explicit, os.environ.get("CLIPROXY_AUTH_DIR"), _pinned_auth_dir()):
        if raw:
            return Path(raw).expanduser()
    for entry in _proxy_processes():
        path = _auth_dir_from_process(entry)
        if path and path.is_dir():
            return path
    home = Path(os.path.expanduser("~"))
    # $HERMES_HOME, else the default ~/.hermes; /opt/data is the official Docker image's home.
    hermes_homes = list(dict.fromkeys(Path(p) for p in (os.environ.get("HERMES_HOME"), home / ".hermes", "/opt/data") if p))
    candidates = [home / ".cli-proxy-api"]
    for hermes_home in hermes_homes:
        # ``cliproxy/auths`` is the layout of github.com/artemiimillier/hermes-codex-pool
        candidates += [hermes_home / "cliproxy" / "auths", hermes_home / ".cli-proxy-api"]
    for path in candidates:
        if _looks_like_auth_dir(path):
            return path
    return _search_auth_dir([h for h in hermes_homes if h.is_dir()] + [home])


# ── reading accounts ─────────────────────────────────────────────────


def _load_auth_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(_read(path))
    except ValueError:
        return None
    if not isinstance(data, dict) or str(data.get("type", "")).lower() not in SUPPORTED_TYPES:
        return None
    return data if isinstance(data.get("access_token"), str) and data["access_token"] else None


def _label(data: Dict[str, Any], path: Path) -> str:
    for key in ("email", "label", "name"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return path.stem


# ── provider usage endpoints ─────────────────────────────────────────


def _get_json(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:  # noqa: S310 — fixed https URLs
        return json.loads(response.read().decode("utf-8", "replace") or "{}") or {}


def _epoch(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return int(value / 1000 if value > 1e11 else value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return int((parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp())


def _percent(value: Any, *, fraction: bool = False) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    used = float(value) * 100 if fraction and value <= 1 else float(value)
    return round(max(0.0, min(100.0, used)), 1)


def _codex_usage(data: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {data['access_token']}", "Accept": "application/json",
               "User-Agent": "codex-cli"}
    if isinstance(data.get("account_id"), str) and data["account_id"]:
        headers["ChatGPT-Account-ID"] = data["account_id"]
    payload = _get_json(CODEX_USAGE_URL, headers)
    rate_limit = payload.get("rate_limit") or {}
    windows = []
    for key, fallback in CODEX_WINDOWS_POSITIONAL:
        window = rate_limit.get(key)
        if not isinstance(window, dict) or window.get("used_percent") is None:
            continue
        seconds = window.get("limit_window_seconds")
        label = CODEX_WINDOWS_BY_SECONDS.get(int(seconds), fallback) if isinstance(seconds, (int, float)) else fallback
        reset = _epoch(window.get("reset_at"))
        if reset is None and isinstance(window.get("reset_after_seconds"), (int, float)):
            reset = int(datetime.now(timezone.utc).timestamp() + window["reset_after_seconds"])
        windows.append([label, _percent(window.get("used_percent")), reset])
    details = []
    credits = payload.get("credits") or {}
    if credits.get("has_credits") and isinstance(credits.get("balance"), (int, float)):
        details.append(f"Credits: ${float(credits['balance']):.2f}")
    return {"p": payload.get("plan_type"), "w": windows, "d": details}


def _claude_usage(data: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {data['access_token']}", "Accept": "application/json",
               "Content-Type": "application/json", "anthropic-beta": "oauth-2025-04-20",
               "User-Agent": "claude-code/2.1.0"}
    payload = _get_json(CLAUDE_USAGE_URL, headers)
    windows = []
    for key, label in CLAUDE_WINDOWS:
        window = payload.get(key)
        if isinstance(window, dict) and window.get("utilization") is not None:
            windows.append([label, _percent(window.get("utilization"), fraction=True), _epoch(window.get("resets_at"))])
    details = []
    extra = payload.get("extra_usage") or {}
    if extra.get("is_enabled") and isinstance(extra.get("used_credits"), (int, float)) \
            and isinstance(extra.get("monthly_limit"), (int, float)):
        details.append(f"Extra usage: {extra['used_credits']:.2f} / {extra['monthly_limit']:.2f} "
                       f"{extra.get('currency') or 'USD'}")
    return {"p": None, "w": windows, "d": details}


_FETCHERS = {"codex": _codex_usage, "claude": _claude_usage}


# ── API-key subscriptions (GLM, Kimi) ────────────────────────────────

GLM_QUOTA_PATH = "/api/monitor/usage/quota/limit"   # GET, Bearer key; percentage = used %
GLM_UNIT_SECONDS = {3: 3600, 5: 86400, 6: 604800}   # observed unit codes: hour / day / week
KIMI_USAGE_URL = "https://api.kimi.com/coding/v1/usages"
KIMI_USER_AGENT = "KimiCLI/1.6"                     # without it the endpoint answers 404
_GLM_HOST = re.compile(r"(^|\.)z\.ai$|bigmodel\.cn$", re.I)
_KIMI_HOST = re.compile(r"kimi\.(com|ai)$", re.I)


def _openai_compat_entries(text: str) -> List[Tuple[str, str, List[str]]]:
    """``(name, base-url, [api-key, …])`` of the config's ``openai-compatibility`` block.

    PyYAML when importable; otherwise a line parser for the block's fixed shape
    (``openai-compatibility:`` opens the block, ``- name:`` starts an entry,
    ``base-url:`` / ``- api-key:`` are its fields, any other indent-0 key closes it).
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        try:
            data = yaml.safe_load(text) or {}
        except Exception:
            data = {}
        out = []
        for item in data.get("openai-compatibility") or []:
            if not isinstance(item, dict):
                continue
            keys = [k["api-key"] for k in (item.get("api-key-entries") or [])
                    if isinstance(k, dict) and isinstance(k.get("api-key"), str) and k["api-key"]]
            base = item.get("base-url")
            if isinstance(base, str) and base and keys:
                out.append((str(item.get("name") or ""), base, keys))
        return out

    entries: List[Tuple[str, str, List[str]]] = []
    inside = False
    name = base = ""
    keys: List[str] = []

    def flush() -> None:
        if base and keys:
            entries.append((name, base, keys))

    for line in text.splitlines():
        if not line.strip():
            continue
        stripped = line.strip()
        if not line[0].isspace():
            flush()
            inside = stripped.rstrip(":") == "openai-compatibility"
            name = base = ""
            keys = []
            continue
        if not inside:
            continue
        if stripped.startswith("- name:"):
            flush()
            name, base, keys = stripped[len("- name:"):].strip().strip("'\""), "", []
        elif base == "" and stripped.startswith("base-url:"):
            base = stripped[len("base-url:"):].strip().strip("'\"")
        elif stripped.startswith(("- api-key:", "api-key:")):
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            if value:
                keys.append(value)
    flush()
    return entries


def _proxy_config_file() -> Optional[Path]:
    for entry in _proxy_processes():
        config = _proxy_config(entry)
        if config and config.is_file():
            return config
    return None


def _hermes_env_key(name: str) -> Optional[str]:
    """``NAME=…`` from the Hermes ``.env`` ($HERMES_HOME, else ~/.hermes)."""
    homes = [os.environ.get("HERMES_HOME"), str(Path.home() / ".hermes")]
    for home in dict.fromkeys(h for h in homes if h):
        match = re.search(rf"^{name}=(.+)$", _read(Path(home) / ".env"), re.M)
        if match:
            return match.group(1).strip().strip("'\"")
    return None


def _glm_usage(key: str, origin: str) -> Dict[str, Any]:
    payload = _get_json(origin + GLM_QUOTA_PATH, {"Authorization": f"Bearer {key}",
                                                  "Accept": "application/json"})
    windows = []
    for limit in (payload.get("data") or {}).get("limits") or []:
        if not isinstance(limit, dict) or limit.get("percentage") is None:
            continue
        unit = limit.get("unit")
        seconds = GLM_UNIT_SECONDS.get(unit, 0) * int(limit.get("number") or 1) if isinstance(unit, int) else 0
        if limit.get("type") == "TIME_LIMIT":
            label = "Инструменты"          # search-prime / web-reader / zread share this one
        elif seconds == 18000:
            label = "5 часов"
        elif seconds == 604800:
            label = "Неделя"
        else:
            label = f"Окно {limit.get('number')}×{limit.get('unit')}"
        windows.append([label, _percent(limit.get("percentage")), _epoch(limit.get("nextResetTime"))])
    return {"p": "coding plan", "w": windows, "d": []}


def _kimi_usage(key: str) -> Dict[str, Any]:
    payload = _get_json(KIMI_USAGE_URL, {"Authorization": f"Bearer {key}",
                                         "Accept": "application/json",
                                         "User-Agent": KIMI_USER_AGENT})
    usages = payload.get("usages") or {}
    windows = []
    for field, label in (("limit_5h", "5 часов"), ("limit_month_total", "Месяц")):
        entry = usages.get(field)
        if isinstance(entry, dict) and entry.get("used_ratio") is not None:
            windows.append([label, _percent(entry.get("used_ratio"), fraction=True),
                            _epoch(entry.get("reset_time"))])
    code = usages.get("limit_month_code")
    total = usages.get("limit_month_total")
    if isinstance(code, dict) and isinstance(total, dict) \
            and code.get("used_ratio") is not None and code.get("used_ratio") != total.get("used_ratio"):
        windows.append(["Код, месяц", _percent(code.get("used_ratio"), fraction=True),
                        _epoch(code.get("reset_time"))])
    details = []
    parallel = payload.get("parallel") or {}
    if isinstance(parallel.get("limit"), (int, float)):
        details.append(f"Параллельных запросов: {int(parallel['limit'])}")
    return {"p": "coding plan", "w": windows, "d": details}


def _api_subscriptions() -> List[Tuple[str, str, Dict[str, Any]]]:
    """``(kind, label, fetcher_data)`` for every API-key subscription found on the host."""
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    config = _proxy_config_file()
    if config:
        for name, base, keys in _openai_compat_entries(_read(config)):
            origin = f"{urlparse(base).scheme}://{urlparse(base).netloc}"
            if _GLM_HOST.search(urlparse(base).netloc):
                for key in keys:
                    out.append(("glm", name or "GLM", {"origin": origin, "key": key}))
            elif _KIMI_HOST.search(urlparse(base).netloc):
                for key in keys:  # a future Kimi entry inside the proxy config
                    out.append(("kimi", name or "Kimi", {"key": key}))
    env_key = os.environ.get("KIMI_API_KEY") or _hermes_env_key("KIMI_API_KEY")
    if env_key:
        out.append(("kimi", "Kimi", {"key": env_key}))
    return out


def _api_account(index: int, kind: str, label: str, data: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"i": index, "t": kind, "l": label[:64], "p": None,
                           "off": False, "w": [], "d": [], "e": None}
    try:
        if kind == "glm":
            row.update(_glm_usage(data["key"], data["origin"]))
        elif kind == "kimi":
            row.update(_kimi_usage(data["key"]))
    except urllib.error.HTTPError as exc:
        row["e"] = "key rejected" if exc.code in (401, 403) else f"HTTP {exc.code}"
    except Exception as exc:
        row["e"] = type(exc).__name__
    return row


def _account(index: int, path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(data.get("type", "")).lower()
    row: Dict[str, Any] = {"i": index, "t": kind, "l": _label(data, path)[:64], "p": None,
                           "off": bool(data.get("disabled")), "w": [], "d": [], "e": None}
    if row["off"]:
        return row  # the proxy does not route to it; don't spend a request on it
    try:
        row.update(_FETCHERS[kind](data))
    except urllib.error.HTTPError as exc:
        # 401 = the stored access token has lapsed; the proxy refreshes it on its own schedule.
        row["e"] = "token expired (proxy will refresh it)" if exc.code == 401 else f"HTTP {exc.code}"
    except Exception as exc:  # network, JSON, anything — one bad account must not sink the rest
        row["e"] = type(exc).__name__
    return row


def collect(auth_dir: Optional[str] = None) -> Dict[str, Any]:
    directory = find_auth_dir(auth_dir)
    files: List[Tuple[Path, Dict[str, Any]]] = []
    if directory is not None and directory.is_dir():
        files = [(path, data) for path in sorted(directory.glob("*.json")) if (data := _load_auth_file(path))]
    subs = _api_subscriptions()
    if not files and not subs:
        if directory is None or not directory.is_dir():
            return {"ok": False, "error": "CLIProxyAPI auth directory not found", "accounts": []}
        return {"ok": False, "error": "no codex/claude auth files in the pool directory", "accounts": []}
    accounts: List[Dict[str, Any]] = []
    workers = min(MAX_WORKERS, len(files) + len(subs))
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        if files:
            accounts += list(pool.map(lambda item: _account(item[0], item[1][0], item[1][1]),
                                      enumerate(files[:MAX_ACCOUNTS], 1)))
        start = len(accounts) + 1
        accounts += list(pool.map(lambda item: _api_account(start + item[0], item[1][0], item[1][1], item[1][2]),
                                  enumerate(subs[:MAX_ACCOUNTS - len(accounts)])))
    return {"ok": True, "n": len(files) + len(subs), "accounts": accounts}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--auth-dir", default=None, help="CLIProxyAPI auth directory (default: auto-detect)")
    args = parser.parse_args(argv)
    try:
        result = collect(args.auth_dir)
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200], "accounts": []}
    sys.stdout.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
