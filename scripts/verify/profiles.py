"""Installed-wheel profile tests: the real install, not a simulation (042 T22).

``tests/unit/test_install_profiles.py`` blocks imports to imitate a base
install. This script does the real thing. It builds the wheel once, then for
each profile creates a throwaway venv, installs the wheel with that profile's
extras (the repository lock as constraints, so results are reproducible) and
runs the profile's checks inside the venv from an empty directory, so the
source tree cannot be imported by accident.

Each profile proves three things:

* the packages its extras provide are installed and import;
* the packages it does not provide are absent, ``semantic-kernel``,
  ``azure-ai-evaluation`` and ``evaluate`` in every profile (AC-02, AC-13);
* the features its extras carry work, and a feature whose extra is missing
  fails with the install command and no traceback (AC-01).

No model credentials, Docker or network beyond package downloads are needed.

Usage::

    uv run python scripts/verify/profiles.py                    # every profile
    uv run python scripts/verify/profiles.py --profile base --profile qdrant
    uv run python scripts/verify/profiles.py --group stores     # one CI group
    uv run python scripts/verify/profiles.py --keep             # keep the venvs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess  # noqa: S404 - runs fixed local tooling (uv, the venv's python)
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if not __package__:
    # Run as a file path (PEP 366): import the package so relative imports work.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.verify"  # noqa: A001

from .procs import ROOT, Check, scrubbed_env  # noqa: E402

PROBE = ROOT / "scripts" / "verify" / "profile_probe.py"
MARKER = "@@PROBE@@ "
DEFAULT_PYTHON = "3.11"  # the lowest supported: catches resolution differences
INSTALL_TIMEOUT = 1200.0
PROBE_TIMEOUT = 900.0
CLI_TIMEOUT = 180.0

# Command groups of ``holodeck --help``; the base smoke runs ``<group> --help``.
COMMAND_GROUPS = (
    "chat",
    "config",
    "deploy",
    "init",
    "mcp",
    "serve",
    "test",
    "vectorstore",
    "worker",
)
# Files that must be inside the wheel (templates, schema, dashboard assets).
BUNDLED_PATHS = (
    "templates/conversational",
    "templates/customer-support",
    "templates/research",
    "templates/conversational/.env.example.j2",
    "templates/customer-support/.env.example.j2",
    "templates/research/.env.example.j2",
    "dashboard/assets/01-tokens.css",
    "dashboard/assets/02-holodeck.css",
)

# -- What each extra provides ---------------------------------------------------

# Modules that import when the extra is installed. Every runtime extra in
# pyproject.toml needs a row: ``test_verify_profiles`` fails otherwise, so a new
# extra cannot land without a profile check.
EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "eval": ("deepeval", "anthropic", "ollama"),
    "nlp": ("nltk", "sacrebleu", "rouge_score"),
    "optimizer": ("optuna",),
    "opensearch": ("opensearchpy",),
    "pinecone": ("pinecone",),
    "azure-ai-search": ("azure.search.documents", "azure.identity"),
    "postgres": ("psycopg", "psycopg_pool", "pgvector"),
    "qdrant": ("qdrant_client",),
    "chromadb": ("chromadb",),
    "weaviate": ("weaviate",),
    "azure-cosmos": ("azure.cosmos",),
    "vectorstores": (
        "pinecone",
        "azure.search.documents",
        "azure.identity",
        "psycopg",
        "psycopg_pool",
        "pgvector",
        "qdrant_client",
        "chromadb",
        "weaviate",
        "azure.cosmos",
    ),
    "otel-prometheus": ("opentelemetry.exporter.prometheus", "prometheus_client"),
    "otel-azmon": ("azure.monitor.opentelemetry.exporter",),
    "otel-all": (
        "opentelemetry.exporter.prometheus",
        "prometheus_client",
        "azure.monitor.opentelemetry.exporter",
    ),
    "claude-otel": ("opentelemetry.instrumentation.claude_agent_sdk",),
    "openai-agents": ("agents", "openai"),
    "dashboard": ("dash", "plotly", "pandas"),
    "deploy": ("docker",),
    "deploy-aws": ("boto3",),
    "deploy-gcp": ("google.cloud.run_v2",),
    "deploy-azure": ("azure.mgmt.appcontainers", "azure.identity"),
    "deploy-all": (
        "docker",
        "boto3",
        "google.cloud.run_v2",
        "azure.mgmt.appcontainers",
        "azure.identity",
    ),
    "s3": ("boto3",),
    "azure-blob": ("azure.storage.blob",),
    "all-sources": ("boto3", "azure.storage.blob"),
    "temporal": ("temporalio",),
}
# The aggregate extra: every row above.
ALL_EXTRA = "all"


@dataclass(frozen=True)
class Package:
    """A package a base install must not have.

    ``extras`` lists the extras that legitimately provide it. An empty tuple
    means no extra does, in any profile.
    """

    module: str
    dist: str
    extras: tuple[str, ...] = ()


PACKAGES: tuple[Package, ...] = (
    # Never installed, in any profile (AC-02, AC-13).
    Package("semantic_kernel", "semantic-kernel"),
    Package("azure.ai.evaluation", "azure-ai-evaluation"),
    Package("evaluate", "evaluate"),
    Package("mkdocs", "mkdocs"),
    Package("twine", "twine"),
    # Installed only by their extra.
    Package("deepeval", "deepeval", ("eval",)),
    Package("anthropic", "anthropic", ("eval",)),
    Package("ollama", "ollama", ("eval",)),
    Package("nltk", "nltk", ("nlp",)),
    Package("sacrebleu", "sacrebleu", ("nlp",)),
    Package("rouge_score", "rouge-score", ("nlp",)),
    Package("optuna", "optuna", ("optimizer",)),
    Package("opensearchpy", "opensearch-py", ("opensearch",)),
    Package("dash", "dash", ("dashboard",)),
    Package("agents", "openai-agents", ("openai-agents",)),
    Package("temporalio", "temporalio", ("temporal",)),
    Package("qdrant_client", "qdrant-client", ("qdrant", "vectorstores")),
    Package("chromadb", "chromadb", ("chromadb", "vectorstores")),
    Package("psycopg", "psycopg", ("postgres", "vectorstores")),
    Package("pinecone", "pinecone", ("pinecone", "vectorstores")),
    Package("weaviate", "weaviate-client", ("weaviate", "vectorstores")),
    Package(
        "azure.search.documents",
        "azure-search-documents",
        ("azure-ai-search", "vectorstores"),
    ),
    Package("azure.cosmos", "azure-cosmos", ("azure-cosmos", "vectorstores")),
)

# Vector store extras and the provider each one carries.
STORE_PROVIDERS: dict[str, str] = {
    "qdrant": "qdrant",
    "chromadb": "chromadb",
    "postgres": "postgres",
    "pinecone": "pinecone",
    "weaviate": "weaviate",
    "azure-ai-search": "azure-ai-search",
    "azure-cosmos": "azure-cosmos-nosql",
}
# Embedded stores: a real upsert and vector search, no server.
ROUNDTRIP_STORES = ("qdrant", "chromadb")


@dataclass(frozen=True)
class Profile:
    """One install profile: the extras to install and the checks to run.

    ``probes`` run inside the venv in one interpreter; ``cli`` run as
    ``holodeck`` processes from an empty directory.
    """

    name: str
    extras: tuple[str, ...]
    group: str
    probes: tuple[str, ...] = ()
    cli: tuple[str, ...] = ()
    store_extra: str | None = None


def _store_profile(extra: str) -> Profile:
    roundtrip = extra in ROUNDTRIP_STORES
    return Profile(
        name=extra,
        extras=(extra,),
        group="stores",
        probes=("adapter-roundtrip" if roundtrip else "adapter-constructs",),
        store_extra=extra,
    )


PROFILES: tuple[Profile, ...] = (
    Profile(
        "base",
        (),
        "base",
        probes=(
            "bundled-data",
            "config-load",
            "vectorstore-roundtrip",
            "missing-clients",
            "qdrant-tool-names-extra",
        ),
        cli=(
            "version",
            "help",
            "command-help",
            "init-scaffold",
            "test-run-needs-eval",
            "optimize-needs-optimizer",
            "view-needs-dashboard",
        ),
    ),
    Profile(
        "eval-nlp",
        ("eval", "nlp"),
        "features",
        probes=(
            "deepeval-constructs",
            "deepeval-provider-sdks",
            "nlp-scores",
            "meteor",
            "executor-builds-evaluators",
        ),
    ),
    Profile("optimizer", ("optimizer",), "features", probes=("optimizer-study",)),
    Profile("opensearch", ("opensearch",), "features", probes=("opensearch-provider",)),
    Profile(
        "openai-agents",
        ("openai-agents",),
        "features",
        probes=("backends-import",),
    ),
    Profile(
        "dashboard",
        ("dashboard",),
        "features",
        probes=("dashboard-imports",),
        cli=("view-help",),
    ),
    Profile(
        "temporal",
        ("temporal",),
        "features",
        probes=("temporal-imports",),
        cli=("worker-help",),
    ),
    Profile(
        "deploy",
        ("deploy-all",),
        "features",
        probes=("deploy-imports",),
        cli=("deploy-help",),
    ),
    *(_store_profile(extra) for extra in STORE_PROVIDERS),
    Profile(
        "all",
        (ALL_EXTRA,),
        "all",
        probes=(
            "import-every-module",
            "bundled-data",
            "backends-import",
            "dashboard-imports",
            "executor-builds-evaluators",
        ),
        cli=("version", "help", "command-help", "view-help", "optimize-help"),
    ),
)
PROFILE_NAMES = tuple(profile.name for profile in PROFILES)
GROUPS = tuple(dict.fromkeys(profile.group for profile in PROFILES))


# -- Pure helpers ---------------------------------------------------------------


def expand_extras(extras: Sequence[str]) -> set[str]:
    """Return the extras a profile installs, with ``all`` expanded."""
    expanded = set(extras)
    if ALL_EXTRA in expanded:
        expanded.update(EXTRA_MODULES)
    return expanded


def present_modules(profile: Profile) -> list[str]:
    """Return the modules the profile's extras provide, sorted."""
    modules: set[str] = set()
    for extra in expand_extras(profile.extras):
        modules.update(EXTRA_MODULES.get(extra, ()))
    return sorted(modules)


def absent_packages(profile: Profile) -> list[Package]:
    """Return the packages the profile must not have installed."""
    installed = expand_extras(profile.extras)
    return [pkg for pkg in PACKAGES if not installed.intersection(pkg.extras)]


def profile_by_name(name: str) -> Profile:
    """Return the profile called *name* or raise ``KeyError``."""
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise KeyError(name)


def constraints_from_export(export: str) -> str:
    """Turn ``uv export`` output into a constraints file.

    Keeps the ``name==version ; marker`` lines. Drops comments, blank lines,
    hash continuations and option or editable lines (``-e .``), none of which
    a constraints file may hold.
    """
    lines: list[str] = []
    for raw in export.splitlines():
        line = raw.strip().rstrip("\\").strip()
        if not line or line.startswith(("#", "-", "--hash")):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def overrides_from_pyproject(pyproject: Mapping[str, object]) -> str:
    """Return the ``[tool.uv] override-dependencies`` as an overrides file."""
    tool = pyproject.get("tool")
    uv = tool.get("uv") if isinstance(tool, dict) else None
    overrides = uv.get("override-dependencies", []) if isinstance(uv, dict) else []
    return "".join(f"{requirement}\n" for requirement in overrides)


def install_target(wheel: Path, extras: Sequence[str]) -> str:
    """Return the requirement that installs *wheel* with *extras*."""
    return f"{wheel}[{','.join(extras)}]" if extras else str(wheel)


def parse_probe_output(stdout: str) -> dict[str, tuple[bool, str]]:
    """Return ``name -> (ok, detail)`` from the probe's marked lines."""
    results: dict[str, tuple[bool, str]] = {}
    for line in stdout.splitlines():
        if not line.startswith(MARKER):
            continue
        try:
            row = json.loads(line[len(MARKER) :])
        except json.JSONDecodeError:
            continue
        results[str(row["name"])] = (bool(row["ok"]), str(row["detail"]))
    return results


def pip_check_problems(output: str, overridden: Sequence[str]) -> list[str]:
    """Return the ``uv pip check`` incompatibilities that no override explains.

    ``[tool.uv] override-dependencies`` replaces a requirement a dependency
    declares on purpose (``bkflow-feel`` pins ``pytz<2024``), so the check
    reports it. That conflict is documented, not a broken install.
    """
    names = {re.sub(r"[-_.]+", "-", name).lower() for name in overridden}
    problems: list[str] = []
    for line in output.splitlines():
        match = re.match(r"The package `([^`]+)` requires `([A-Za-z0-9._-]+)", line)
        if match is None:
            continue
        if re.sub(r"[-_.]+", "-", match.group(2)).lower() in names:
            continue
        problems.append(line.strip())
    return problems


def human_size(size: int) -> str:
    """Format *size* bytes as MB."""
    return f"{size / (1024 * 1024):.0f} MB"


@dataclass
class ProfileResult:
    """Outcome of one profile."""

    name: str
    checks: list[Check] = field(default_factory=list)
    packages: int = 0
    size_bytes: int = 0
    seconds: float = 0.0

    @property
    def failed(self) -> list[Check]:
        """Return the failed checks."""
        return [check for check in self.checks if check.status == "FAIL"]

    @property
    def ok(self) -> bool:
        """Return True when no check failed."""
        return not self.failed


def render_profile(result: ProfileResult) -> str:
    """Render one profile's PASS/FAIL table."""
    title = (
        f"profile {result.name}: {result.packages} packages, "
        f"{human_size(result.size_bytes)}, {result.seconds:.0f}s"
    )
    width = max([len(check.name) for check in result.checks] + [5])
    lines = ["", title, f"  {'check'.ljust(width)}  result  detail"]
    for check in result.checks:
        detail = check.detail.splitlines()[0] if check.detail else ""
        lines.append(f"  {check.name.ljust(width)}  {check.status:<6}  {detail[:140]}")
    return "\n".join(lines)


def render_summary(results: Sequence[ProfileResult]) -> str:
    """Render the one-row-per-profile summary."""
    width = max([len(result.name) for result in results] + [7])
    lines = [
        "",
        "summary",
        f"  {'profile'.ljust(width)}  result  checks  packages  site-packages  time",
    ]
    for result in results:
        passed = sum(1 for check in result.checks if check.status == "PASS")
        total = sum(1 for check in result.checks if check.status != "SKIP")
        lines.append(
            f"  {result.name.ljust(width)}  {'PASS' if result.ok else 'FAIL':<6}  "
            f"{passed:>2}/{total:<4}  {result.packages:>8}  "
            f"{human_size(result.size_bytes):>13}  {result.seconds:>3.0f}s"
        )
    return "\n".join(lines)


# -- Running a profile ----------------------------------------------------------


@dataclass
class Workspace:
    """Files shared by every profile of one run."""

    root: Path
    wheel: Path
    constraints: Path
    overrides: Path
    data: Path  # agent YAML files; never the working directory of a check
    overridden_names: tuple[str, ...] = ()


def run(
    args: Sequence[str | Path],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Run *args*, capturing text output; a timeout is exit code 124."""
    try:
        return subprocess.run(  # noqa: S603 - fixed local tooling
            [str(arg) for arg in args],
            cwd=cwd,
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        out = error.stdout.decode() if isinstance(error.stdout, bytes) else ""
        return subprocess.CompletedProcess(
            args, 124, out, f"timed out after {timeout}s"
        )


def tool_env(home: Path) -> dict[str, str]:
    """Return the environment for installs and checks.

    Credentials are scrubbed, ``HOME`` is empty (so ``~/.holodeck/.env`` is not
    read) and nothing points at the development environment.
    """
    env = scrubbed_env({"HOME": str(home), "NO_COLOR": "1"})
    for key in list(env):
        if key in {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"} or key.startswith(
            "UV_PROJECT"
        ):
            del env[key]
    env["OPENAI_API_KEY"] = "sk-profile-check"
    return env


def last_lines(text: str, count: int = 6) -> str:
    """Return the last *count* non-empty lines of *text*, for failure details."""
    lines = [line for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-count:])


def build_wheel(out_dir: Path) -> Path:
    """Build the wheel once into *out_dir* and return its path."""
    result = run(
        ["uv", "build", "--wheel", "--out-dir", out_dir, ROOT],
        cwd=ROOT,
        env=os.environ,
        timeout=INSTALL_TIMEOUT,
    )
    wheels = sorted(out_dir.glob("holodeck_ai-*.whl"))
    if result.returncode != 0 or not wheels:
        raise RuntimeError(f"wheel build failed: {last_lines(result.stderr)}")
    return wheels[-1]


def export_constraints(workspace_root: Path) -> tuple[Path, Path]:
    """Write the lock-derived constraints and the uv overrides; return both."""
    result = run(
        [
            "uv",
            "export",
            "--frozen",
            "--all-extras",
            "--no-dev",
            "--no-hashes",
            "--no-emit-project",
            "--quiet",
        ],
        cwd=ROOT,
        env=os.environ,
        timeout=INSTALL_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uv export failed: {last_lines(result.stderr)}")
    constraints = workspace_root / "constraints.txt"
    constraints.write_text(constraints_from_export(result.stdout), encoding="utf-8")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    overrides = workspace_root / "overrides.txt"
    overrides.write_text(overrides_from_pyproject(pyproject), encoding="utf-8")
    return constraints, overrides


def write_data_files(data: Path) -> None:
    """Write the agent YAML files the checks use (outside every cwd)."""
    data.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "tests" / "fixtures" / "agents" / "minimal_agent.yaml", data)

    def agent(metrics: str) -> str:
        return (
            "name: profile-agent\n"
            "model:\n  provider: openai\n  name: gpt-4o\n"
            "instructions:\n  inline: You are a helpful assistant.\n"
            f"evaluations:\n  metrics:\n{metrics}"
            'test_cases:\n  - input: "Hello"\n    ground_truth: "Hi"\n'
        )

    geval = "    - {type: geval, name: Helpful, criteria: 'Is the answer helpful'}\n"
    equality = "    - {type: standard, metric: equality}\n"
    everything = (
        geval
        + "    - {type: rag, metric_type: faithfulness}\n"
        + "    - {type: standard, metric: bleu}\n"
        + "    - {type: standard, metric: rouge}\n"
        + "    - {type: standard, metric: meteor}\n"
        + equality
    )
    (data / "agent_eval.yaml").write_text(agent(geval), encoding="utf-8")
    (data / "agent_equality.yaml").write_text(agent(equality), encoding="utf-8")
    (data / "agent_metrics.yaml").write_text(agent(everything), encoding="utf-8")


@dataclass
class VenvEnv:
    """A profile's venv and how to run things in it."""

    python: Path
    holodeck: Path
    cwd: Path  # empty; the source tree is not importable from here
    env: dict[str, str]
    data: Path
    probe_spec: Path


def venv_python(venv: Path) -> Path:
    """Return the interpreter of *venv*."""
    scripts = "Scripts" if os.name == "nt" else "bin"
    return venv / scripts / ("python.exe" if os.name == "nt" else "python")


def directory_size(path: Path) -> int:
    """Return the apparent size in bytes of the files under *path*."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).lstat().st_size
            except OSError:
                continue
    return total


def site_packages(python: Path, env: Mapping[str, str], cwd: Path) -> Path | None:
    """Return the venv's ``site-packages`` directory."""
    result = run(
        [
            python,
            "-I",
            "-c",
            "import sysconfig;print(sysconfig.get_paths()['purelib'])",
        ],
        cwd=cwd,
        env=env,
        timeout=60,
    )
    return Path(result.stdout.strip()) if result.returncode == 0 else None


def run_probes(profile: Profile, venv: VenvEnv, workspace: Workspace) -> list[Check]:
    """Run the profile's in-venv checks and return one ``Check`` per result."""
    absent = absent_packages(profile)
    spec: dict[str, Any] = {
        "checks": [
            "absent",
            "holodeck-imports",
            *(["present"] if present_modules(profile) else []),
            *profile.probes,
        ],
        "absent_modules": [pkg.module for pkg in absent],
        "absent_dists": [pkg.dist for pkg in absent],
        "present_modules": present_modules(profile),
        "bundled_paths": list(BUNDLED_PATHS),
        "agent_yaml": str(venv.data / "minimal_agent.yaml"),
        "metrics_yaml": str(venv.data / "agent_metrics.yaml"),
        "missing_store_extras": sorted(
            {
                extra
                for pkg in absent
                for extra in pkg.extras
                if extra in STORE_PROVIDERS
            }
        ),
        "store_provider": (
            STORE_PROVIDERS[profile.store_extra] if profile.store_extra else None
        ),
        "openai_agents": "openai-agents" in expand_extras(profile.extras),
    }
    venv.probe_spec.write_text(json.dumps(spec), encoding="utf-8")
    result = run(
        [venv.python, "-I", PROBE, venv.probe_spec],
        cwd=venv.cwd,
        env=venv.env,
        timeout=PROBE_TIMEOUT,
    )
    parsed = parse_probe_output(result.stdout)
    checks: list[Check] = []
    for name in spec["checks"]:
        if name in parsed:
            ok, detail = parsed[name]
            checks.append(Check(name, "PASS" if ok else "FAIL", detail))
        else:
            reason = last_lines(result.stderr or result.stdout) or "no result"
            checks.append(Check(name, "FAIL", f"probe died: {reason}"))
    return checks


# -- CLI checks -----------------------------------------------------------------


def _cli(
    venv: VenvEnv, args: Sequence[str], cwd: Path | None = None
) -> tuple[int, str]:
    result = run(
        [venv.holodeck, *args],
        cwd=cwd or venv.cwd,
        env=venv.env,
        timeout=CLI_TIMEOUT,
    )
    return result.returncode, result.stdout + result.stderr


def _expect_exit(
    name: str, code: int, output: str, *, want_zero: bool = True
) -> Check | None:
    if want_zero and code != 0:
        return Check(name, "FAIL", f"exit {code}: {last_lines(output)}")
    if "Traceback" in output:
        return Check(name, "FAIL", f"traceback: {last_lines(output)}")
    return None


def cli_version(venv: VenvEnv) -> Check:
    """``holodeck --version`` prints a version."""
    code, out = _cli(venv, ["--version"])
    return _expect_exit("cli --version", code, out) or Check(
        "cli --version", "PASS", out.strip()
    )


def cli_help(venv: VenvEnv) -> Check:
    """``holodeck --help`` lists every command."""
    code, out = _cli(venv, ["--help"])
    failed = _expect_exit("cli --help", code, out)
    if failed:
        return failed
    missing = [group for group in COMMAND_GROUPS if group not in out]
    if missing:
        return Check("cli --help", "FAIL", f"commands not listed: {missing}")
    return Check("cli --help", "PASS", f"{len(COMMAND_GROUPS)} commands listed")


def cli_command_help(venv: VenvEnv) -> Check:
    """Every command group's ``--help`` exits 0 with no traceback."""
    problems: list[str] = []
    for group in COMMAND_GROUPS:
        code, out = _cli(venv, [group, "--help"])
        failed = _expect_exit(group, code, out)
        if failed:
            problems.append(f"{group}: {failed.detail}")
    if problems:
        return Check("cli <command> --help", "FAIL", "; ".join(problems))
    return Check("cli <command> --help", "PASS", f"{len(COMMAND_GROUPS)} commands")


def cli_init_scaffold(venv: VenvEnv) -> Check:
    """``holodeck init`` scaffolds a project from the wheel's templates."""
    target = venv.cwd.parent / "scaffold"
    target.mkdir(exist_ok=True)
    code, out = _cli(
        venv,
        ["init", "--name", "profile-demo", "--non-interactive", "--force"],
        cwd=target,
    )
    failed = _expect_exit("cli init", code, out)
    if failed:
        return failed
    agent = target / "profile-demo" / "agent.yaml"
    if not agent.is_file():
        return Check("cli init", "FAIL", f"no {agent.name} written: {last_lines(out)}")
    return Check("cli init", "PASS", "project scaffolded from the packaged template")


def _needs_extra(name: str, venv: VenvEnv, args: Sequence[str], command: str) -> Check:
    code, out = _cli(venv, args)
    if code == 0:
        return Check(name, "FAIL", "exited 0 without the extra")
    if "Traceback" in out:
        return Check(name, "FAIL", f"traceback: {last_lines(out)}")
    if command not in out:
        return Check(name, "FAIL", f"no '{command}' in: {last_lines(out)}")
    return Check(name, "PASS", f"exit {code}; prints {command}")


def cli_test_run_needs_eval(venv: VenvEnv) -> Check:
    """A DeepEval metric without ``eval`` fails with the install command."""
    return _needs_extra(
        "test run: deepeval metric",
        venv,
        ["test", "run", str(venv.data / "agent_eval.yaml")],
        "pip install 'holodeck-ai[eval]'",
    )


def cli_optimize_needs_optimizer(venv: VenvEnv) -> Check:
    """``test optimize`` without ``optimizer`` fails with the install command."""
    return _needs_extra(
        "test optimize",
        venv,
        ["test", "optimize", str(venv.data / "agent_equality.yaml")],
        "pip install 'holodeck-ai[optimizer]'",
    )


def cli_view_needs_dashboard(venv: VenvEnv) -> Check:
    """``test view`` without ``dashboard`` fails with the install command."""
    return _needs_extra(
        "test view",
        venv,
        ["test", "view", "--seed", "--no-browser"],
        "holodeck-ai[dashboard]",
    )


def cli_view_help(venv: VenvEnv) -> Check:
    """``holodeck test view --help`` works."""
    code, out = _cli(venv, ["test", "view", "--help"])
    return _expect_exit("test view --help", code, out) or Check(
        "test view --help", "PASS", "exit 0"
    )


def cli_optimize_help(venv: VenvEnv) -> Check:
    """``holodeck test optimize --help`` works."""
    code, out = _cli(venv, ["test", "optimize", "--help"])
    return _expect_exit("test optimize --help", code, out) or Check(
        "test optimize --help", "PASS", "exit 0"
    )


def _help(name: str, venv: VenvEnv, args: Sequence[str]) -> Check:
    code, out = _cli(venv, [*args, "--help"])
    return _expect_exit(name, code, out) or Check(name, "PASS", "exit 0")


def cli_worker_help(venv: VenvEnv) -> Check:
    """``holodeck worker --help`` works with the ``temporal`` extra."""
    return _help("worker --help", venv, ["worker"])


def cli_deploy_help(venv: VenvEnv) -> Check:
    """``holodeck deploy --help`` works with the deploy extras."""
    return _help("deploy --help", venv, ["deploy"])


CLI_CHECKS: dict[str, Callable[[VenvEnv], Check]] = {
    "version": cli_version,
    "help": cli_help,
    "command-help": cli_command_help,
    "init-scaffold": cli_init_scaffold,
    "test-run-needs-eval": cli_test_run_needs_eval,
    "optimize-needs-optimizer": cli_optimize_needs_optimizer,
    "view-needs-dashboard": cli_view_needs_dashboard,
    "view-help": cli_view_help,
    "optimize-help": cli_optimize_help,
    "worker-help": cli_worker_help,
    "deploy-help": cli_deploy_help,
}


def run_profile(profile: Profile, workspace: Workspace, python: str) -> ProfileResult:
    """Install *profile* into a fresh venv and run its checks."""
    started = time.monotonic()
    result = ProfileResult(profile.name)
    home = workspace.root / f"home-{profile.name}"
    cwd = workspace.root / f"cwd-{profile.name}"
    home.mkdir()
    cwd.mkdir()
    env = tool_env(home)
    venv_dir = workspace.root / f"venv-{profile.name}"

    created = run(
        ["uv", "venv", "--python", python, "--quiet", venv_dir],
        cwd=cwd,
        env=env,
        timeout=INSTALL_TIMEOUT,
    )
    if created.returncode != 0:
        result.checks.append(Check("venv", "FAIL", last_lines(created.stderr)))
        result.seconds = time.monotonic() - started
        return result
    py = venv_python(venv_dir)

    installed = run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            py,
            "--quiet",
            "--constraint",
            workspace.constraints,
            "--override",
            workspace.overrides,
            install_target(workspace.wheel, profile.extras),
        ],
        cwd=cwd,
        env=env,
        timeout=INSTALL_TIMEOUT,
    )
    if installed.returncode != 0:
        result.checks.append(Check("install", "FAIL", last_lines(installed.stderr)))
        result.seconds = time.monotonic() - started
        return result
    extras = ",".join(profile.extras) or "none"
    result.checks.append(Check("install", "PASS", f"wheel with extras: {extras}"))

    holodeck = py.with_name("holodeck.exe" if os.name == "nt" else "holodeck")
    venv = VenvEnv(
        py, holodeck, cwd, env, workspace.data, workspace.root / f"{profile.name}.json"
    )
    result.checks.extend(run_probes(profile, venv, workspace))
    for name in profile.cli:
        result.checks.append(CLI_CHECKS[name](venv))

    checked = run(
        ["uv", "pip", "check", "--python", py],
        cwd=cwd,
        env=env,
        timeout=CLI_TIMEOUT,
    )
    broken = pip_check_problems(
        checked.stdout + checked.stderr, workspace.overridden_names
    )
    result.checks.append(
        Check(
            "uv pip check",
            "FAIL" if broken else "PASS",
            "; ".join(broken) or "no broken requirements (documented overrides aside)",
        )
    )

    listing = run(
        ["uv", "pip", "list", "--python", py, "--format", "json"],
        cwd=cwd,
        env=env,
        timeout=CLI_TIMEOUT,
    )
    if listing.returncode == 0:
        result.packages = len(json.loads(listing.stdout))
    packages_dir = site_packages(py, env, cwd)
    if packages_dir is not None:
        result.size_bytes = directory_size(packages_dir)
    result.seconds = time.monotonic() - started
    return result


def select_profiles(names: Sequence[str], groups: Sequence[str]) -> list[Profile]:
    """Return the profiles chosen by ``--profile`` and ``--group`` (all if none)."""
    if not names and not groups:
        return list(PROFILES)
    chosen = {profile.name for profile in PROFILES if profile.group in groups}
    chosen.update(names)
    return [profile for profile in PROFILES if profile.name in chosen]


def main(argv: Sequence[str] | None = None) -> int:
    """Build the wheel, run the chosen profiles and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        choices=PROFILE_NAMES,
        help="profile to run (repeatable); default: all",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        choices=GROUPS,
        help="CI group of profiles to run (repeatable)",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PROFILES_PYTHON", DEFAULT_PYTHON),
        help=f"interpreter of every venv (default {DEFAULT_PYTHON})",
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the venvs and wheel for debugging"
    )
    args = parser.parse_args(argv)

    profiles = select_profiles(args.profile, args.group)
    root = Path(tempfile.mkdtemp(prefix="holodeck-profiles-"))
    started = time.monotonic()
    try:
        print(
            f"building the wheel (python {args.python}, workspace {root})", flush=True
        )
        wheel = build_wheel(root / "dist")
        constraints, overrides = export_constraints(root)
        data = root / "data"
        write_data_files(data)
        workspace = Workspace(
            root,
            wheel,
            constraints,
            overrides,
            data,
            tuple(
                re.split(r"[<>=!~ ;\[]", line, maxsplit=1)[0]
                for line in overrides.read_text(encoding="utf-8").splitlines()
            ),
        )

        results: list[ProfileResult] = []
        for profile in profiles:
            print(f"profile {profile.name} ...", flush=True)
            result = run_profile(profile, workspace, args.python)
            results.append(result)
            print(render_profile(result), flush=True)
    finally:
        if args.keep:
            print(f"\nkept {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)

    print(render_summary(results))
    failed = [result.name for result in results if not result.ok]
    elapsed = time.monotonic() - started
    if failed:
        print(f"\nFAILED profiles: {', '.join(failed)} ({elapsed:.0f}s)")
        return 1
    print(f"\nAll {len(results)} profiles passed ({elapsed:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
