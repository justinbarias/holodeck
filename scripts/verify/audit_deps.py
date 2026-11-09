"""Audit locked Python dependencies (and the CopilotKit sample) for advisories.

Exports the full ``uv.lock`` (all extras and groups), runs ``pip-audit`` on it,
and applies ``security/audit-ignores.toml``. The audit fails when a reported
vulnerability is not ignored, when an ignore has expired, or when an ignore no
longer matches any reported vulnerability.

Usage::

    uv run python scripts/verify/audit_deps.py
    uv run python scripts/verify/audit_deps.py --audit-json report.json --skip-npm
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess  # noqa: S404 - runs fixed local tooling (uv, pip-audit, npm)
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
IGNORES = ROOT / "security" / "audit-ignores.toml"
NPM_PROJECT = ROOT / "sample" / "copilotkit"
REQUIRED_FIELDS = ("id", "package", "reason", "expires", "owner")


@dataclass(frozen=True)
class Vuln:
    """One advisory reported by pip-audit for one locked package."""

    package: str
    version: str
    id: str
    aliases: tuple[str, ...] = ()
    fix_versions: tuple[str, ...] = ()

    @property
    def ids(self) -> set[str]:
        """Return the advisory id and every alias."""
        return {self.id, *self.aliases}


@dataclass(frozen=True)
class Ignore:
    """One accepted advisory from the ignore file."""

    id: str
    package: str
    reason: str
    expires: date
    owner: str


@dataclass
class Findings:
    """Result of applying the ignore file to an audit report."""

    unignored: list[Vuln] = field(default_factory=list)
    expired: list[Ignore] = field(default_factory=list)
    stale: list[Ignore] = field(default_factory=list)
    ignored: list[Vuln] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True when nothing needs attention."""
        return not (self.unignored or self.expired or self.stale)


def normalize(name: str) -> str:
    """Normalize a package name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def load_ignores(path: Path) -> list[Ignore]:
    """Load and validate ignore entries.

    Raises:
        ValueError: If an entry is missing a field or has a bad date.
    """
    if not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = data.get("ignore", [])
    if not isinstance(entries, list):
        raise ValueError(f"{path}: 'ignore' must be an array of tables")
    ignores: list[Ignore] = []
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: ignore #{index + 1} must be a table")
        entry = cast(dict[str, object], raw)
        missing = [key for key in REQUIRED_FIELDS if not entry.get(key)]
        if missing:
            raise ValueError(
                f"{path}: ignore #{index + 1} is missing {', '.join(missing)}"
            )
        expires = entry["expires"]
        if isinstance(expires, date):
            expiry = expires
        else:
            try:
                expiry = date.fromisoformat(str(expires))
            except ValueError as exc:
                raise ValueError(
                    f"{path}: ignore {entry['id']} has expires={expires!r}; "
                    "use YYYY-MM-DD"
                ) from exc
        ignores.append(
            Ignore(
                id=str(entry["id"]),
                package=str(entry["package"]),
                reason=str(entry["reason"]),
                expires=expiry,
                owner=str(entry["owner"]),
            )
        )
    return ignores


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in cast(list[object], value))


def parse_audit(report: str) -> list[Vuln]:
    """Parse ``pip-audit -f json`` output into unique vulnerabilities."""
    data = json.loads(report)
    if not isinstance(data, dict):
        raise ValueError("pip-audit JSON must be an object")
    dependencies = cast(dict[str, object], data).get("dependencies", [])
    vulns: dict[tuple[str, str], Vuln] = {}
    for raw_dep in cast(list[object], dependencies):
        dep = cast(dict[str, object], raw_dep)
        name = str(dep.get("name", ""))
        version = str(dep.get("version", ""))
        for raw_vuln in cast(list[object], dep.get("vulns", [])):
            vuln = cast(dict[str, object], raw_vuln)
            vuln_id = str(vuln.get("id", ""))
            key = (normalize(name), vuln_id)
            if key in vulns:
                continue
            vulns[key] = Vuln(
                package=name,
                version=version,
                id=vuln_id,
                aliases=_strings(vuln.get("aliases")),
                fix_versions=_strings(vuln.get("fix_versions")),
            )
    return list(vulns.values())


def skipped_dependencies(report: str) -> list[str]:
    """Return ``name: reason`` for dependencies pip-audit could not audit."""
    data = cast(dict[str, object], json.loads(report))
    skipped: list[str] = []
    for raw_dep in cast(list[object], data.get("dependencies", [])):
        dep = cast(dict[str, object], raw_dep)
        if "skip_reason" in dep:
            skipped.append(f"{dep.get('name')}: {dep.get('skip_reason')}")
    return skipped


def evaluate(vulns: list[Vuln], ignores: list[Ignore], today: date) -> Findings:
    """Apply ignores to vulnerabilities.

    A vulnerability is ignored when an entry for the same package matches its id
    or any alias. Expired entries never suppress a vulnerability.
    """
    findings = Findings()
    used: set[int] = set()
    for vuln in vulns:
        matched = False
        for index, ignore in enumerate(ignores):
            if normalize(ignore.package) != normalize(vuln.package):
                continue
            if ignore.id not in vuln.ids:
                continue
            used.add(index)
            if ignore.expires >= today:
                matched = True
        if matched:
            findings.ignored.append(vuln)
        else:
            findings.unignored.append(vuln)
    for index, ignore in enumerate(ignores):
        if ignore.expires < today:
            findings.expired.append(ignore)
        elif index not in used:
            findings.stale.append(ignore)
    return findings


def report_findings(findings: Findings, ignores_path: Path) -> list[str]:
    """Render remediation messages for every failing finding."""
    lines: list[str] = []
    for vuln in findings.unignored:
        fixes = ", ".join(vuln.fix_versions) or "no fixed release"
        aliases = f" ({', '.join(vuln.aliases)})" if vuln.aliases else ""
        lines.append(
            f"vulnerable: {vuln.package} {vuln.version} {vuln.id}{aliases}; "
            f"fix versions: {fixes}. Upgrade it in uv.lock, or add a reviewed "
            f"entry with reason, owner and expires to {ignores_path.name}."
        )
    for ignore in findings.expired:
        lines.append(
            f"ignore {ignore.id} for {ignore.package} expired on "
            f"{ignore.expires.isoformat()}: re-review and extend with a new "
            "date, or remove it"
        )
    for ignore in findings.stale:
        lines.append(
            f"stale ignore {ignore.id} ({ignore.package}): no current "
            "vulnerability matches it: remove it"
        )
    return lines


def run_pip_audit() -> str:
    """Export the full lock and return pip-audit JSON output."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not installed")
    exported = subprocess.run(  # noqa: S603
        [
            uv,
            "export",
            "--all-extras",
            "--all-groups",
            "--format",
            "requirements-txt",
            "--no-hashes",
            "--no-emit-project",
            "--quiet",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        requirements = Path(tmp) / "requirements.txt"
        requirements.write_text(exported.stdout, encoding="utf-8")
        audit = subprocess.run(  # noqa: S603
            [
                uv,
                "run",
                "--quiet",
                "pip-audit",
                "-r",
                str(requirements),
                "--no-deps",
                "--disable-pip",
                "--progress-spinner",
                "off",
                "-f",
                "json",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    # pip-audit exits 1 when it finds vulnerabilities; anything else is an error.
    if audit.returncode not in (0, 1) or not audit.stdout.strip():
        raise RuntimeError(
            f"pip-audit failed (exit {audit.returncode}):\n{audit.stderr.strip()}"
        )
    return audit.stdout


def run_npm_audit(project: Path) -> int:
    """Run ``npm audit`` for the CopilotKit sample, or skip when absent."""
    if not (project / "package-lock.json").exists():
        print(
            f"npm audit: skipped, {project.relative_to(ROOT)}/package-lock.json "
            "does not exist"
        )
        return 0
    npm = shutil.which("npm")
    if npm is None:
        print("npm audit: FAILED, package-lock.json exists but npm is not installed")
        return 1
    result = subprocess.run(  # noqa: S603
        [npm, "audit", "--audit-level=high", "--prefix", str(project)],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        print(
            "npm audit: FAILED, high or critical advisories in "
            f"{project.relative_to(ROOT)}. Run `npm audit fix --prefix "
            f"{project.relative_to(ROOT)}` and commit the lock file."
        )
        return 1
    print("npm audit: ok (no high or critical advisories)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the dependency audit and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ignores", type=Path, default=IGNORES)
    parser.add_argument(
        "--audit-json",
        type=Path,
        help="Use an existing pip-audit JSON report instead of running pip-audit",
    )
    parser.add_argument(
        "--today", type=date.fromisoformat, default=None, help="YYYY-MM-DD"
    )
    parser.add_argument("--skip-npm", action="store_true")
    args = parser.parse_args(argv)
    today: date = args.today or date.today()
    ignores_path: Path = args.ignores

    try:
        ignores = load_ignores(ignores_path)
        if args.audit_json is not None:
            report = Path(args.audit_json).read_text(encoding="utf-8")
        else:
            report = run_pip_audit()
        vulns = parse_audit(report)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"pip-audit: ERROR, {exc}", file=sys.stderr)
        return 1

    findings = evaluate(vulns, ignores, today)
    for vuln in findings.ignored:
        print(f"pip-audit: ignored {vuln.package} {vuln.version} {vuln.id}")
    # A package pip-audit could not look up is unaudited, not clean.
    problems = [
        f"not audited: {skipped}; make it resolvable on PyPI or audit it another "
        "way before accepting it"
        for skipped in skipped_dependencies(report)
    ]
    problems += report_findings(findings, ignores_path)
    for line in problems:
        print(f"pip-audit: {line}", file=sys.stderr)
    status = 1 if problems else 0
    if not problems:
        print(
            f"pip-audit: ok ({len(vulns)} reported, {len(findings.ignored)} "
            "ignored, 0 unignored)"
        )
    if not args.skip_npm:
        status = max(status, run_npm_audit(NPM_PROJECT))
    return status


if __name__ == "__main__":
    sys.exit(main())
