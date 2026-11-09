"""Decision logic for scripts/verify/audit_deps.py (offline, fixture JSON)."""

import json
from datetime import date
from pathlib import Path

import pytest

from scripts.verify.audit_deps import (
    evaluate,
    load_ignores,
    main,
    parse_audit,
    report_findings,
)

pytestmark = pytest.mark.unit

TODAY = date(2026, 11, 9)
REPORT = {
    "dependencies": [
        {
            "name": "chromadb",
            "version": "1.0.21",
            "vulns": [
                {
                    "id": "PYSEC-2026-311",
                    "fix_versions": [],
                    "aliases": ["CVE-2026-45829", "GHSA-f4j7-r4q5-qw2c"],
                },
                {
                    "id": "PYSEC-2026-311",
                    "fix_versions": [],
                    "aliases": ["CVE-2026-45829", "GHSA-f4j7-r4q5-qw2c"],
                },
            ],
        },
        {"name": "requests", "version": "2.0.0", "vulns": []},
    ]
}


def _ignores(tmp_path: Path, *entries: str) -> Path:
    path = tmp_path / "ignores.toml"
    path.write_text("\n".join(entries), encoding="utf-8")
    return path


def _entry(vuln_id: str, package: str = "chromadb", expires: str = "2026-12-31") -> str:
    return (
        f'[[ignore]]\nid = "{vuln_id}"\npackage = "{package}"\n'
        f'reason = "server only"\nexpires = "{expires}"\nowner = "me"\n'
    )


def test_parse_dedupes_repeated_advisories() -> None:
    vulns = parse_audit(json.dumps(REPORT))
    assert [(v.package, v.id) for v in vulns] == [("chromadb", "PYSEC-2026-311")]
    assert "CVE-2026-45829" in vulns[0].ids


def test_alias_match_ignores_vulnerability(tmp_path: Path) -> None:
    ignores = load_ignores(_ignores(tmp_path, _entry("CVE-2026-45829")))
    findings = evaluate(parse_audit(json.dumps(REPORT)), ignores, TODAY)
    assert findings.ok
    assert [v.id for v in findings.ignored] == ["PYSEC-2026-311"]


def test_unignored_vulnerability_fails_with_fix_versions(tmp_path: Path) -> None:
    report = {
        "dependencies": [
            {
                "name": "Jinja2",
                "version": "3.1.0",
                "vulns": [{"id": "GHSA-x", "fix_versions": ["3.1.6"], "aliases": []}],
            }
        ]
    }
    # An ignore for another package must not match, even with the same id.
    ignores = load_ignores(_ignores(tmp_path, _entry("GHSA-x", package="flask")))
    findings = evaluate(parse_audit(json.dumps(report)), ignores, TODAY)
    assert [v.id for v in findings.unignored] == ["GHSA-x"]
    lines = report_findings(findings, Path("ignores.toml"))
    assert "vulnerable: Jinja2 3.1.0 GHSA-x; fix versions: 3.1.6" in lines[0]
    assert (
        lines[1]
        == "stale ignore GHSA-x (flask): no current vulnerability matches it: remove it"
    )


def test_expired_ignore_fails_and_stops_suppressing(tmp_path: Path) -> None:
    ignores = load_ignores(
        _ignores(tmp_path, _entry("CVE-2026-45829", expires="2026-11-08"))
    )
    findings = evaluate(parse_audit(json.dumps(REPORT)), ignores, TODAY)
    assert [v.id for v in findings.unignored] == ["PYSEC-2026-311"]
    assert [i.id for i in findings.expired] == ["CVE-2026-45829"]
    assert findings.stale == []
    assert (
        "ignore CVE-2026-45829 for chromadb expired on 2026-11-08: re-review and "
        "extend with a new date, or remove it"
    ) in report_findings(findings, Path("ignores.toml"))


def test_ignore_expiring_today_is_still_valid(tmp_path: Path) -> None:
    ignores = load_ignores(
        _ignores(tmp_path, _entry("CVE-2026-45829", expires="2026-11-09"))
    )
    assert evaluate(parse_audit(json.dumps(REPORT)), ignores, TODAY).ok


def test_stale_ignore_fails(tmp_path: Path) -> None:
    ignores = load_ignores(
        _ignores(tmp_path, _entry("PYSEC-2026-311"), _entry("CVE-2099-0001"))
    )
    findings = evaluate(parse_audit(json.dumps(REPORT)), ignores, TODAY)
    assert findings.unignored == []
    assert [i.id for i in findings.stale] == ["CVE-2099-0001"]
    assert not findings.ok


def test_missing_field_is_rejected(tmp_path: Path) -> None:
    path = _ignores(tmp_path, '[[ignore]]\nid = "X"\npackage = "p"\n')
    with pytest.raises(ValueError, match="missing reason, expires, owner"):
        load_ignores(path)


def test_main_uses_injected_today(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "audit.json"
    report.write_text(json.dumps(REPORT), encoding="utf-8")
    ignores = _ignores(tmp_path, _entry("CVE-2026-45829"))
    base = ["--audit-json", str(report), "--ignores", str(ignores), "--skip-npm"]
    assert main([*base, "--today", "2026-12-31"]) == 0
    assert main([*base, "--today", "2027-01-01"]) == 1
    assert "expired on 2026-12-31" in capsys.readouterr().err


def test_unaudited_dependency_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    skipped = {"name": "private-pkg", "skip_reason": "not found on PyPI"}
    data = {**REPORT, "dependencies": [*REPORT["dependencies"], skipped]}
    report = tmp_path / "audit.json"
    report.write_text(json.dumps(data), encoding="utf-8")
    ignores = _ignores(tmp_path, _entry("CVE-2026-45829"))
    args = ["--audit-json", str(report), "--ignores", str(ignores), "--skip-npm"]
    assert main([*args, "--today", "2026-12-31"]) == 1
    assert "not audited: private-pkg" in capsys.readouterr().err


def test_seeded_ignore_file_is_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    ignores = load_ignores(root / "security" / "audit-ignores.toml")
    assert {i.id for i in ignores} == {
        "CVE-2026-45829",
        "CVE-2026-45830",
        "CVE-2026-45831",
        "CVE-2026-45833",
    }
