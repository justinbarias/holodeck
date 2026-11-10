"""Offline coverage for the installed-wheel profile verifier (042 T22).

The verifier itself builds a wheel and installs it per profile, so it runs in
CI and under ``make verify-profiles``. These tests cover its pure helpers and
keep its profile table in step with ``pyproject.toml``: a new extra cannot be
added without a profile check, and every vector store extra has a profile.
"""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from holodeck.lib.retrieval.registry import PROVIDERS
from scripts.verify.procs import Check
from scripts.verify.profile_probe import CHECKS as PROBE_CHECKS
from scripts.verify.profile_probe import OFFLINE_SETTINGS, format_result
from scripts.verify.profiles import (
    ALL_EXTRA,
    CLI_CHECKS,
    EXTRA_MODULES,
    GROUPS,
    PACKAGES,
    PROFILE_NAMES,
    PROFILES,
    STORE_PROVIDERS,
    ProfileResult,
    absent_packages,
    constraints_from_export,
    expand_extras,
    install_target,
    overrides_from_pyproject,
    parse_probe_output,
    pip_check_problems,
    present_modules,
    profile_by_name,
    render_profile,
    render_summary,
    select_profiles,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
EXTRAS: dict[str, list[str]] = PYPROJECT["project"]["optional-dependencies"]
RUNTIME_EXTRAS = set(EXTRAS) - {"dev", ALL_EXTRA}
NEVER_INSTALLED = {"semantic_kernel", "azure.ai.evaluation", "evaluate"}


# -- The profile table matches pyproject.toml ---------------------------------


def test_every_runtime_extra_has_a_module_row() -> None:
    """A new extra needs a row here, so the ``all`` profile checks it imports."""
    assert set(EXTRA_MODULES) == RUNTIME_EXTRAS


def test_all_profile_installs_every_runtime_extra() -> None:
    all_profile = profile_by_name("all")
    assert all_profile.extras == (ALL_EXTRA,)
    assert expand_extras(all_profile.extras) == RUNTIME_EXTRAS | {ALL_EXTRA}
    expected = {module for modules in EXTRA_MODULES.values() for module in modules}
    assert set(present_modules(all_profile)) == expected


def test_every_extra_is_installed_by_some_profile() -> None:
    covered: set[str] = set()
    for profile in PROFILES:
        covered |= expand_extras(profile.extras)
    assert covered >= RUNTIME_EXTRAS


def test_profile_names_are_unique_and_grouped() -> None:
    assert len(set(PROFILE_NAMES)) == len(PROFILE_NAMES)
    assert {profile.group for profile in PROFILES} == set(GROUPS)
    assert GROUPS == ("base", "features", "stores", "all")


def test_every_profile_extra_exists() -> None:
    for profile in PROFILES:
        for extra in profile.extras:
            assert extra in EXTRAS, f"{profile.name}: unknown extra {extra}"


def test_profiles_name_only_checks_that_exist() -> None:
    for profile in PROFILES:
        for probe in profile.probes:
            assert probe in PROBE_CHECKS, f"{profile.name}: unknown probe {probe}"
        for cli in profile.cli:
            assert cli in CLI_CHECKS, f"{profile.name}: unknown cli check {cli}"


def test_store_extras_match_the_registry() -> None:
    registry = {entry.extra: name for name, entry in PROVIDERS.items() if entry.extra}
    assert registry == STORE_PROVIDERS
    assert set(STORE_PROVIDERS) <= RUNTIME_EXTRAS
    assert set(STORE_PROVIDERS) <= {profile.name for profile in PROFILES}
    assert set(OFFLINE_SETTINGS) >= set(STORE_PROVIDERS.values())
    for extra, provider in STORE_PROVIDERS.items():
        profile = profile_by_name(extra)
        assert profile.store_extra == extra
        assert provider in OFFLINE_SETTINGS


def test_the_base_profile_installs_no_extra() -> None:
    assert profile_by_name("base").extras == ()


def test_blocked_packages_name_real_extras_and_rows() -> None:
    for package in PACKAGES:
        for extra in package.extras:
            assert extra in EXTRAS, f"{package.module}: unknown extra {extra}"
            assert package.module in EXTRA_MODULES[extra] or package.module in {
                "psycopg",
                "agents",
            }, f"{package.module} is missing from EXTRA_MODULES[{extra!r}]"


def test_blocked_distributions_leave_the_base_requirements() -> None:
    base = {
        requirement.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip()
        for requirement in PYPROJECT["project"]["dependencies"]
    }
    normalized = {name.lower().replace("_", "-") for name in base}
    for package in PACKAGES:
        assert package.dist not in normalized, f"{package.dist} is a base requirement"


# -- Absence rules ------------------------------------------------------------


def test_the_retired_stacks_are_absent_in_every_profile() -> None:
    for profile in PROFILES:
        absent = {package.module for package in absent_packages(profile)}
        assert absent >= NEVER_INSTALLED, profile.name


def test_base_absent_list_covers_every_feature_package() -> None:
    absent = {package.module for package in absent_packages(profile_by_name("base"))}
    assert absent >= {
        "deepeval",
        "optuna",
        "opensearchpy",
        "nltk",
        "sacrebleu",
        "rouge_score",
        "anthropic",
        "ollama",
        "mkdocs",
        "twine",
        "dash",
        "agents",
        "temporalio",
        "qdrant_client",
        "chromadb",
        "psycopg",
        "pinecone",
        "weaviate",
        "azure.search.documents",
        "azure.cosmos",
    }


def test_a_store_profile_keeps_every_other_store_client_absent() -> None:
    clients = {
        "qdrant": "qdrant_client",
        "chromadb": "chromadb",
        "postgres": "psycopg",
        "pinecone": "pinecone",
        "weaviate": "weaviate",
        "azure-ai-search": "azure.search.documents",
        "azure-cosmos": "azure.cosmos",
    }
    assert set(clients) == set(STORE_PROVIDERS)
    for extra, own in clients.items():
        absent = {pkg.module for pkg in absent_packages(profile_by_name(extra))}
        assert own not in absent
        assert set(clients.values()) - {own} <= absent


def test_the_all_profile_leaves_only_the_never_installed_packages() -> None:
    absent = {pkg.module for pkg in absent_packages(profile_by_name("all"))}
    assert absent == NEVER_INSTALLED | {"mkdocs", "twine"}


def test_eval_nlp_provides_its_packages_and_nothing_else() -> None:
    profile = profile_by_name("eval-nlp")
    assert set(present_modules(profile)) == {
        "deepeval",
        "anthropic",
        "ollama",
        "nltk",
        "sacrebleu",
        "rouge_score",
    }
    absent = {pkg.module for pkg in absent_packages(profile)}
    assert {"optuna", "opensearchpy", "dash", "qdrant_client"} <= absent
    assert not absent & set(present_modules(profile))


# -- Constraints and install target -------------------------------------------

EXPORT = """\
# This file was autogenerated by uv via the following command:
#    uv export --frozen --all-extras --no-hashes --no-emit-project
annotated-types==0.7.0
    # via pydantic
aiohttp==3.14.4 \\
    --hash=sha256:abc
colorama==0.4.6 ; sys_platform == 'win32'
-e .

pydantic==2.12.0
"""


def test_constraints_keep_pins_and_markers_only() -> None:
    constraints = constraints_from_export(EXPORT)
    assert constraints.splitlines() == [
        "annotated-types==0.7.0",
        "aiohttp==3.14.4",
        "colorama==0.4.6 ; sys_platform == 'win32'",
        "pydantic==2.12.0",
    ]
    assert constraints.endswith("\n")


def test_overrides_come_from_pyproject() -> None:
    assert overrides_from_pyproject(PYPROJECT) == "pytz>=2024\n"
    assert overrides_from_pyproject({}) == ""
    assert overrides_from_pyproject({"tool": {"uv": {}}}) == ""


def test_install_target_adds_extras_to_the_wheel() -> None:
    wheel = Path("dist/holodeck_ai-1.0-py3-none-any.whl")
    assert install_target(wheel, ()) == str(wheel)
    assert install_target(wheel, ("eval", "nlp")) == f"{wheel}[eval,nlp]"


def test_select_profiles_by_name_and_group() -> None:
    assert [p.name for p in select_profiles([], [])] == list(PROFILE_NAMES)
    assert [p.name for p in select_profiles(["qdrant", "base"], [])] == [
        "base",
        "qdrant",
    ]
    stores = {p.name for p in select_profiles([], ["stores"])}
    assert stores == set(STORE_PROVIDERS)
    both = {p.name for p in select_profiles(["base"], ["stores"])}
    assert both == stores | {"base"}


# -- Probe output and pip check -----------------------------------------------


def test_probe_output_ignores_log_noise() -> None:
    stdout = (
        "2026-11-10 - holodeck - INFO - noise\n"
        '@@PROBE@@ {"name": "absent", "ok": true, "detail": "23 modules"}\n'
        "@@PROBE@@ not json\n"
        '@@PROBE@@ {"name": "present", "ok": false, "detail": "boom"}\n'
    )
    assert parse_probe_output(stdout) == {
        "absent": (True, "23 modules"),
        "present": (False, "boom"),
    }
    assert parse_probe_output("") == {}


def test_probe_marker_round_trips() -> None:
    line = format_result({"name": "x", "ok": True, "detail": "d"})
    assert parse_probe_output(line) == {"x": (True, "d")}


PIP_CHECK = """\
Using Python 3.11.15 environment at: /tmp/venv
Checked 143 packages in 22ms
Found 2 incompatibilities
The package `bkflow-feel` requires `pytz<2024`, but `2026.6` is installed
The package `foo` requires `bar>=2`, but `1.0` is installed
"""


def test_pip_check_ignores_only_overridden_requirements() -> None:
    assert pip_check_problems(PIP_CHECK, ["pytz"]) == [
        "The package `foo` requires `bar>=2`, but `1.0` is installed"
    ]
    assert len(pip_check_problems(PIP_CHECK, [])) == 2
    assert pip_check_problems("All installed packages are compatible", []) == []


# -- Tables -------------------------------------------------------------------


def _result(name: str, *checks: Check) -> ProfileResult:
    return ProfileResult(
        name, list(checks), packages=143, size_bytes=772 * 1024 * 1024, seconds=31.6
    )


def test_profile_table_lists_every_check_with_its_status() -> None:
    result = _result(
        "base",
        Check("install", "PASS", "wheel with extras: none"),
        Check("absent", "FAIL", "module deepeval is importable\nsecond line"),
    )
    table = render_profile(result)
    assert "profile base: 143 packages, 772 MB, 32s" in table
    assert "install" in table
    assert "PASS" in table
    assert "FAIL" in table
    assert "module deepeval is importable" in table
    assert "second line" not in table
    assert not result.ok
    assert [check.name for check in result.failed] == ["absent"]


def test_summary_counts_passes_and_skips() -> None:
    summary = render_summary(
        [
            _result("base", Check("a", "PASS"), Check("b", "PASS")),
            _result("qdrant", Check("a", "PASS"), Check("b", "FAIL", "x")),
            _result("eval-nlp", Check("a", "PASS"), Check("b", "SKIP")),
        ]
    )
    rows = {line.split()[0]: line for line in summary.splitlines()[3:]}
    assert "PASS" in rows["base"]
    assert " 2/2" in rows["base"]
    assert "FAIL" in rows["qdrant"]
    assert " 1/2" in rows["qdrant"]
    assert " 1/1" in rows["eval-nlp"]
    assert "143" in rows["base"]
    assert "772 MB" in rows["base"]


# -- Packaging regression ------------------------------------------------------


def test_no_tracked_package_file_is_gitignored() -> None:
    """The wheel leaves out a gitignored file, even a tracked one.

    ``.env.*`` in ``.gitignore`` once dropped the ``.env.example.j2`` templates
    from the wheel, so ``holodeck init`` failed on every installed copy.
    """
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    result = subprocess.run(  # noqa: S603
        [git, "ls-files", "-ci", "--exclude-standard", "src/holodeck"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split() == []
