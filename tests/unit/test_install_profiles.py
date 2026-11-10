"""Install profiles: feature extras and the missing-extra error (042 T21).

``pip install holodeck-ai`` leaves out DeepEval (``eval``), the NLP metric
libraries (``nlp``), Optuna (``optimizer``) and opensearch-py
(``opensearch``). The development environment has every extra installed, so
these tests simulate a base install by blocking the modules in a fresh
interpreter (``sys.modules[name] = None``) and check that:

* a feature whose extra is missing raises ``MissingExtraError`` with the exact
  install command, before any agent or model call;
* the CLI prints that message in one line, without a traceback, and exits
  non-zero;
* unrelated commands still start and run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from holodeck.lib.extras import MissingExtraError, install_command, require_extra
from holodeck.lib.retrieval.errors import MissingClientError

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
EXTRAS: dict[str, list[str]] = PYPROJECT["project"]["optional-dependencies"]

# Top-level modules that each feature extra provides.
EXTRA_MODULES = {
    "eval": ("deepeval", "anthropic", "ollama"),
    "nlp": ("nltk", "sacrebleu", "rouge_score"),
    "optimizer": ("optuna",),
    "opensearch": ("opensearchpy",),
}
# What a base install does not have: every feature extra, plus the dashboard
# and the docs/release tooling that left the runtime requirements.
BASE_BLOCKED = (
    *(module for modules in EXTRA_MODULES.values() for module in modules),
    "dash",
    "mkdocs",
    "twine",
)
MOVED_DISTRIBUTIONS = {
    "eval": {"deepeval", "anthropic", "ollama"},
    "nlp": {"nltk", "rouge-score", "sacrebleu"},
    "optimizer": {"optuna"},
    "opensearch": {"opensearch-py"},
    "dev": {"mkdocs-material", "twine"},
}

AGENT_YAML = """\
name: profile-agent
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: You are a helpful assistant.
evaluations:
  metrics:
    - {metric}
test_cases:
  - input: "Hello"
    ground_truth: "Hi"
"""
METRICS = {
    "geval": "{type: geval, name: Helpful, criteria: 'Is the answer helpful'}",
    "rag": "{type: rag, metric_type: faithfulness}",
    "bleu": "{type: standard, metric: bleu}",
    "rouge": "{type: standard, metric: rouge}",
    "meteor": "{type: standard, metric: meteor}",
    "equality": "{type: standard, metric: equality}",
}


def _distribution(requirement: str) -> str:
    """Return the normalized distribution name of a requirement string."""
    match = re.match(r"[A-Za-z0-9._-]+", requirement)
    assert match is not None, requirement
    return match.group(0).lower().replace("_", "-")


def _self_referenced_extras(requirements: list[str]) -> set[str]:
    names: set[str] = set()
    for requirement in requirements:
        match = re.fullmatch(r"holodeck-ai\[([^\]]+)\]", requirement)
        assert match is not None, f"'all' must only self-reference: {requirement}"
        names.update(name.strip() for name in match.group(1).split(","))
    return names


def _run(code: str, *, blocked: tuple[str, ...], cwd: Path = ROOT) -> str:
    """Run *code* in a fresh interpreter with *blocked* modules absent."""
    prelude = f"import sys\nfor _name in {blocked!r}:\n    sys.modules[_name] = None\n"
    env = {
        **os.environ,
        "OPENAI_API_KEY": "sk-profile-test",
        "HOLODECK_VERIFY_ANTHROPIC_URL": "http://127.0.0.1:9",
    }
    result = subprocess.run(
        [sys.executable, "-c", prelude + code],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=cwd,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


# -- Package metadata ---------------------------------------------------------


def test_all_covers_every_runtime_extra() -> None:
    runtime_extras = set(EXTRAS) - {"dev", "all"}
    assert _self_referenced_extras(EXTRAS["all"]) == runtime_extras


@pytest.mark.parametrize("extra", sorted(MOVED_DISTRIBUTIONS))
def test_moved_packages_left_the_base_install(extra: str) -> None:
    base = {_distribution(r) for r in PYPROJECT["project"]["dependencies"]}
    in_extra = {_distribution(r) for r in EXTRAS[extra]}
    assert MOVED_DISTRIBUTIONS[extra] <= in_extra
    assert not MOVED_DISTRIBUTIONS[extra] & base


# -- The error and the helper -------------------------------------------------


def test_missing_extra_error_names_the_install_command() -> None:
    error = MissingExtraError("DeepEval metrics", "eval")
    assert error.feature == "DeepEval metrics"
    assert error.extra == "eval"
    assert error.install_command == "pip install 'holodeck-ai[eval]'"
    assert str(error) == (
        "DeepEval metrics needs the 'eval' extra, which is not installed. "
        "Install it with: pip install 'holodeck-ai[eval]'"
    )
    assert "\n" not in str(error)


def test_require_extra_returns_the_module() -> None:
    assert require_extra("json", "eval", "x").__name__ == "json"


def test_require_extra_reports_only_the_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "holodeck_absent_pkg", None)
    with pytest.raises(MissingExtraError) as raised:
        require_extra("holodeck_absent_pkg.sub", "nlp", "Feature")
    assert raised.value.install_command == install_command("nlp")


def test_require_extra_does_not_hide_a_broken_install(tmp_path: Path) -> None:
    """A dependency missing *inside* an installed package is not an extra."""
    (tmp_path / "holodeck_broken_pkg.py").write_text("import holodeck_absent_dep\n")
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(ModuleNotFoundError, match="holodeck_absent_dep"):
            require_extra("holodeck_broken_pkg", "nlp", "Feature")
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("holodeck_broken_pkg", None)


def test_vector_store_client_error_is_a_missing_extra() -> None:
    error = MissingClientError("qdrant", "qdrant")
    assert isinstance(error, MissingExtraError)
    assert error.provider == "qdrant"
    assert error.extra == "qdrant"
    assert error.install_command == "pip install 'holodeck-ai[qdrant]'"
    assert "pip install 'holodeck-ai[qdrant]'" in str(error)


# -- A missing extra fails before any agent call ------------------------------

EXECUTOR_CHECK = """
from unittest.mock import patch
from holodeck.lib.extras import MissingExtraError
from holodeck.lib.test_runner.executor import TestExecutor

with patch("holodeck.lib.backends.selector.BackendSelector.select") as select:
    try:
        TestExecutor(agent_config_path={path!r})
    except MissingExtraError as error:
        assert not select.called
        print("EXTRA", error.extra, error.install_command)
    else:
        print("NO-ERROR")
"""


@pytest.mark.parametrize(
    ("metric", "blocked", "extra"),
    [
        ("geval", ("deepeval",), "eval"),
        ("rag", ("deepeval",), "eval"),
        ("bleu", ("sacrebleu",), "nlp"),
        ("rouge", ("rouge_score",), "nlp"),
        ("meteor", ("nltk",), "nlp"),
    ],
)
def test_metric_without_its_extra_fails_at_executor_construction(
    tmp_path: Path, metric: str, blocked: tuple[str, ...], extra: str
) -> None:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(AGENT_YAML.format(metric=METRICS[metric]))

    out = _run(EXECUTOR_CHECK.format(path=str(agent_yaml)), blocked=blocked)

    assert f"EXTRA {extra} pip install 'holodeck-ai[{extra}]'" in out


@pytest.mark.parametrize(
    ("provider", "blocked"),
    [("anthropic", ("anthropic",)), ("ollama", ("ollama",))],
)
def test_deepeval_provider_sdk_names_the_eval_extra(
    provider: str, blocked: tuple[str, ...]
) -> None:
    code = (
        "from holodeck.lib.extras import MissingExtraError\n"
        "from holodeck.lib.evaluators.deepeval.config import DeepEvalModelConfig\n"
        "from holodeck.models.llm import ProviderEnum\n"
        f"config = DeepEvalModelConfig(provider=ProviderEnum({provider!r}), "
        "model_name='m', api_key='k')\n"
        "try:\n"
        "    config.to_deepeval_model()\n"
        "except MissingExtraError as error:\n"
        "    print('EXTRA', error.extra)\n"
    )
    assert "EXTRA eval" in _run(code, blocked=blocked)


def test_optimizer_without_optuna_names_its_extra() -> None:
    code = (
        "from holodeck.lib.extras import MissingExtraError\n"
        "from holodeck.optimizer.proposers.numeric import NumericProposer\n"
        "try:\n"
        "    NumericProposer([], seed=1)\n"
        "except MissingExtraError as error:\n"
        "    print('EXTRA', error.extra, error.install_command)\n"
    )
    out = _run(code, blocked=("optuna",))
    assert "EXTRA optimizer pip install 'holodeck-ai[optimizer]'" in out


def test_opensearch_index_without_client_names_its_extra() -> None:
    code = (
        "from unittest.mock import MagicMock\n"
        "from holodeck.lib.extras import MissingExtraError\n"
        "from holodeck.lib.keyword_search import HybridSearchExecutor, "
        "OpenSearchKeywordProvider\n"
        "from holodeck.models.tool import KeywordIndexConfig\n"
        "for build in (\n"
        "    lambda: OpenSearchKeywordProvider('https://h:9200', 'i'),\n"
        "    lambda: HybridSearchExecutor(\n"
        "        MagicMock(provider='in-memory', **{\n"
        "            'capabilities.supports.return_value': False}),\n"
        "        keyword_index_config=KeywordIndexConfig(\n"
        "            provider='opensearch', endpoint='https://h:9200',\n"
        "            index_name='i'),\n"
        "    ),\n"
        "):\n"
        "    try:\n"
        "        build()\n"
        "    except MissingExtraError as error:\n"
        "        print('EXTRA', error.extra)\n"
    )
    out = _run(code, blocked=("opensearchpy",))
    assert out.count("EXTRA opensearch") == 2


# -- The CLI prints one line and exits non-zero --------------------------------

CLI = "from holodeck.cli.main import main\nmain({argv!r})\n"


def _cli(
    argv: list[str], *, blocked: tuple[str, ...], cwd: Path
) -> subprocess.CompletedProcess[str]:
    prelude = f"import sys\nfor _name in {blocked!r}:\n    sys.modules[_name] = None\n"
    return subprocess.run(
        [sys.executable, "-c", prelude + CLI.format(argv=argv)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=cwd,
        env={**os.environ, "OPENAI_API_KEY": "sk-profile-test"},
    )


def test_test_run_with_a_deepeval_metric_prints_the_install_command(
    tmp_path: Path,
) -> None:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(AGENT_YAML.format(metric=METRICS["geval"]))

    result = _cli(["test", "run", str(agent_yaml)], blocked=BASE_BLOCKED, cwd=tmp_path)

    assert result.returncode == 2, result.stdout + result.stderr
    assert "pip install 'holodeck-ai[eval]'" in result.stderr
    assert "Traceback" not in result.stdout + result.stderr


def test_optimize_without_optuna_prints_the_install_command(tmp_path: Path) -> None:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(AGENT_YAML.format(metric=METRICS["equality"]))

    result = _cli(
        ["test", "optimize", str(agent_yaml)], blocked=BASE_BLOCKED, cwd=tmp_path
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert "pip install 'holodeck-ai[optimizer]'" in result.stderr
    assert "Traceback" not in result.stdout + result.stderr


def test_optimize_without_optuna_keeps_the_json_fatal_event(tmp_path: Path) -> None:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(AGENT_YAML.format(metric=METRICS["equality"]))

    result = _cli(
        ["test", "optimize", str(agent_yaml), "--progress", "json"],
        blocked=BASE_BLOCKED,
        cwd=tmp_path,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert events[-1]["fatal"] is True
    assert "pip install 'holodeck-ai[optimizer]'" in events[-1]["message"]
    assert "pip install 'holodeck-ai[optimizer]'" in result.stderr


# -- Unrelated commands still work on a base install ---------------------------


@pytest.mark.parametrize(
    "argv",
    [["--help"], ["--version"], ["init", "--help"], ["test", "run", "--help"]],
    ids=" ".join,
)
def test_base_commands_start_without_feature_extras(
    tmp_path: Path, argv: list[str]
) -> None:
    result = _cli(argv, blocked=BASE_BLOCKED, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_base_install_loads_config_and_builds_a_metricless_executor(
    tmp_path: Path,
) -> None:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(AGENT_YAML.format(metric=METRICS["equality"]))
    code = (
        "from holodeck.config.loader import ConfigLoader\n"
        "from holodeck.lib.test_runner.executor import TestExecutor\n"
        "smoke = ConfigLoader().load_agent_yaml("
        f"{str(ROOT / 'tests/fixtures/verify/smoke-claude.yaml')!r})\n"
        f"executor = TestExecutor(agent_config_path={str(agent_yaml)!r})\n"
        "print('OK', smoke.name, sorted(executor.evaluators))\n"
    )
    out = _run(code, blocked=BASE_BLOCKED)
    assert "OK verify-smoke-claude ['equality']" in out


def _run_test_against_fake_provider(
    tmp_path: Path, yaml_name: str, *, extra_tools_yaml: str = ""
) -> subprocess.CompletedProcess[str]:
    """Run ``holodeck test run`` on a smoke fixture, offline, on a base install.

    *extra_tools_yaml* is inserted at the top of the fixture's ``tools:`` list.
    """
    pytest.importorskip("agents")
    import shutil

    from scripts.verify.procs import background, free_port, scrubbed_env, wait_http

    agents_dir = tmp_path / "agents"
    shutil.copytree(ROOT / "tests" / "fixtures" / "verify", agents_dir)
    agent_path = agents_dir / yaml_name
    if extra_tools_yaml:
        text = agent_path.read_text(encoding="utf-8")
        marker = "tools:\n"
        assert marker in text
        agent_path.write_text(
            text.replace(marker, marker + extra_tools_yaml, 1), encoding="utf-8"
        )
    home = tmp_path / "home"
    home.mkdir()
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    provider_args = [
        sys.executable,
        str(ROOT / "scripts" / "verify" / "fake_provider.py"),
        "--port",
        str(port),
        "--log",
        str(tmp_path / "provider.jsonl"),
    ]
    blocked = (*BASE_BLOCKED, "qdrant_client")
    with background(
        provider_args, cwd=ROOT, env=scrubbed_env({}), log=tmp_path / "provider.log"
    ) as provider:
        assert wait_http(f"{url}/health", provider, timeout=30) is None
        prelude = (
            f"import sys\nfor _name in {blocked!r}:\n    sys.modules[_name] = None\n"
        )
        return subprocess.run(
            [
                sys.executable,
                "-c",
                prelude + CLI.format(argv=["test", "run", str(agent_path)]),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=agents_dir,
            env=scrubbed_env(
                {
                    "HOME": str(home),
                    "OPENAI_AGENTS_DISABLE_TRACING": "1",
                    "OPENAI_API_KEY": "sk-verify-smoke-fake",
                    "OPENAI_BASE_URL": f"{url}/v1",
                }
            ),
        )


def test_base_install_runs_a_test_case_against_the_fake_provider(
    tmp_path: Path,
) -> None:
    """``holodeck test run`` with no extra-backed metric, end to end, offline."""
    result = _run_test_against_fake_provider(tmp_path, "smoke-openai.yaml")
    assert result.returncode == 0, result.stdout + result.stderr


QDRANT_TOOL = """\
  - name: kb
    type: vectorstore
    description: Knowledge base on Qdrant.
    source: rag/kb
    top_k: 2
    database:
      provider: qdrant
      connection_string: http://127.0.0.1:6333
"""


def test_test_run_with_a_qdrant_tool_and_no_extra_prints_the_install_command(
    tmp_path: Path,
) -> None:
    """The missing extra reaches the CLI unwrapped: one line, exit 2."""
    result = _run_test_against_fake_provider(
        tmp_path, "smoke-openai.yaml", extra_tools_yaml=QDRANT_TOOL
    )
    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "pip install 'holodeck-ai[qdrant]'" in result.stderr
    assert "Traceback" not in output
