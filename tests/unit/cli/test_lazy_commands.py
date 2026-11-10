"""The CLI imports a command's module only when that command is used.

Regression guard: eager command imports made every invocation load pandas,
DeepEval, Semantic Kernel and the agent SDKs, so ``holodeck init`` took as
long to start as ``holodeck test`` and subprocess tests timed out on slow CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading

import click
import pytest
from click.testing import CliRunner

from holodeck.cli.lazy_group import LazyGroup
from holodeck.cli.main import main
from holodeck.lib.extras import MissingExtraError
from holodeck.lib.retrieval import RetiredProviderError

HEAVY = ("pandas", "deepeval", "claude_agent_sdk", "agents")
# Modules from optional install extras (042 T21) and from dev-only tooling.
# No base command's --help may import them: a base install does not have them.
EXTRA_MODULES = (
    "deepeval",
    "anthropic",
    "ollama",
    "nltk",
    "sacrebleu",
    "rouge_score",
    "optuna",
    "opensearchpy",
    "dash",
    "mkdocs",
    "twine",
)
COMMANDS = [
    "chat",
    "config",
    "deploy",
    "init",
    "mcp",
    "serve",
    "test",
    "vectorstore",
    "worker",
]


def _heavy_modules_after(command: str | None) -> list[str]:
    """Import the CLI (and resolve *command*) in a fresh interpreter."""
    resolve = f"main.get_command(None, {command!r})" if command else "None"
    code = (
        "import json, sys\n"
        "from holodeck.cli.main import main\n"
        f"{resolve}\n"
        f"print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    loaded: list[str] = json.loads(out.stdout.strip().splitlines()[-1])
    return loaded


@pytest.mark.unit
@pytest.mark.parametrize(
    "command", [None, "init", "config", "serve", "vectorstore", "worker"]
)
def test_light_commands_do_not_import_heavy_stacks(command: str | None) -> None:
    assert _heavy_modules_after(command) == []


def _extra_modules_after_help(argv: list[str]) -> list[str]:
    """Run ``holodeck <argv> --help`` in a fresh interpreter."""
    code = (
        "import json, sys\n"
        "from holodeck.cli.main import main\n"
        "try:\n"
        f"    main({[*argv, '--help']!r}, standalone_mode=False)\n"
        "except SystemExit:\n"
        "    pass\n"
        f"print(json.dumps([m for m in {EXTRA_MODULES!r} if m in sys.modules]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    loaded: list[str] = json.loads(out.stdout.strip().splitlines()[-1])
    return loaded


@pytest.mark.unit
@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["init"],
        ["chat"],
        ["serve"],
        ["test"],
        ["test", "run"],
        ["test", "optimize"],
        ["mcp"],
        ["vectorstore"],
        ["config"],
    ],
    ids=lambda argv: " ".join(argv) or "root",
)
def test_help_does_not_import_optional_extras(argv: list[str]) -> None:
    assert _extra_modules_after_help(argv) == []


@pytest.mark.unit
def test_help_lists_every_command() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    for name in COMMANDS:
        assert f"  {name} " in result.output


@pytest.mark.unit
@pytest.mark.parametrize("name", COMMANDS)
def test_each_command_resolves_to_its_click_command(name: str) -> None:
    command = main.get_command(click.Context(main), name)
    assert isinstance(command, click.Command)
    assert command.name == name


@pytest.mark.unit
def test_unknown_command_still_errors() -> None:
    result = CliRunner().invoke(main, ["nope"])
    assert result.exit_code == 2
    assert "No such command 'nope'" in result.output


@pytest.mark.unit
def test_lazy_target_that_is_not_a_command_fails_loudly() -> None:
    group = LazyGroup(name="g", lazy_subcommands={"bad": "json:dumps"})
    with pytest.raises(TypeError, match="not a click.Command"):
        group.get_command(click.Context(group), "bad")


@pytest.mark.unit
def test_concurrent_first_lookups_all_get_the_command() -> None:
    """Threads racing on a command's first load must not raise."""
    group = LazyGroup(
        name="g", lazy_subcommands={"init": "holodeck.cli.commands.init:init"}
    )
    barrier = threading.Barrier(8)
    results: list[object] = []

    def lookup() -> None:
        barrier.wait()
        try:
            results.append(group.get_command(click.Context(group), "init"))
        except Exception as exc:  # noqa: BLE001 - the test records any failure
            results.append(exc)

    threads = [threading.Thread(target=lookup) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8
    assert all(isinstance(r, click.Command) for r in results), results
    assert len({id(r) for r in results}) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            MissingExtraError("Qdrant", "qdrant"),
            "Error: Qdrant needs the 'qdrant' extra",
        ),
        (
            RetiredProviderError("faiss", "Use 'chromadb' instead."),
            "Config Error: Vector store provider 'faiss' is no longer supported. "
            "Use 'chromadb' instead.",
        ),
    ],
    ids=["missing-extra", "retired-provider"],
)
def test_user_fixable_errors_print_one_line_and_exit_2(
    error: Exception, expected: str
) -> None:
    @click.group(cls=LazyGroup)
    def group() -> None:
        pass

    @group.command()
    def boom() -> None:
        raise error

    result = CliRunner().invoke(group, ["boom"])
    assert result.exit_code == 2
    assert expected in result.output
    assert "Traceback" not in result.output
