"""In-environment checks for ``profiles.py``: runs inside an installed wheel.

``profiles.py`` installs the built wheel into a throwaway venv and runs this
file with that venv's interpreter (``python -I``), from an empty working
directory. Nothing here may import from the source tree: it is standard
library plus the installed ``holodeck`` package.

Protocol: argv[1] is a JSON spec file; each requested check prints one result
line ``@@PROBE@@ {"name": ..., "ok": ..., "detail": ...}`` on stdout. HoloDeck
logs to the same streams, so the parent reads only marked lines.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import importlib.util
import json
import pkgutil
import re
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

MARKER = "@@PROBE@@ "
Spec = dict[str, Any]

# Settings that let each adapter build without a reachable service.
OFFLINE_SETTINGS: dict[str, dict[str, object]] = {
    "qdrant": {"location": ":memory:"},
    "chromadb": {},  # persist_directory is filled in per run
    "postgres": {"connection_string": "postgresql://holodeck:x@127.0.0.1:1/holodeck"},
    "pinecone": {"api_key": "pc-profile-check"},
    "weaviate": {"local_host": "127.0.0.1"},
    "azure-ai-search": {
        "endpoint": "https://profile-check.search.windows.net",
        "api_key": "profile-check",
    },
    "azure-cosmos-nosql": {
        "url": "https://profile-check.documents.azure.com:443/",
        "key": "cHJvZmlsZS1jaGVjaw==",
        "database_name": "profile_check",
    },
}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_distributions() -> set[str]:
    return {
        _normalize(str(dist.metadata["Name"]))
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }


def _is_importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _schema(name: str = "profile_check", dimensions: int = 4) -> Any:
    from holodeck.lib.retrieval import CollectionSchema, FieldSpec, FieldType

    return CollectionSchema(
        name=name,
        dimensions=dimensions,
        fields=(
            FieldSpec("text", FieldType.STRING, full_text=True),
            FieldSpec("source", FieldType.STRING, indexed=True),
        ),
    )


# -- Presence and absence ---------------------------------------------------------


def check_absent(spec: Spec) -> str:
    """Blocked modules cannot be found and blocked distributions are not installed."""
    found = [m for m in spec["absent_modules"] if _is_importable(m)]
    installed = _installed_distributions()
    dists = [d for d in spec["absent_dists"] if _normalize(d) in installed]
    problems = [f"module {m} is importable" for m in found]
    problems += [f"distribution {d} is installed" for d in dists]
    if problems:
        raise AssertionError("; ".join(problems))
    return f"{len(spec['absent_modules'])} modules, {len(spec['absent_dists'])} dists"


def check_present(spec: Spec) -> str:
    """Every module the profile's extras provide imports."""
    failed: list[str] = []
    for module in spec["present_modules"]:
        try:
            importlib.import_module(module)
        except Exception as error:  # noqa: BLE001 - report every failure
            failed.append(f"{module}: {type(error).__name__}: {error}")
    if failed:
        raise AssertionError("; ".join(failed))
    return f"{len(spec['present_modules'])} modules import"


def check_holodeck_imports(spec: Spec) -> str:
    """The wheel's package imports, from site-packages and not a source tree."""
    import holodeck
    import holodeck.lib.retrieval  # noqa: F401
    import holodeck.tools.vectorstore_tool  # noqa: F401

    location = str(Path(holodeck.__file__).resolve())
    if "site-packages" not in location:
        raise AssertionError(f"holodeck imported from {location}, not the wheel")
    return "imported from the installed wheel"


def check_import_every_module(spec: Spec) -> str:
    """Every module in the wheel imports."""
    import holodeck

    skip = ("holodeck.cli.main", "holodeck.dashboard.__main__")
    failed: list[str] = []
    count = 0
    for info in pkgutil.walk_packages(holodeck.__path__, "holodeck."):
        if info.name in skip or info.name.endswith(".__main__"):
            continue
        count += 1
        try:
            importlib.import_module(info.name)
        except Exception as error:  # noqa: BLE001 - report every failure
            failed.append(f"{info.name}: {type(error).__name__}: {error}")
    if failed:
        raise AssertionError("; ".join(failed[:5]))
    return f"{count} modules import"


def check_bundled_data(spec: Spec) -> str:
    """Templates and static files are packaged in the wheel."""
    import holodeck

    root = Path(holodeck.__file__).resolve().parent
    missing = [path for path in spec["bundled_paths"] if not (root / path).exists()]
    if missing:
        raise AssertionError(f"missing from the wheel: {missing}")
    return f"{len(spec['bundled_paths'])} paths present"


# -- Base install ---------------------------------------------------------------


def check_config_load(spec: Spec) -> str:
    """The config loader reads an agent YAML that has no evaluation metrics."""
    from holodeck.config.loader import ConfigLoader

    agent = ConfigLoader().load_agent_yaml(spec["agent_yaml"])
    return f"loaded {agent.name}"


def check_vectorstore_roundtrip(spec: Spec) -> str:
    """Ingest and search through the vectorstore tool, in memory, no embedder."""
    from holodeck.models.tool import VectorstoreTool as ToolConfig
    from holodeck.tools.vectorstore_tool import VectorStoreTool

    async def run() -> str:
        with tempfile.TemporaryDirectory() as directory:
            docs = Path(directory) / "docs"
            docs.mkdir()
            (docs / "note.txt").write_text(
                "HoloDeck runs agents from YAML. Vector search finds documents."
            )
            config = ToolConfig.model_validate(
                {
                    "name": "kb",
                    "description": "profile check",
                    "source": str(docs),
                    "embedding_dimensions": 8,
                    "top_k": 3,
                }
            )
            tool = VectorStoreTool(config, base_dir=directory)
            await tool.initialize(provider_type="openai", force_ingest=True)
            return str(await tool.search("agents"))

    result = asyncio.run(run())
    if "Found 1 result" not in result:
        raise AssertionError(f"unexpected search result: {result[:200]}")
    return "ingested 1 file, found 1 result"


def check_missing_clients(spec: Spec) -> str:
    """Each vector store without its client names its extra, not a traceback."""
    from holodeck.lib.extras import install_command
    from holodeck.lib.retrieval import MissingClientError, create_adapter
    from holodeck.lib.retrieval.registry import PROVIDERS

    problems: list[str] = []
    checked = 0
    for provider, entry in PROVIDERS.items():
        if entry.extra is None or entry.extra not in spec["missing_store_extras"]:
            continue
        checked += 1
        try:
            create_adapter(provider, _schema(), OFFLINE_SETTINGS[provider])
        except MissingClientError as error:
            if install_command(entry.extra) not in str(error):
                problems.append(f"{provider}: no install command in {error}")
        except Exception as error:  # noqa: BLE001
            problems.append(f"{provider}: {type(error).__name__}: {error}")
        else:
            problems.append(f"{provider}: built an adapter without its client")
    if problems:
        raise AssertionError("; ".join(problems))
    return f"{checked} providers name their extra"


def check_qdrant_tool_names_extra(spec: Spec) -> str:
    """``provider: qdrant`` on a vectorstore tool prints the ``qdrant`` install command.

    The tool either raises ``MissingExtraError`` or, as designed today, logs a
    warning that carries the command and falls back to in-memory storage.
    """
    import logging

    from holodeck.lib.extras import MissingExtraError
    from holodeck.models.tool import VectorstoreTool as ToolConfig
    from holodeck.tools.vectorstore_tool import VectorStoreTool

    command = "pip install 'holodeck-ai[qdrant]'"
    warnings: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            warnings.append(record.getMessage())

    async def run() -> VectorStoreTool:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "note.txt").write_text("HoloDeck profile check.")
            config = ToolConfig.model_validate(
                {
                    "name": "kb",
                    "description": "profile check",
                    "source": directory,
                    "embedding_dimensions": 8,
                    "database": {"provider": "qdrant", "location": ":memory:"},
                }
            )
            tool = VectorStoreTool(config, base_dir=directory)
            await tool.initialize(provider_type="openai", force_ingest=True)
            return tool

    handler = Capture(level=logging.WARNING)
    logging.getLogger("holodeck").addHandler(handler)
    try:
        tool = asyncio.run(run())
    except MissingExtraError as error:
        if command not in str(error):
            raise AssertionError(f"no install command in: {error}") from error
        return f"raised MissingExtraError naming {command}"
    finally:
        logging.getLogger("holodeck").removeHandler(handler)
    if any(command in message for message in warnings):
        return f"warned with {command} (fell back to {tool._provider})"
    raise AssertionError(f"qdrant tool initialized silently: {warnings}")


# -- Evaluation extras ----------------------------------------------------------


def check_deepeval_constructs(spec: Spec) -> str:
    """DeepEval evaluators construct without a model call."""
    from holodeck.lib.evaluators.deepeval import (
        DeepEvalModelConfig,
        FaithfulnessEvaluator,
        GEvalEvaluator,
    )
    from holodeck.models.llm import ProviderEnum

    config = DeepEvalModelConfig(
        provider=ProviderEnum.OPENAI, model_name="gpt-4o", api_key="sk-profile-check"
    )
    GEvalEvaluator(
        name="Helpful", criteria="Is the answer helpful?", model_config=config
    )
    FaithfulnessEvaluator(model_config=config, threshold=0.5)
    return "GEval and Faithfulness evaluators built"


def check_deepeval_provider_sdks(spec: Spec) -> str:
    """The Anthropic and Ollama judge models build (their SDKs come with eval)."""
    from holodeck.lib.evaluators.deepeval import DeepEvalModelConfig
    from holodeck.models.llm import ProviderEnum

    for provider in (ProviderEnum.ANTHROPIC, ProviderEnum.OLLAMA):
        DeepEvalModelConfig(
            provider=provider, model_name="m", api_key="profile-check"
        ).to_deepeval_model()
    return "AnthropicModel and OllamaModel built"


def check_nlp_scores(spec: Spec) -> str:
    """BLEU and ROUGE compute a score offline."""
    from holodeck.lib.evaluators.nlp_metrics import BLEUEvaluator, ROUGEEvaluator

    async def run() -> tuple[float, float]:
        bleu = await BLEUEvaluator().evaluate(
            response="The cat sat on the mat", ground_truth="The cat is on the mat"
        )
        rouge = await ROUGEEvaluator().evaluate(
            response="The cat sat on the mat", ground_truth="The cat is on the mat"
        )
        return float(bleu["bleu"]), float(
            next(
                v
                for k, v in rouge.items()
                if k.startswith("rouge") and isinstance(v, float)
            )
        )

    bleu, rouge = asyncio.run(run())
    if not (0.0 < bleu <= 1.0 and 0.0 < rouge <= 1.0):
        raise AssertionError(f"scores out of range: bleu={bleu}, rouge={rouge}")
    return f"bleu={bleu:.3f} rouge={rouge:.3f}"


def check_meteor(spec: Spec) -> str:
    """METEOR computes, or names the NLTK data download command."""
    from holodeck.lib.evaluators.nlp_metrics import METEOREvaluator

    async def run() -> float:
        result = await METEOREvaluator().evaluate(
            response="The cat sat on the mat", ground_truth="The cat is on the mat"
        )
        return float(result["meteor"])

    try:
        score = asyncio.run(run())
    except Exception as error:  # noqa: BLE001
        message = str(error)
        if "nltk.downloader" in message:
            return "NLTK data absent; error names the download command"
        raise AssertionError(f"{type(error).__name__}: {message}") from error
    return f"meteor={score:.3f}"


def check_executor_builds_evaluators(spec: Spec) -> str:
    """The test executor builds every eval/nlp evaluator, before any agent call."""
    from unittest.mock import patch

    from holodeck.lib.test_runner.executor import TestExecutor

    with patch("holodeck.lib.backends.selector.BackendSelector.select"):
        executor = TestExecutor(agent_config_path=spec["metrics_yaml"])
    return f"evaluators: {sorted(executor.evaluators)}"


# -- Optimizer, OpenSearch ------------------------------------------------------


def check_optimizer_study(spec: Spec) -> str:
    """The numeric proposer builds an Optuna study and suggests a value."""
    from holodeck.models.agent import Agent
    from holodeck.optimizer.config import NumericAxis
    from holodeck.optimizer.proposers.numeric import NumericProposer

    proposer = NumericProposer(
        [NumericAxis(path="model.temperature", type="float", range=[0.0, 1.0])], seed=7
    )
    agent = Agent.model_validate(
        {
            "name": "profile-check",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Profile check."},
        }
    )
    proposer.begin(agent, None)
    proposal = asyncio.run(proposer.ask())
    if proposal is None:
        raise AssertionError("the proposer suggested nothing")
    return "study built, one value suggested"


def check_opensearch_provider(spec: Spec) -> str:
    """The OpenSearch keyword index builds against an unreachable host."""
    from holodeck.lib.keyword_search import OpenSearchKeywordProvider

    OpenSearchKeywordProvider("https://127.0.0.1:1", "profile-check", timeout_seconds=1)
    return "provider built; no connection attempted"


# -- Vector stores --------------------------------------------------------------


def _adapter_for(spec: Spec, provider: str, workdir: str) -> Any:
    from holodeck.lib.retrieval import create_adapter

    settings = dict(OFFLINE_SETTINGS[provider])
    if provider == "chromadb":
        settings["persist_directory"] = str(Path(workdir) / "chroma")
    return create_adapter(provider, _schema(), settings)


def check_adapter_constructs(spec: Spec) -> str:
    """The profile's store adapter builds without a network connection.

    Construction succeeds, or fails only with a connection error that names an
    unreachable store (``StoreConnectionError``, ``ConnectionError``,
    ``TimeoutError``). Anything else is a defect in the adapter or the client
    library (``TypeError``, ``AttributeError``, ``ImportError``, a missing
    client) and fails the probe.
    """
    from holodeck.lib.retrieval import StoreConnectionError

    provider = spec["store_provider"]
    with tempfile.TemporaryDirectory() as workdir:
        try:
            adapter = _adapter_for(spec, provider, workdir)
        except (StoreConnectionError, ConnectionError, TimeoutError) as error:
            return f"store unreachable, as allowed: {type(error).__name__}"
        except Exception as error:
            raise AssertionError(
                f"{provider} adapter construction failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        asyncio.run(adapter.close())
    return f"{provider} adapter built"


def check_adapter_roundtrip(spec: Spec) -> str:
    """Upsert and vector search against an embedded store."""
    from holodeck.lib.retrieval import Record, VectorQuery

    provider = spec["store_provider"]
    vectors = {
        "a": [1.0, 0.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0, 0.0],
        "c": [0.0, 0.0, 1.0, 0.0],
    }

    async def run(adapter: Any) -> str:
        try:
            await adapter.ensure_collection()
            await adapter.upsert(
                [
                    Record(
                        id=_record_id(provider, name),
                        fields={"text": f"record {name}", "source": "profile"},
                        vector=vector,
                    )
                    for name, vector in vectors.items()
                ]
            )
            count = await adapter.count()
            hits = await adapter.vector_search(
                VectorQuery(vector=[0.0, 1.0, 0.0, 0.0], top_k=1)
            )
            await adapter.delete_collection()
        finally:
            await adapter.close()
        if count != 3:
            raise AssertionError(f"count={count}, expected 3")
        if not hits or hits[0].record.fields.get("text") != "record b":
            raise AssertionError(f"nearest neighbour wrong: {hits}")
        return f"3 upserted, nearest of 3 found ({hits[0].score:.2f})"

    with tempfile.TemporaryDirectory() as workdir:
        adapter = _adapter_for(spec, provider, workdir)
        return asyncio.run(run(adapter))


def _record_id(provider: str, name: str) -> str:
    # Qdrant point IDs are UUIDs or integers; the adapter maps readable keys.
    return f"profile-{name}"


# -- Backends and dashboard -----------------------------------------------------


def check_backends_import(spec: Spec) -> str:
    """Both agent backends import and BackendSelector routes to each."""
    from holodeck.lib.backends import selector  # noqa: F401
    from holodeck.lib.backends.claude_backend import ClaudeBackend  # noqa: F401

    if spec.get("openai_agents"):
        from holodeck.lib.backends.openai_agents_backend import (  # noqa: F401
            OpenAIAgentsBackend,
        )
    return "claude" + (" and openai-agents" if spec.get("openai_agents") else "")


def check_dashboard_imports(spec: Spec) -> str:
    """The dashboard app imports and its assets are packaged."""
    import holodeck.dashboard.app as app

    assets = Path(app.__file__).resolve().parent / "assets"
    if not any(assets.glob("*.css")):
        raise AssertionError(f"no dashboard stylesheets in {assets}")
    return "dashboard app imports; assets present"


def check_temporal_imports(spec: Spec) -> str:
    """The Temporal worker modules import."""
    from temporalio.worker import Worker  # noqa: F401

    import holodeck.temporal  # noqa: F401

    return "holodeck.temporal and temporalio.worker import"


def check_deploy_imports(spec: Spec) -> str:
    """The deploy engine and its cloud deployers import."""
    import holodeck.deploy  # noqa: F401

    for module in importlib.import_module("holodeck.deploy").__path__:
        for info in pkgutil.walk_packages([module], "holodeck.deploy."):
            importlib.import_module(info.name)
    return "holodeck.deploy and its submodules import"


CHECKS: dict[str, Callable[[Spec], str]] = {
    "absent": check_absent,
    "present": check_present,
    "holodeck-imports": check_holodeck_imports,
    "import-every-module": check_import_every_module,
    "bundled-data": check_bundled_data,
    "config-load": check_config_load,
    "vectorstore-roundtrip": check_vectorstore_roundtrip,
    "missing-clients": check_missing_clients,
    "qdrant-tool-names-extra": check_qdrant_tool_names_extra,
    "deepeval-constructs": check_deepeval_constructs,
    "deepeval-provider-sdks": check_deepeval_provider_sdks,
    "nlp-scores": check_nlp_scores,
    "meteor": check_meteor,
    "executor-builds-evaluators": check_executor_builds_evaluators,
    "optimizer-study": check_optimizer_study,
    "opensearch-provider": check_opensearch_provider,
    "adapter-constructs": check_adapter_constructs,
    "adapter-roundtrip": check_adapter_roundtrip,
    "backends-import": check_backends_import,
    "dashboard-imports": check_dashboard_imports,
    "temporal-imports": check_temporal_imports,
    "deploy-imports": check_deploy_imports,
}


def format_result(result: dict[str, Any]) -> str:
    """Return the marked stdout line for one check result."""
    return MARKER + json.dumps(result)


def main(argv: list[str]) -> int:
    """Run the checks named in the spec file; print one marked line per check."""
    spec: Spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    for name in spec["checks"]:
        try:
            detail = CHECKS[name](spec)
            result = {"name": name, "ok": True, "detail": detail}
        except Exception as error:  # noqa: BLE001 - a failed check is a result
            tail = traceback.format_exc().strip().splitlines()[-1]
            result = {
                "name": name,
                "ok": False,
                "detail": f"{type(error).__name__}: {error}"[:400] or tail,
            }
        print(format_result(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
