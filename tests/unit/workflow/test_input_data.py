"""Tests for pre-execution ``input_data`` validation (036, T2a).

FR-025: every declared fact of record is validated against its JSON Schema
before any node executes; a missing or schema-invalid fact fails loudly with a
typed error naming the fact.
"""

import http.server
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from holodeck.lib.errors import InputDataError
from holodeck.lib.workflow.input_data import validate_input_data
from holodeck.models.workflow import Workflow

PRIOR_STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"demerits": {"type": "integer", "minimum": 0}},
    "required": ["demerits"],
}


def _workflow(input_data: dict[str, Any] | None) -> Workflow:
    """A one-node workflow, optionally declaring facts of record."""
    data: dict[str, Any] = {
        "name": "wf",
        "version": "2026-07-01.1",
        "nodes": [
            {
                "id": "zone",
                "decision": "tables/zone.dmn.yaml",
                "inputs": list(input_data) if input_data else ["evidence"],
            }
        ],
    }
    if input_data is None:
        data["nodes"].append(
            {
                "id": "evidence",
                "edge": {"agent": "agents/e/agent.yaml"},
                "gate": {"schema": "schemas/e.json"},
            }
        )
    else:
        data["input_data"] = input_data
    return Workflow.model_validate(data)


def _write_schema(base: Path, name: str, schema: Any) -> None:
    """Write a schema document under ``base/schemas/<name>.json``."""
    target = base / "schemas" / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(schema))


@pytest.mark.unit
class TestValidateInputData:
    """FR-025: declared facts validate before any node runs."""

    def test_valid_fact_is_returned(self, tmp_path: Path) -> None:
        # Arrange
        _write_schema(tmp_path, "prior_state", PRIOR_STATE_SCHEMA)
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act
        validated = validate_input_data(wf, {"prior_state": {"demerits": 3}}, tmp_path)

        # Assert
        assert validated == {"prior_state": {"demerits": 3}}

    def test_missing_fact_raises_naming_it(self, tmp_path: Path) -> None:
        # Arrange
        _write_schema(tmp_path, "prior_state", PRIOR_STATE_SCHEMA)
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act / Assert
        with pytest.raises(InputDataError) as exc:
            validate_input_data(wf, {}, tmp_path)
        assert exc.value.name == "prior_state"
        assert "missing" in str(exc.value)

    def test_schema_invalid_value_raises_naming_the_fact(self, tmp_path: Path) -> None:
        # Arrange
        _write_schema(tmp_path, "prior_state", PRIOR_STATE_SCHEMA)
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act / Assert
        with pytest.raises(InputDataError) as exc:
            validate_input_data(wf, {"prior_state": {"demerits": -1}}, tmp_path)
        assert exc.value.name == "prior_state"
        assert "schema-invalid" in str(exc.value)

    def test_missing_required_property_raises(self, tmp_path: Path) -> None:
        # Arrange
        _write_schema(tmp_path, "prior_state", PRIOR_STATE_SCHEMA)
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act / Assert
        with pytest.raises(InputDataError) as exc:
            validate_input_data(wf, {"prior_state": {}}, tmp_path)
        assert exc.value.name == "prior_state"

    def test_absent_schema_file_raises(self, tmp_path: Path) -> None:
        # Arrange — nothing written to disk.
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act / Assert
        with pytest.raises(InputDataError) as exc:
            validate_input_data(wf, {"prior_state": {"demerits": 1}}, tmp_path)
        assert exc.value.name == "prior_state"
        assert "cannot read schema" in str(exc.value)

    def test_unparseable_schema_file_raises(self, tmp_path: Path) -> None:
        # Arrange
        target = tmp_path / "schemas" / "prior_state.json"
        target.parent.mkdir(parents=True)
        target.write_text("{not json")
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act / Assert
        with pytest.raises(InputDataError) as exc:
            validate_input_data(wf, {"prior_state": {"demerits": 1}}, tmp_path)
        assert "not valid JSON" in str(exc.value)

    def test_non_object_schema_document_raises(self, tmp_path: Path) -> None:
        # Arrange
        _write_schema(tmp_path, "prior_state", ["not", "a", "schema"])
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act / Assert
        with pytest.raises(InputDataError) as exc:
            validate_input_data(wf, {"prior_state": {"demerits": 1}}, tmp_path)
        assert "must be a JSON object" in str(exc.value)

    def test_invalid_json_schema_raises(self, tmp_path: Path) -> None:
        # Arrange — valid JSON, invalid as a JSON Schema.
        _write_schema(tmp_path, "prior_state", {"type": "not-a-type"})
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act / Assert
        with pytest.raises(InputDataError) as exc:
            validate_input_data(wf, {"prior_state": {"demerits": 1}}, tmp_path)
        assert "invalid JSON Schema" in str(exc.value)

    def test_every_declared_fact_is_checked(self, tmp_path: Path) -> None:
        # Arrange — the second fact is the bad one.
        _write_schema(tmp_path, "prior_state", PRIOR_STATE_SCHEMA)
        _write_schema(tmp_path, "excuse", {"type": "string"})
        wf = _workflow(
            {
                "prior_state": {"schema": "schemas/prior_state.json"},
                "excuse": {"schema": "schemas/excuse.json"},
            }
        )

        # Act / Assert
        with pytest.raises(InputDataError) as exc:
            validate_input_data(
                wf, {"prior_state": {"demerits": 0}, "excuse": 7}, tmp_path
            )
        assert exc.value.name == "excuse"

    def test_undeclared_payload_keys_are_dropped(self, tmp_path: Path) -> None:
        # Arrange
        _write_schema(tmp_path, "prior_state", PRIOR_STATE_SCHEMA)
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act
        validated = validate_input_data(
            wf, {"prior_state": {"demerits": 2}, "smuggled": "value"}, tmp_path
        )

        # Assert — only declared facts reach the spine.
        assert validated == {"prior_state": {"demerits": 2}}

    def test_workflow_without_input_data_validates_to_empty(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        wf = _workflow(None)

        # Act
        validated = validate_input_data(wf, {"anything": 1}, tmp_path)

        # Assert
        assert validated == {}

    def test_unresolvable_internal_ref_raises_input_data_error(
        self, tmp_path: Path
    ) -> None:
        # Arrange — a typo'd internal $ref, the most common authoring mistake.
        _write_schema(tmp_path, "prior_state", {"$ref": "#/definitions/Typo"})
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act / Assert
        with pytest.raises(InputDataError) as exc:
            validate_input_data(wf, {"prior_state": {"demerits": 1}}, tmp_path)
        assert exc.value.name == "prior_state"
        assert "unresolvable" in str(exc.value)

    def test_unresolvable_relative_file_ref_raises_input_data_error(
        self, tmp_path: Path
    ) -> None:
        # Arrange — the natural way to share fact definitions across entries,
        # but the default registry has no retrieve hook so it cannot resolve.
        _write_schema(tmp_path, "prior_state", {"$ref": "common.json#/PS"})
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act / Assert
        with pytest.raises(InputDataError) as exc:
            validate_input_data(wf, {"prior_state": {"demerits": 1}}, tmp_path)
        assert exc.value.name == "prior_state"
        assert "unresolvable" in str(exc.value)

    def test_remote_ref_raises_without_network_call(self, tmp_path: Path) -> None:
        """A remote $ref must fail loudly, never fetch (SSRF + FR-028)."""
        # Arrange — a local HTTP server that records every request it serves.
        hits: list[str] = []

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib-mandated name
                hits.append(self.path)
                body = json.dumps({"type": "object"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass  # Silence stdlib request logging during the test.

        server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_port
            _write_schema(
                tmp_path,
                "prior_state",
                {"$ref": f"http://127.0.0.1:{port}/gate.json"},
            )
            wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

            # Act / Assert
            with pytest.raises(InputDataError) as exc:
                validate_input_data(wf, {"prior_state": {"demerits": 1}}, tmp_path)
            assert exc.value.name == "prior_state"
            assert "unresolvable" in str(exc.value)
            assert hits == []
        finally:
            server.shutdown()
            thread.join()

    def test_internal_ref_still_resolves_and_validates(self, tmp_path: Path) -> None:
        # Arrange — a schema that $refs a local $defs entry, not the network.
        schema = {
            "$defs": {
                "PriorState": {
                    "type": "object",
                    "properties": {"demerits": {"type": "integer"}},
                    "required": ["demerits"],
                }
            },
            "$ref": "#/$defs/PriorState",
        }
        _write_schema(tmp_path, "prior_state", schema)
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act
        validated = validate_input_data(wf, {"prior_state": {"demerits": 3}}, tmp_path)

        # Assert
        assert validated == {"prior_state": {"demerits": 3}}

    def test_undecodable_schema_file_raises_input_data_error(
        self, tmp_path: Path
    ) -> None:
        # Arrange — a byte that is not valid UTF-8 anywhere in the sequence.
        target = tmp_path / "schemas" / "prior_state.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b'{"type": "str\xe9ing"}')
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act / Assert
        with pytest.raises(InputDataError) as exc:
            validate_input_data(wf, {"prior_state": {"demerits": 1}}, tmp_path)
        assert exc.value.name == "prior_state"
        assert "cannot read schema" in str(exc.value)

    def test_date_time_format_constraint_is_enforced(self, tmp_path: Path) -> None:
        """``format: date-time`` is enforced by a declared dependency, not luck.

        Before pinning ``jsonschema[format-nongpl]`` this format only happened
        to be checked because ``rfc3339-validator`` was pulled in transitively
        by an unrelated dependency (openapi-core); an unrelated dependency bump
        could silently have dropped the check with no test failure. This pins
        the guarantee explicitly.
        """
        # Arrange
        _write_schema(
            tmp_path,
            "prior_state",
            {
                "type": "object",
                "properties": {
                    "recorded_at": {"type": "string", "format": "date-time"}
                },
                "required": ["recorded_at"],
            },
        )
        wf = _workflow({"prior_state": {"schema": "schemas/prior_state.json"}})

        # Act / Assert
        with pytest.raises(InputDataError) as exc:
            validate_input_data(
                wf, {"prior_state": {"recorded_at": "not-a-date-time"}}, tmp_path
            )
        assert exc.value.name == "prior_state"
        assert "schema-invalid" in str(exc.value)

    def test_undeclared_payload_keys_warn_via_logging(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange — no input_data block at all: the whole payload is dropped.
        wf = _workflow(None)

        # Act
        with caplog.at_level("WARNING"):
            validated = validate_input_data(wf, {"smuggled": "value"}, tmp_path)

        # Assert
        assert validated == {}
        assert any(
            "smuggled" in record.getMessage() for record in caplog.records
        ), caplog.text
