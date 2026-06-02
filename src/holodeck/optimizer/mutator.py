"""Apply optimizer axes to an Agent without mutating the original.

Every mutation deep-copies the agent first, so the caller's ``Agent`` (and the
original ``agent.yaml`` it came from) is never touched. Axis paths support
dotted attribute access and a ``tools[name=X]`` list selector:

    model.temperature
    tools[name=kb].top_k
    instructions.inline
"""

import re
from typing import Any

from holodeck.lib.errors import OptimizerError
from holodeck.models.agent import Agent

# A path segment of the form ``tools[name=kb]`` — selects a list element by a
# field value rather than a positional index.
_SELECTOR_RE = re.compile(r"^(?P<attr>\w+)\[(?P<key>\w+)=(?P<value>[^\]]+)\]$")


def _navigate(obj: Any, segment: str, full_path: str) -> Any:
    """Resolve one path segment against ``obj``.

    Args:
        obj: Current object in the walk.
        segment: A plain attribute name or a ``attr[key=value]`` selector.
        full_path: The full axis path, for error messages.

    Returns:
        The resolved child object.

    Raises:
        OptimizerError: If the attribute, collection, or selector target is
            missing.
    """
    match = _SELECTOR_RE.match(segment)
    if match:
        attr = match.group("attr")
        key = match.group("key")
        value = match.group("value")
        collection = getattr(obj, attr, None)
        if collection is None:
            raise OptimizerError(
                f"Cannot resolve axis '{full_path}': '{attr}' is empty or absent."
            )
        for item in collection:
            if str(getattr(item, key, None)) == value:
                return item
        raise OptimizerError(
            f"Cannot resolve axis '{full_path}': no element in '{attr}' "
            f"with {key}={value!r}."
        )

    if not hasattr(obj, segment):
        raise OptimizerError(
            f"Cannot resolve axis '{full_path}': '{segment}' is not a valid field."
        )
    return getattr(obj, segment)


def _set_path(root: Any, path: str, value: Any) -> None:
    """Set the leaf of a dotted/selector ``path`` on ``root`` to ``value``.

    Raises:
        OptimizerError: If any segment or the leaf field is invalid.
    """
    segments = path.split(".")
    *parents, leaf = segments
    obj = root
    for segment in parents:
        obj = _navigate(obj, segment, path)
    if not hasattr(obj, leaf):
        raise OptimizerError(
            f"Cannot resolve axis '{path}': '{leaf}' is not a valid field."
        )
    setattr(obj, leaf, value)


def get_path(root: Any, path: str) -> Any:
    """Read the value at a dotted/selector ``path`` on ``root``.

    Args:
        root: The object to read from (e.g. an Agent).
        path: Dotted/selector path (e.g. ``instructions.inline``).

    Returns:
        The resolved value.

    Raises:
        OptimizerError: If any segment of the path is invalid.
    """
    obj = root
    for segment in path.split("."):
        obj = _navigate(obj, segment, path)
    return obj


def apply_axes(agent: Agent, params: dict[str, Any]) -> Agent:
    """Return a new Agent with the given numeric-axis values applied.

    Args:
        agent: The baseline agent (left unmodified).
        params: Mapping of axis path → value (e.g. ``{"model.temperature": 0.7}``).

    Returns:
        A deep copy of ``agent`` with each axis set to its value.

    Raises:
        OptimizerError: If any axis path cannot be resolved.
    """
    candidate = agent.model_copy(deep=True)
    for path, value in params.items():
        _set_path(candidate, path, value)
    return candidate


def overlay_axes(template: Agent, source: Agent, axis_paths: list[str]) -> Agent:
    """Return ``template`` with each axis path's value copied from ``source``.

    Used to rebuild the optimized agent on top of the original *unsubstituted*
    config so ``${VAR}`` secret placeholders survive into ``best.yaml``: only the
    optimizer's tuned axes are taken from the env-resolved ``source``; every other
    field — including templated credentials — stays as it is in ``template``.

    Args:
        template: The unsubstituted source agent (``${VAR}`` intact); not modified.
        source: The env-resolved, optimized agent to read axis values from.
        axis_paths: Axis paths to copy from ``source`` onto ``template``.

    Returns:
        A deep copy of ``template`` with the axis values overlaid.

    Raises:
        OptimizerError: If any axis path cannot be resolved on either agent.
    """
    values = {path: get_path(source, path) for path in axis_paths}
    return apply_axes(template, values)


def apply_textual_edit(agent: Agent, axis: str, new_text: str) -> Agent:
    """Return a new Agent with a single instruction axis rewritten.

    Args:
        agent: The baseline agent (left unmodified).
        axis: Path to the instruction text (e.g. ``instructions.inline``).
        new_text: Replacement text.

    Returns:
        A deep copy of ``agent`` with the targeted text replaced.

    Raises:
        OptimizerError: If the axis path cannot be resolved.
    """
    return apply_axes(agent, {axis: new_text})
