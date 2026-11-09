"""Click group that imports each subcommand only when it is used.

Command modules pull in heavy optional stacks (evaluators with pandas and
DeepEval, the agent SDKs, Semantic Kernel, the HTTP server). Importing all of
them on every invocation made ``holodeck init`` take as long to start as
``holodeck test``. A lazy group imports a command module the first time the
command runs or ``--help`` lists it.
"""

from __future__ import annotations

import importlib

import click


class LazyGroup(click.Group):
    """A ``click.Group`` whose subcommands are given as ``"module:attribute"``."""

    def __init__(
        self,
        *args: object,
        lazy_subcommands: dict[str, str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._lazy_subcommands = dict(lazy_subcommands or {})

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Return all command names sorted, as ``click.Group`` does."""
        return sorted({*super().list_commands(ctx), *self._lazy_subcommands})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Return the command, importing its module on first use."""
        if cmd_name in self._lazy_subcommands:
            return self._load(cmd_name)
        return super().get_command(ctx, cmd_name)

    def _load(self, cmd_name: str) -> click.Command:
        module_name, attribute = self._lazy_subcommands[cmd_name].split(":", 1)
        module = importlib.import_module(module_name)
        command = vars(module).get(attribute)
        if not isinstance(command, click.Command):
            raise TypeError(
                f"lazy command {cmd_name!r}: {module_name}:{attribute} is "
                f"{type(command).__name__}, not a click.Command"
            )
        # Cache as an eager command so later lookups skip the import path.
        self.add_command(command, cmd_name)
        del self._lazy_subcommands[cmd_name]
        return command
