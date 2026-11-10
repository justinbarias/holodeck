"""Click group that imports each subcommand only when it is used.

Command modules pull in heavy optional stacks (evaluators with pandas and
DeepEval, the agent SDKs, Semantic Kernel, the HTTP server). Importing all of
them on every invocation made ``holodeck init`` take as long to start as
``holodeck test``. A lazy group imports a command module the first time the
command runs or ``--help`` lists it.
"""

from __future__ import annotations

import importlib
import threading

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
        self._load_lock = threading.Lock()

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Return all command names sorted, as ``click.Group`` does."""
        return sorted({*super().list_commands(ctx), *self._lazy_subcommands})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Return the command, importing its module on first use."""
        if cmd_name in self.commands or cmd_name not in self._lazy_subcommands:
            return super().get_command(ctx, cmd_name)
        # Serialize first loads: concurrent lookups must all get the command.
        with self._load_lock:
            loaded = self.commands.get(cmd_name)
            return loaded if loaded is not None else self._load(cmd_name)

    def _load(self, cmd_name: str) -> click.Command:
        module_name, attribute = self._lazy_subcommands[cmd_name].split(":", 1)
        module = importlib.import_module(module_name)
        command = vars(module).get(attribute)
        if not isinstance(command, click.Command):
            raise TypeError(
                f"lazy command {cmd_name!r}: {module_name}:{attribute} is "
                f"{type(command).__name__}, not a click.Command"
            )
        # Cache as an eager command so later lookups skip the lock and import.
        self.add_command(command, cmd_name)
        return command
