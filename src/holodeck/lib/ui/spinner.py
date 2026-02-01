"""Spinner animation utilities.

Provides mixin class for spinner animation functionality used in progress indicators.
"""

from typing import ClassVar


class SpinnerMixin:
    """Mixin providing spinner animation functionality.

    Provides braille spinner characters and rotation logic for progress indicators.
    Classes using this mixin should initialize _spinner_index = 0 in their __init__.

    Class Attributes:
        SPINNER_CHARS: List of braille characters for spinner animation.

    Instance Attributes:
        _spinner_index: Current position in spinner rotation (must be initialized).
    """

    SPINNER_CHARS: ClassVar[list[str]] = [
        "\u280b",  # ⠋
        "\u2819",  # ⠙
        "\u2839",  # ⠹
        "\u2838",  # ⠸
        "\u283c",  # ⠼
        "\u2834",  # ⠴
        "\u2826",  # ⠦
        "\u2827",  # ⠧
        "\u2807",  # ⠇
        "\u280f",  # ⠏
    ]
    _spinner_index: int

    def get_spinner_char(self) -> str:
        """Get current spinner character and advance rotation.

        Returns:
            Current spinner character from the braille sequence.
        """
        char = self.SPINNER_CHARS[self._spinner_index % len(self.SPINNER_CHARS)]
        self._spinner_index += 1
        return char
