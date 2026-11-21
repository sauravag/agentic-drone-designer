"""Streamlit theme: readable contrast (WCAG 2.2 AA) and shipped in the Docker image."""

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
THEME_PATH = REPO_ROOT / ".streamlit" / "config.toml"
WCAG_AA_TEXT = 4.5  # minimum contrast ratio for body text (WCAG 2.2 AA)


def _luminance(hex_color: str) -> float:
    channels = [int(hex_color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _theme() -> dict[str, Any]:
    import tomllib

    return tomllib.loads(THEME_PATH.read_text())["theme"]


def test_theme_is_light_with_readable_text() -> None:
    theme = _theme()
    assert theme["base"] == "light"
    for surface in ("backgroundColor", "secondaryBackgroundColor"):
        assert _contrast(theme["textColor"], theme[surface]) >= WCAG_AA_TEXT, surface
    assert _contrast("#ffffff", theme["primaryColor"]) >= WCAG_AA_TEXT  # primary button label


def test_theme_is_shipped_in_the_docker_image() -> None:
    dockerfile_path = REPO_ROOT / "Dockerfile"
    if not dockerfile_path.is_file():  # running inside the image, which ships no Dockerfile: the theme must be here
        assert THEME_PATH.is_file()
        return
    dockerfile = dockerfile_path.read_text()
    assert "COPY .streamlit ./.streamlit" in dockerfile
    assert ".streamlit" not in (REPO_ROOT / ".dockerignore").read_text()
