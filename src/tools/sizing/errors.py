"""Sizing errors shared by the sizing loop and the propulsion matching (kept apart to avoid an import cycle)."""


class SizingDivergenceError(RuntimeError):
    """The mission cannot close with this powertrain/airframe (mass spiral or beyond Group 3)."""
