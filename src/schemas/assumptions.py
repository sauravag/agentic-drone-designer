"""Technology constants the sizing chain relies on, exposed as one validated, frozen model.

Defaults reproduce the historical hard-coded values exactly. Field descriptions double as UI
help text. This module must not import from `src.tools` (seed_data imports it, not vice versa).
"""

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

Chemistry = Literal["li_ion", "lipo"]
SERIES_HYBRID: Final[str] = "series_hybrid"


class DesignAssumptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    liion_specific_energy_wh_kg: float = Field(
        200.0, ge=100.0, le=500.0, description="Li-ion pack specific energy (Wh/kg)."
    )
    liion_energy_density_wh_l: float = Field(
        250.0, ge=100.0, le=1000.0, description="Li-ion pack energy density (Wh/L)."
    )
    liion_max_c_rate: float = Field(
        3.0, ge=1.0, le=30.0, description="Li-ion max continuous discharge rate (C, 1/h)."
    )
    lipo_specific_energy_wh_kg: float = Field(
        155.0, ge=80.0, le=300.0, description="LiPo pack specific energy (Wh/kg)."
    )
    lipo_energy_density_wh_l: float = Field(
        280.0, ge=100.0, le=800.0, description="LiPo pack energy density (Wh/L)."
    )
    lipo_max_c_rate: float = Field(
        25.0, ge=5.0, le=100.0, description="LiPo max continuous discharge rate (C, 1/h)."
    )
    battery_reserve_fraction: float = Field(
        0.20, ge=0.10, le=0.40, description="Unusable battery energy kept in reserve (fraction of capacity)."
    )
    fuel_reserve_fraction: float = Field(
        0.10, ge=0.05, le=0.40, description="Fuel kept in reserve for hybrids (fraction of fuel load)."
    )
    bsfc_kg_per_kwh: float = Field(
        0.50, ge=0.25, le=1.20, description="Generator brake-specific fuel consumption (kg/kWh)."
    )
    drivetrain_efficiency: float = Field(
        0.68, ge=0.40, le=0.95, description="Cruise electrical-to-thrust efficiency (fraction)."
    )
    hover_figure_of_merit: float = Field(
        0.72, ge=0.40, le=0.90, description="Rotor hover figure of merit (fraction)."
    )
    lift_motor_efficiency: float = Field(
        0.85, ge=0.50, le=0.98, description="VTOL lift motor + ESC efficiency (fraction)."
    )

    def reserve_for(self, propulsion: str) -> float:
        """Fuel reserve for series hybrids, battery reserve for everything else."""
        if propulsion == SERIES_HYBRID:
            return self.fuel_reserve_fraction
        return self.battery_reserve_fraction

    def specific_energy_wh_kg(self, chem: Chemistry) -> float:
        return self.liion_specific_energy_wh_kg if chem == "li_ion" else self.lipo_specific_energy_wh_kg

    def energy_density_wh_l(self, chem: Chemistry) -> float:
        return self.liion_energy_density_wh_l if chem == "li_ion" else self.lipo_energy_density_wh_l

    def max_c_rate(self, chem: Chemistry) -> float:
        return self.liion_max_c_rate if chem == "li_ion" else self.lipo_max_c_rate

    def non_default(self) -> dict[str, tuple[float, float]]:
        """Fields that differ from the defaults, as field -> (default, value)."""
        defaults = type(self)()
        return {
            name: (getattr(defaults, name), getattr(self, name))
            for name in type(self).model_fields
            if getattr(self, name) != getattr(defaults, name)
        }

    def with_reserve_floor(self, floor: float) -> "DesignAssumptions":
        """Copy with both reserve fractions raised to at least `floor`."""
        return self.model_copy(
            update={
                "battery_reserve_fraction": max(self.battery_reserve_fraction, floor),
                "fuel_reserve_fraction": max(self.fuel_reserve_fraction, floor),
            }
        )


DEFAULT_ASSUMPTIONS: Final[DesignAssumptions] = DesignAssumptions()
