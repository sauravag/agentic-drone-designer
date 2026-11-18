"""Technology constants the sizing chain relies on, exposed as one validated, frozen model.

Defaults reproduce the historical hard-coded values exactly. Field descriptions double as UI
help text. This module must not import from `src.tools` (seed_data imports it, not vice versa).
"""

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.requirements import ARK_JETSON_PAB

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

    # --- avionics (defaults researched 2026-11-18; docs/superpowers/specs/2026-11-18-companion-computer-design.md)
    avionics_base_mass_kg: float = Field(
        0.227, ge=0.0, le=5.0, description="Shared avionics allowance: GPS, radios, power module, wiring (kg)."
    )
    avionics_mass_per_mtow: float = Field(
        0.01, ge=0.0, le=0.05, description="Shared avionics allowance growth (kg per kg of MTOW)."
    )
    avionics_base_power_w: float = Field(
        10.0, ge=1.0, le=200.0, description="Shared avionics power incl. the flight controller's own draw (W)."
    )
    avionics_power_per_kg_w: float = Field(
        0.4, ge=0.0, le=5.0, description="Shared avionics power growth (W per kg of MTOW)."
    )
    cube_orange_plus_mass_kg: float = Field(
        0.073, ge=0.03, le=0.5, description="CubePilot Cube Orange+ with ADS-B carrier board (kg)."
    )
    ark_pab_stack_mass_kg: float = Field(
        0.166, ge=0.08, le=1.0, description="ARK Jetson PAB carrier + ARKV6X + Jetson Orin NX + heatsink (kg)."
    )
    companion_computer_mass_kg: float = Field(
        0.2, ge=0.02, le=3.0, description="Separate Jetson Orin NX + its carrier + heatsink, Cube only (kg)."
    )
    jetson_power_w: float = Field(
        25.0, ge=1.0, le=100.0, description="Jetson Orin NX power, drawn for the whole flight when fitted (W)."
    )

    def avionics_mass_kg(self, mtow_kg: float, flight_controller: str, companion_computer: bool) -> float:
        """Shared allowance + the chosen flight-controller stack (+ the separate Jetson for a Cube)."""
        is_ark: bool = flight_controller == ARK_JETSON_PAB
        stack: float = self.ark_pab_stack_mass_kg if is_ark else self.cube_orange_plus_mass_kg
        companion: float = self.companion_computer_mass_kg if companion_computer and not is_ark else 0.0
        return self.avionics_base_mass_kg + stack + self.avionics_mass_per_mtow * mtow_kg + companion

    def avionics_power_w(self, mtow_kg: float, jetson_fitted: bool) -> float:
        """Shared allowance power (+ the Jetson when one is fitted), drawn for the whole flight."""
        jetson: float = self.jetson_power_w if jetson_fitted else 0.0
        return self.avionics_base_power_w + self.avionics_power_per_kg_w * mtow_kg + jetson

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
