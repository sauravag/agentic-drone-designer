"""COTS hardware schemas. Column names follow SRS §4.2; `capacity_wh` and `bsfc_kg_kwh` are
catalog (I/O-boundary) units and are exposed in SI via `capacity_j` / `bsfc_kg_per_j`."""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from src.schemas.requirements import CUBE_ORANGE_PLUS, FlightControllerId
from src.units import KWH_TO_J, WH_TO_J

# Any length >= 3 m on a single component almost certainly means a mm/inch unit leak.
Length = Annotated[float, Field(gt=0, lt=3.0)]
Power = Annotated[float, Field(gt=0, le=250_000.0)]
Voltage = Annotated[float, Field(gt=0, le=1_000.0)]

MIN_SPECIFIC_ENERGY_WH_KG: float = 80.0
MAX_SPECIFIC_ENERGY_WH_KG: float = 500.0
MIN_VTOL_ROTORS: int = 4


class _Component(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Optional[int] = None
    name: str = Field(..., min_length=1, max_length=120)
    mass_kg: float = Field(..., gt=0, le=200.0)


class CruiseMotor(_Component):
    max_power_w: Power
    continuous_power_w: Power
    kv: float = Field(..., gt=0, le=5_000.0)
    voltage_v: Voltage
    diameter_m: Length
    length_m: Length

    @model_validator(mode="after")
    def _continuous_within_max(self) -> "CruiseMotor":
        if self.continuous_power_w > self.max_power_w:
            raise ValueError("continuous_power_w cannot exceed max_power_w")
        return self


class VtolMotor(_Component):
    max_thrust_n: float = Field(..., gt=0, le=20_000.0)
    max_power_w: Power
    voltage_v: Voltage
    diameter_m: Length
    length_m: Length


class IceGenerator(_Component):
    continuous_power_w: Power
    bsfc_kg_kwh: float = Field(..., ge=0.2, le=1.5)
    fuel_type: Literal["gasoline", "jp8", "diesel"]
    length_m: Length
    width_m: Length
    height_m: Length

    @property
    def bsfc_kg_per_j(self) -> float:
        return self.bsfc_kg_kwh / KWH_TO_J


class Battery(_Component):
    chemistry: Literal["li_ion", "lipo"]
    capacity_wh: float = Field(..., gt=0, le=100_000.0)
    max_c_rate: float = Field(..., gt=0, le=100.0)
    voltage_v: Voltage
    length_m: Length
    width_m: Length
    height_m: Length

    @model_validator(mode="after")
    def _physical_specific_energy(self) -> "Battery":
        e_spec: float = self.capacity_wh / self.mass_kg
        if not MIN_SPECIFIC_ENERGY_WH_KG <= e_spec <= MAX_SPECIFIC_ENERGY_WH_KG:
            raise ValueError(f"non-physical specific energy {e_spec:.0f} Wh/kg for {self.name}")
        return self

    @property
    def capacity_j(self) -> float:
        return self.capacity_wh * WH_TO_J

    @property
    def specific_energy_wh_kg(self) -> float:
        return self.capacity_wh / self.mass_kg

    @property
    def max_power_w(self) -> float:
        # A C-rate of 1 drains the full capacity in one hour, so P_max = E[Wh] * C.
        return self.capacity_wh * self.max_c_rate


class Propeller(_Component):
    diameter_m: float = Field(..., gt=0, le=2.5)
    pitch_m: float = Field(..., gt=0, le=2.0)
    max_power_w: Power
    application: Literal["cruise", "vtol"]


SensorType = Literal["eo", "eo_ir", "sar", "lidar", "multispectral", "hyperspectral", "sigint", "comms_relay"]


class Payload(_Component):
    sensor_type: SensorType
    power_w: float = Field(..., ge=0, le=5_000.0)
    length_m: Length
    width_m: Length
    height_m: Length
    vendor: Optional[str] = None  # None: a generic sensor class (what Custom payloads are matched against)
    reference_price_usd: Optional[str] = None  # display only, researched 2026-11-18


class _Avionics(_Component):
    """Not a catalog lookup: built from DesignAssumptions; the price is reference text, never a design input."""

    power_w: Optional[float] = Field(default=None, ge=0, le=1_000.0)  # None: inside the shared avionics allowance
    reference_price_usd: str = Field(..., min_length=1, max_length=120)


class FlightController(_Avionics):
    pass


class CompanionComputer(_Avionics):
    pass


class ComponentNeeds(BaseModel):
    """SI requirements emitted by the sizing engine and consumed by the matcher."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cruise_power_w: float = Field(..., gt=0)
    battery_energy_j: float = Field(..., gt=0)
    battery_peak_power_w: float = Field(..., gt=0)
    payload_mass_kg: float = Field(..., gt=0)
    payload_sensor_type: Optional[SensorType] = None
    payload_power_w: Optional[float] = Field(default=None, ge=0)
    battery_chemistry: Optional[Literal["li_ion", "lipo"]] = None
    vtol_thrust_per_motor_n: Optional[float] = Field(default=None, gt=0)
    vtol_motor_count: int = Field(default=MIN_VTOL_ROTORS, ge=MIN_VTOL_ROTORS, le=8)
    generator_power_w: Optional[float] = Field(default=None, gt=0)
    flight_controller: FlightControllerId = CUBE_ORANGE_PLUS
    companion_computer: bool = False


class SelectedBOM(BaseModel):
    """Discrete COTS bill of materials. Serialises directly to bom.json."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cruise_motor: CruiseMotor
    cruise_propeller: Propeller
    battery: Battery
    payload: Payload
    vtol_motor: Optional[VtolMotor] = None
    vtol_propeller: Optional[Propeller] = None
    vtol_motor_count: int = 0
    ice_generator: Optional[IceGenerator] = None
    flight_controller: Optional[FlightController] = None  # None only in bom.json written before the avionics choice
    companion_computer: Optional[CompanionComputer] = None

    @model_validator(mode="after")
    def _consistent_vtol_set(self) -> "SelectedBOM":
        # Either a complete lift system (motor + prop + >=4 rotors) or none at all.
        has_motor: bool = self.vtol_motor is not None
        has_prop: bool = self.vtol_propeller is not None
        complete: bool = has_motor and has_prop and self.vtol_motor_count >= MIN_VTOL_ROTORS
        absent: bool = not has_motor and not has_prop and self.vtol_motor_count == 0
        if not (complete or absent):
            raise ValueError(
                f"Inconsistent VTOL set: motor={has_motor}, propeller={has_prop}, count={self.vtol_motor_count}; "
                f"need all three (count >= {MIN_VTOL_ROTORS}) or none (count == 0)"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_hardware_mass_kg(self) -> float:
        single: float = sum(
            part.mass_kg
            for part in (self.cruise_motor, self.cruise_propeller, self.battery, self.payload, self.ice_generator,
                         self.flight_controller, self.companion_computer)
            if part is not None
        )
        rotor_set: float = sum(p.mass_kg for p in (self.vtol_motor, self.vtol_propeller) if p is not None)
        return single + self.vtol_motor_count * rotor_set
