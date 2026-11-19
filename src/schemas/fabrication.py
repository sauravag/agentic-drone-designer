"""Fabrication settings for the laser-cut airframe kit: bed, stock, and structural allowances.

All values are SI (metres, kg/m^3, kg/m^2, Pa). Defaults describe a 600 x 400 mm laser bed,
birch ply sheet, and carbon tube / rod stock. Must not import from `src.tools`.
"""

import math
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PartFamily = Literal["rib", "former", "crutch", "plate", "doubler", "clamp", "bulkhead"]
ALL_FAMILIES: Final[tuple[str, ...]] = (
    "rib", "former", "crutch", "plate", "doubler", "clamp", "bulkhead",
)

MM: Final[float] = 1e-3
BIRCH_DENSITY_KG_M3: Final[float] = 680.0


class SheetStock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Sheet stock label shown in the kit manifest.")
    thickness_m: float = Field(gt=0.0, description="Sheet thickness (m).")
    density_kg_m3: float = Field(gt=0.0, description="Sheet material density (kg/m^3).")
    families: tuple[PartFamily, ...] = Field(description="Part families cut from this sheet.")


class TubeStock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    od_m: float = Field(gt=0.0, description="Outer diameter (m).")
    id_m: float = Field(ge=0.0, description="Inner diameter (m); 0 means a solid rod.")

    @model_validator(mode="after")
    def _id_below_od(self) -> "TubeStock":
        if self.id_m >= self.od_m:
            raise ValueError(f"id_m ({self.id_m}) must be below od_m ({self.od_m})")
        return self

    @property
    def area_m2(self) -> float:
        """Cross-section area of the material (m^2)."""
        return math.pi / 4.0 * (self.od_m**2 - self.id_m**2)

    @property
    def second_moment_m4(self) -> float:
        """Second moment of area about a diameter (m^4)."""
        return math.pi / 64.0 * (self.od_m**4 - self.id_m**4)


def _tubes_from_mm(pairs: tuple[tuple[float, float], ...]) -> tuple[TubeStock, ...]:
    return tuple(TubeStock(od_m=od * MM, id_m=idm * MM) for od, idm in pairs)


_THIN_WALL_MM: Final[tuple[tuple[float, float], ...]] = (
    (6, 4), (8, 6), (10, 8), (12, 10), (14, 12), (16, 14), (18, 16), (20, 18), (22, 20), (25, 23),
)
_THICK_WALL_MM: Final[tuple[tuple[float, float], ...]] = (
    (20, 16), (24, 20), (28, 24), (32, 28), (35, 31), (36, 32), (40, 36),
)

_DEFAULT_SHEETS: Final[tuple[SheetStock, ...]] = (
    SheetStock(
        name="Birch ply 3 mm",
        thickness_m=3.0 * MM,
        density_kg_m3=BIRCH_DENSITY_KG_M3,
        families=("former", "crutch", "plate", "doubler", "clamp", "bulkhead"),
    ),
    SheetStock(
        name="Birch ply 1.5 mm",
        thickness_m=1.5 * MM,
        density_kg_m3=BIRCH_DENSITY_KG_M3,
        families=("rib",),
    ),
)


class FabricationSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bed_width_m: float = Field(0.600, gt=0.0, description="Laser bed width (m).")
    bed_height_m: float = Field(0.400, gt=0.0, description="Laser bed height (m).")
    margin_m: float = Field(0.005, ge=0.0, description="Clear margin kept around the bed edge and between parts (m).")
    kerf_m: float = Field(0.00015, ge=0.0, description="Laser kerf width (m).")
    sheets: tuple[SheetStock, ...] = Field(_DEFAULT_SHEETS, description="Sheet stock; each part family on exactly one sheet.")
    tubes: tuple[TubeStock, ...] = Field(
        _tubes_from_mm(_THIN_WALL_MM + _THICK_WALL_MM), description="Available carbon tube stock (sorted by OD)."
    )
    rods_m: tuple[float, ...] = Field((0.003, 0.004, 0.005), description="Available solid carbon rod diameters (m).")
    stringer_side_m: float = Field(0.003, gt=0.0, description="Square balsa stringer side (m).")
    stringer_density_kg_m3: float = Field(160.0, gt=0.0, description="Stringer wood density (kg/m^3).")
    le_side_m: float = Field(0.006, gt=0.0, description="Leading-edge square stock side (m).")
    te_height_m: float = Field(0.005, gt=0.0, description="Trailing-edge stock height (m).")
    te_chord_m: float = Field(0.015, gt=0.0, description="Trailing-edge stock chordwise width (m).")
    film_kg_m2: float = Field(0.09, gt=0.0, description="Covering film areal density (kg/m^2).")
    fuselage_cover_kg_m2: float = Field(0.24, gt=0.0, description="Fuselage skin areal density (kg/m^2).")
    fuselage_cover_thickness_m: float = Field(0.0015, gt=0.0, description="Fuselage skin thickness (m).")
    glue_fraction: float = Field(0.10, ge=0.0, le=1.0, description="Adhesive mass as a fraction of structure mass.")
    carbon_density_kg_m3: float = Field(1600.0, gt=0.0, description="Carbon tube / rod density (kg/m^3).")
    sigma_allow_pa: float = Field(500e6, gt=0.0, description="Allowable carbon stress (Pa).")
    safety_factor: float = Field(1.5, ge=1.0, description="Ultimate-load safety factor.")
    n_limit: float = Field(4.0, gt=0.0, description="Limit load factor (g).")
    insertion_fraction: float = Field(0.20, ge=0.0, le=1.0, description="Spar insertion depth as a fraction of its length.")
    fit_clearance_m: float = Field(0.0002, ge=0.0, description="Slot / hole clearance for press fits (m).")

    @model_validator(mode="after")
    def _validate_sheets_and_sort_tubes(self) -> "FabricationSettings":
        listed = [family for sheet in self.sheets for family in sheet.families]
        for family in ALL_FAMILIES:
            count = listed.count(family)
            if count != 1:
                raise ValueError(f"part family '{family}' must be on exactly one sheet, found {count}")
        # Frozen model: bypass the frozen guard once to store the sorted copy.
        object.__setattr__(self, "tubes", tuple(sorted(self.tubes, key=lambda t: t.od_m)))
        return self

    def sheet_for(self, family: PartFamily) -> SheetStock:
        """Return the single sheet that part family is cut from."""
        return next(sheet for sheet in self.sheets if family in sheet.families)


DEFAULT_FABRICATION: Final[FabricationSettings] = FabricationSettings()
