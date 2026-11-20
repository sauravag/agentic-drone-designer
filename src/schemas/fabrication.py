"""Fabrication settings for the laser-cut airframe kit: bed, stock, and structural allowances.

All values are SI (metres, kg/m^3, kg/m^2, Pa). Defaults describe a 600 x 400 mm laser bed,
birch ply sheet, and carbon tube / rod stock. Must not import from `src.tools`.
"""

import math
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PartFamily = Literal["rib", "former", "crutch", "plate", "doubler", "clamp", "bulkhead"]
ALL_FAMILIES: Final[tuple[str, ...]] = (
    "rib", "former", "crutch", "plate", "doubler", "clamp", "bulkhead",
)

MM: Final[float] = 1e-3
BIRCH_DENSITY_KG_M3: Final[float] = 680.0
G_PER_KG: Final[float] = 1000.0


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

    @property
    def polar_moment_m4(self) -> float:
        """Polar second moment of area J = pi (OD^4 - ID^4) / 32 (m^4)."""
        return math.pi / 32.0 * (self.od_m**4 - self.id_m**4)


STOCK_DECIMALS: Final[int] = 9  # catalogue sizes are whole mm: round away the float dust of mm * 1e-3 (1e-9 m)


def _tubes_from_mm(pairs: tuple[tuple[float, float], ...]) -> tuple[TubeStock, ...]:
    return tuple(TubeStock(od_m=round(od * MM, STOCK_DECIMALS), id_m=round(idm * MM, STOCK_DECIMALS))
                 for od, idm in pairs)


_THIN_WALL_MM: Final[tuple[tuple[float, float], ...]] = (
    (6, 4), (8, 6), (10, 8), (12, 10), (14, 12), (16, 14), (18, 16), (20, 18), (22, 20), (25, 23),
)
_THICK_WALL_MM: Final[tuple[tuple[float, float], ...]] = (
    (20, 16), (24, 20), (28, 24), (32, 28), (36, 32), (40, 36),
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
        _tubes_from_mm(tuple(sorted(_THIN_WALL_MM + _THICK_WALL_MM))), min_length=1,
        description="Available carbon tube stock (sorted by OD)."
    )
    rods_m: tuple[float, ...] = Field((0.003, 0.004, 0.005), min_length=1, description="Available solid carbon rod diameters (m).")
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
    carbon_modulus_pa: float = Field(
        70e9, gt=0.0, description="Carbon tube / rod bending (Young's) modulus (Pa), for boom stiffness."
    )
    sigma_allow_pa: float = Field(500e6, gt=0.0, description="Allowable carbon stress (Pa).")
    safety_factor: float = Field(1.5, ge=1.0, description="Ultimate-load safety factor.")
    n_limit: float = Field(4.0, gt=0.0, description="Limit load factor (g).")
    min_insertion_diameters: float = Field(
        3.0, gt=0.0,
        description="Minimum spar insertion into the fuselage socket (skin to centreline), in root-tube diameters.",
    )
    fit_clearance_m: float = Field(0.0002, ge=0.0, description="Slot / hole clearance for press fits (m).")
    carbon_shear_allow_pa: float = Field(
        60e6, gt=0.0, description="Allowable in-plane shear of roll-wrapped carbon tube (Pa), for spar torsion."
    )
    ply_bearing_allow_pa: float = Field(15e6, gt=0.0, description="Allowable birch-ply edge bearing stress (Pa).")
    ply_shear_allow_pa: float = Field(
        7e6, gt=0.0, description="Allowable birch aviation-ply panel shear stress (Pa), for shear-out checks."
    )
    overwrap_layer_thickness_m: float = Field(
        0.25e-3, gt=0.0, description="Cured thickness of one wet-laid carbon cloth layer (m); must exceed the dry cloth's areal mass at the cured density."
    )
    overwrap_cloth_g_m2: float = Field(
        200.0, gt=0.0, description="Areal weight of the dry carbon cloth one overwrap layer is laid from (g/m^2)."
    )
    overwrap_allow_pa: float = Field(250e6, gt=0.0, description="Allowable stress of hand-laid 0/90 cloth (Pa).")
    overwrap_density_kg_m3: float = Field(1500.0, gt=0.0, description="Cured overwrap density (kg/m^3).")
    overwrap_max_layers: int = Field(12, ge=1, description="Most cloth layers a centreline overwrap may have.")
    epoxy_lap_shear_pa: float = Field(15e6, gt=0.0, description="Epoxy lap-shear strength (Pa); sizes the wrap lap.")
    crutch_rail_min_m: float = Field(
        0.008, gt=0.0, description="Minimum crutch side-rail width beside a component opening (m)."
    )

    @model_validator(mode="after")
    def _validate_family_coverage(self) -> "FabricationSettings":
        listed = [family for sheet in self.sheets for family in sheet.families]
        for family in ALL_FAMILIES:
            count = listed.count(family)
            if count != 1:
                raise ValueError(f"part family '{family}' must be on exactly one sheet, found {count}")
        return self

    @model_validator(mode="after")
    def _cloth_fits_its_layer(self) -> "FabricationSettings":
        """The dry cloth must weigh less than the cured layer it makes (fibre mass fraction below 1)."""
        cured_g_m2 = self.overwrap_layer_thickness_m * self.overwrap_density_kg_m3 * G_PER_KG
        if self.overwrap_cloth_g_m2 >= cured_g_m2:
            raise ValueError(f"overwrap_cloth_g_m2 ({self.overwrap_cloth_g_m2:g}) must be below the cured layer's "
                             f"areal mass ({cured_g_m2:g} g/m^2 from overwrap_layer_thickness_m x "
                             f"overwrap_density_kg_m3)")
        return self

    @field_validator("tubes", mode="after")
    @classmethod
    def _sort_tubes(cls, v: tuple[TubeStock, ...]) -> tuple[TubeStock, ...]:
        return tuple(sorted(v, key=lambda t: (t.od_m, t.id_m)))

    def sheet_for(self, family: PartFamily) -> SheetStock:
        """Return the single sheet that part family is cut from."""
        return next(sheet for sheet in self.sheets if family in sheet.families)


DEFAULT_FABRICATION: Final[FabricationSettings] = FabricationSettings()
