"""Fabrication settings for the laser-cut airframe kit: bed, stock, and structural allowances.

All values are SI (metres, kg/m^3, kg/m^2, Pa). Defaults describe a 600 x 400 mm laser bed,
birch ply sheet, and carbon tube / rod stock. Must not import from `src.tools`.
"""

import math
from typing import Final, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PartFamily = Literal["rib", "former", "crutch", "plate", "doubler", "clamp", "bulkhead"]
ALL_FAMILIES: Final[tuple[str, ...]] = get_args(PartFamily)

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

DEFAULT_SHEETS: Final[tuple[SheetStock, ...]] = (
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


# --------------------------------------------------------------------- defaults (D1: named, not literals) ---
DEFAULT_BED_WIDTH_M: Final[float] = 0.600
DEFAULT_BED_HEIGHT_M: Final[float] = 0.400
DEFAULT_MARGIN_M: Final[float] = 0.005
DEFAULT_KERF_M: Final[float] = 0.00015
DEFAULT_TUBES: Final[tuple[TubeStock, ...]] = _tubes_from_mm(tuple(sorted(_THIN_WALL_MM + _THICK_WALL_MM)))
DEFAULT_RODS_M: Final[tuple[float, ...]] = (0.003, 0.004, 0.005)
DEFAULT_STRINGER_SIDE_M: Final[float] = 0.003
DEFAULT_STRINGER_DENSITY_KG_M3: Final[float] = 160.0  # balsa
DEFAULT_LE_SIDE_M: Final[float] = 0.006
DEFAULT_TE_HEIGHT_M: Final[float] = 0.005
DEFAULT_TE_CHORD_M: Final[float] = 0.015
DEFAULT_FILM_KG_M2: Final[float] = 0.09
DEFAULT_FUSELAGE_COVER_KG_M2: Final[float] = 0.24  # 1.5 mm balsa sheet
DEFAULT_FUSELAGE_COVER_THICKNESS_M: Final[float] = 0.0015
DEFAULT_GLUE_FRACTION: Final[float] = 0.10
DEFAULT_CARBON_DENSITY_KG_M3: Final[float] = 1600.0
DEFAULT_CARBON_MODULUS_PA: Final[float] = 70e9
DEFAULT_SIGMA_ALLOW_PA: Final[float] = 500e6
DEFAULT_SAFETY_FACTOR: Final[float] = 1.5
DEFAULT_N_LIMIT: Final[float] = 4.0
DEFAULT_MIN_INSERTION_DIAMETERS: Final[float] = 3.0
DEFAULT_FIT_CLEARANCE_M: Final[float] = 0.0002
DEFAULT_CARBON_SHEAR_ALLOW_PA: Final[float] = 60e6
DEFAULT_PLY_BEARING_ALLOW_PA: Final[float] = 15e6
DEFAULT_PLY_SHEAR_ALLOW_PA: Final[float] = 7e6
DEFAULT_OVERWRAP_LAYER_THICKNESS_M: Final[float] = 0.25e-3
DEFAULT_OVERWRAP_CLOTH_G_M2: Final[float] = 200.0
DEFAULT_OVERWRAP_ALLOW_PA: Final[float] = 250e6
DEFAULT_OVERWRAP_DENSITY_KG_M3: Final[float] = 1500.0
DEFAULT_OVERWRAP_MAX_LAYERS: Final[int] = 12
DEFAULT_EPOXY_LAP_SHEAR_PA: Final[float] = 15e6
DEFAULT_CRUTCH_RAIL_MIN_M: Final[float] = 0.008
MARGIN_SIDES: Final[float] = 2.0  # the margin is kept on both edges of each bed side


class FabricationSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bed_width_m: float = Field(DEFAULT_BED_WIDTH_M, gt=0.0, description="Laser bed width (m).")
    bed_height_m: float = Field(DEFAULT_BED_HEIGHT_M, gt=0.0, description="Laser bed height (m).")
    margin_m: float = Field(
        DEFAULT_MARGIN_M, ge=0.0,
        description="Clear margin kept around the bed edge and between parts (m).",
    )
    kerf_m: float = Field(DEFAULT_KERF_M, ge=0.0, description="Laser kerf width (m).")
    sheets: tuple[SheetStock, ...] = Field(
        DEFAULT_SHEETS,
        description="Sheet stock; each part family on exactly one sheet.",
    )
    tubes: tuple[TubeStock, ...] = Field(
        DEFAULT_TUBES, min_length=1,
        description="Available carbon tube stock (sorted by OD)."
    )
    rods_m: tuple[float, ...] = Field(
        DEFAULT_RODS_M, min_length=1,
        description="Available solid carbon rod diameters (m).",
    )
    stringer_side_m: float = Field(DEFAULT_STRINGER_SIDE_M, gt=0.0, description="Square balsa stringer side (m).")
    stringer_density_kg_m3: float = Field(
        DEFAULT_STRINGER_DENSITY_KG_M3, gt=0.0,
        description="Stringer wood density (kg/m^3).",
    )
    le_side_m: float = Field(DEFAULT_LE_SIDE_M, gt=0.0, description="Leading-edge square stock side (m).")
    te_height_m: float = Field(DEFAULT_TE_HEIGHT_M, gt=0.0, description="Trailing-edge stock height (m).")
    te_chord_m: float = Field(DEFAULT_TE_CHORD_M, gt=0.0, description="Trailing-edge stock chordwise width (m).")
    film_kg_m2: float = Field(DEFAULT_FILM_KG_M2, gt=0.0, description="Covering film areal density (kg/m^2).")
    fuselage_cover_kg_m2: float = Field(
        DEFAULT_FUSELAGE_COVER_KG_M2, gt=0.0,
        description="Fuselage skin areal density (kg/m^2).",
    )
    fuselage_cover_thickness_m: float = Field(
        DEFAULT_FUSELAGE_COVER_THICKNESS_M, gt=0.0,
        description="Fuselage skin thickness (m).",
    )
    glue_fraction: float = Field(
        DEFAULT_GLUE_FRACTION, ge=0.0, le=1.0,
        description="Adhesive mass as a fraction of structure mass.",
    )
    carbon_density_kg_m3: float = Field(
        DEFAULT_CARBON_DENSITY_KG_M3, gt=0.0,
        description="Carbon tube / rod density (kg/m^3).",
    )
    carbon_modulus_pa: float = Field(
        DEFAULT_CARBON_MODULUS_PA, gt=0.0,
        description="Carbon tube / rod bending (Young's) modulus (Pa), for boom stiffness."
    )
    sigma_allow_pa: float = Field(DEFAULT_SIGMA_ALLOW_PA, gt=0.0, description="Allowable carbon stress (Pa).")
    safety_factor: float = Field(DEFAULT_SAFETY_FACTOR, ge=1.0, description="Ultimate-load safety factor.")
    n_limit: float = Field(DEFAULT_N_LIMIT, gt=0.0, description="Limit load factor (g).")
    min_insertion_diameters: float = Field(
        DEFAULT_MIN_INSERTION_DIAMETERS, gt=0.0,
        description="Minimum spar insertion into the fuselage socket (skin to centreline), in root-tube diameters.",
    )
    fit_clearance_m: float = Field(
        DEFAULT_FIT_CLEARANCE_M, ge=0.0,
        description="Slot / hole clearance for press fits (m).",
    )
    carbon_shear_allow_pa: float = Field(
        DEFAULT_CARBON_SHEAR_ALLOW_PA, gt=0.0,
        description="Allowable in-plane shear of roll-wrapped carbon tube (Pa), for spar torsion."
    )
    ply_bearing_allow_pa: float = Field(
        DEFAULT_PLY_BEARING_ALLOW_PA, gt=0.0,
        description="Allowable birch-ply edge bearing stress (Pa).",
    )
    ply_shear_allow_pa: float = Field(
        DEFAULT_PLY_SHEAR_ALLOW_PA, gt=0.0,
        description="Allowable birch aviation-ply panel shear stress (Pa), for shear-out checks."
    )
    overwrap_layer_thickness_m: float = Field(
        DEFAULT_OVERWRAP_LAYER_THICKNESS_M, gt=0.0,
        description="Cured thickness of one wet-laid carbon cloth layer (m)."
    )
    overwrap_cloth_g_m2: float = Field(
        DEFAULT_OVERWRAP_CLOTH_G_M2, gt=0.0,
        description="Areal weight of the dry carbon cloth one overwrap layer is laid from (g/m^2)."
    )
    overwrap_allow_pa: float = Field(
        DEFAULT_OVERWRAP_ALLOW_PA, gt=0.0,
        description="Allowable stress of hand-laid 0/90 cloth (Pa).",
    )
    overwrap_density_kg_m3: float = Field(
        DEFAULT_OVERWRAP_DENSITY_KG_M3, gt=0.0,
        description="Cured overwrap density (kg/m^3).",
    )
    overwrap_max_layers: int = Field(
        DEFAULT_OVERWRAP_MAX_LAYERS, ge=1,
        description="Most cloth layers a centreline overwrap may have.",
    )
    epoxy_lap_shear_pa: float = Field(
        DEFAULT_EPOXY_LAP_SHEAR_PA, gt=0.0,
        description="Epoxy lap-shear strength (Pa); sizes the wrap lap.",
    )
    crutch_rail_min_m: float = Field(
        DEFAULT_CRUTCH_RAIL_MIN_M, gt=0.0, description="Minimum crutch side-rail width beside a component opening (m)."
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

    @model_validator(mode="after")
    def _margin_leaves_a_bed(self) -> "FabricationSettings":
        """D4: 2 x margin_m must stay below the bed's smaller side, or no part fits anywhere."""
        smaller = min(self.bed_width_m, self.bed_height_m)
        if MARGIN_SIDES * self.margin_m >= smaller:
            raise ValueError(f"margin_m ({self.margin_m:g}) leaves no usable bed: twice it must be below the bed's "
                             f"smaller side ({smaller:g})")
        return self

    @field_validator("tubes", mode="after")
    @classmethod
    def _sort_tubes(cls, v: tuple[TubeStock, ...]) -> tuple[TubeStock, ...]:
        """Sorted by OD (then ID); D5: no tube listed twice."""
        ordered = tuple(sorted(v, key=lambda t: (t.od_m, t.id_m)))
        for a, b in zip(ordered, ordered[1:]):
            if a == b:
                raise ValueError(f"duplicate tube {a.od_m / MM:g} x {a.id_m / MM:g} mm")
        return ordered

    @field_validator("rods_m", mode="after")
    @classmethod
    def _rods_positive_and_unique(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        """D3: every rod diameter positive; no rod listed twice."""
        if any(r <= 0.0 for r in v):
            raise ValueError("rod diameters must be positive")
        seen: set[float] = set()
        for r in v:
            if r in seen:
                raise ValueError(f"duplicate rod diameter {r / MM:g} mm")
            seen.add(r)
        return v

    def sheet_for(self, family: PartFamily) -> SheetStock:
        """Return the single sheet that part family is cut from."""
        return next(sheet for sheet in self.sheets if family in sheet.families)

    @property
    def usable_bed_m(self) -> tuple[float, float]:
        """(width, height) of the bed inside `margin_m` on every edge: where nested parts may go."""
        return self.bed_width_m - 2.0 * self.margin_m, self.bed_height_m - 2.0 * self.margin_m


DEFAULT_FABRICATION: Final[FabricationSettings] = FabricationSettings()


def changed_fields(before: FabricationSettings, after: FabricationSettings) -> tuple[str, ...]:
    """Names of the settings that differ between `before` and `after`, in field order."""
    return tuple(name for name in FabricationSettings.model_fields if getattr(before, name) != getattr(after, name))
