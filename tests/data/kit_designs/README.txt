Saved designs that still build the laser-cut kit (tests/reference_designs.py: saved_kit_design, KIT_DESIGN_CASES).

g1, g2, g2_vtail, g1_hand: the reference missions as master df94b76 sized them (before propulsion matching), the
designs the kit, CAD-packing and fabrication tests were written for. Since the 2026-11-25 matched-propulsion
re-baseline these references build no kit (user: "accept, kit fixes later"): g1 crutch slot < 2 mm, g2 anti-rotation
pin, g2_vtail Group 3, g1_hand aileron reversal. Written on 2026-11-25 from a checkout of that commit with:

    git worktree add /tmp/master df94b76 && cd /tmp/master
    python -c "import json, sys, tempfile; from pathlib import Path; sys.path.insert(0, '.'); \
from src.tools.database.matcher import select_bom; from src.tools.database.seed_data import seed_database; \
from src.tools.sizing.mission_solver import component_needs, size_aircraft; from tests.reference_designs import CASES; \
db = seed_database(Path(tempfile.mkdtemp()) / 'c.db'); \
[[(Path(c) / f'{n}.json').write_text(json.dumps(r.model_dump(mode='json', \
exclude=set(type(r).model_computed_fields)), indent=2) + '\n') for n, r in (('requirements', q), ('sizing', s), \
('bom', select_bom(db, component_needs(s, q))))] for c in ('g1', 'g2', 'g2_vtail', 'g1_hand') \
if Path(c).mkdir(exist_ok=True) is None for q in [CASES[c]] for s in [size_aircraft(q)]]"

When the kit-fix branch makes the matched reference designs build kits again, point the tests back at the
reference designs and delete this folder.
