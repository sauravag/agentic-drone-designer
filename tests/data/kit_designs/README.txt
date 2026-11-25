Saved designs that still build the laser-cut kit (tests/reference_designs.py: saved_kit_design).

g2/ : the G2 reference mission as master df94b76 sized it (19.88 kg, before propulsion matching), written on
2026-11-25 from a checkout of that commit with:

    git worktree add /tmp/master df94b76 && cd /tmp/master
    python -c "import json, sys, tempfile; from pathlib import Path; sys.path.insert(0, '.'); \
from src.tools.database.matcher import select_bom; from src.tools.database.seed_data import seed_database; \
from src.tools.sizing.mission_solver import component_needs, size_aircraft; from tests.reference_designs import CASES; \
req = CASES['g2']; s = size_aircraft(req); d = tempfile.mkdtemp(); \
b = select_bom(seed_database(Path(d) / 'c.db'), component_needs(s, req)); \
[Path(f'{n}.json').write_text(json.dumps(r.model_dump(mode='json', exclude=set(type(r).model_computed_fields)), \
indent=2) + '\n') for n, r in (('requirements', req), ('sizing', s), ('bom', b))]"

No quadplane sized by the matched propulsion builds a kit yet (kit structural limits are fixed in a later branch);
once one does, point the quadplane kit tests at it and delete this folder.
