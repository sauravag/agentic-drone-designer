# Airframe materials for Group 1 and Group 2 fixed-wing ISR drones

Date: 2026-11-21. Status: research only; no source or test file changed.

## 0. How to read this report

- Tags: **[DS]** manufacturer datasheet read in full; **[DS-d]** datasheet or distributor copy read through a search summary only, not verified against the original; **[P]** paper; **[Retail]** retailer or distributor page; **[Forum]** hobby forum or blog, practice only; **[Est]** my estimate, reasoning given. Anything untagged is derived by me from tagged numbers, with the arithmetic shown.
- Source gaps (stated up front, because the user's brief assumes things I could not confirm):
  - No JSP ARPRO datasheet was readable at jsp.com. The EPP numbers come from a distributor copy of it and from the BASF Neopolen data as reproduced by Knauf, both through search summaries [DS-d].
  - No datasheet exists in public for hobby "EPO". It is a trade term, and the polymer blend varies by moulder.
  - "EPF": I assumed it is a typo for EPO. I ran no search for a material by that name, so "no such material" is untested.
  - Public construction data for Raven, Puma, ScanEagle, Aerosonde and Integrator is thin (section 4). The brief's pairing "Group 1 = EPO/EPP foam" is true for hobby-derived kits like the Skywalker X8. For the AeroVironment Group 1 aircraft I found only "Kevlar composite" statements, not foam.
  - I could not read the Hexcel M79 mechanical tables, only its cure and Tg page.

## 1. Executive summary

| Topic | Headline |
|---|---|
| EPP | 30 to 60 g/l are the usable grades. Compressive stress at 10% strain is about 0.12 to 0.3 MPa and tensile strength about 0.4 to 0.9 MPa. Modulus is a few MPa (E about 4 MPa at 30 g/l, about 8 MPa at 80 g/l). Shear modulus is not published; my estimate is 1.5 to 3 MPa. This is 40 to 90 times softer in shear than balsa (the tool's 130 MPa). |
| EPO | No datasheet. Treat as EPP-like with a smoother surface and somewhat lower durability [Retail]. Density estimate 25 to 40 g/l. |
| Group 1 stiffness | The foam is not a structural member. The carbon tube must carry bending and torsion. The foam's solid-section torsion adds about 25% of a 16x14 mm tube's GJ on a 250 mm chord (my calculation, section 5), so it is worth counting as a small credit but not worth relying on. |
| Group 2 sandwich | 200 g/m² 3K plain carbon cures to 0.20 mm and about 290 g/m² (Hexcel's 50% Vf convention). ROHACELL 51/71 are the working cores. 71 IG-F: 75 kg/m³, G 29 MPa, shear strength 1.3 MPa. 71 WF: G 34 MPa. |
| Cure | Hexcel M79 cures at 70 to 120 C (Tg dry 110 C, wet 100 C). Its nominal autoclave cycle is 5 bar, which exceeds the 0.3 MPa (3 bar) limit of ROHACELL IG-F. Use WF (0.7 MPa, 130 C) for any pressure cure, or vacuum-bag only. 8552 at 180 C needs WF-HT. |
| Torsion in Group 2 | A closed carbon skin gives GJ about 1,600 N m² on a 250 mm chord, about 100 times a 16 mm tube. The tool's spar-only torsion model would badly under-credit a sandwich wing. |

Three most important implications for the tool: see section 7 (summary at its top).

## 2. EPO and EPP (Group 1)

### 2.1 Material data

**EPP, JSP ARPRO distributor copy [DS-d]** (imperial values converted by me):

| Density | Comp. 10% | Comp. 25% | Tensile | Elongation |
|---|---|---|---|---|
| 30 g/l | 18 psi = 124 kPa | 23.5 psi = 162 kPa | 55.5 psi = 383 kPa | 18% |
| 60 g/l | 44 psi = 303 kPa | 57 psi = 393 kPa | 89 psi = 614 kPa | 15% |

**BASF Neopolen P 9235, via Knauf [DS-d]:** at 35 to 70 kg/m³, compressive stress at 10% (ISO 844) is 150 to 390 kPa and tensile strength (ISO 1798) is 530, 600, 740, 880, 1020 kPa across the density band. P 9270 at 70 to 110 kg/m³: tensile 1020 to 1530 kPa, elongation 18 to 25%, water absorption under 1 vol.%.

**Where they disagree:** at about 60 g/l the distributor copy gives 0.61 MPa tensile, while Neopolen interpolates to roughly 0.85 MPa (the five tensile steps are presumably evenly spaced in density, which I did not verify). A third datasheet in the search results gives 0.62 MPa tensile and 19 MPa flexural modulus at 60 g/L, but its table was garbled, so I do not use it. The spread is about 30%; use the lower value for design allowables.

**Modulus [P]:** Static EPP study (PMC7825443): E = 3.5 MPa at 20 g/dm³, 4.3 at 30, 8.2 at 80, 16.0 at 120, 92 at 200. These are the elastic constants the authors fed into FE models, not measured values, and the paper says nothing about tension or shear. Compressive "strength" 0.24 MPa at 20 g/dm³, 0.42 at 30, 0.95 at 80.

**Shear:** no published shear modulus or strength found for EPP. [Est] Isotropic closed-cell foam with nu about 0.3 gives G = E / 2.6, so 1.5 to 3 MPa at 30 to 60 g/l. Shear strength [Est] about 0.7 x tensile, so 0.3 to 0.5 MPa. Both need a coupon test (ASTM C273) before they are used for any allowable.

**Density to use:** hobby and kit airframes are 30 to 60 g/l. [Est] I would use 35 (wings, fuselage skins), 45 (stiffer wings), 60 (belly or nose zones), because the Neopolen and ARPRO grids bracket these. Moulded parts have a denser skin and a lighter core, so measured part density is typically 10 to 20% above the bead density [Est, not sourced].

**EPO:** no datasheet found. A retailer describes it as sitting between EPS and EPP, with a smoother surface than EPP, and calls it less durable than EPP [Retail]. [Est] density 25 to 40 g/l, mechanical properties 70 to 100% of EPP at equal density. Treat as unverified.

**Temperature limits:** none of the sources I read states the EPP or EPO service temperature. [Est] polypropylene-based bead foam is generally quoted to about 100 C continuous, but that is general knowledge and not from these sources. For a drone parked in a hot car or sun (60 to 80 C) the foam is fine; it will creep under sustained load, so do not preload it.

### 2.2 Adhesives and repair practice

All of this is [Forum] and [Retail] practice, not datasheet data.

- Foam-safe CA is the common field repair for EPP and EPO. A retailer says standard CA works on EPO; other sources warn that ordinary CA (with acetone carriers) can melt EPO, so foam-safe is the safer default [Retail, Forum].
- Beacon Foam-Tac, UHU Por, polyurethane glue and medium CA are listed as working on EPP; hot glue is mentioned for EPP repair [Retail, Forum]. Foam-Tac stays flexible, which suits impact-prone joints.
- **Epoxy on EPP/EPO: caveat to the brief.** The user's brief lists CA/epoxy. I found no source on epoxy bond strength to polyolefin foam. [Est] polypropylene has a low surface energy, so bonds are largely mechanical grip into the open cell surface; epoxy works for bonding carbon tube into a routed channel (it fills the cut cells) but gives poor peel strength to a smooth skin. A flexible PU or Foam-Tac joint is the safer default for foam-to-foam.
- Repair: EPP tears rather than cracks and bounces back from impact [Forum]; a stock all-EPP wing is floppy without a stiff skin or spar [Forum]. Belly-skid abrasion is the main wear mechanism in practice.

### 2.3 Moulding versus laser-cut or hot-wire

- No quotes or lead times for EPP steam-chest tools were found. Moulds are traditionally machined from aluminium-alloy castings (paper in the search results). A Virginia Tech thesis reports additively manufactured bead-foam tooling cutting cost by up to 63% and lead time by up to 70% against aluminium tooling for low volumes [P, secondary]. It does not give absolute figures.
- [Est, no source] Aluminium steam-chest tool for a 2 m wing half: tens of thousands of USD and 6 to 12 weeks; a CNC hot-wire core or laser-cut balsa kit: near-zero tooling and days. A moulded airframe pays off only at volumes in the hundreds or more. This is a reasoned guess and should be replaced by a real moulder quote.

### 2.4 Foam wing with carbon spar versus balsa rib wing: what the model must treat differently

Current model (wing_stiffness.py): only the carbon tube carries bending (EI = E_carbon x I_tube); torsion is the tube plus an optional balsa D-box; skin shear is otherwise ignored.

**Does foam torsion count?** My calculation, treating the foam as a solid elliptical section (J = pi a³b³ / (a² + b²), a = c/2, b = t/2), chord 0.25 m, t/c = 0.12:

- J about 3.35e-4 c⁴ = 1.3e-6 m⁴. With G = 1.5 to 3 MPa [Est], GJ_foam is about 2 to 4 N m².
- A 16x14 mm carbon tube with the tool's G = 5 GPa has J = pi/32 (0.016⁴ - 0.014⁴) = 2.66e-9 m⁴ and GJ about 13 N m².
- So the foam adds roughly 15 to 30% of a typical tube's torsional stiffness. It is not negligible, but it rests on an unmeasured G, so credit at most half of it, or not at all in the first model. Foam adds nothing useful in bending (E about 4 MPa against 70 GPa).
- **A balsa D-box is a poor comparison.** Balsa shear modulus in the tool is 130 MPa, 40 to 90 times the foam's. The existing D-box branch (R45) has no foam analogue; foam wings lack a stiff nose skin.

**Skin flex and aileron reversal.** One study of a 15 kg UAV foam-core wing found that only the main spar contributed significantly to wing stiffness [P, Southampton/Cambridge, from search summary]. Reversal is controlled by GJ (sources in the search results agree); foam wings in practice reverse earlier than rib-and-skin wings unless the spar is large or a stiff skin is added. The tool's reversal check (R43) is the right check for foam wings, but its inputs change: tube GJ plus a small foam term, no D-box, and the elastic axis may sit behind the AC because the spar is mid-chord in a deep foam section (the tool assumes spar at 25% chord).

**Local checks that do not transfer:** plate bearing, shear-out and pin checks (R27, R31, R32) are written for birch ply. Foam crush strength is 0.12 to 0.3 MPa, about 50 to 100 times lower than the tool's 15 MPa ply bearing allowable, so the spar tube must be bonded over a long length and the load shared along it.

## 3. Group 2 composite sandwich

### 3.1 Prepreg and fabric

From the Hexcel UAV brochure [DS], whose thickness is "the theoretical thickness of a cured ply with 50% fiber volume content":

| Fabric | Weave | gsm | Cured ply mm |
|---|---|---|---|
| 46075 IM7 6K | plain | 75 | 0.07 |
| 43098 AS4 3K | plain | 98 | 0.09 |
| 43161 AS4C 3K | plain | 160 | 0.16 |
| 43199 AS4C 3K | plain | 200 | 0.20 |
| 43280 AS4 3K | 5H satin | 280 | 0.29 |
| G0986 AS4C 6K | twill 2x2 | 285 | 0.29 |
| E-glass 1080 | plain | 48 | 0.06 |
| E-glass 7581 | 8H satin | 300 | 0.23 |
| S2 6781 | 8H satin | 305 | 0.23 |

- Hexcel 8552 AS4 plain weave, 196 g/m²: 0.199 mm at 37% resin by weight, 55.6% Vf [DS-d, from search summary].
- **Cured areal mass of a 200 gsm fabric ply:** at 0.20 mm and Vf 0.5, density = 0.5 x 1790 (AS4, Hexcel public value, from memory [Est]) + 0.5 x 1150 (M79 matrix 1.1 to 1.2 g/cm³ [DS]) = about 1,470 kg/m³, so 0.20 mm x 1470 = **294 g/m² cured** (fabric 200 + resin 94). A purchased prepreg at 42% resin is about 345 g/m² before cure bleed.

### 3.2 Foam cores: ROHACELL

IG-F, April 2022 [DS, read in full] (typical values):

| Property | 31 IG-F | 51 IG-F | 71 IG-F | 110 IG-F |
|---|---|---|---|---|
| Density kg/m³ | 32 +/- 7 | 52 +/- 12 | 75 +/- 15 | 110 +/- 21 |
| Compressive strength MPa | 0.4 | 0.9 | 1.5 | 3.0 |
| Compressive modulus MPa | 17 | 43 | 73 | 120 |
| Tensile strength MPa | 1.0 | 1.9 | 2.8 | 3.5 |
| Tensile modulus MPa | 36 | 70 | 92 | 160 |
| Shear strength MPa | 0.4 | 0.8 | 1.3 | 2.4 |
| Shear modulus MPa | 13 | 19 | 29 | 50 |

Processing limit for IG-F: 130 C and **0.3 MPa** (resin infusion, RTM, prepreg).

WF / WF-HT, May 2026 [DS, read through a PDF text extraction with some cells elided]: typical (minimum in brackets):

| Property | 51 WF | 71 WF | 110 WF |
|---|---|---|---|
| Density kg/m³ | 52 +/- 12 | 75 +/- 15 | 110 +/- 21 |
| Shear modulus MPa | 20 (14) | 34 (24.1) | 60 (40) |
| Shear strength MPa | 0.8 (0.5) | 1.3 (1.0) | 2.4 (1.75) |
| Elastic modulus MPa | 75 (45) | 105 (75) | 180 (135) |
| Compressive strength MPa | 0.6 (0.4) | 1.3 (1.05) | not read |

WF cures to 130 C at up to **0.7 MPa**; WF-HT to 180 C at 0.7 MPa. Evonik states the WF foam is isotropic, giving "no need for any kind of potting or reinforcement" at load introductions and edges (marketing statement; see section 3.6 for the contrary design practice). 31 WF is not in the May 2026 WF sheet.

**Disagreement:** IG-F and WF 51/71 differ (WF is stiffer and Evonik quotes minimum values for it). A retailer copy gave the 31 IG-F shear modulus as 13 GPa, a units typo for 13 MPa. For design with a knockdown, use the WF minimums, not the typical values.

### 3.3 Nomex honeycomb alternative

Hexcel HexWeb HRH-10, 3.2 mm cell, room temperature, typical [DS, read in full]:

| Density kg/m³ | Bare comp. MPa | L shear MPa / G MPa | W shear MPa / G MPa |
|---|---|---|---|
| 29 | 0.64 | 0.62 / 26 | 0.34 / 10 |
| 48 | 2.07 | 1.21 / 41 | 0.69 / 24 |
| 64 | 3.3 | 1.76 / 59 | 0.97 / 32 |
| 80 | 4.8 | 2.24 / 70 | 1.21 / 37 |

Nomex softens at 230 to 260 C. Properties are strongly anisotropic (W direction about half of L). [Est] against foam: honeycomb has better shear stiffness per unit mass above 48 kg/m³ and is lighter at equal shear modulus, but it is harder to form to a thin curved wing section, can crush under vacuum or press pressure, and thin skins can dimple into cells. Foam gives a smoother skin and is machined or hot-wired to the airfoil. Not sourced; engineering judgement.

### 3.4 Vacuum bag, autoclave, out-of-autoclave

From Hexcel UAV brochure table [DS]: cure and Tg (dry onset DMA):

| Resin | Cure | Dry Tg | Autoclave | Vacuum-only |
|---|---|---|---|---|
| M91 | 180 C / 120 min | 185 C | yes | no |
| 8552 | 180 C / 120 min | 195 C | yes | no |
| M56 | 180 C / 120 min | 195 C | yes | yes (OOA, low-density resin) |
| 913 | 125 C / 60 min | 125 C | yes | no |
| M79 | 70 C / 480 min or 120 C / 60 min | 110 C | yes | subject to fabric style |
| M49 | 120 C / 60 min | 105 C | no | subject to fabric style |
| M901 | 130 to 160 C / 15 to 30 min | 175 C | yes | subject to fabric style |

M79 datasheet [DS]: dry Tg 110 +/- 5 C, wet 100 +/- 5 C; nominal autoclave cycle full vacuum plus **5 bar**, 60 min at 120 C; out life 60 days at 23 C; shelf life 24 months at -18 C; "designed for autoclave cure to obtain optimum mechanical performance". Its UAV positioning makes it the natural small-run choice.

**Foam interaction (derived from the datasheets above):**
- 5 bar is 0.5 MPa. That exceeds IG-F's 0.3 MPa limit but is inside WF's 0.7 MPa. So an M79 autoclave cure needs WF, or the autoclave pressure must be reduced to 3 bar or less.
- 180 C prepregs (8552, M56, M91) exceed WF's 130 C limit. They need WF-HT (180 C). That is why 120 C systems (M79, M49, 913) are the natural pairing with standard cores.
- Vacuum-bag only is about 0.1 MPa, inside every limit, but is only allowed for M79/M49/M56/M901 depending on fabric.
- [Est, not sourced] Small-run economics: a vacuum-bag oven cure needs no autoclave and suits 10s of units; autoclave time is bought per run from a contract autoclave. PMI cores should be dried before cure; check Evonik's guidance, which I did not read.

### 3.5 Aramid belly and leading edge

- Kevlar fabric [Retail, Aircraft Spruce via search summary]: style 120 is a 1.8 oz/yd² plain weave (about 61 g/m², 0.0035 in dry); style 285 is a 5 oz/yd² 4-harness satin (about 170 g/m², 0.010 in dry). Kevlar 49 density about 1.44 g/cm³ [Est, general knowledge].
- No source I found gives ply counts for belly or leading-edge protection. [Est] Practice: 1 to 2 plies of 170 g/m² aramid outside the carbon skin along the belly strip and the leading edge, each about 0.28 kg/m² cured (170 / 0.6 fibre mass fraction). Aramid is hard to cut (use scissors) and absorbs moisture, so it is often co-cured under a finishing layer. A belly strip of 0.3 m x 1 m (g2 scale) with 2 plies adds about 170 g.
- Kevlar prepreg is used because aramid resists abrasion and impact tearing where carbon shatters; the Hexcel brochure lists "impact" as a key attribute for wings and fuselage but does not discuss aramid.

### 3.6 CNC aluminium hardpoints for catapult hooks

I found no source specific to catapult hooks. What was found: potted inserts are the most common attachment in sandwich panels; compressive load passes through the face sheets around the potting; the bracket or foot should extend beyond the potting; apply a 1.15 fitting factor to potted inserts; check the shear stress at the potting perimeter against core shear strength (Abbott Aerospace CMH-17 style notes, via search summary [P-secondary]). The ESA Insert Design Handbook is the primary reference and I did not read it. Hexcel lists HexBond EA9394 paste adhesive for potting and bonding, ST1480 and 679 film adhesives for metal-to-metal and honeycomb bonding [DS].

- Alloy [Est]: 7075-T73 (stress-corrosion resistant) or 6061-T6 (cheaper, tougher, easier to anodise); 2024-T3 if fatigue-critical. 7075-T6 is higher strength but more prone to stress-corrosion cracking. Not sourced.
- Bond prep [Est]: grit-blast and anodise or primer; paste adhesive for potting.
- Rules of thumb [Est]: replace the core locally with a solid laminate or filled-epoxy potting; the flange spreads the load; design ultimate hook load = MTOW x g x launch load factor x 1.5, with a launch factor of 3 to 6 (a guess; I found no catapult acceleration source). Size potting-perimeter shear against the core's design shear (use the minimum value). Evonik's claim that WF needs no potting conflicts with the standard insert practice above; I would follow the standard practice for a hook that sees tens of g-times-MTOW.

## 4. Real examples

| Aircraft | Group | What is publicly known | Source |
|---|---|---|---|
| RQ-11 Raven | 1 | 1.4 m span; "constructed primarily of Kevlar" (weak source). No foam statement found. | globalmilitary.net, AV datasheet [Retail] |
| Puma AE | 1 | 110 in span, 12.8 lb; structure "modular, Kevlar composite" in a Puma-AE system manual; Puma 3 "principally a Kevlar-based composite". No foam statement found. | system manual PDF (eff.org copy), SAE Mobility Engineering article |
| Skywalker X8 / X-UAV | 1 (hobby kit) | Moulded EPO, carbon spars (builders say 10 mm stock), retailers list 2,000 g airframe, 80 dm² wing, 3,000 to 3,500 g typical flying weight, 3,500 g max AUW. Fuselage material not confirmed. | Retail, ArduPilot forum [Forum] |
| ScanEagle | 2 | 3.1 m span, five field-replaceable modules (nose, fuselage, avionics, wings, propulsion). Cobham made winglets by hand layup and bladder moulding and the nose housing in carbon/epoxy. Wing and fuselage skin materials not found. | CompositesWorld, Wikipedia |
| Aerosonde Mk 4.4 | 2 | Pod fuselage, twin booms, inverted V tail, "constructed mainly of composites". Fibre and core not named. | NASA Airborne Science |
| Integrator | 2 | Nothing found. | none |
| Hunter UAV | 2 | Centre wing in woven carbon/epoxy prepreg. | CompositesWorld |

Weak spot in the brief: the user's mapping of Group 1 to moulded EPP/EPO is supported only for hobby-derived kits. A DTIC report "Airframe Structural Materials for Drone Applications" (AD0736616) appeared in the results and I did not read it; it is the first lead to follow for real military foam-airframe data.

## 5. What the tool models today (read from the five files)

| Area | Current parameters (defaults) |
|---|---|
| Sheets (fabrication.py) | Birch ply 680 kg/m³, 3 mm and 1.5 mm; birch bearing allowable 15 MPa, shear 7 MPa |
| Wood | stringer density 160 (balsa), 3 mm stringer, LE 6 mm, TE 5x15 mm; balsa shear modulus 130 MPa; D-box sheet 1.5 mm |
| Carbon tube/rod | density 1600 kg/m³, E 70 GPa, G 5 GPa, allowable 500 MPa, shear allowable 60 MPa |
| Overwrap | 200 g/m² cloth, 0.25 mm layer, 1500 kg/m³, allowable 250 MPa, up to 12 layers, epoxy lap shear 15 MPa |
| Covering | film 0.09 kg/m²; fuselage cover 0.24 kg/m² (1.5 mm balsa); glue fraction 0.10 |
| Loads | safety factor 1.5; n_limit 4 |
| Mass model (mass_model.py) | wing areal density = (0.9 + 0.006 x wing loading[N/m²]) x (AR/10)^0.6 kg/m²; tail 0.6 x that; fuselage (1.0 + 0.002 x MTOW) kg/m² of wetted area; gear 4% (fixed wing) or 1.5% (quadplane) of MTOW; systems 5% of MTOW |
| Stiffness (wing_stiffness.py) | EI = E_carbon x I_tube; GJ = G_carbon x J_tube (+ balsa Bredt D-box); tip deflection limit 5% of semi-span at 1 g; reversal speed must exceed 1.15 x V_NE; spar on aerodynamic centre; no flutter, gust, or skin shear |
| Kit mass (kit_mass.py) | Mass of laser-cut parts, tubes, linear stock, overwrap, covering, glue (10% of structure, not of film, fuselage cover, overwrap) |

Observation: the whole stiffness and strength chain is hard-wired to a carbon tube as the single bending member, with balsa and birch named in the parameters.

## 6. Proposed parameter values

All new names are suggestions. Existing field names are used where they fit.

### 6.1 Group 1: EPP / EPO foam + pultruded tube

| Parameter | Value | Basis | Fits existing name? |
|---|---|---|---|
| foam density | 35 / 45 / 60 kg/m³ | section 2.1 | NEW (foam_density_kg_m3); the nearest is stringer_density_kg_m3 |
| foam shear modulus | 2 MPa (range 1.5 to 3) | [Est] E/2.6 from [P] | NEW; analogous to balsa_shear_modulus_pa |
| foam compressive allowable | 0.12 MPa at 30 g/l; 0.30 MPa at 60 g/l | [DS-d] 10% strain | NEW; replaces ply_bearing_allow_pa for tube-in-foam bearing |
| foam shear allowable | 0.3 MPa | [Est] | NEW; replaces ply_shear_allow_pa |
| carbon_density_kg_m3 | keep 1600 | typical | existing |
| carbon_modulus_pa | 70 GPa conservative for roll-wrapped; 100 to 130 GPa for pultruded UD | [Est, not sourced] | existing; tube catalogue entry should carry its own modulus |
| carbon_shear_modulus_pa | 5 GPa roll-wrapped; about 3 GPa pultruded UD | [Est] | existing |
| film_kg_m2 | 0.03 to 0.09 (tape or film) | [Est] | existing |
| fuselage_cover_kg_m2 | 0 for self-skinned moulded EPP | assumption | existing |
| glue_fraction | 0.03 to 0.05 | [Est]; foam glue joints are light | existing |
| dbox_* | not applicable (no stiff nose skin) | | remove or make optional |
| gear fraction | 0 to 0.02 (belly skid) | [Est] | existing, GEAR_FRACTION |

Wing mass model check [Est]: foam volume per planform area is about 0.68 x (t/c) x c. For t/c 0.10, c = 0.25 m: 0.017 m x 35 kg/m³ = 0.6 kg/m² foam alone, plus 0.05 to 0.09 film or tape, plus the tube. That is comparable to or below the model's 0.9 + 0.006 x W/S (1.26 kg/m² at 60 N/m², AR 10). Unlike the existing model, foam wing mass scales with chord and with foam density, not with wing loading. The X8 data point (airframe 2,000 g over 0.80 m², about 2.5 kg/m² of wing area, if the retailer's figure includes the fuselage and spars) suggests moulded EPO kits are heavier than the balsa-kit model. That single Retail figure is ambiguous and I would not calibrate to it.

### 6.2 Group 2: carbon/glass skins on PMI foam (or Nomex)

| Parameter | Value | Basis | Fits existing name? |
|---|---|---|---|
| skin ply cured thickness | 0.20 mm (200 gsm 3K plain); 0.29 mm (280 to 285 gsm); glass 7581 0.23 mm | [DS] Hexcel | overwrap_layer_thickness_m (0.25 mm default) is close; make it per material |
| skin cloth areal weight | 200 g/m² dry, about 294 g/m² cured | [DS] plus derived | overwrap_cloth_g_m2 (200) fits; note the existing validator requires cloth gsm < thickness x density |
| cured density | 1,470 kg/m³ (Vf 0.5) | derived | overwrap_density_kg_m3 (1500) fits |
| prepreg woven allowable | 250 MPa for hand-laid wet fabric (current); [Est] 350 to 450 MPa design allowable for cured prepreg fabric, to be set from supplier coupon data | no datasheet allowable read | overwrap_allow_pa; new field for prepreg |
| skin in-plane shear modulus (+/-45 fabric) | about 20 GPa | [Est] | NEW |
| core density | 52 (51 WF), 75 (71 WF) | [DS] | NEW (core_density) |
| core shear modulus | 20 and 34 MPa typical; 14 and 24 MPa minimum | [DS] | NEW; analogue of balsa_shear_modulus_pa |
| core shear allowable | 0.5 and 1.0 MPa (WF minimum), 0.8 and 1.3 typical | [DS] | NEW |
| aramid belly/LE | 0.28 kg/m² per ply; 1 to 2 plies; Kevlar 1.44 g/cm³ | [Est] plus [Retail] | NEW local area mass item |
| hardpoint | aluminium 7075-T73 or 6061-T6; mass from machined volume x 2,810 or 2,700 kg/m³ | [Est] | NEW; its own checks (section 7) |
| film adhesive/potting | paste EA9394, film ST1480 or 679 | [DS] | glue_fraction 0.05 to 0.10 for co-cured [Est] |

Wing mass model check [Est]: two faces x two plies of 200 gsm cured at 294 g/m² gives 1.18 kg/m², plus 5 to 8 mm of 71 foam at 75 kg/m³ (0.38 to 0.60 kg/m²), plus adhesive and finish about 0.1 to 0.2: roughly 1.7 to 2.0 kg/m². The existing formula gives (0.9 + 0.006 x 250) x (AR/10)^0.6 = 2.4 kg/m² at W/S 250 N/m² and AR 10, so a Group 2 sandwich wing of this build is lighter than the existing formula by about 20%. Within the formula's accuracy; no change needed except for the AR and loading terms, which a sandwich design drives through skin ply count.

## 7. Implications for the tool

**The three that matter most:**

1. **The torsion model is wrong for both groups, in opposite directions.** Foam wings (Group 1) are spar-dominated but the tool ignores a foam torsion term worth 15 to 30% of the tube. Sandwich wings (Group 2) are skin-dominated: a closed 0.4 mm carbon (+/-45, G about 20 GPa) skin on a 250 mm chord gives GJ = 4 A² G t / p = 4 x (5.1e-3 m²)² x 20e9 x 0.0004 / 0.515 m about 1,600 N m², about 100 times the 13 N m² of a 16 mm tube. The Bredt branch exists (balsa D-box, R45) so the machinery can be reused with the skin's G and t and the whole airfoil contour as the cell. Reversal (R43) then becomes a non-issue for sandwich wings and the check should say so rather than fail on a tube-only number.
2. **Replace "spar tube carries everything" with a material library.** Names like stringer_density_kg_m3, balsa_shear_modulus_pa, dbox_sheet_thickness_m, ply_bearing_allow_pa, ply_shear_allow_pa are wood-specific. A material record (density, E, G, allowables, areal mass) selected per component (core, skin, local reinforcement) would carry foam and PMI as well as balsa and ply. The tool's units and SI conventions already fit.
3. **Several checks need new physics, not new numbers:**
   - Foam core shear check and shear deflection (not needed for balsa rib wings, where ribs carry load).
   - Sandwich skin bending (EI from face sheets plus core, replacing or adding to the tube EI), including wrinkling and face-sheet stress. Tip deflection (R44) will change by a large factor for sandwich wings.
   - Tube-in-foam bearing and bond length for Group 1 (replacing R27/R31/R32 birch plate checks).
   - Hardpoint check: potting-perimeter shear against core shear, bracket flange bearing, 1.15 fitting factor, launch load case. Current load cases contain flight and hover only.
   - Cure compatibility rule: core limit pressure and temperature against the prepreg cycle (e.g. IG-F 0.3 MPa against M79's 5 bar).

**Structural mass fraction.** The current model is (wing + tail + fuselage areal masses) + gear 4% + systems 5%, with structure derived as empty mass minus propulsion, avionics and systems. Effects:
- Group 1 foam: mass scales with chord and foam density (about 0.6 kg/m² at 35 g/l, 10% t/c, 0.25 m chord), not with wing loading; tail and fuselage can use the same foam model; gear fraction falls toward 0 to 2%; glue fraction falls. A single data point (X8) suggests foam kits can sit above the balsa-kit formula; do not calibrate on it alone.
- Group 2 sandwich: about 1.7 to 2.0 kg/m² wing at the build above, so about 20% below the formula at W/S 250 N/m²; add aramid strip mass (about 0.28 kg/m² per ply over the strip area) and hardpoint mass explicitly.

**Stiffness checks.**
- R44 tip deflection: unchanged for Group 1 (the tube still governs). For Group 2, use sandwich EI; the existing check would be very conservative.
- R43 reversal: Group 1 needs a foam torsion term (optional, credit half) and drops the D-box; Group 2 uses skin GJ. Note the tool's assumption of spar on the aerodynamic centre may not hold in a deep foam section.
- R25 interaction ratio uses carbon tube stress allowables; for sandwich wings the skin stress replaces it.

**Open questions for the user:**
1. Should Group 1 be tool-buildable as moulded EPP at all, given that no moulding tool exists in the kit model? Hot-wire or CNC-cut EPP cores need a different manufacturing output than a laser bed.
2. Is the Group 1 definition really EPP/EPO foam? Public sources for Raven and Puma say Kevlar composite.
3. Are coupon tests acceptable to fix the EPP shear modulus, since no published value exists?
4. What launch load factor and hook geometry should the hardpoint be designed to?
5. Is a prepreg allowable (not a wet-layup one) available from a chosen supplier?

## 8. Sources

Datasheets read in full: Evonik ROHACELL IG-F, April 2022; Evonik ROHACELL WF/WF-HT, May 2026 (some cells elided in my extraction); Hexcel HexWeb HRH-10 (2017); Hexcel HexPly M79 (2026, cure and Tg page only); Hexcel UAV Solutions brochure (2026).
Search summaries only: JSP ARPRO distributor copy (psi values); Knauf Neopolen P 9235 and P 9270; HexPly 8552 AS4 plain weave; Abbott Aerospace insert notes; CompositesWorld, NASA and Wikipedia pages on the example aircraft; Aircraft Spruce Kevlar fabric pages.
Papers: PMC7825443 (static mechanical properties of EPP crushable foam); Virginia Tech thesis on rapid tooling for bead foam; Southampton/Cambridge morphing UAV wing study; MDPI 2504-446X/9/2/99 (PVC foam and carbon UAV wing).
Forum and retail: ArduPilot forum (X8 builds), RCUniverse, Model Aviation "Stuck on Foam", motionrc, FMS Hobby, hobbyking.
