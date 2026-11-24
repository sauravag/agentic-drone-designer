# UIUC propeller data (test fixtures)

Source: J. B. Brandt, R. W. Deters, G. K. Ananda and M. S. Selig, *UIUC Propeller Data Site*,
Department of Aerospace Engineering, University of Illinois at Urbana-Champaign.
Index: https://m-selig.ae.illinois.edu/props/propDB.html (Volume 1, APC Thin Electric).
Files downloaded unchanged from https://m-selig.ae.illinois.edu/props/volume-1/data/<file> on 2026-11-24.

Static files: columns RPM, CT, CP. Advance-ratio files: columns J, CT, CP, eta. The tunnel sweeps at one rpm
were run in two parts (low J and high J), so each propeller has two advance-ratio files at about 5000 rpm.
Test conditions were not given in the files; the tests use sea-level air (rho 1.225 kg/m^3, mu 1.81e-5 Pa s,
a 340.3 m/s) since CT, CP and J are non-dimensional.

| Propeller | D x P (in) | Static file | Advance-ratio files |
|---|---|---|---|
| APC 9x6E | 9 x 6 | apce_9x6_static_rd0987.txt | apce_9x6_rd0989_5013.txt, apce_9x6_rd0990_5000.txt |
| APC 10x7E | 10 x 7 | apce_10x7_static_pg0811.txt | apce_10x7_pg0813_5018.txt, apce_10x7_pg0814_5001.txt |
| APC 11x7E | 11 x 7 | apce_11x7_static_kt0534.txt | apce_11x7_kt0538_4997.txt, apce_11x7_kt0539_4997.txt |
