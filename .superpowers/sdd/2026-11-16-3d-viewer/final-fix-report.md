# Final-review fix report

1. Stale mesh: mesh_export.py export_viewer_model unlinks the target before tessellating; nodes.py _build_viewer and pipeline.py _execute build the viewer only when cad_artifacts is set with viewer_model_path not None (write_viewer unchanged). Tests: test_agent (stale model with cad None / viewer_model_path None -> no viewer.html, no warning; existing viewer tests now supply cad_artifacts; restored no-mesh report assertion), test_pipeline (stale model, no cad), test_mesh_export (stale file removed on failed export).
2. viewer_warning catches Exception (build.py); test_viewer parametrized with RuntimeError/ValueError.
3. viewer_panel.py catches (OSError, UnicodeError), caption uses VIEWER_FILE; test_ui invalid-UTF8 test.
4. README outputs row reworded.

RED (src stashed, new tests only): 8 failed, 136 passed. Failures: stale_viewer x2, pipeline stale, mesh_export stale, viewer_warning RuntimeError/ValueError, ui utf8, plus my first draft of the no-mesh report assertion (test bug, fixed: report lists viewer.html by file existence, so the assertion uses a run with no viewer.html).
GREEN: covering files 144 passed; full host suite `.venv/bin/python -m pytest -q -p no:cacheprovider`: 511 passed, 13 skipped. Docker isr-forge: tests/test_aero_cad.py tests/test_mesh_export.py: 60 passed.
