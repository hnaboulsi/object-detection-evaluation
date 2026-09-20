# Furniture assets

Sixteen separate OBJ/MTL pairs from Quaternius' Ultimate House Interior Pack
(June 2020), released under CC0 1.0. See `CC0-LICENSE.txt`.

`scenes/assets.py` defines labels, source-axis scales, display colors and the
Y-up to Z-up rotation. Floor offsets are computed from source vertex bounds.
The renderer uses one mesh per furniture instance and flat registry colors;
MTL files are retained for provenance but are not interpreted.

Place personal assets under the ignored `assets/local/` directory. The curated
benchmark references only the vendored files, so a fresh clone is sufficient.
