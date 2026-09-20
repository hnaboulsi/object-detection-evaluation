# Local OBJ assets

Put optional `.obj` files here. They are ignored by Git because research assets can be large or have separate licenses.

The loader treats one OBJ file as one semantic object. It does not load MTL files or textures. OBJ origin, units, and scale are not standardized, so adjust `--target-scale` and the generated target pose when an asset appears displaced or incorrectly sized.
