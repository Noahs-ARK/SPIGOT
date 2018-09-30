Python bindings for ARGMAX_STE
=======================

Author: Andreas Mueller <amueller@ais.uni-bonn.de>

Build Instructions
------------------
The Python bindings require Cython.
To build the Python bindings use the following commands at the top level:

```bash
python setup.py install
```

to install the bindings systemwide

or


```bash
python setup.py build_clib
python setup.py build_ext -i
```

to install them in ARGMAX_STE/python/argmax_ste directory


See ``example_grid.py`` or the notebook for an example.
