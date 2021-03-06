# FT8 Notes

A collection of Jupyter notebooks to explore the transmission and
reception of the FT8 digital mode.

## Overview

FT8 is a digital mode used by amateur radio operators that has
become quite popular since it was first introduced in 2017.
It was created by Dr. Joe Taylor (K1JT) and Dr. Steven Franke (K9AN)
and their collaborators who develop the
[WSJT-X](https://physics.princeton.edu/pulsar/k1jt/wsjtx.html) program.

I was interested in how FT8 works, so I decided to read the source
code of WSJT-X and document what I learned in a series of Jupyter
notebooks, as there is no formal specification for FT8.
As part of this process I have re-implemented most of the FT8 protocols
used by WSJT-X in a Python module called [ft8](ft8.py).

The main notebooks in this repository are:

* [Background](https://nbviewer.jupyter.org/github/vk3jpk/ft8-notes/blob/master/Background.ipynb) - Background material to level set an audience prior to diving into how FT8 works.
* [Transmit](https://nbviewer.jupyter.org/github/vk3jpk/ft8-notes/blob/master/Transmit.ipynb) - A walkthrough of how FT8 signals are transmitted.
* [Receive](https://nbviewer.jupyter.org/github/vk3jpk/ft8-notes/blob/master/Receive.ipynb) - A walkthrough of how FT8 signals are received.

The github renderer for Jupyter notebooks has some limitations that I
haven't been able to work around.
Therefore, it is best to view notebooks via
[nbviewer](https://nbviewer.jupyter.org).
This repository can be accessed through nbviewer
[here](https://nbviewer.jupyter.org/github/vk3jpk/ft8-notes).

Each of the main notebooks includes the additional metadata required
to enable the notebooks to be presented using the presentation mode of
nbviewer.

The remaining notebooks focus on narrow aspects of FT8 and do not
include any presentation metadata.

This repository is a work in progress and does not yet fully implement
all the features of the FT8 implementation in WSJT-X.
There are also some aspects of the WSJT-X FT8 implementation that I do
not yet fully understand.

## Acknowledgements

These notebooks are based on reviewing the source code of WSJT-X.
I thank Dr. Joe Taylor and his collaborators for open sourcing
WSJT-X to enable others to study the code and learn from it.

The code is this repository uses similar algorithms to those used
in WSJT-X in many instances.
However, in some instances the algorithms have been modified
significantly to enable better performance in the Python
programming language.

## Related Work

There are a few other repositories on Github that may be of interest:

* [rtmrtmrtmrtm/basicft8](https://github.com/rtmrtmrtmrtm/basicft8)
is a basic Python implementation of a FT8 receiver.
Note that it has not been updated to reflect changes to the FT8
protocol that were made in 2018 so it will not work with the signals
generated by the current version of WSJT-X.
* [rtmrtmrtmrtm/weakmon](https://github.com/rtmrtmrtmrtm/weakmon)
is a very comprehensive set of Python scripts that explore FT8 and
other weak signal digital modes.
* [kgoba/ft8_lib](https://github.com/kgoba/ft8_lib) is a C++ library
intended for use with microcontrollers.

## License

Copyright (C) 2019 James Kelly, VK3JPK

The [ft8 Python module](ft8.py) in this repository is licensed under the
[GNU General Public License Version 3.0](LICENSE.gpl).

All other materials are licensed under the
[Creative Commons Attribution-ShareAlike 4.0
International License](LICENSE.ccbysa).
