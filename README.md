# Paleopy

### Installation

    pip3 install git+https://github.com/tedwards2412/paleopy
    
You will also need the [`WIMpy`](https://github.com/bradkav/WIMpy_NREFT) package for calculating the DM and neutrino spectra.
    
### Overview

The core of the code is in the [`paleopy`](paleopy) module. Data for converting recoil energies to track lengths, along with tables of background distributions are loaded from [`Data/`](Data). This then allows you to calculate all the relevant track length distributions. The currently supported minerals are Nchwaningite, Sinjarite, Halite, Olivine, Gypsum and Phlogopite (see [`Data/MineralList.txt`](Data/MineralList.txt)).
