# Data files

The FlexBid code is under the MIT license. The two data files here come from public sources and keep their own terms.

| file | what it holds | source |
|---|---|---|
| `caiso_netload.npz` | Net load of the California ISO, z-scored, at 5-minute resolution. Six windows of one week each in 2025, starting 10 January, March, May, July, September and November. | California ISO, OASIS public data. |
| `ocv_soc.npz` | Open-circuit voltage against state of charge of one NMC cell, 201 points, downsampled from 36,848 measured points. | EV-EcoSim battery data, cell W8 (`OCV/SOC_corr_W8_1`). Balogun et al., "EV-EcoSim: A grid-aware co-simulation platform for the design and optimization of electric vehicle charging infrastructure", IEEE Transactions on Smart Grid, 2023. https://github.com/ebalogun01/EV-EcoSim |

No charging-session data is shipped. The arrival state of charge and the pack sizes are drawn from fitted distributions whose parameters are written in the code (`sim/fleet.py`, `sim/config.py`).
