# Automatic Multi-Agent Reinforcement Learning Telecom Radio Planner

An open-source LTE radio frequency planner that uses multi-agent reinforcement learning to automatically determine the optimal placement and parameters for cell towers — replacing expensive proprietary RF planning tools.

> Final Year Project (FYP) — Engineering Degree

---

## Abstract

RF planning is the process of assigning frequencies, transmitter locations, and parameters to a wireless communications system to evaluate coverage and capacity. This process is usually performed by radio planning engineers or using automatic planning tools, which are either expensive or inefficient.

This project provides an open-source RF planner that leverages multi-agent reinforcement learning to determine the optimal placement and parameters of LTE cells.

With the rapid advancement of technology, the increasing number of mobile devices, and the growth of the IoT field, more devices require reliable internet access. These devices demand excellent coverage and high bandwidth to ensure seamless connectivity. Careful radio frequency planning is essential to meet these needs.

RF planning consists of two main phases:
1. **Initial Radio Link Budgeting** — approximating the coverage area using statistical models.
2. **Detailed RF Propagation Modeling** — determining the number of sites, site locations and heights, antenna directions and downtilts, neighbor cells, and mobility parameters for each site.

Traditionally, this process is carried out by radio experts or through expensive proprietary software. This project aims to address this challenge for new mobile operators by providing an efficient and cost-effective open-source solution.

---

## Proposed Solution

The area of interest is divided into **hexagonal clusters of 7 LTE cells with 3 sectors each**, where every sector is served by a dedicated antenna. Soft Frequency Reuse (SFR) logic is implemented to reduce inter-cell interference.

Using multi-agent reinforcement learning, the system learns the optimal antenna placement, height, and transmit power within each cluster.

![LTE Hexagonal Cluster](<images/Lte cluster.png>)

---

## Prerequisites

### 1. GRASS GIS
A powerful open-source GIS used for spatial data management and RF propagation analysis. You will need to prepare the following maps for your target area:

| Map | Description |
|-----|-------------|
| **DEM Map** | Digital Elevation Model — represents terrain elevation |
| **Population Map** | Population density — used for capacity analysis |
| **Clutter Map** | Land use and obstacles — affects signal propagation |

**Lebanon example maps used in this project:**

![Lebanon DEM](<images/Lebanon DEM.png>)
*Digital Elevation Model*

![Lebanon Population](<images/Lebanon population.png>)
*Population density map*

![Lebanon Clutter](<images/Lebanon clutter.png>)
*Clutter/land-use map*

### 2. Raplat
A radio propagation tool integrated with GRASS GIS for RF coverage simulation.

### 3. Ubuntu or WSL (Windows Subsystem for Linux)
A Linux environment is required to run GRASS GIS and Raplat.

### 4. Antenna Files
Example antenna configuration files are included in the `antennas/` folder.

### 5. Python Dependencies
```bash
pip install -r requirements.txt
```
Key packages: `ray`, `gymnasium`.

---

## Running the Application

### Step 1 — Configure paths
Update the constants at the top of `Environment_logic.py` to match your GRASS GIS setup and file paths:
```python
GISBASE        = '/usr/lib/grass78'
GISDBASE       = '/home/your_user/grassdata'
AGENTS_CSV     = '/home/your_user/raplat/agents.csv'
...
```

### Step 2 — Run training
```bash
python FYP.py
```

This restores from the latest checkpoint and continues training. To start fresh, remove the `trainer.restore(...)` line in `FYP.py`.

### Step 3 — Raplat command reference
The environment internally calls Raplat after each step. The equivalent manual command is:
```bash
r.raplat csv_file=~/raplat/agents.csv \
         dem_map=n34_e036_1arc_v3@PERMANENT \
         antmap_file=~/raplat/antenna_diagrams/antennamap.csv \
         clutter_map=loss@PERMANENT \
         out_map=final \
         db_driver=csv \
         out_table=testoo \
         rx_threshold=-80 --o
```

| Parameter | Description |
|-----------|-------------|
| `csv_file` | Input CSV containing agent/antenna data |
| `dem_map` | Digital Elevation Model map |
| `antmap_file` | Antenna configuration file |
| `clutter_map` | Land use and obstacles map |
| `out_map` | Output coverage map name |
| `rx_threshold` | Minimum received power in dBm |

---

## Results

![Training Results](<images/Results.png>)

---

## Project Structure

```
├── Environment_logic.py   # Gymnasium multi-agent environment (Planner class)
├── FYP.py                 # Training entry point — PPO config and policy mapping
├── requirements.txt       # Python dependencies
├── antennas/              # Antenna diagram files
└── images/                # Maps and result visualizations
```

---

## Note

Due to the high complexity of this project, some logic may be difficult to follow and there may be areas for improvement. Feel free to open an issue if you encounter any problems.

---

## Keywords

Multi-Agent Reinforcement Learning · Telecommunications · Radio Frequency Planning · LTE · Radio Network Optimization · Open-Source RF Planner
