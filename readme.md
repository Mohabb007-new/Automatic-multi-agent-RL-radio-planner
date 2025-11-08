# Automatic Multi-Agent Reinforcement Learning Telecom Radio Planner

This repository contains the implementation of an open-source RF planner that leverages multi-agent reinforcement learning to optimize the placement and parameters of LTE cells. The project combines AI techniques with telecommunications to provide a cost-effective and efficient solution for radio frequency planning.

Explore the code, logic, and tools used to achieve optimal RF planning in this repository.

This project was done as part my engineering degree final year project. 

## Abstract

RF planning is the process of assigning frequencies, transmitter locations, and parameters to a wireless communications system to evaluate coverage and capacity. This process is usually performed by radio planning engineers or using automatic planning tools, which are either expensive or inefficient.

In this work, we provide an open-source RF planner that leverages multi-agent reinforcement learning to determine the optimal placement and parameters of LTE cells. This project contains the logic and code behind it.

With the rapid advancement of technology, the increasing number of mobile devices, and the growth of the IoT field, more devices require reliable internet access. These devices demand excellent coverage and high bandwidth to ensure seamless connectivity. Careful radio frequency planning is essential to meet these needs.

RF planning involves assigning frequencies, transmitter locations, and parameters to evaluate coverage and capacity. It consists of two main phases:
1. **Initial Radio Link Budgeting**: Approximating the coverage area using statistical models.
2. **Detailed RF Propagation Modeling**: Determining the number of sites, site locations and heights, antenna directions and downtilts, neighbor cells, and mobility parameters (handover and cell re-selection) for each site.

Traditionally, this process is carried out by radio experts or through expensive proprietary software. Our project aims to address this challenge for new mobile operators by providing an efficient and cost-effective solution.

### Proposed Solution

To solve this problem, we divided the area of interest into hexagonal clusters of 7 LTE cells of 3 sectors, and each sector is served by a different antenna. We also used soft frequency reuse (SFR) logic and implemented it in our code. Finally, using reinforcement learning, specifically multi-agent reinforcement learning, we developed an algorithm to calculate the optimal placement and parameters for antennas and cells (location, height, power) within each cluster.
Evrything used in this project is open-source

## Prerequisites

To run this project, ensure you have the following installed and set up:

1. **GRASS GIS**: A powerful open-source geographical information system used for spatial data management and analysis.  
    You also need to set up your geolocation with the following maps:  
    - **DEM Map**: A Digital Elevation Model to represent terrain elevation.  
    - **Population Map**: To analyze coverage and capacity based on population density.  
    - **Clutter Map**: To account for land use and obstacles affecting signal propagation.  
    Ensure these maps are prepared and integrated into your environment before proceeding.

2. **Raplat**: A radio propagation tool integrated with GRASS GIS for RF planning.  
3. **Ubuntu or WSL (Windows Subsystem for Linux)**: A Linux environment is required to run the tools and scripts effectively.  
4. **Antenna Files**: Example antenna configuration files are included in the `antennas` folder.  
5. **Python Dependencies**: Install the required Python libraries for reinforcement learning by running:  
    ```bash
    pip install -r requirements.txt
    ```  
    The `requirements.txt` file includes essential packages like `gym` and `ray`. Ensure you have Python installed before proceeding.


### Running the reinforcement learning Application

Below is an example command to execute the application:

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

- `csv_file`: Path to the input CSV file containing agent data.
- `dem_map`: Digital Elevation Model map for terrain elevation.
- `antmap_file`: Path to the antenna configuration file.
- `clutter_map`: Map representing land use and obstacles.
- `out_map`: Name of the output map.
- `db_driver`: Database driver for output storage.
- `out_table`: Name of the output table.
- `rx_threshold`: Minimum allowed received power (in dBm).

Make sure to adjust the paths and parameters as needed for your specific setup.

The core logic and environment setup for the multi-agent reinforcement learning system are implemented in the `environment_logic.py` file. This script defines the environment, agents, and their interactions. To run the application, execute the `FYP.py` file, which serves as the main entry point and includes the necessary configuration.

Before running the application, ensure all dependencies are properly installed and configured. This includes setting up the required maps (DEM, Population, and Clutter) and verifying the availability of necessary files like `cells.csv`. For additional details, refer to the Raplat documentation.


## Note 

Due to the high complexity of this project some logic might be confusing and there might exist some areas of improvement. 
I am open to helping you solve any issue you might face while using this code.

## Keywords
- Multi-Agent Reinforcement Learning
- Telecommunications
- Radio Frequency Planning
- Reinforcement Learning (RL)
- AI
- Radio Network Optimization
- Open-Source RF Planner