import os
import csv
import copy
import time
import random
import bisect
import subprocess

import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces
from gymnasium.spaces import Discrete, Tuple, MultiDiscrete
from grass.script import core as grass
import grass.script.setup as gsetup
from PIL import Image
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# ── GRASS GIS configuration ───────────────────────────────────────────────────
GISBASE        = '/usr/lib/grass78'
GISDBASE       = '/home/master/grassdata'
GRASS_LOCATION = 'New'
GRASS_MAPSET   = 'Project1'

# ── File paths ────────────────────────────────────────────────────────────────
AGENTS_CSV      = '/home/master/raplat/agents.csv'
ANTENNA_MAP     = '/home/master/raplat/antenna_diagrams/antennamap.csv'
DEM_MAP         = 'n34_e036_1arc_v3@PERMANENT'
CLUTTER_MAP     = 'loss@PERMANENT'
POPULATION_MAP  = 'lbn_general_2020@Project1'
POPULATION_PATH = '/home/master/grassdata/New/population'
COVERAGE_PATH   = '/home/master/grassdata/New/coverage'

# ── Visualization paths (WSL ↔ Windows) ──────────────────────────────────────
VIZ_COVERAGE_WSL   = '/mnt/c/Users/owner/Desktop/temp_coverage_map.png'
VIZ_COVERAGE_WIN   = r'C:\Users\owner\Desktop\temp_coverage_map.png'
VIZ_POPULATION_WSL = '/mnt/c/Users/owner/Desktop/temp_population_map.png'
VIZ_POPULATION_WIN = r'C:\Users\owner\Desktop\temp_population_map.png'
VIZ_COMBINED_WSL   = '/mnt/c/Users/owner/Desktop/temp_combined_map.png'
VIZ_COMBINED_WIN   = r'C:\Users\owner\Desktop\temp_combined_map.png'

class Planner(MultiAgentEnv):
    """Custom Environment that follows gym interface."""

    #metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,config=None):
        super().__init__()
        self.iter=0
        self.accepted=0

        self.location = config.get("location") #This is a string specifying if the location is rural,urban,...
        self.west = config.get("w")
        self.east = config.get("e")
        self.south = config.get("s")
        self.north = config.get("n")
        self.max_output = config.get("max_output")
        self.user_bandwidth = config.get("user_bandwidth")
        self.steps=0
        self.observation_space={
                      "agent0" : spaces.Box(low=0, high=1000000, shape=(13,),dtype=np.int32),
                      "agent1" : spaces.Box(low=0.0, high=self.north, shape=(9,)),
                      "agent2" : spaces.Box(low=0.0, high=self.north, shape=(9,)),
                      "agent3" : spaces.Box(low=0.0, high=self.north, shape=(9,)),
                      "agent4" : spaces.Box(low=0.0, high=self.north, shape=(9,)),
                      "agent5" : spaces.Box(low=0.0, high=self.north, shape=(9,)),
                      "agent6" : spaces.Box(low=0.0, high=self.north, shape=(9,)),
                      "agent7" : spaces.Box(low=0.0, high=self.north, shape=(9,))
            }
        
        #In hexadecimal planning, we have a central base station and 6 others around it so 7 agents in total and i added one more
        #this one more agent decides the position of the agents based on interference.
        self._agent_ids = {"agent0","agent1","agent2","agent3","agent4","agent5","agent6","agent7"}
        #the coverage area is set to be (4 radius of agent1 * 4 radius of agent1) where radius decided by the distance from center to pint where pr is minimal
        '''self.action_space={
                      "agent0" : Tuple([Discrete(3), Discrete(3),Discrete(3),Discrete(3),Discrete(3), Discrete(3),Discrete(3),Discrete(3),Discrete(3), Discrete(3),Discrete(3),Discrete(3)]),
                      "agent1" : Tuple([Discrete(3), Discrete(3),Discrete(3),Discrete(3)]),
                      "agent2" : Tuple([Discrete(3), Discrete(3),Discrete(3),Discrete(3)]),
                      "agent3" : Tuple([Discrete(3), Discrete(3),Discrete(3),Discrete(3)]),
                      "agent4" : Tuple([Discrete(3), Discrete(3),Discrete(3),Discrete(3)]),
                      "agent5" : Tuple([Discrete(3), Discrete(3),Discrete(3),Discrete(3)]),
                      "agent6" : Tuple([Discrete(3), Discrete(3),Discrete(3),Discrete(3)]),
                      "agent7" : Tuple([Discrete(3), Discrete(3),Discrete(3),Discrete(3)])
            }
        '''
        self.action_space={
                      "agent0" : MultiDiscrete([3,3,3,3,3,3,3,3,3,3,3,3]),
                      "agent1" : MultiDiscrete([3,3,3,3]),
                      "agent2" : MultiDiscrete([3,3,3,3]),
                      "agent3" : MultiDiscrete([3,3,3,3]),
                      "agent4" : MultiDiscrete([3,3,3,3]),
                      "agent5" : MultiDiscrete([3,3,3,3]),
                      "agent6" : MultiDiscrete([3,3,3,3]),
                      "agent7" : MultiDiscrete([3,3,3,3])
                      }
        
        
        # Initialize GRASS GIS environment
        self.grass = grass
        gsetup.init(GISBASE, GISDBASE, GRASS_LOCATION, GRASS_MAPSET)
        grass.run_command('g.region', e=self.east, w=self.west, n=self.north, s=self.south)
        grass.run_command('r.out.xyz', input=POPULATION_MAP, output=POPULATION_PATH, separator=',', overwrite=True)
        grass.run_command('g.remove', type="raster", pattern="agent*", flags='f')
        grass.run_command('g.remove', type="raster", pattern="_*", flags='f')

        self.df2 = pd.read_csv(POPULATION_PATH, header=None)
        

    def step(self, action):
        ...
        #first, let's take the actions of normal agents wich is change in the csv file named cells, its height and power
        #agent 1-7 action space{+-pow1,+-pow2,+-pow3,+-height} multidiscrete(3,3,3,3)
        #agent 0 action space{+x2/-x2,+y2/-y2,+x3/-x3,+y3/-y3,+x4/-x4,+y4/-y4,+x5/-x5,+y5/-y5,+x6/-x6,+y6/-y6,+x7/-x7,+y7/-y7} multidiscrete(3,3,3,3,3,3,3,3,3,3,3,3)
       

        self.steps+=1
        self.iter+=1
        observation=copy.deepcopy(self.observation)
        with open(AGENTS_CSV, 'r+') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)
            i=1
            while i <19:
                agent=data[i][0]
                k=int(i/3)
                k=k*2
                for j in range(3):
                    try:
                        if (action["agent0"][k]==1):   #to update location
                            data[i+j][3]=int(data[i+j][3])+29
                        elif (action["agent0"][k]==2):
                            data[i+j][3]=int(data[i+j][3])-29
                        if (action["agent0"][k+1]==1):
                            data[i+j][4]=int(data[i+j][4])+29
                        elif (action["agent0"][k+1]==2):
                            data[i+j][4]=int(data[i+j][4])-29
                    except:
                        pass
                    
                    try:
                        if (action[agent][j]==1): #to update the power
                            data[i+j][10]=int(data[i+j][10])+1
                        elif (action[agent][j]==2):
                            data[i+j][10]=int(data[i+j][10])-1
                        observation[agent][j+3]=int(data[i+j][10])

                        if (action[agent][3]==1):#to update height
                            data[i+j][5]=int(data[i+j][5])+1 
                        elif (action[agent][3]==-1):
                            data[i+j][5]=int(data[i+j][5])-1
                    except:
                        pass
                        
                observation[agent][0]=int(data[i][3])
                observation[agent][1]=int(data[i][4])
                observation[agent][2]=int(data[i][5])  #AT this line we have x,y,height,pow1,pow2,pow3 for agents 2-7
                observation[agent][6]=0
                observation[agent][7]=0
                observation[agent][8]=0#set capacities to 0
                i=i +3

            try:   
                for j in range(3):
                        if (action['agent1'][j]==1): #to update the power
                            data[19+j][10]=int(data[19+j][10])+1 
                        elif (action['agent1'][j]==2):
                            data[19+j][10]=int(data[19+j][10])-1
                        observation["agent1"][j+3]=int(data[19+j][10])

                        if (action['agent1'][3]==1): #to update height
                            data[19+j][5]=int(data[19+j][5])+1 
                        elif (action['agent1'][3]==-1):
                            data[19+j][5]=int(data[19+j][5])-1
            except:
                pass
            
            observation["agent1"][0]=int(data[19][3])
            observation["agent1"][1]=int(data[19][4])
            observation["agent1"][2]=int(data[19][5]) #AT this line we have x,y,height,pow1,pow2,pow3 for agent1
            observation["agent1"][6]=0
            observation["agent1"][7]=0
            observation["agent1"][8]=0 #set the capacities to 0
            csvfile.seek(0)
            csvfile.truncate()
            writer=csv.writer(csvfile)
            while len(data) > 22:
                data.pop()
            writer.writerows(data)

        #After all the updates according to actions, we run the r.raplat script and get new rewards which will be based on the new states
        #the most important thing for rewards is the interference from agent0 and the antenna capacities in normal agents

        grass.parse_command(
            'r.raplat',
            csv_file=AGENTS_CSV, antmap_file=ANTENNA_MAP,
            dem_map=DEM_MAP, clutter_map=CLUTTER_MAP,
            out_map='coverage', db_driver='csv', out_table='coverage',
            rx_threshold='-80', overwrite=True, flags='r',
        )
        
        '''Here i will calculate the number of users for each antenna and the total coverage by following certain steps:
           -We will calculate the region around the central base station using the formula we said before
           -The next thing we will do is get the population distribution in the whole region
           -I will iterate over the coverage csv file and in each point, i will see weather this point is inside the boundaries around the bts
            and i will search for this point in the population csv.
           -If it is in the population csv, we will add the number of people in this point to the entry related to antenna number in the num dictionary.
           -We will know the antennas that have coverage in the same point in the coverage csv and we wil add these to the interference in agent0
        '''

        

        df1 = pd.read_csv(COVERAGE_PATH, header=None)
        df1['composite_key'] =(-df1[1]).tolist()

        min_x=float('inf')
        min_y=float('inf')
        max_x=0
        max_y=0
        
        for agent in observation:
            if agent != 'agent0':
                observation[agent][6]=observation[agent][7]=observation[agent][8]=0
                if int(observation[agent][0]) < min_x :
                    min_x=int(observation[agent][0])
                elif observation[agent][0] > max_x:
                    max_x=int(observation[agent][0])
                if observation[agent][1] < min_y :
                    min_y=int(observation[agent][1])
                elif observation[agent][1] > max_y:
                    max_y=int(observation[agent][1])

        observation['agent0']=np.zeros(13,dtype=np.int32)
        
        area_points=((max_x-min_x)/29) * ((max_y-min_y)/29) # Rewards for agent0
        while max_y > min_y:
            lef_idx=bisect.bisect_left(df1['composite_key'],-max_y)
            rig_idx=bisect.bisect_right(df1['composite_key'],-max_y+29)
            l_idx=lef_idx
            r_idx=rig_idx
            while r_idx-l_idx > 1:
                    mid_idx=(l_idx+r_idx)//2
                    if df1.iloc[mid_idx][0] >= min_x:
                        r_idx=mid_idx
                    else:
                        l_idx=mid_idx
            left_bound=r_idx

            while rig_idx-lef_idx > 1:
                    mid_idx=(lef_idx+rig_idx)//2
                    if df1.iloc[mid_idx][0] >= max_x:
                        rig_idx=mid_idx
                    else:
                        lef_idx=mid_idx
            right_bound=lef_idx
            max_y-=29
            observation['agent0'][12]+=(right_bound-left_bound)
        
        # Define a function for binary search on the sorted DataFrame
        def binary_search(df, key):
            left_idx = bisect.bisect_left(df['composite_key'], key[0])
            right_idx= bisect.bisect_right(df['composite_key'], key[0]+29)
            if left_idx!= len(df) and  df['composite_key'][left_idx] <= (key[0]+29):
                while right_idx-left_idx > 1:
                    mid_idx=(left_idx+right_idx)//2
                    if df.iloc[mid_idx][0] >= key[1]:
                        right_idx=mid_idx
                    else:
                        left_idx=mid_idx
                if right_idx !=len(df) and df.iloc[right_idx][0] <= (key[1]+29):
                    return right_idx
                
            return None
        #inetrference_points(1-2,1-3,1-4,1-5,1-6,1-7,2-1,2-3,2-7,3-1,3-2,3-4,4-1,4-3,4-5,5-1,5-4,5-6,6-1,6-5,6-7,7-1,7-6,7-2)
        #inetrference_points=np.zeros(24,dtype=np.float32)
        interference_points={
                     'agent1' :np.zeros(6,dtype=np.float32),
                     'agent2': np.zeros(3,dtype=np.float32),
                     'agent3' :np.zeros(3,dtype=np.float32),
                     'agent4': np.zeros(3,dtype=np.float32),
                     'agent5' :np.zeros(3,dtype=np.float32),
                     'agent6': np.zeros(3,dtype=np.float32),
                     'agent7' :np.zeros(3,dtype=np.float32)
                    }  
        def interference(current,interfere,row):
            
                    if (current=='agent1'):
                        if (interfere=='agent2'):
                            interference_points[current][0]+=row[2]
                            observation['agent0'][0]+=row[2]
                        elif (interfere=='agent3'):
                            interference_points[current][1]+=row[2]
                            observation['agent0'][3]+=row[2]
                        elif (interfere=='agent4'):
                            interference_points[current][2]+=row[2]
                            observation['agent0'][5]+=row[2]
                        elif (interfere=='agent5'):
                            interference_points[current][3]+=row[2]
                            observation['agent0'][7]+=row[2]
                        elif (interfere=='agent6'):
                            interference_points[current][4]+=row[2]
                            observation['agent0'][9]+=row[2]
                        elif (interfere=='agent7'):
                            interference_points[current][5]+=row[2]
                            observation['agent0'][11]+=row[2]

                    elif (current=='agent2'):
                        if (interfere=='agent1'):
                            interference_points[current][0]+=row[2]
                            observation['agent0'][0]+=row[2]
                        elif (interfere=='agent7'):
                            interference_points[current][1]+=row[2]
                            observation['agent0'][2]+=row[2]
                        elif (interfere=='agent3'):
                            interference_points[current][2]+=row[2]
                            observation['agent0'][1]+=row[2]
                            
                    elif (current=='agent3'):
                        if (interfere=='agent1'):
                            interference_points[current][0]+=row[2]
                            observation['agent0'][3]+=row[2]
                        elif (interfere=='agent2'):
                            interference_points[current][1]+=row[2]
                            observation['agent0'][1]+=row[2]
                        elif (interfere=='agent4'):
                            interference_points[current][2]+=row[2]
                            observation['agent0'][4]+=row[2]

                    elif (current=='agent4'):
                        if (interfere=='agent1'):
                            interference_points[current][0]+=row[2]
                            observation['agent0'][5]+=row[2]
                        elif (interfere=='agent3'):
                            interference_points[current][1]+=row[2]
                            observation['agent0'][4]+=row[2]
                        elif (interfere=='agent5'):
                            interference_points[current][2]+=row[2]
                            observation['agent0'][6]+=row[2]

                    elif (current=='agent5'):
                        if (interfere=='agent1'):
                            interference_points[current][0]+=row[2]
                            observation['agent0'][7]+=row[2]
                        elif (interfere=='agent4'):
                            interference_points[current][1]+=row[2]
                            observation['agent0'][6]+=row[2]
                        elif (interfere=='agent6'):
                            interference_points[current][2]+=row[2]
                            observation['agent0'][8]+=row[2]

                    elif (current=='agent6'):
                        if (interfere=='agent1'):
                            interference_points[current][0]+=row[2]
                            observation['agent0'][9]+=row[2]
                        elif (interfere=='agent5'):
                            interference_points[current][1]+=row[2]
                            observation['agent0'][8]+=row[2]
                        elif (interfere=='agent7'):
                            interference_points[current][2]+=row[2]
                            observation['agent0'][10]+=row[2]

                    elif (current=='agent7'):
                        if (interfere=='agent1'):
                            interference_points[current][0]+=row[2]
                            observation['agent0'][11]+=row[2]
                        elif (interfere=='agent6'):
                            interference_points[current][1]+=row[2]
                            observation['agent0'][10]+=row[2]
                        elif (interfere=='agent2'):
                            interference_points[current][2]+=row[2]
                            observation['agent0'][2]+=row[2]

        def interference1(current,interfere,row):#Here, we include only the interference points in agent0 wothout assigning the points to other antennas
            
                    if (current=='agent1'):
                        if (interfere=='agent2'):
                            observation['agent0'][0]+=row[2]
                        elif (interfere=='agent3'):
                            observation['agent0'][3]+=row[2]
                        elif (interfere=='agent4'):
                            observation['agent0'][5]+=row[2]
                        elif (interfere=='agent5'):
                            observation['agent0'][7]+=row[2]
                        elif (interfere=='agent6'):
                            observation['agent0'][9]+=row[2]
                        elif (interfere=='agent7'):
                            observation['agent0'][11]+=row[2]

                    elif (current=='agent2'):
                        if (interfere=='agent1'):
                            observation['agent0'][0]+=row[2]
                        elif (interfere=='agent7'):
                            observation['agent0'][2]+=row[2]
                        elif (interfere=='agent3'):
                            observation['agent0'][1]+=row[2]
                            
                    elif (current=='agent3'):
                        if (interfere=='agent1'):
                            observation['agent0'][3]+=row[2]
                        elif (interfere=='agent2'):
                            observation['agent0'][1]+=row[2]
                        elif (interfere=='agent4'):
                            observation['agent0'][4]+=row[2]

                    elif (current=='agent4'):
                        if (interfere=='agent1'):
                            observation['agent0'][5]+=row[2]
                        elif (interfere=='agent3'):
                            observation['agent0'][4]+=row[2]
                        elif (interfere=='agent5'):
                            observation['agent0'][6]+=row[2]

                    elif (current=='agent5'):
                        if (interfere=='agent1'):
                            observation['agent0'][7]+=row[2]
                        elif (interfere=='agent4'):
                            observation['agent0'][6]+=row[2]
                        elif (interfere=='agent6'):
                            observation['agent0'][8]+=row[2]

                    elif (current=='agent6'):
                        if (interfere=='agent1'):
                            observation['agent0'][9]+=row[2]
                        elif (interfere=='agent5'):
                            observation['agent0'][8]+=row[2]
                        elif (interfere=='agent7'):
                            observation['agent0'][10]+=row[2]

                    elif (current=='agent7'):
                        if (interfere=='agent1'):
                            observation['agent0'][11]+=row[2]
                        elif (interfere=='agent6'):
                            observation['agent0'][10]+=row[2]
                        elif (interfere=='agent2'):
                            observation['agent0'][2]+=row[2]
        

        for index, row in self.df2.iterrows():
            
            # Create the composite key based on the current row's values
            search_key = (-row[1]-15,row[0]-15)  # Adjust column names as necessary
            result_row = binary_search(df1, search_key)
            
            if result_row is not None:
                cur_row=df1.iloc[result_row] #current row in the coverage csv which is df1
                current=cur_row[3]
                current=current.strip("'")
                loc=cur_row[4]-int(current[5])*10+5
                observation[current][loc]+=row[2] #here row[4] is the antid and current[5] gives the number of the agent;the equation gives 1,2or3 +5
                
                if (cur_row[9] > -200):
                  interfere=cur_row[7]
                  interfere=interfere.strip("'")
                  interference(current,interfere,row)  
                  if (cur_row[13] > -200):
                        interfere=cur_row[11]
                        interfere=interfere.strip("'")
                        interference1(current,interfere,row)
                        current=cur_row[7]
                        current=current.strip("'")
                        interference1(current,interfere,row)
                

        self.observation=copy.deepcopy(observation)
        

        terminateds={ 'agent0': False,
                     'agent1' :False,
                     'agent2': False,
                     'agent3' :False,
                     'agent4': False,
                     'agent5' :False,
                     'agent6': False,
                     'agent7' :False,
                     "__all__":False}
        
        truncateds={ 'agent0': False,
                     'agent1' :False,
                     'agent2': False,
                     'agent3' :False,
                     'agent4': False,
                     'agent5' :False,
                     'agent6': False,
                     'agent7' :False,
                     "__all__":False}
        
        rewards={ 'agent0': 0,
                     'agent1' :0,
                     'agent2': 0,
                     'agent3' :0,
                     'agent4': 0,
                     'agent5' :0,
                     'agent6': 0,
                     'agent7' :0
                    }

        

        available_space={    #where each value for each antenna
                     'agent2': np.zeros(3,dtype=np.float32),
                     'agent3' :np.zeros(3,dtype=np.float32),
                     'agent4': np.zeros(3,dtype=np.float32),
                     'agent5' :np.zeros(3,dtype=np.float32),
                     'agent6': np.zeros(3,dtype=np.float32),
                     'agent7' :np.zeros(3,dtype=np.float32)
                    }


        max_users=self.max_output/self.user_bandwidth #Maximum number of users for each antenna considering 75 MHZ
        max_users/=0.62
        max_users*=3.8
 
        for agent in observation: #The number of available space in each antenna cconsidering the max space is 80 beacuase max edge cell bandwidth=80MHZ
            if agent !='agent0' and agent!= 'agent1':
                if observation[agent][6] < max_users:
                    if max_users-observation[agent][6] < 80:
                        available_space[agent][0]=max_users-observation[agent][6]
                    else:
                        available_space[agent][0]=80
                        
                if observation[agent][7] < max_users:
                    if max_users-observation[agent][7] < 80:
                        available_space[agent][1]=max_users-observation[agent][7]
                    else:
                        available_space[agent][1]=80
                        
                if observation[agent][8] < max_users:
                    if max_users-observation[agent][8] < 80:
                        available_space[agent][2]=max_users-observation[agent][8]
                    else:
                        available_space[agent][2]=80

                if agent =='agent2': #here we declare the available space in antennas that do not interfere with others as 0
                    available_space[agent][0]=0
                    z1=available_space[agent][1]-interference_points[agent][0]-interference_points[agent][2]
                    available_space[agent][1]=z1 if z1>0 else 0
                    z2=available_space[agent][2]-interference_points[agent][1]
                    available_space[agent][2]=z2 if z2>0 else 0
                elif agent =='agent3': 
                    available_space[0]=0
                    z1=available_space[agent][2]-interference_points[agent][0]-interference_points[agent][1]
                    available_space[agent][2]=z1 if z1>0 else 0
                    z2=available_space[agent][1]-interference_points[agent][2]
                    available_space[agent][1]=z2 if z2>0 else 0
                    
                elif agent =='agent4': 
                    available_space[agent][1]=0
                    z1=available_space[agent][2]-interference_points[agent][0]-interference_points[agent][2]
                    available_space[agent][2]=z1 if z1>0 else 0
                    z2=available_space[agent][0]-interference_points[agent][1]
                    available_space[agent][0]=z2 if z2>0 else 0
                elif agent =='agent5':
                    available_space[1]=0
                    z1=available_space[agent][0]-interference_points[agent][0]-interference_points[agent][1]
                    available_space[agent][0]=z1 if z1>0 else 0
                    z2=available_space[agent][2]-interference_points[agent][2]
                    available_space[agent][2]=z2 if z2>0 else 0

                elif agent =='agent6': 
                    available_space[agent][2]=0
                    z1=available_space[agent][0]-interference_points[agent][0]-interference_points[agent][2]
                    available_space[agent][0]=z1 if z1>0 else 0
                    z2=available_space[agent][1]-interference_points[agent][1]
                    available_space[agent][1]=z2 if z2>0 else 0
                elif agent =='agent7':
                    available_space[2]=0
                    z1=available_space[agent][1]-interference_points[agent][0]-interference_points[agent][1]
                    available_space[agent][1]=z1 if z1>0 else 0
                    z2=available_space[agent][0]-interference_points[agent][2]
                    available_space[agent][0]=z2 if z2>0 else 0
  

        for agent in available_space:#in this block of code , we will try to check where interference exists then load balance with neighboring cells
            if agent=='agent2' :  #Do the same process for all agents 
                if available_space[agent][1] > 0:
                    z1=80-interference_points['agent1'][0]-interference_points['agent1'][1] #checkif the antenna we want to take users from is over capacity or not
                    if z1<0:
                        temp_interference=interference_points['agent1'][0]
                        temp_space=available_space[agent][1]
                        interference_points['agent1'][0]-=available_space[agent][1]
                        available_space[agent][1]-=interference_points['agent1'][0]
                        if interference_points['agent1'][0] < 0:
                            interference_points['agent1'][0]=0
                            observation['agent1'][6]-=temp_interference
                            observation[agent][7]+=temp_interference
                        else :
                            available_space[agent][1]=0
                            observation[agent][7]+=temp_space
                            observation['agent1'][6]-=temp_space
                    if available_space[agent][1] >0 :
                        z2=80-interference_points['agent3'][0]-interference_points['agent3'][1]
                        if z2<0:
                            temp_interference=interference_points['agent3'][1]
                            temp_space=available_space[agent][1]
                            interference_points['agent3'][1]-=available_space[agent][1]
                            available_space[agent][1]-=interference_points['agent3'][1]
                            if interference_points['agent3'][1] < 0:
                                interference_points['agent3'][1]=0
                                observation['agent3'][8]-=temp_interference
                                observation[agent][7]+=temp_interference
                            else :
                                available_space[agent][1]=0
                                observation[agent][7]+=temp_space
                                observation['agent3'][8]-=temp_space
                                
                if available_space[agent][2] >0:
                    z3=80 - interference_points['agent7'][2]
                    if z3 <0:
                        temp_interference=interference_points['agent7'][2]
                        temp_space=available_space[agent][2]
                        interference_points['agent7'][2]-=available_space[agent][2]
                        available_space[agent][2]-=interference_points['agent7'][2]
                        if interference_points['agent7'][2] < 0:
                            interference_points['agent7'][2]=0
                            observation['agent7'][6]-=temp_interference
                            observation[agent][8]+=temp_interference
                        else :
                            available_space[agent][2]=0
                            observation[agent][8]+=temp_space
                            observation['agent7'][6]-=temp_space

            elif agent=='agent3':
                if available_space[agent][2] > 0:
                    z1=80-interference_points['agent1'][0]-interference_points['agent1'][1]
                    if z1<0:
                        temp_interference=interference_points['agent1'][1]
                        temp_space=available_space[agent][2]
                        interference_points['agent1'][1]-=available_space[agent][2]
                        available_space[agent][2]-=interference_points['agent1'][0]
                        if interference_points['agent1'][1] < 0:
                            interference_points['agent1'][1]=0
                            observation['agent1'][6]-=temp_interference
                            observation[agent][8]+=temp_interference
                        else :
                            available_space[agent][2]=0
                            observation[agent][8]+=temp_space
                            observation['agent1'][6]-=temp_space
                    if available_space[agent][2] >0 :
                        z2=80-interference_points['agent2'][0]-interference_points['agent2'][2]
                        if z2<0:
                            temp_interference=interference_points['agent2'][2]
                            temp_space=available_space[agent][2]
                            interference_points['agent2'][2]-=available_space[agent][2]
                            available_space[agent][2]-=interference_points['agent2'][2]
                            if interference_points['agent2'][2] < 0:
                                interference_points['agent2'][2]=0
                                observation['agent2'][7]-=temp_interference
                                observation[agent][8]+=temp_interference
                            else :
                                available_space[agent][2]=0
                                observation[agent][8]+=temp_space
                                observation['agent2'][7]-=temp_space
                                
                if available_space[agent][1] >0:
                    z3=80 - interference_points['agent4'][1]
                    if z3 <0:
                        temp_interference=interference_points['agent4'][1]
                        temp_space=available_space[agent][1]
                        interference_points['agent4'][1]-=available_space[agent][1]
                        available_space[agent][1]-=interference_points['agent4'][1]
                        if interference_points['agent4'][1] < 0:
                            interference_points['agent4'][1]=0
                            observation['agent4'][6]-=temp_interference
                            observation[agent][7]+=temp_interference
                        else :
                            available_space[agent][1]=0
                            observation[agent][7]+=temp_space
                            observation['agent4'][6]-=temp_space

            elif agent=='agent4':
                if available_space[agent][2] > 0:
                    z1=80-interference_points['agent1'][2]-interference_points['agent1'][3]
                    if z1<0:
                        temp_interference=interference_points['agent1'][2]
                        temp_space=available_space[agent][2]
                        interference_points['agent1'][2]-=available_space[agent][2]
                        available_space[agent][2]-=interference_points['agent1'][2]
                        if interference_points['agent1'][2] < 0:
                            interference_points['agent1'][2]=0
                            observation['agent1'][7]-=temp_interference
                            observation[agent][8]+=temp_interference
                        else :
                            available_space[agent][2]=0
                            observation[agent][8]+=temp_space
                            observation['agent1'][7]-=temp_space
                    if available_space[agent][2] >0 :
                        z2=80-interference_points['agent5'][0]-interference_points['agent5'][1]
                        if z2<0:
                            temp_interference=interference_points['agent5'][1]
                            temp_space=available_space[agent][2]
                            interference_points['agent5'][1]-=available_space[agent][2]
                            available_space[agent][2]-=interference_points['agent5'][1]
                            if interference_points['agent5'][1] < 0:
                                interference_points['agent5'][1]=0
                                observation['agent5'][6]-=temp_interference
                                observation[agent][8]+=temp_interference
                            else :
                                available_space[agent][2]=0
                                observation[agent][8]+=temp_space
                                observation['agent5'][6]-=temp_space
                                
                if available_space[agent][0] >0:
                    z3=80 - interference_points['agent3'][2]
                    if z3 <0:
                        temp_interference=interference_points['agent3'][2]
                        temp_space=available_space[agent][0]
                        interference_points['agent3'][2]-=available_space[agent][0]
                        available_space[agent][0]-=interference_points['agent3'][2]
                        if interference_points['agent3'][2] < 0:
                            interference_points['agent2'][2]=0
                            observation['agent3'][7]-=temp_interference
                            observation[agent][6]+=temp_interference
                        else :
                            available_space[agent][0]=0
                            observation[agent][6]+=temp_space
                            observation['agent3'][7]-=temp_space

            elif agent=='agent6':
                if available_space[agent][0] > 0:
                    z1=80-interference_points['agent1'][2]-interference_points['agent1'][3]
                    if z1<0:
                        temp_interference=interference_points['agent1'][3]
                        temp_space=available_space[agent][0]
                        interference_points['agent1'][3]-=available_space[agent][0]
                        available_space[agent][0]-=interference_points['agent1'][3]
                        if interference_points['agent1'][3] < 0:
                            interference_points['agent1'][3]=0
                            observation['agent1'][7]-=temp_interference
                            observation[agent][6]+=temp_interference
                        else :
                            available_space[agent][0]=0
                            observation[agent][6]+=temp_space
                            observation['agent1'][7]-=temp_space
                    if available_space[agent][0] >0 :
                        z2=80-interference_points['agent4'][0]-interference_points['agent4'][2]
                        if z2<0:
                            temp_interference=interference_points['agent4'][2]
                            temp_space=available_space[agent][0]
                            interference_points['agent4'][2]-=available_space[agent][0]
                            available_space[agent][0]-=interference_points['agent4'][2]
                            if interference_points['agent4'][2] < 0:
                                interference_points['agent4'][2]=0
                                observation['agent4'][8]-=temp_interference
                                observation[agent][6]+=temp_interference
                            else :
                                available_space[agent][2]=0
                                observation[agent][6]+=temp_space
                                observation['agent4'][8]-=temp_space
                                
                if available_space[agent][2] >0:
                    z3=80 - interference_points['agent6'][1]
                    if z3 <0:
                        temp_interference=interference_points['agent6'][1]
                        temp_space=available_space[agent][2]
                        interference_points['agent6'][1]-=available_space[agent][2]
                        available_space[agent][2]-=interference_points['agent6'][1]
                        if interference_points['agent6'][1] < 0:
                            interference_points['agent6'][1]=0
                            observation['agent6'][7]-=temp_interference
                            observation[agent][8]+=temp_interference
                        else :
                            available_space[agent][1]=0
                            observation[agent][8]+=temp_space
                            observation['agent6'][7]-=temp_space
                            
            elif agent=='agent7':
                if available_space[agent][1] > 0:
                    z1=80-interference_points['agent1'][4]-interference_points['agent1'][5]
                    if z1<0:
                        temp_interference=interference_points['agent1'][5]
                        temp_space=available_space[agent][1]
                        interference_points['agent1'][5]-=available_space[agent][1]
                        available_space[agent][1]-=interference_points['agent1'][5]
                        if interference_points['agent1'][5] < 0:
                            interference_points['agent1'][5]=0
                            observation['agent1'][8]-=temp_interference
                            observation[agent][7]+=temp_interference
                        else :
                            available_space[agent][1]=0
                            observation[agent][7]+=temp_space
                            observation['agent1'][8]-=temp_space
                    if available_space[agent][1] >0 :
                        z2=80-interference_points['agent6'][0]-interference_points['agent6'][2]
                        if z2<0:
                            temp_interference=interference_points['agent6'][2]
                            temp_space=available_space[agent][1]
                            interference_points['agent6'][2]-=available_space[agent][1]
                            available_space[agent][1]-=interference_points['agent2'][2]
                            if interference_points['agent6'][2] < 0:
                                interference_points['agent6'][2]=0
                                observation['agent6'][6]-=temp_interference
                                observation[agent][7]+=temp_interference
                            else :
                                available_space[agent][1]=0
                                observation[agent][7]+=temp_space
                                observation['agent6'][6]-=temp_space
                                
                if available_space[agent][0] >0:
                    z3=80 - interference_points['agent2'][1]
                    if z3 <0:
                        temp_interference=interference_points['agent2'][1]
                        temp_space=available_space[agent][0]
                        interference_points['agent2'][1]-=available_space[agent][0]
                        available_space[agent][0]-=interference_points['agent2'][1]
                        if interference_points['agent2'][1] < 0:
                            interference_points['agent2'][1]=0
                            observation['agent2'][8]-=temp_interference
                            observation[agent][6]+=temp_interference
                        else :
                            available_space[agent][0]=0
                            observation[agent][6]+=temp_space
                            observation['agent2'][8]-=temp_space


        #total_coverage=100*observation['agent0'][12]/area_points
        #print(total_coverage)

        observation['agent0'][12]=100*observation['agent0'][12]/area_points
        total_coverage=observation['agent0'][12]
        if total_coverage < 65 :
            rewards['agent0']+= (total_coverage-80)*3
        elif total_coverage < 90:
            rewards['agent0']+= (total_coverage -80)*2
        else :
            rewards['agent0']+= (total_coverage - 82)*3
            
        for i in range(12):
                if observation['agent0'][i]==0:
                    rewards['agent0']+=-1
                else :
                    rewards['agent0']+= 40 - observation['agent0'][i]/2
                
        done=0
        for agent in observation:
            fin=0
            
            if agent != 'agent0':
                if  observation[agent][0] < self.west or observation[agent][0] > self.east or observation[agent][1]< self.south or observation[agent][1] > self.north or observation[agent][2] < 28 or observation[agent][2] > 190 or observation[agent][3] >40 or observation[agent][3] <5 or observation[agent][4] >40 or observation[agent][4]<5 or observation[agent][5]>40 or observation[agent][5]<5:
                    rewards[agent]+= -200
                    truncateds[agent]=True
                    truncateds["__all__"]=True

                if observation[agent][6] <0:
                    observation[agent][6]=0
                if observation[agent][7] <0:
                    observation[agent][7]=0
                if observation[agent][8] <0:
                    observation[agent][8]=0
                
                if observation[agent][6] > max_users:
                    rewards[agent]+= (-observation[agent][6]+max_users)/2 + 0.1*max_users
                else:
                    rewards[agent]+= observation[agent][6]/2
                    fin+=1

                if observation[agent][7] > max_users:
                    rewards[agent]+= (-observation[agent][7]+max_users)/2 + 0.1*max_users
                else:
                    rewards[agent]+= observation[agent][7]/2
                    fin+=1

                if observation[agent][8] > max_users:
                    rewards[agent]+= (-observation[agent][8]+max_users)/2 + 0.1*max_users
                else:
                    rewards[agent]+= observation[agent][8]/2
                    fin+=1

                if fin ==3 :
                    #terminateds[agent]=True
                    done+=1

        print(self.iter,self.steps,self.accepted)
        print(observation)
        print(rewards)
        
        if observation['agent0'][12] > 85:
            #terminateds['agent0']=True
            done+=1

        if done ==8 :
            terminateds["__all__"]=True
            print("lek eeeeeeeee")
            self.accepted+=1

        if self.steps >= 70: #Truncated if the episode is finished abruptly like number of steps excedded
            truncateds["__all__"]=True
            
        self._render()

        return observation, rewards, terminateds, truncateds, {}

    def _render(self):
        """Export coverage + population maps, blend them, and display on the Windows desktop."""
        self.grass.run_command('r.out.png', input=f'coverage@{GRASS_MAPSET}', output=VIZ_COVERAGE_WSL, overwrite=True)
        self.grass.run_command('r.out.png', input=POPULATION_MAP, output=VIZ_POPULATION_WSL, overwrite=True)

        time.sleep(1)

        coverage_img   = Image.open(VIZ_COVERAGE_WSL).convert("RGBA")
        population_img = Image.open(VIZ_POPULATION_WSL).convert("RGBA")
        combined       = Image.blend(coverage_img, population_img, alpha=0.5)
        combined.save(VIZ_COMBINED_WSL)

        if os.path.exists(VIZ_COMBINED_WSL):
            subprocess.run(["cmd.exe", "/c", "start", VIZ_COMBINED_WIN])
            time.sleep(5)

        os.remove(VIZ_COVERAGE_WSL)
        os.remove(VIZ_POPULATION_WSL)
        os.remove(VIZ_COMBINED_WSL)

    def reset(self, seed=None, options=None):

        self.steps=0
        
        if (self.location=="suburban"):
            radius=random.randrange(1500,2500,29)
        elif(self.location=="urban"):
            radius=random.randrange(500, 2000,29)
        else :
            radius=random.randrange(5000,30000,29)


        #Agent1 location which is the central bts is in the center of the region x=(w+e)/2 nad y=(s+n)/2
        x=int((self.west + self.east)/2)
        shift_x=random.randrange(int((self.west-x)/4),int((self.east-x)/4),29)
        x=x+shift_x
        y=int((self.south+self.north)/2)
        shift_y=random.randrange(int((self.south-y)/4),int((self.north-y)/4),29)
        y=shift_y+y
        #{x,y,height,power1,power2,power3,capacity1,capacity2,capacity3} for eACH NORMAL AGENT from 1-7
        #{2_1, 2_3, 2_7, 3_1, 3_4, 4_1, 4_5, 5_1, 5_6, 6_1, 6_7, 7_1, Total coverage} for agent 0 which is the group director
        #I have to declare the rest of the agents with x,y randomized according to central bts the power random with bounds and height random with bounds and capacity=0
        
        info={}
        
        self.observation={ "agent0" : np.zeros(13,dtype=np.int32),
                      "agent1" : [x,y,random.randrange(30,180),28,28,28,0,0,0],
                      "agent2" : [x,y+2*radius,random.randrange(30,180),28,28,28,0,0,0],
                      "agent3" : [int(x + 1.7*radius),y+radius,random.randrange(30,180),28,28,28,0,0,0],
                      "agent4" : [int(x + 1.7*radius),y-radius,random.randrange(30,180),28,28,28,0,0,0],
                      "agent5" : [x,y-2*radius,random.randrange(30,180),28,28,28,0,0,0],
                      "agent6" : [int(x - 1.7*radius),y-radius,random.randrange(30,180),28,28,28,0,0,0],
                      "agent7" : [int(x - 1.7*radius),y+radius,random.randrange(30,180),28,28,28,0,0,0]
            }
        with open(AGENTS_CSV, 'r+') as csvfile:
            reader = csv.reader(csvfile)
            writer = csv.writer(csvfile)
            data = list(reader)
            for i in range(1,22):
                data[i][3]=self.observation[data[i][0]][0]
                data[i][4]=self.observation[data[i][0]][1]
                data[i][5]=self.observation[data[i][0]][2]
                data[i][10]=28
            while len(data) > 22:
                data.pop()
            csvfile.seek(0)
            writer.writerows(data)
        
        return self.observation, info
