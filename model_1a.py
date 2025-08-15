"""
Imports
"""
import numpy as np
from landlab import RasterModelGrid, imshow_grid
from landlab.components import FlowAccumulator, FastscapeEroder, DepressionFinderAndRouter, LakeMapperBarnes
import matplotlib.pyplot as plt
from numpy.random import RandomState
import csv
from itertools import islice
import os

"""Import functions created for this model."""
import init_functions

"""
Assign parameter file to variable.
"""
parameter_file = 'input_parameters_1a.csv'

"""
Create a data output directory
"""
output_dir = 'output-1a/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

#Create list of lists for parameters
#Each row in the list contains parameters for a single model run
params = init_functions.load_parameter_file(parameter_file)
number_of_simulations = len(params)

#Create a simulation for each parameter set
for sim in range(number_of_simulations):
    sim_num = params[sim][0]            #Simulation number, integer
    time = params[sim][1]               #Total model time, float, years
    dt = params[sim][2]                 #Time step interval, float, years
    n_sinks = params[sim][3]            #Initial number of sink points, int
    init_sink_depth = params[sim][4]    #Initial sink depth, float, meters
    b_0 = params[sim][5]                #Bare ice melt rate, float, meters/year
    h_c = params[sim][6]                #Critical debris thickness, float, meters
    D = params[sim][7]                  #Debris diffusivity, float, meters^2/year
    S_c = params[sim][8]                #Critical slope threshold, float, meters/meter  
    K_sp = params[sim][9]               #Stream power constant, float 
    m_sp = params[sim][10]              #Stream power m, float
    n_sp = params[sim][11]              #Stream power n, float
    threshold_sp = params[sim][12]      #Threshold stream power, float        
    deb_conc = params[sim][13]          #Englacial debris concentration, float, meters/meter
    sink_loc = params[sim][14]          #Location of sinks, string, "center" or "random"
    C_d = params[sim][15]               #Scaling constant for debris removal from sinks, meters
    rs = params[sim][16]                #Random state
    dz_sink = params[sim][17]           #Lowering rate of the sink point, meters
    ncols = params[sim][18]
    nrows = params[sim][19]
    dx_spacing = params[sim][20]


    #Create a simulation file name
    sim_filename = "sim" + parameter_file[-7:-4] + "_" + str(sim_num) + ".csv"

    """
    Create model grid
    """

    #Define RasterModelGrid dimensions
    nc = ncols    #number of columns (x-direction)
    nr = nrows    #number of rows (y-direction)
    dx = dx_spacing     #METERS grid spacing


    #Create the RasterModelGrid
    mg = RasterModelGrid((nr, nc), xy_spacing=dx)


    #Create ice surface field
    init_elevation = 100    #METERS glacier thickness at terminus
    ns_slope = 0.01         #METER/METER slope increase to north

    s = np.ones(nr*nc) * init_elevation + mg.y_of_node * ns_slope

    #IMPORTANT!
    #The flow routing components act on the 'topographic__elevation'
    #field. For the debris-covered glacier model, we want streams to
    #route over the ice surface. Therefore, we must assign the 
    #'topographic__elevation' field to the ice surface. 
    s = mg.add_field('topographic__elevation', s, at = 'node')


    #Create debris thickness, topographic surface, debris flux, and melt fields
    h = mg.add_zeros('debris_thickness', at = 'node')
    z = mg.add_zeros('topographic_surface', at = 'node')
    qs = mg.add_zeros('debris_flux', at = 'link')
    melt = mg.add_zeros('melt', at = 'node')

    h += 0.1


    #Assign boundary conditions
    #Water and debris will only exit at the bottom edge of the
    #model grid. Here, we set all boundary conditions to 
    #fixed values to work with the flow router. Outlet nodes
    #are assigned manually later on. 
    mg.set_fixed_value_boundaries_at_grid_edges(True, True, True, True)

    #Create englacial sink points
    sink_points = init_functions.assign_sink_nodes(n_sinks, nr, nc, mg, sink_loc = sink_loc, randomstate = rs)
    init_functions.create_sinks(s, sink_points, depth = init_sink_depth)

    z = s + h

    #Add row 2 of RasterModelGrid to sink_points list
    #We explicitly assign this row to be open drainage 
    outlet_drainage = []
    for i in range(nc+1, 2*nc-1):
        outlet_drainage.append(i)

    outlet_drainage = np.array(outlet_drainage) #convert to array
    all_drain_points = np.concatenate((sink_points, outlet_drainage))

    #Explicitly assign the sink_points list of node IDs as OPEN

    mg.set_watershed_boundary_condition_outlet_id(all_drain_points, s, -9999.)

    t_steps = int(time/dt)

    sim_data = np.zeros(shape = (6, t_steps))
    """
    sim_data[0, :] = model time
    sim_data[1, :] = sink drainage area
    sim_data[2, :] = sink depression area
    sim_data[3, :] = sink depth
    sim_data[4, :] = sink_perimeter
    sim_data[5, :] = debris removed at sink
    """
    
    sim_data[0, :] = np.arange(0, time, dt) #fill first row with times
    print('sim = ', sim_num)
    print('D =', D)
    print('K_sp =', K_sp)
    print('C_d = ', C_d)
    sink_points_list = list(sink_points)
    print(sink_points_list)

    #BEGIN SIMULATION
    for ts in range(t_steps):
        #Calculate sub-debris melt
        melt = b_0 * dt * (h_c / (h_c + h)) #hyperbolic melt
        s -= melt

        #Calculate melt-out of englacial debris
        h = h.copy() + melt * deb_conc

        #Update surface topography
        z = h + s

        #Calculate gradient and slope of the topography
        grad = mg.calc_grad_at_link(z)
        slp = mg.calc_slope_at_node(z)
        slope = mg.map_mean_of_link_nodes_to_link(slp)

        # Calculate the "transportable" debris
        s_fill = z.copy() #create a duplicate ice surface
        fa = FlowAccumulator(mg)
        lmb = LakeMapperBarnes(mg, method = "Steepest", fill_flat = True, surface = s, fill_surface = s_fill, track_lakes = True)
        lmb.run_one_step()

        s_fill = s + lmb.lake_depths #calculate the elevation of the filled ice surface
        h_trans = z.copy() - s_fill #calculate transportable debris thickness
        for i in np.where(h_trans <= 0):
            h_trans[i] = 0

        #Map mean debris thickness of adjacent nodes to link
        h_link = mg.map_mean_of_link_nodes_to_link(h_trans)

        #Calculate hillslope debris flux
        qs[mg.active_links] = D * (1 - np.exp(-h_link[mg.active_links] / h_c)) * grad[mg.active_links] / (1 - (slope[mg.active_links]/S_c)**2)


        #Calculate divergence of debris flux
        dhdt = mg.calc_flux_div_at_node(qs) * dt

        #Update the debris thickness
        h = h.copy() + dhdt

        #Update surface topography
        z[mg.core_nodes] = s[mg.core_nodes] + h[mg.core_nodes]

        #Ensure that the sink point nodes are lower than adjacent nodes
        for n in sink_points:
            s[n] = np.amin(s[mg.active_adjacent_nodes_at_node[n]]) - dz_sink

        #Accumulate flow and erode ice surface (s)
        fr = FlowAccumulator(mg, flow_director = 'D8', depression_finder = DepressionFinderAndRouter)
        sp = FastscapeEroder(mg, K_sp = K_sp * b_0, m_sp = m_sp, n_sp = n_sp, threshold_sp = threshold_sp)

        fr.run_one_step()
        sp.run_one_step(dt = dt)

        #Route debris through the sink point, rate scales with drainage area
        #MODEL OUTPUT STORED HERE
        for k in sink_points_list:
            if h[k] >= np.sqrt(mg.at_node['drainage_area'][k]) * C_d * dt:
                h[k] -= np.sqrt(mg.at_node['drainage_area'][k]) * C_d * dt
                
                sim_data[5, ts] += dx * dx * np.sqrt(mg.at_node['drainage_area'][k]) * C_d * dt #meters cubed

            elif h[k] > 0 and h[k] < np.sqrt(mg.at_node['drainage_area'][k]) * C_d * dt:
                sim_data[5, ts] += dx * dx * h[k] #meters cubed
                h[k] = 0

        #Keep track of sink point drainage area and filled depression area
        #MODEL OUTPUT STORED HERE
        for j in sink_points_list:
            #Calculate drainage are at sink point
            drainage_area = mg.at_node['drainage_area'][j] 
            
            mg.status_at_node[j] = 0 #close drainage status at sink point
            df2 = DepressionFinderAndRouter(mg) 
            df2.map_depressions()

            #Calculate depression area at sink point
            depression_area = df2.lake_areas[0] 

            #Calculate depression depth
            outlet_node = df2.lake_outlets[0] 
            outlet_elevation = mg.at_node['topographic__elevation'][outlet_node]
            depth = outlet_elevation - np.amin(z[mg.active_adjacent_nodes_at_node[j]])

            #Calculate the depression perimeter
            depression_nodes = np.where(df2.lake_at_node == True) # Get depression node ids
            # Find boundary nodes of the depression
            boundary_nodes = []
            for node in depression_nodes:
                neighbors = mg.active_adjacent_nodes_at_node[node]
                for neighbor in neighbors:
                    if df2.lake_at_node[neighbor][0] != True:
                        boundary_nodes.append(neighbor)
            perimeter_length = mg.length_of_d8[boundary_nodes].sum()
            
            sim_data[1, ts] = drainage_area
            sim_data[2, ts] = depression_area
            sim_data[3, ts] = depth
            sim_data[4, ts] = perimeter_length

            mg.status_at_node[j] = 1 #re-open sink point drainage

        #Update all the fields
        mg.at_node['debris_thickness'] = h
        mg.at_node['topographic_surface'] = z
        mg.at_node['topographic__elevation'] = s

    #Write the simulation data to CSV
    sim_data_T = sim_data.T
    column_names = ['time_y', 'drainage_area_m2', 'depr_area_m2', 'depr_depth_m', 'depr_perimeter_m', 'debr_removed_m3']
    np.savetxt(os.path.join(output_dir, sim_filename), sim_data_T, delimiter = ',', header=','.join(column_names), comments='')


        

