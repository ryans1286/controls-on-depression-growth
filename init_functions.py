def load_parameter_file(file):
    """
    Loads parameters from CSV file. Skips the first 16 rows in the CSV.
    
    RETURNS: list of lists containing parameters for each simulation
    
    PARAMETERS
    ----------
    file: path to parameter file; CSV
    """
    import csv
    from itertools import islice

    skiprows = 0  # Initialize skiprows variable

    with open(file, newline='') as f:
        for line_number, line in enumerate(f, 1):
            skiprows += 1  # Increment skiprows for each comment line
            if line.startswith("#"):
                break
    params = []
    with open(file, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for r in islice(reader, skiprows + 1, None):
            params.append(r)
    number_of_simulations = len(params)
    for sim in range(number_of_simulations):
        params[sim][0] = int(params[sim][0])        #Simulation number, integer
        params[sim][1] = float(params[sim][1])      #Total model time, float, years
        params[sim][2] = float(params[sim][2])      #Time step interval, float, years
        params[sim][3] = int(params[sim][3])        #Initial number of sink points, int
        params[sim][4] = float(params[sim][4])      #Initial sink depth, float, meters
        params[sim][5] = float(params[sim][5])      #Bare ice melt rate, float, meters/year
        params[sim][6] = float(params[sim][6])      #Critical debris thickness, float, meters
        params[sim][7] = float(params[sim][7])      #Debris diffusivity, float, meters/year
        params[sim][8] = float(params[sim][8])      #Critical slope threshold, float, meters/meter  
        params[sim][9] = float(params[sim][9])      #Stream power constant, float 
        params[sim][10] = float(params[sim][10])    #Stream power m, float
        params[sim][11] = float(params[sim][11])    #Stream power n, float
        params[sim][12] = float(params[sim][12])    #Threshold stream power, float        
        params[sim][13] = float(params[sim][13])    #Englacial debris concentration, float, meters/meter
        params[sim][14] = str(params[sim][14])      #Location of sinks, string, "center" or "random"
        params[sim][15] = float(params[sim][15])    #Scaling constant for debris removal from sinks, meters
        params[sim][16] = int(params[sim][16])      #Random state
        params[sim][17] = float(params[sim][17])    #Lowering rate of the sink point, meters
        params[sim][18] = int(params[sim][18])      #number of columns, int
        params[sim][19] = int(params[sim][19])      #number of rows, int
        params[sim][20] = float(params[sim][20])    #grid spacing, float, meters
    return params   

def assign_sink_nodes(n_sinks, nr, nc, grid, sink_loc, randomstate = None):
    """
    RETURNS: Numpy array of sink point node IDs.
    
    PARAMETERS
    ----------
    n_sinks: integer; number of sink points 
    nr: integer; number of rows in model grid
    nc: integer; number of columns in model grid
    grid: ModelGrid; Landlab model grid object
    sink_loc: string; "center" or "random"; sets sink point locations
    randomstate: (OPTIONAL, default None); integer; 
                allows user control of random sink id assignment
    """
    from numpy.random import RandomState
    import numpy as np
    if sink_loc == 'random':
        nodes = grid.number_of_nodes
        if randomstate == None:
            print("WARNING: Random state not set.")
            print("Default random state == 1")
            randomstate = 1
        r = np.random.RandomState(randomstate)
        sink_nodes = r.randint(low = 0, high = nodes, size = n_sinks)
    
    elif sink_loc == 'center':
        if n_sinks > 1:
            print("ERROR: You can only have a single node at the center.")
            return
        center_node = nr//2 * nc + nc//2
        sink_nodes = np.array([center_node])
    return sink_nodes

def create_sinks(topo_elevation_grid, sink_nodes, depth = 0.1):
    """
    Creates depressions in the topographic__elevation grid.
    
    RETURNS: None
    
    PARAMETERS
    ----------
    topo_elevation_grid: DataField that stores "topographic__elevation";
                        IMPORTANT: For the debris-covered glacier model, this layer must be
                        the ice surface. The ice surface should be assigned to the 
                        "topographic__elevation" layer. 
    sink_nodes: numpy array; array of sink node IDs
    depth: (OPTIONAL, default 0.1 m) float; 
            allows user control of random sink point assignment;
            units = meters
    """
    for i in sink_nodes:
        topo_elevation_grid[i] -= depth
    return

def remove_debris_from_sinks(sink_nodes, C, dt):
    """
    Removes debris from sink points. Debris removal scales with the square root
    of the drainage area at the sink point. 

    RETURNS: None

    PARAMETERS
    ----------
    sink_nodes: Numpy array of sink points
    C: Scaling constant for debris removal
    dt: Time step length, units = model time units
    """
    for i in sink_nodes:
        #Calculate how much debris should be removed
        remove_debris = np.sqrt(mg.at_node['drainage_area'][i]) * C * dt
        
        #Remove that amount of debris if there is enough on the sink node
        if h[i] >= remove_debris:
            h[i] -= remove_debris

        #Otherwise, remove all the debris from the sink node
        elif h[i] < remove_debris and h[i] > 0:
            h[i] = 0
    return