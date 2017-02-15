""" Algorithms for sorting polygons within a writefield """

# import some polygon functions
import numpy as np
from ebeamtools.polygons import polyCOM, polyUtility, get_starting_points, get_ending_points

# check if ortools is installed
import importlib
ortools_spec = importlib.util.find_spec("ortools")
if ortools_spec is None:
    ORTOOLS = False
else:
    ORTOOLS = True
    from ortools.constraint_solver import pywrapcp
    from ortools.constraint_solver import routing_enums_pb2

###################################
#### travelling ebeam solution ####
###################################

# reference: https://developers.google.com/optimization/routing/tsp/tsp

### Cost/Distance Callbacks ###

def euclidean_distance(vi, vj):
    """ vi and vj are polygon vertices"""
    return np.sqrt(np.sum(((vi-vj)**2)))

# one can also imaging a cost function
# that gives slight preference to jumping between 
# polygons of the same width
#
# def weighted_distance(vi, vj, wi, wj):
#     ...
#     return ...

class EuclideanDistanceCallback(object):
    """Create callback to calculate distances between points."""

    def __init__(self, start_points, end_points):
        """ Initialize distance array.
        
            start_points is the starting point of each polygon write
            end_points is the ending point of each polygon write. 
            
            results seem to be best when """

        if len(start_points)!=len(end_points):
            raise ValueError('Input arrays must be the same length.')
            
        # create matrix with one additional node
        # this node has distance zero to all other nodes and will be
        # removed at the end of the calculation
        # this is a hack to remove the restriction that the salesman
        # return to his home city
        self.tsp_size = len(start_points)+1
        self.matrix = np.zeros((self.tsp_size, self.tsp_size))

        for from_node in range(len(start_points)):
            for to_node in range(len(start_points)):
                if from_node == to_node:
                    self.matrix[from_node][to_node] = 0
                else:
                    self.matrix[from_node][to_node] = euclidean_distance(end_points[from_node], start_points[to_node])

    def Distance(self, from_node, to_node):
        return self.matrix[from_node][to_node]
        
        
def travelling_ebeam_sort(poly_list, timeout = 0):
    """ Use Google's ortools package to sort polygons in a layer by finding a solution
        to the travelling salesman problem.
        
        Args:
            poly_list (list): list of 2D numpy arrays defining the vertices of each polygon
                
        Kwargs: 
            timeout (float): timeout in seconds to wait for TSP solution
                
        Returns:
            array: numpy array of indices that sort poly_list """
        
    if not ORTOOLS:
        raise ImportError('Cannot use TSP sort without the \'ortools\' package')
    
    # not clear if using center of mass
    # or actual starting and ending points is better
    com = polyUtility(poly_list, polyCOM)
    starts = get_starting_points(poly_list)
    ends = get_ending_points(poly_list)
    
    # create distance matrix and define callback function
    matrix = EuclideanDistanceCallback(com, com)
    matrix_callback = matrix.Distance

    # create solver instance
    routing = pywrapcp.RoutingModel(matrix.tsp_size, 1, 0) # problem size, number of routes allowed, starting node

    # create search parameters
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

    # Setting first solution heuristic (cheapest addition).
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
    # option to give more time to escape local minima
    # this will use all of the allotted timeout to keep searching
    if timeout!=0:
        search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit_ms = int(timeout*1000)

    # Setting the cost function.
    # Put a callback to the distance accessor here. The callback takes two
    # arguments (the from and to node inidices) and returns the distance between
    # these nodes.
    routing.SetArcCostEvaluatorOfAllVehicles(matrix_callback)

    # Solve, returns a solution if any.
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
        # Solution cost.
#         print("Total distance: " + str(assignment.ObjectiveValue()) + " um\n")
        # Inspect solution.
        # Only one route here; otherwise iterate from 0 to routing.vehicles() - 1
        route_number = 0
        index = routing.Start(route_number) # Index of the variable for the starting node.
        route = np.empty(matrix.tsp_size+1, dtype=np.int) # list to hold route results
        i = 0
        while not routing.IsEnd(index):
            # Convert variable indices to node indices in the displayed route.
            route[i] = routing.IndexToNode(index)
            index = assignment.Value(routing.NextVar(index))
            i+=1
        route[i] = routing.IndexToNode(index)
    else:
        raise RuntimeError('No solution found when attempting to sort polygons.')
    
    # edit the result to fit my problem
    # remove the last point
    route = np.delete(route, -1)
    # find the index of the fake node
    zero_node = np.argwhere(route==len(com))[0][0]
    # roll the array so the zero node is at the start
    route = np.roll(route, -zero_node)
    # remove the zero_node
    route = np.delete(route, 0)
    
    return route

#########################
#### old/depreciated ####
#########################

def typewriter_sort(poly_list, eps):
    """ Sort polygons left to right, top to bottom, based on the location of
        their center of mass. This turned out to be useful when the SEM
        in use had a serious stage drift problem. 
        
        The TSP solution is probably a better choice.
        
        Args:
            poly_list (list): list of 2D numpy arrays defining the vertices of each polygon
                
        Kwargs: 
            n (float): grid in microns to round COM coordinates to
                
        Returns:
            array: numpy array of indices that sort poly_list """

    com = polyUtility(poly_list, polyCOM)

    X = -np.floor(com/eps)[:,0]*eps
    Y = -np.floor(com/eps)[:,1]*eps
    return np.lexsort((X, Y))[::-1]
    
def euclidean_distance_list(pnts0, pnts1):
    return np.sqrt(np.sum(((pnts0-pnts1)**2), axis=1))
    
def walking_sort(poly_list, starting_point = None):
    """ This is pretty much a lazy travelling salesman solution.
        Start at a given point and jump to the nearest polygon 
        (ignoring those that have already been written. 
        
        I see no reason why you should use this over the TSP solution. 
        
        Args:
            poly_list (list): list of 2D numpy arrays defining the vertices of each polygon
                
        Kwargs: 
            starting_point (array): start with the polygon closest to this point
                
        Returns:
            array: numpy array of indices that sort poly_list """
    
    com = polyUtility(poly_list, polyCOM)

    if not starting_point:
       start = np.amin(com, axis=0) # lower left corner
    else:
        start = starting_point
        
    sorted_idx = np.empty(len(com_list), dtype=np.int)
    sorted_idx[:] = np.nan

    for i in range(len(com_list)):
        # find index of com closest to start
        for j in np.argsort(euclidean_distance_list(com_list, start)):
                if j not in sorted_idx:
                    sorted_idx[i] = j
                    start = com[j]
                    break
                else:
                    continue
    return sorted_idx