""" this module contains functions useful for polygon statistics """

import numpy as np
import warnings

####################
### Polygon math ###
####################

def polyArea(verts0):
    """ Find area of a polygon that has vertices in a numpy array
        
        Args:
            verts (array): np.array([x0 y0], [x1 y1], ....) 
        Returns:
            float: Area of polygon. Sign gives orientation (<0 clockwise). """
            
    verts1 = np.roll(verts0, -1, axis=0)
    return 0.5*np.sum(verts0[:,0]*verts1[:,1] - verts1[:,0]*verts0[:,1])

def polyCOM(verts0):
    """ Find center of mass of a polygon that has vertices in a numpy array
    
        Args:
            verts (array): np.array([x0 y0], [x1 y1], ....) 
        Returns:
            array: np.array([x_com, y_com])"""
            
    A = 1/(6*polyArea(verts0))
    verts1 = np.roll(verts0, -1, axis=0)
    C = verts0[:,0]*verts1[:,1] - verts1[:,0]*verts0[:,1]
    X = np.sum((verts0[:,0] + verts1[:,0])*C)
    Y = np.sum((verts0[:,1] + verts1[:,1])*C)
    return A*np.array([X, Y])

def polyPerimeter(verts0):
    """ Find perimeter length of a polygon that has vertices in a numpy array.
    
        Args:
            verts (array): np.array([x0 y0], [x1 y1], ....) 
        Returns:
            float: length of the polygon perimenter. """
            
    verts1 = np.roll(verts0, -1, axis=0)
    return np.sum(np.hypot(verts0[:,0] - verts1[:,0],verts0[:,1] - verts1[:,1]))

def polyUtility(poly_list, polyFunc):
    """ Takes an array full of polygon vertices, as created by 
        get_vertices, and returns an array full of values returned by 
        polyFunc
        
        Args:
            poly_list (list): list of 2D numpy arrays defining the vertices of a number of polygons
            polyFun (function): a function to apply to the list of polygons
        Returns:
            list: output of polyFunc for each polygon in poly_list """
            
    return np.array([polyFunc(v) for v in poly_list])
    
########################################################################
### Functions to normalize everything into a standard polygon format ###
########################################################################

def remove_duplicate_vertices(verts, eps):
    """ Look for duplicate points (closing point excluded) in lists of polygon vertices.
    
        Args:
            verts (np.ndarray): x,y coordinates for each vertex of a polygon
            
        Returns 
            np.ndarray: modified verts with duplicate points removed """
            
    idx = np.ones(verts.shape, dtype=bool)
    idx[0:-1] = (np.abs(verts - np.roll(verts,-1, axis=0))>eps)[0:-1]
    return verts[np.logical_or(idx[:,0],idx[:,1])]

def contains_closing_point(verts, eps):
    """ Check that the polygon described by verts contains
        a closing point.
        
        Args:
            verts (list): a list of vertices in the form np.array([x,y])
        Returns:
            bool: True if verts contains a closing point. False otherwise. """
        
    return np.all([abs(v)<eps for v in verts[0]-verts[-1]])
        
def close_all_polygons(poly_list, eps, warn = True):
    """ Go through poly_list and look for polygons that are not closed
        (first point the same as last point). 
        
        Args:
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons
        Kwargs:
            warn (bool): True if you want warn to print to the terminal
        
        returns: list of polygons with one of the duplicates removed """

    for i in range(len(poly_list)):
        if not contains_closing_point(poly_list[i], eps):
            poly_list[i] = np.vstack((poly_list[i], poly_list[i][0]))
            # if warn:
                # print('POLYGON CLOSED ({0})'.format(i))

    return poly_list

def sort_by_position(verts, eps):
    """ Sort polygons left to right, top to bottom, based on the location of
        their center of mass.
        
        Args:
            pnts (array): 2D numpy array of coordinates 
                
        Kwargs: 
            n (float): grid in microns to round COM coordinates to
                
        Returns:
            array: numpy array of indices that sort com """

    X = -np.floor(verts/eps)[:,0]*eps
    Y = -np.floor(verts/eps)[:,1]*eps
    return np.lexsort((X, Y))[::-1]

def same_shape(verts0, verts1, eps):
    """ Check if two lists of vertices contain the same points. 
    
        Args:
            verts0 (list): list of (x,y) vertices for polygon 0
            verts1 (list): list of (x,y) vertices for polygon 1
            
        Returns: 
            bool: True if verts0 and vert1 describe the same polygon """
            
    # get out of here immediately if the number of points is different
    if verts0.shape!=verts1.shape:
        return False
    
    # sort points in some known order
    ind0 = sort_by_position(verts0, eps)
    ind1 = sort_by_position(verts1, eps)
    verts0 = verts0[ind0]
    verts1 = verts1[ind1]
    
    # check distance between points
    dist = np.linalg.norm(verts0-verts1, axis=1)
    return np.all([d<eps for d in dist])

def remove_duplicate_polygons(poly_list, eps, warn=True):
    """ Look through the list of polygons to see if any are repeated. Print warning if they are. 
        
        Args:
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons
        Kwargs:
            warn (bool): True if you want warn to print to the terminal
            
        Returns: 
            list: modified poly_list with duplicates removed """
    
    ind = []
    for i in range(len(poly_list)):
        for j in range(len(poly_list)):
            if j>=i:
                pass
            else:
                if same_shape(poly_list[i], poly_list[j], eps):
                    if warn:
                        com = polyCOM(poly_list[i])
                        print('DUPLICATE POLYGON REMOVED AT ({0:.1f}, {1:.1f})'.format(com[0],com[1]))
                    ind.append(i)
    return [vert for i, vert in enumerate(poly_list) if i not in ind]
    
def normalize_polygon_orientation(poly_list, warn = True):
    """ Make sure all polygons have their vertices listed in counter-clockwise order.
    
        Args:
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons
        Kwargs:
            warn (bool): True if you want warn to print to the terminal
            
        Returns: 
            list: modified poly_list with properly rotated polygons """
            
    for i in range(len(poly_list)):
        if polyArea(poly_list[i])<0:
            poly_list[i] = poly_list[i][::-1]
        
    return poly_list
    
def rotate_to_longest_side(verts, eps):
    """ Rotate the order in which vertices are listed such that the two points defining
        the longest side of the polygon come first. In NPGS, this vertex ordering defines
        the direction in which the electron beam sweeps to fill the area.
        
        Args:
            verts (list): a list of vertices in the form np.array([x,y])
        Returns:
            list: modified verts """
            
    verts = verts[:-1] # remove closing point

    lower_left = np.array([verts[:,0].min(), verts[:,1].min()]) # lower left corner of bounding box
    side_lengths = np.sqrt(np.sum((verts-np.roll(verts,-1, axis=0))**2, axis=1)) # lengths of sides
    centers = 0.5*(verts+np.roll(verts,-1, axis=0)) # center point of each side

    max_length = side_lengths.max() # length of longest side
    long_ind = np.abs(side_lengths - max_length) < eps # find indices of all sides with this length

    c_to_ll = np.sqrt(np.sum((centers-lower_left)**2,axis=1)) # distance from center points to lower left corner
    c_to_ll[~long_ind] = np.inf
    start_ind = c_to_ll.argmin()

    verts = np.roll(verts, -start_ind, axis=0)
    return np.vstack((verts,verts[0]))
    
def choose_scan_side(poly_list, eps):
    """ A function to wrap rotate_to_longest_side such that it can operate on a list.
    
        Args:
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons
        Returns:
            list: modified poly_list """
            
    return [rotate_to_longest_side(vert, eps) for vert in poly_list]

def line2poly_const(centers, width, warn=True):
    """ Convert lines of constant width to filled polygons. 
    
        Args:
            centers (array of tuples): vertices defining center of line
            width (float): width of line
        Returns:
            np.ndarray: list of x,y coordinates for polygon vertices """
    
    lower = np.zeros(centers.shape) # to hold vertices for lower parallel line
    upper = np.zeros(centers.shape) # to hold vertices for upper parallel line

    diff = np.roll(centers,-1, axis=0)-centers # vectors representing each line segement
    phi = np.arctan2(diff[:,1],diff[:,0]) # angle each line segment makes with x-axis
    m = np.tan(phi) # slope of each line segment to avoid div by 0
    b_lower = centers[:,1]-m*centers[:,0]-0.5*width/np.cos(phi) # intercepts of lower parallel line
    b_upper = centers[:,1]-m*centers[:,0]+0.5*width/np.cos(phi) # intercepts of upper parallel lines

    # find all intersections, ignore endpoints
    eps = 1e9

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
    
        try:
            for i in range(1,len(centers)-1):
                if np.abs(m[i])<eps:
                    a = m[i]
                    bl = b_lower[i]
                    bu = b_upper[i]
                elif np.abs(m[i-1])<eps:
                    a = m[i-1]
                    bl = b_lower[i-1]
                    bu = b_upper[i-1]
                lower[i,0] = ((b_lower[i]-b_lower[i-1])/(m[i-1]-m[i]))
                lower[i,1] = a*((b_lower[i]-b_lower[i-1])/(m[i-1]-m[i]))+bl
                upper[i,0] = ((b_upper[i]-b_upper[i-1])/(m[i-1]-m[i]))
                upper[i,1] = a*((b_upper[i]-b_upper[i-1])/(m[i-1]-m[i]))+bu

            # find endpoints
            lower[0,0] = centers[0,0]+0.5*width*np.sin(phi[0])
            lower[0,1] = centers[0,1]-0.5*width*np.cos(phi[0])
            upper[0,0] = centers[0,0]-0.5*width*np.sin(phi[0])
            upper[0,1] = centers[0,1]+0.5*width*np.cos(phi[0])

            lower[-1,0] = centers[-1,0]+0.5*width*np.sin(phi[-2])
            lower[-1,1] = centers[-1,1]-0.5*width*np.cos(phi[-2])
            upper[-1,0] = centers[-1,0]-0.5*width*np.sin(phi[-2])
            upper[-1,1] = centers[-1,1]+0.5*width*np.cos(phi[-2])

            return np.vstack((lower, upper[::-1,:], [lower[0,:]]))
    
        except Warning:
            if warn:
                print('LINE CONVERSION FAILED. {0}'.format(verts[:,0:2]))
            return None
            
def split_line2poly(verts):
    """ take a n irregularly shaped polygon (created from a line entity) and split it 
        into individual polygons each with 4 sides 
        
        Args:
            verts (np.ndarray): list of vertices defining polygon
        
        Returns: 
            list: list of np.ndarrays defining the vertices of each individual polygon """
            
    verts = verts[:-1] # drop closing point

    n = int((verts.__len__()-2)/2) # number of resulting polygons
    out = []
    for i in range(n):
        out.append(np.array([verts[i],verts[i+1],verts[-i-2],verts[-i-1],verts[i]]))
    return out
    
def pnt_pnt_dist(vi, vj):
    """ vi and vj are polygon vertices"""
    return np.sqrt(np.sum(((vi-vj)**2), axis=0))
    
# def rounded_argmax(a, min_val = dxfasc.SMALLEST_SCALE):
#     a = (a/min_val).astype('int')
#     return np.argmax(a) # returns the first occurance of the maximum 
#                         # should be the right choice for squares
                        
def last_point(verts):
    x0 = verts[0] # vertex 0
    x1 = verts[1] # vertex 1
    d = np.zeros(len(verts)-3)
    for i, x_vert in enumerate(verts[2:-1]):
        # find point along 0->1 that forms perpendicular line with 0->1 and intersects COM
        A = np.array([[x0[1]-x1[1],x1[0]-x0[0]],[x1[0]-x0[0],x1[1]-x0[1]]])
        b = np.array([(x1[1]-x0[1])*x0[0]-(x1[0]-x0[0])*x0[1],-(x1[1]-x0[1])*x_vert[1]-(x1[0]-x0[0])*x_vert[0]])
        x_intersect = np.linalg.solve(A,-b)
        d[i] = pnt_pnt_dist(x_vert, x_intersect)
    return verts[np.argmax(d)+2]
    
def get_ending_points(poly_list):
    return np.array([last_point(v) for v in poly_list])

def get_starting_points(poly_list):
    return np.array([v[0] for v in poly_list])

def list_to_nparray_safe(poly_list):
    """ Safe way to convert a list of polygon verticies to a 
        numpy array full of numpy arrays. 
        
        Args:
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons
        Returns:
            numpy.ndarray: poly_list converted to an ndarray full of ndarrays """
            
    out = np.empty(len(poly_list), dtype=np.ndarray)
    for i in range(len(out)):
        out[i] = poly_list[i]
    return out