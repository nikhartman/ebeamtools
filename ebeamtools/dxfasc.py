""" This module offers basic support for converting DXF drawings to 
    the ASCII formats supported by Raith and NPGS ebeam lithography software.
    
    The module has not been extensively tested. It may only work in a few use cases. 
    
    The package ezdxf is required for DXF read/write operations.  """

import itertools
import numpy as np
import ezdxf
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import warnings

SMALLEST_SCALE = 5e-4 # um, distances smaller than this number are considered zero

##############################################
### Functions for dealing with layer names ###
##############################################

def get_layer_names(dxf):

    """ Get list of layer names. Only lists layers that contain objects.
        Any artwork in layer 0 is ignored. Layer 0 however is added as an
        empty layer in this list, as this is required for some versions of the
        Raith software. 
        
        All characters are converted to caps and spaces to _. 
        
        Args:
            dxf (dxfgrabber object): dxfgrabber object refering to the drawing of interest 
        Returns:
            list (str): list of layer names """
            
    layers = ['0'] # this empty layer is required
    for i, ent in enumerate(dxf.entities):

        l = ent.dxf.layer.upper().replace (" ", "_")

        if i==0: 
            layers.append(l) # definitely add the first layer name
        elif l not in layers:
            layers.append(l) # add the layer name if it is not already included. 
    return layers

def print_layer_names(filename):
    """ Print all layers in a DXF file that contain artwork. Layer 0 is added 
        as an empty layer. 
        
        Args: 
            filename (str): name of DXF file """

    dxf = ezdxf.readfile(filename)

    layers = get_layer_names(dxf)
    for i, l in enumerate(layers):
        print('{0}:  {1}'.format(i, l))

##################################################################################
### Functions to normalize a list of DXF shapes into a standard polygon format ###
##################################################################################
    
def remove_duplicate_vertices(verts):
    """ Look for duplicate points (closing point excluded) in lists of polygon vertices.
    
        Args:
            verts (np.ndarray): x,y coordinates for each vertex of a polygon
            
        Returns 
            np.ndarray: modified verts with duplicate points removed """
            
    eps = SMALLEST_SCALE
    idx = np.ones(verts.shape, dtype=bool)
    idx[0:-1] = (np.abs(verts - np.roll(verts,-1, axis=0))>eps)[0:-1]
    return verts[np.logical_or(idx[:,0],idx[:,1])]
    
def import_polyline(ent, warn=True):
    """ A fucntion to import polygon entities from a drawing. Remove z coordinates, 
        convert list to numpy array, remove any duplicate vertices.
        
        Args:
            ent (dxfgrabber.entity): object representing the polygon
        
        Returns
            np.ndarray: list of x,y coordinates for polygon vertices """
            
            
    # this thing does not work right
    # I think it will need the same kind of logic as lwpolyline        
    with ent.points() as pnts:
        verts = np.array(pnts)
        return [remove_duplicate_vertices(verts[:,0:2])]
        
def import_lwpolyline(ent, split_lines=True, warn=True):
    # should always return a list of numpy.ndarrays, even if there is only one list element

    with ent.points() as pnts:
        verts = np.array(pnts)
        
    # logic to sort out what type of object ent is
    closed = ent.closed # true if shape is closed
    
    if(ent.dxf.const_width>SMALLEST_SCALE):
        const_width = True # ent.dxf.const_width is the global linewidth
        width = ent.dxf.const_width
    elif(np.count_nonzero(abs(verts[:,2:4].flatten()-verts[0,2]) > SMALLEST_SCALE)==0):
        # if all of the elements in columns 2+3 of verts are equal
        # this is a line of constant width (could be width=0)
        const_width = True
        width = verts[0,2]
    elif(np.count_nonzero(abs(verts[:,2:4].flatten()-verts[0,2]) > SMALLEST_SCALE) > 0):
        # this is a line of variable width
        const_width = False
        width = np.nan
        if warn:
            print('VARIABLE WIDTH POLYLINES NOT SUPPORTED. DXFTYPE = LWPOLYLINE. {0}'.format(verts[:,0:2]))
        return []
    
    verts = verts[:,0:2] # strip the width information now that we know it
    
    if (width<SMALLEST_SCALE and closed): 
        # closed polygons, lines have no width
        # return vertices as they are without the closing point
        return [remove_duplicate_vertices(verts)]
    elif (width<SMALLEST_SCALE and not closed):
        # most likely an unclosed polygon
        # add it and it will be fixed later.
        return [remove_duplicate_vertices(verts)]
    elif (width>SMALLEST_SCALE and not closed): # lines with constant width
        verts = line2poly_const(verts, width)
        if verts is not None:
            if split_lines:
                return split_line2poly(verts)
            else:
                return [verts]
    
def contains_closing_point(verts):
    """ Check that the polygon described by verts contains
        a closing point.
        
        Args:
            verts (list): a list of vertices in the form np.array([x,y])
        Returns:
            bool: True if verts contains a closing point. False otherwise. """
        
    eps = SMALLEST_SCALE # anything within 5A is the same point
    return np.all([abs(v)<eps for v in verts[0]-verts[-1]])
        
def close_all_polygons(poly_list, warn = True):
    """ Go through poly_list and look for polygons that are not closed
        (first point the same as last point). 
        
        Args:
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons
        Kwargs:
            warn (bool): True if you want warn to print to the terminal
        
        returns: list of polygons with one of the duplicates removed """

    for i in range(len(poly_list)):
        if not contains_closing_point(poly_list[i]):
            poly_list[i] = np.vstack((poly_list[i], poly_list[i][0]))
            # if warn:
                # print('POLYGON CLOSED ({0})'.format(i))

    return poly_list

def sort_by_position(pnts, n = SMALLEST_SCALE):
    """ Sort polygons left to right, top to bottom, based on the location of
        their center of mass.
        
        Args:
            pnts (array): 2D numpy array of coordinates 
                
        Kwargs: 
            n (float): grid in microns to round COM coordinates to
                
        Returns:
            array: numpy array of indices that sort com """

    X = -np.floor(pnts/n)[:,0]*n
    Y = -np.floor(pnts/n)[:,1]*n
    return np.lexsort((X, Y))[::-1]

def same_shape(verts0,verts1):
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
    ind0 = sort_by_position(verts0)
    ind1 = sort_by_position(verts1)
    verts0 = verts0[ind0]
    verts1 = verts1[ind1]
    
    # check distance between points
    eps = SMALLEST_SCALE # anything within 5A is the same point
    dist = np.linalg.norm(verts0-verts1, axis=1)
    return np.all([d<eps for d in dist])

def remove_duplicate_polygons(poly_list, warn=True):
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
                if same_shape(poly_list[i], poly_list[j]):
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
    
def rotate_to_longest_side(verts):
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
    long_ind = np.abs(side_lengths - max_length) < SMALLEST_SCALE # find indices of all sides with this length

    c_to_ll = np.sqrt(np.sum((centers-lower_left)**2,axis=1)) # distance from center points to lower left corner
    c_to_ll[~long_ind] = np.inf
    start_ind = c_to_ll.argmin()

    verts = np.roll(verts, -start_ind, axis=0)
    return np.vstack((verts,verts[0]))
    
def choose_scan_side(poly_list):
    """ A function to wrap rotate_to_longest_side such that it can operate on a list.
    
        Args:
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons
        Returns:
            list: modified poly_list """
            
    return [rotate_to_longest_side(vert) for vert in poly_list]

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

def get_vertices(dxf, layer, warn=True):
    """ Get list of vertices from dxf object. 
    
        This is certainly full of bugs. It has only been tested with Illustrator CS5 
        and AutoCAD 2015. There are many object types that are not supported. Ideally, something
        useful will be printed to notify you about what is missing. 

        Args:
            dxf (dxfgrabber object): dxfgrabber object refering to the drawing of interest 
            layer (str): string defining which layer will be imported
                
        Returns:
            list: list of polygon vertices as 2D numpy arrarys. """
        
    # get all layer names in dxf
    all_layers = get_layer_names(dxf)
            
    # loop through layers to create poly_list
    poly_list = []
    i = 0
    layer = layer.upper().replace(' ','_')
    
    if layer not in all_layers:
        if warn:
            print('LAYER NOT FOUND IN DRAWING -- {0}'.format(layer))
        return poly_list
    elif (layer=='0' or layer==0):
        if warn:
            print('DO NOT USE LAYER 0 FOR DRAWINGS')
        return poly_list
     
    for ent in dxf.entities:
        if ent.dxf.layer.upper().replace(' ', '_') == layer:
            i+=1
#             print(ent.dxftype())
            if ent.dxftype() == 'POLYLINE':
#                 poly_list += import_polyline(ent)
                pass
            elif ent.dxftype() == 'LWPOLYLINE':
                poly_list += import_lwpolyline(ent)
                        
            # add additional dxftypes here
                
            else:
                if warn:
                    print('NOT A KNOWN TYPE ({0}) -- LAYER: {1}'.format(ent.dxftype(), layer))
    
    poly_list = close_all_polygons(poly_list, warn=warn) # make sure all polygons are closed
    poly_list = remove_duplicate_polygons(poly_list, warn=warn) # remove duplicates
    poly_list = normalize_polygon_orientation(poly_list, warn=warn) # orient all polygons counter-clockwise
    poly_list = choose_scan_side(poly_list)
    return list_to_nparray_safe(poly_list)
    
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
    
#####################################
### Operations on multiple layers ###
#####################################
    
def import_multiple_layers(dxf, layers, warn=True):
    """ Import multiple layers from dxf drawing into a dictionary.
    
        Args:
            dxf (dxfgrabber object): obejct representing the dxf drawing
            layers (list): str or list of string containing names of layers to import
            
        Kwargs: 
            warn (bool): print warnings
            
        Returns:
            dict: dictionary containing layer names as keys and polygon lists
                  as values """
                  
    if type(layers)==type(''):
        layers = [layers]
    elif type(layers)==type([]):
        pass
    else:
        print("Layers should be a string or list of strings")
        
    all_layers = get_layer_names(dxf) # get list of layers contained in dxf
    layers = [l.upper().replace (" ", "_") for l in layers] # fix string formatting
    
    poly_dict = {}
    for l in layers:
        if l in all_layers:
            poly_dict[l] = get_vertices(dxf, l, warn=warn)
        else:
            if warn:
                print('LAYER: {0} NOT CONTAINED IN DXF'.format(l))
                
    return poly_dict
    
def vstack_all_vertices(poly_dict):
    """ All vertices in the layers contained in poly_dict are stacked to create one long
        list of x,y coordinates.
        
        Args:
            poly_dict (dict): dictionary containing layer names as keys and polygon lists
                                as values
        
        Returns: 
            np.ndarray: list of x,y coordinates """
            
    verts = np.zeros((sum([len(v) for key, val in poly_dict.items() for v in val]),2))
    m = 0
    for key, val in poly_dict.items():
        n = m+sum([len(v) for v in val])
        verts[m:n] = np.vstack(val)
        m = n
    return verts
    
    
def bounding_box(poly_dict, origin='ignore'):
    """ Find bounding box and proper coordinates 

        Args:
            origin -- where the (0,0) coordinate should be located 
        
        Returns:
            ll (np.array): x,y coordiates of lower left corner of drawing after shift
            ur (np.array): x,y coordiates of upper right corner of drawing after shift
            center (np.array): x,y coordinates of center point after shift
            bsize (float): size of smallest bounding box (nearest micron)
            shift (np.array): all x,y coordinates must be shifted by this vector """


    verts = vstack_all_vertices(poly_dict)

    xmin = verts[:,0].min()
    xmax = verts[:,0].max()
    ymin = verts[:,1].min()
    ymax = verts[:,1].max()

    ll = np.array([xmin, ymin])
    ur = np.array([xmax, ymax])
    center = np.array([xmin+xmax, ymin+ymax])/2.0
    bsize = np.ceil(max(xmax-xmin, ymax-ymin))

    if origin=='lower':
        shift = (-1)*(center-bsize/2.0)
        return ll+shift, ur+shift, center+shift, bsize, shift
    elif origin=='center':
        shift = (-1)*center
        return ll+shift, ur+shift, center+shift, bsize, shift
    else:
        shift = np.array([0,0])
        return ll, ur, center, bsize, shift
        
##############################################
### Calculations to determine dose scaling ###
##############################################

# the idea here was to write a function that scales the dose 
# according to the width of the polygon, 
# which is estimated from the vertex positions
# see Nik's thesis for some details

# def geometry_to_dose(verts, doseMin, doseMax):
#     """ calculate approximate width of polygon. scale ebeam dose accordingly. """
#     
#     widths = 2*polyUtility(
#                         polyverts, polyArea)/polyUtility(
#                                             polyverts, polyPerimeter)

# def geometry_to_dose(verts, doseMin, doseMax):
#     """ takes an array of polygon vertices. returns and array of dose values calculated
#         by dividing perimeter by area and using some empirical evidence to scale to the 
#         proper range of doses. the total doses are scaled and limited by doseMin and doseMax. """
# 
#     data = polyUtility(verts, polyPerimeter)/abs(polyUtility(verts, polyArea))
#     
#     #different size scales for different writefields
#     if get_writefield(verts) == 1000:
#         pMin = 0.04; pMax = 1.1
#     else:
#         pMin = 1.0; pMax = 7.0
#     
#     #  split up range into 20 steps, round steps to nearest 10
#     resolution =max(np.floor((doseMax-doseMin)/200)*10, 1.0)
#     
#     m = (doseMax-doseMin)/(pMax-pMin)
#     b = doseMax - m*pMax
#     
#     #  clip data to within limits to make sure nothing gets a totally ridiculous dose
#     #  round to nearest multiple of 'resolution' because this method can't be very accurate
#     return np.clip(np.round(np.array([m*x + b for x in data])/resolution)*resolution, doseMin, doseMax)

######################################################
### Functions to define a write order for polygons ###
######################################################

def typewriter_sort(com, n = SMALLEST_SCALE):
    """ Sort polygons left to right, top to bottom, based on the location of
        their center of mass.
        
        Args:
            com (array): 2D numpy array of the center of mass coordinates for
                each polygon
                
        Kwargs: 
            n (float): grid in microns to round COM coordinates to
                
        Returns:
            array: numpy array of indices that sort com """

    X = -np.floor(com/n)[:,0]*n
    Y = -np.floor(com/n)[:,1]*n
    return np.lexsort((X, Y))[::-1]
    
def walking_sort(com_list, starting_point = None):
    if not starting_point:
        # start in the lower left corner
        # need some code for that
        pass
    else:
        start = starting_point
        
    sorted_idx = np.empty(len(com_list), dtype=np.int)
    sorted_idx[:] = np.nan

    for i in range(len(com_list)):
        # find index of com closest to start
        for j in np.argsort(pnt_pnt_dist(com_list, start)):
                if j not in sorted_idx:
                    sorted_idx[i] = j
                    start = com[j]
                    break
                else:
                    continue
    return sorted_idx

#####################################
### ASC output for Raith software ###
#####################################

def verts_block_asc(verts):
    """ verticies to block of text """
    s = ''
    for v in verts:
        s += '{0:.4f} {1:.4f} \n'.format(v[0], v[1])
        
    # some versions of dxf give different results here....
    # add the first vertex to the end of the list, unless it is 
    # already there
    if '{0:.4f} {1:.4f} \n'.format(verts[0][0], verts[0][1]) == \
        '{0:.4f} {1:.4f} \n'.format(verts[-1][0], verts[-1][1]):
        return s
    else:
        return s + '{0:.4f} {1:.4f} \n'.format(verts[0][0], verts[0][1])

def write_layer_asc(f, poly_list, dose, layer, setDose=None):
    """ Writes all vertices in a layer to an ASCII file.

        Args: f: open file object
              verts (array): numpy array of vertex lists
              dose: array of doses or single dose
              layer (int): ASCII layer number
        Kwargs:
              setDose: doses will be scaled to a % of this value
                       if setDose is None doses will be written as
                       passed to this function.

        Returns: None """
      
    if isinstance(dose, np.ndarray):
        pass
    elif isinstance(dose, (list, tuple)):
        dose = np.array(dose)
    elif type(dose) in (int, float):
        dose = np.ones(len(poly_list), dtype=np.float)*dose
    else:
        raise TypeError('Unknown type for dose.')
              
    for i in range(len(poly_list)):
        if setDose:
            d = dose[i]/setDose*100.0
        else:
            d = dose[i]
        f.write('1 {0:.3f} {1:d} \n'.format(d, layer))
        f.write(verts_block_asc(poly_list[i]) + '# \n')

####################################
### DC2 output for NPGS software ###
####################################

def write_header_dc2(f, ll, ur, layers):
    """ Write header for dc2 file. 
    
        Args:
            f (file object): file in which the header will be written 
            ll (array): x,y coordinates of lower left boundary
            ur (array): x,y coordinates of upper right boundary
            layers (str or list): string or list of layers to be included """
    header = '{0:.4f} {1:.4f} {2:.4f} {3:.4f} 0 -0.0000 0.0000\r\n'.format(
                ll[0]*8, ll[1]*8, (ur[0]-ll[0])*8, (ur[1]-ll[1])*8)
    header +=  ('42 20 0 0 0 0\r\n'
                '8.000000\r\n'
                '8.000000, 0.800000\r\n'
                '8.000000\r\n'
                '3\r\n'
                '16.000000\r\n'
                '0.000000\r\n'
                '0.000000\r\n'
                '1.000000\r\n'
                '1\r\n'
                '1\r\n'
                '1\r\n'
                'SIMPLEX2.VFN\r\n'
                '0.00000000 0.00000000 0.00000000 0.00000000\r\n'
                '0.00000000 0.00000000 0.00000000 0.00000000\r\n'
                '0.00000000 0.00000000 0.00000000 0.00000000\r\n'
                '0.00000000 0.00000000 0.00000000 0.00000000\r\n'
                '1 0 0 0 0 0 0 0\r\n'
                '0.00000000 0.00000000 0.00000000 0.00000000\r\n'
                '0.00000000 0.00000000 0.00000000 0.00000000\r\n'
                '; DesignCAD Drawing Comments /w \';\' as 1st char.\r\n')
    
    header += '23 {0} 0 0 0 0 \r\n'.format(len(layers)+1)
    header += 'DO NOT USE\r\n' # don't use layer 0
    for l in layers: # print other names
        header+='{0}\r\n'.format(l)
    
    f.write(header)
        
def verts_block_dc2(vert, color):
    """ Create block of text that defines each closed polygon. This assumes 
        that all objects have been converted to closed polygons. 
        
        Args:
            vert (array): array defining x,y coordinates for each vertex of a polygon
            color (array): 1D array of length 3 defining the color (dose) for this polygon 
            
        Returns
            str: formatted string representing the polygon in DC2 format """
            
    # format for each polygon:
    # (type=line) (num of points in polygon) (hatching) (line width) (line type) (13) (0) (1) (R G B) (0) (1)
    # (x) (y) 0
    
    # check that color has the correct chape
    if color.shape != (3,):
        raise TypeError('color is not an RGB array')
    
    line_hatch = 0.1 # 100nm hatching
    line_width = 0 # line width=0 for closed polygons
    line_type = 1 # 0 solid, 1 dashed (solid for wide lines, 0 for closed/filled polygons)
    block = '1 {0:d} {1:.4f} {2:.4f} {3:d} 13 0 1 0 {4:d} {5:d} {6:d} 0 1\r\n'.format(
            len(vert), line_hatch*8, line_width*8, line_type, color[0], color[1], color[2])
    for v in vert:
        block += '{0:.4f} {1:.4f} 0\r\n'.format(v[0]*8, v[1]*8)
    return block
    
def write_layer_dc2(f, layer_num, poly_list, colors):
    """ Writes all vertices in a layer to an DC2 file.

        Args: 
            f (file object): file in which the header will be written 
            layer_num (int): number of layer to be written (these should be sequential)
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons
            colors (array): 1 or 2D array giving RGB values for polygons

        Returns: None """ 
    
    if colors.shape == (len(poly_list),3):
        pass 
    elif colors.shape == (3,):
        # a single RGB value was given
        # assume it can be applied to all polygons
        colors = np.ones((len(poly_list),3), dtype=np.int)*colors
    else:
        # colors is not a valid shape
        raise TypeError('colors is not an acceptable shape for an RGB array.')
    
    layer_txt = '21 {0} 0 0 0 0\r\n'.format(layer_num)
    for vert, color in zip(poly_list, colors):
        layer_txt += verts_block_dc2(vert, color)
    f.write(layer_txt)
    
def write_alignment_layers_dc2(f, poly_list, layer_names):
    """ Write layer for manual marker scans on the NPGS software. Each scan is defined
        by a square which much be in its own layer. Along with the squares each layer must 
        contain some lines of 0 width that mark the center point. 
    
        Args: 
            f (file object): file in which the header will be written 
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons
            layer_names (list): list of layer names as strings. one layer name for each alignment marker scan
            
        Returns:
            None """
            
    color = np.array([255,255,255]) # executive decision: alignment marks are white
    
    for j, v, al in zip(range(len(layer_names)),poly_list,layer_names):
    
        layer_num = j+1
        layer_txt = '21 {0} 0 0 0 0\r\n'.format(layer_num) # identify layer
        layer_txt += verts_block_dc2(v, color) # add block for marker scan

        # define and add lines for cross inside box
        com = polyCOM(v) # find center of box
        side = np.sqrt(polyArea(v)) # length of one side of the box (or close enough)
        line0 = np.array([com-np.array([side/4.0,0]), # horizontal line
                          com+np.array([side/4.0,0])])
        line1 = np.array([com-np.array([0,side/4.0]), # vertical line
                          com+np.array([0,side/4.0])])
        cross = [line0, line1]

        # write 
        line_hatch = 0.1 # not sure what I need here
        line_width = 0 # line width=0 as required
        line_type = 0 # 0 solid (solid for wide lines)

        for line in cross:
            layer_txt += '1 {0:d} {1:.4f} {2:.4f} {3:d} 13 0 1 0 {4:d} {5:d} {6:d} 0 1\r\n'.format(
                      len(line), line_hatch*8, line_width*8, line_type, color[0], color[1], color[2])
            for point in line:
                layer_txt += '{0:.4f} {1:.4f} 0\r\n'.format(point[0]*8, point[1]*8)
        f.write(layer_txt)
    
def save_alignment_info(file, layername, poly_list):
    """ Saves a text file with information about the manual alignment mark scans. NPGS
        needs to know the coordinates and vectors between them. A txt file is created
        containing this information. 
        
        Args:
            file (str): name of dxf file containing original drawing
            poly_list (list): list of 2D numpy arrays that contain x,y vertices defining polygons """
            
    with open(file[:-4]+'_{0}.txt'.format(layername), 'w') as f:
        com = polyUtility(poly_list, polyCOM)
        f.write('MARKER LOCATIONS: \r\n')
        f.write(str(np.round(com*1000)/1000))
        f.write('\r\nVECTOR FROM MARKER 0: \r\n')
        f.write(str(np.round((com-com[0])*1000)/1000))

##########################
### Plotting functions ###
##########################

def plot_layers(ax, filename, layers, extent=None):
    """ Plot the layers from filename on ax with bounds given by size. 
    
        Args:
            ax (matplotlib.axes): axis on which the plot will appear
            filename (dxf filename): name of file containing the drawing
            layers (list): str or list of strings containing layer names
            extent (list): [xmin, xmax, ymin, ymax] """
       
    d = Layers(f, layers)

    d.plot(ax, extent = extent)
        
#############################################################################
### Class to deal with importing/editing/exporting a few layers at a time ###
#############################################################################

class Layers:
    """ class used to process layers for ebeam writing """
    
    def __init__(self, filename, layers):
    
        self.filename = filename
        self.dxf = ezdxf.readfile(filename)
        
        if type(layers)==type(''):
            layers = [layers]
        elif type(layers)==type([]):
            pass
        else:
            print("Layers should be a string or list of strings")

        self.layers = [l.upper().replace (" ", "_") for l in layers] # fix strings

        all_layers = get_layer_names(self.dxf)
        for l in layers:
            if l not in all_layers:
                raise KeyError('{0} IS NOT A LAYERNAME'.format(l))
                        
        self.poly_dict = import_multiple_layers(self.dxf, self.layers, warn=True)
        
    def estimate_writetime(self, dose, current):
        """ Estimate write time for given layers.
        
        Args:
            dose (float) -- dose in uC/cm^2
            current (float) -- beam current in pA
                            
        Returns: 
            float: time to write patter in minutes """
        
        for layer in self.layers:
            if 'ALIGN' in layer:
                continue
            verts = self.poly_dict[layer]
            total_area = polyUtility(verts, polyArea).sum() # areas are in um^2
        
            print('Time to write {0}: {1:.1f} min'.format(layer, (dose*(total_area*1e-8)/(current*1e-6))/60.0))
    
    def find_writefield_centers(self):
        """ Locate the center of the writefield for given layers.
            Results are given using the coordinates of the original drawing
            unless offset != (0,0). In which case, center is (coordinates
            of the drawing) - offset.
        
        Args:
            filename (str): str containing filename of dxf file
            layer (str) -- string or list of layer names """

        # find center of all layers together
        *junk, center = bounding_box(self.poly_dict, origin='center')
        self.writefield_centers = {'__ALL__': center}
        
        # find centers of individual layers
        for key, val in self.poly_dict.items():
            *junk, center = bounding_box({key: val}, origin='center')
            self.writefield_centers[key] = center

        for key, val in self.writefield_centers.items():
            print('{0}: {1:.1f}, {2:.1f}'.format(key, val[0], val[1]))
        
    def plot(self, ax, extent=None, layers=None):
        """ Plot the layers from filename on ax with bounds given by size. 
    
            Args:
                ax (matplotlib.axes): axis on which the plot will appear
                filename (dxf filename): name of file containing the drawing
                layers (list): str or list of strings containing layer names
                extent (list): [xmin, xmax, ymin, ymax] """

        if layers is None:
            working_dict = self.poly_dict
        else:
            if type(layers)==type(''):
                layers = [layers]
            elif type(layers)==type([]):
                pass
            else:
                print("Layers should be a string or list of strings")
            working_dict = {k: self.poly_dict[k] for k in layers}

        ll, ur, center, bsize, shift = bounding_box(working_dict, origin='center')

        pmin = np.floor(ll.min()/10)*10
        pmax = np.ceil(ur.max()/10)*10
    
        colors = itertools.cycle([plt.cm.Accent(i) for i in np.linspace(0, 1, 6)])
        for key, val in working_dict.items():            
            verts = np.array([v+shift for v in val])
        
            if 'ALIGN' in key:
                alpha = 0.5
            else:
                alpha = 1.0
        
            polycol = PolyCollection(verts, facecolor=next(colors))
            polycol.set_alpha(alpha)
            ax.add_collection(polycol)

            if extent:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
            else:
                ax.set_xlim(pmin, pmax)
                ax.set_ylim(pmin, pmax)
        
            ax.grid('on')
            
    def save_as_dxf(self, origin='ignore'):
        """ Load dxf file(s), convert all objects to polygons, 
            order elements by location, export dxf file.
        
            Args:
                file (list): str (or list of str) containing filenames of dxf file(s)
                layers (list) -- list of strings, layers to be included in .dc2 file(s)
            
            Kwargs:
                origin (str): where the (0,0) coordinate should be located. 'lower' -- 
                                lower left corner, 'center' -- center of drawing, 'ignore' --
                                do not shift drawing
                            
            Returns: 
                None """
    
        colors = np.arange(0,len(self.layers))+1

        # create dxf drawing objects
        dwg = ezdxf.new('AC1015')
        msp = dwg.modelspace()
    
        ll, ur, center, bsize, shift = bounding_box(self.poly_dict, origin=origin)

        # loop over layers
        for i, l, c in zip(range(len(self.layers)), self.layers, colors):

            # define new layer in dxf
            dwg.layers.new(name=l, dxfattribs={'color': c})

            # get and sort vertices
            verts = self.poly_dict[l]
            verts = np.array([v+shift for v in verts]) 
#             com = polyUtility(verts, polyCOM)
#             ind_sorted = sort_by_position(com)
#             verts = verts[ind_sorted]
        
            for v in verts:
                msp.add_lwpolyline(v, dxfattribs={'layer':l})

        dwg.saveas(file[:-4]+'_edited.dxf')
            
    def process_files_for_npgs(self, layers = None, origin='ignore'):
        # fix this
        """ order elements by location, export DC2 files
        
            Args:
                file (list): str (or list of str) containing filenames of dxf file(s)
                layers (list) -- list of strings, layers to be included in .dc2 file(s)
            
            Kwargs:
                pos_sort_n (float) = distance in microns to round polygon center points to
                                        when sorting by location
                origin (str): where the (0,0) coordinate should be located. 'lower' -- 
                                lower left corner, 'center' -- center of drawing, 'ignore' --
                                do not shift drawing
                            
            Returns: 
                None """

    
        if layers is None:
            working_dict = self.poly_dict
        else:
            if type(layers)==type(''):
                layers = [layers]
            elif type(layers)==type([]):
                pass
            else:
                print("Layers should be a string or list of strings")
            working_dict = {k: self.poly_dict[k] for k in layers}
    
        colors = (plt.cm.Accent(np.linspace(0,1,len(layers)))[:,:-1]*256).astype(dtype='uint8')   
        
        ll, ur, center, bsize, shift = bounding_box(working_dict, origin=origin)
    
        id = '-'.join([l for l in layers if 'ALIGN' not in l])

        if id != '':
            f = open(self.filename[:-4]+'_{0}.dc2'.format(id), 'w')
            write_header_dc2(f, ll, ur, self.layers)

        for i, l, c in zip(range(len(layers)), layers, colors):
        
            # get and sort polygons for this layer
            verts = working_dict[l]
            verts = np.array([v+shift for v in verts]) 
#             com = polyUtility(verts, polyCOM)
#             ind_sorted = sort_by_position(com)
#             verts = verts[ind_sorted]
            
            if 'ALIGN' in l:            
                # open file, write header
                af = open(self.filename[:-4]+'_{0}.dc2'.format(l), 'w')
                align_layer_names = ['MARKER{0:d}'.format(i) for i in range(len(verts))]
                write_header_dc2(af, ll, ur, align_layer_names) # write alignment file header
                write_alignment_layers_dc2(af, verts, align_layer_names)
                print('alignment output: ' + self.filename[:-4]+'_{0}.dc2'.format(l) + 
                                          ', ' + self.filename[:-4]+'_{0}.txt'.format(l))
                af.close()
            
                # record vectors pointing from alignment mark 0 to others
                save_alignment_info(self.filename, l, verts)
            
            else:
                write_layer_dc2(f, i+1, verts, c)
                
        if id != '':
            print('pattern output: ' + self.filename[:-4]+'_{0}.dc2'.format(id))
            f.close()
            
#     def process_files_for_raith(self):
#         do not currently have a Raith system to test
#         should look very similar to process_files_for_npgs
