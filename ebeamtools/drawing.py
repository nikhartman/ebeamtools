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
from ebeamtools import sorting
from ebeamtools import proximity
import ebeamtools.polygons as pg

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

###################################
### Import polygons from layers ###
###################################

def import_polyline(ent, split_lines = True, warn=True):
    """ A fucntion to import polygon entities from a drawing. Remove z coordinates,
        convert list to numpy array, remove any duplicate vertices.

        Args:
            ent (ezdxf.entity): object representing the polygon

        Returns
            np.ndarray: list of x,y coordinates for polygon vertices """



    verts = np.zeros((ent.__len__(), 5))
    for i, v in enumerate(ent.vertices()):
        verts[i,0:2] = v.dxf.location[0:2]
        verts[i,2:] = (v.dxf.start_width, v.dxf.end_width, v.dxf.bulge)

    closed = ent.is_closed

    check_width = (np.count_nonzero(abs(verts[:,2:4].flatten()-verts[0,2]) > SMALLEST_SCALE))
    if(check_width==0):
        # if all of the elements in columns 2+3 of verts are equal
        # this is a line of constant width (could be width=0)
        const_width = True
        width = verts[0,2]
    elif(check_width > 0):
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
        return [pg.remove_duplicate_vertices(verts, SMALLEST_SCALE)]
    elif (width<SMALLEST_SCALE and not closed):
        # most likely an unclosed polygon
        # add it and it will be fixed later.
        return [pg.remove_duplicate_vertices(verts, SMALLEST_SCALE)]
    elif (width>SMALLEST_SCALE and not closed): # lines with constant width
        verts = pg.line2poly_const(verts, width)
        if verts is not None:
            if split_lines:
                return pg.split_line2poly(verts)
            else:
                return [verts]

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
        return [pg.remove_duplicate_vertices(verts, SMALLEST_SCALE)]
    elif (width<SMALLEST_SCALE and not closed):
        # most likely an unclosed polygon
        # add it and it will be fixed later.
        return [pg.remove_duplicate_vertices(verts, SMALLEST_SCALE)]
    elif (width>SMALLEST_SCALE and not closed): # lines with constant width
        verts = pg.line2poly_const(pg.remove_duplicate_vertices(verts, SMALLEST_SCALE), width)
        if verts is not None:
            if split_lines:
                return pg.split_line2poly(verts)
            else:
                return [verts]

def get_vertices(dxf, layer, split_lines = True, warn=True):
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
            if ent.dxftype() == 'POLYLINE':
                poly_list += import_polyline(ent, split_lines=split_lines)
                pass
            elif ent.dxftype() == 'LWPOLYLINE':
                try:
                    poly_list += import_lwpolyline(ent, split_lines=split_lines)
                except Exception as e:
                    print(e)
            # add additional dxftypes here

            else:
                if warn:
                    print('NOT A KNOWN TYPE ({0}) -- LAYER: {1}'.format(ent.dxftype(), layer))

    poly_list = [p for p in poly_list if len(p)>1]

    poly_list = pg.close_all_polygons(poly_list, SMALLEST_SCALE, warn=warn) # make sure all polygons are closed
    poly_list = pg.remove_duplicate_polygons(poly_list, SMALLEST_SCALE, warn=warn) # remove duplicates
    poly_list = pg.normalize_polygon_orientation(poly_list, warn=warn) # orient all polygons counter-clockwise
    poly_list = pg.choose_scan_side(poly_list, SMALLEST_SCALE)
    return pg.list_to_nparray_safe(poly_list)

#####################################
### Operations on multiple layers ###
#####################################

def import_multiple_layers(dxf, layers, split_lines=True, warn=True):
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
            poly_dict[l] = get_vertices(dxf, l, split_lines=split_lines, warn=warn)
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
        # a color was given for each polygon
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
        com = pg.polyCOM(v) # find center of box
        side = np.sqrt(pg.polyArea(v)) # length of one side of the box (or close enough)
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
        com = pg.polyUtility(poly_list, pg.polyCOM)
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

    def __init__(self, filename, layers, split_lines = True):

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
        for l in self.layers:
            if l not in all_layers:
                raise KeyError('{0} IS NOT A LAYERNAME'.format(l))

        self.poly_dict = import_multiple_layers(self.dxf, self.layers, split_lines=split_lines, warn=True)

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
            total_area = pg.polyUtility(verts, pg.polyArea).sum() # areas are in um^2

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
        self.writefield_centers = {'__ALL__': -center}

        # find centers of individual layers
        for key, val in self.poly_dict.items():
            *junk, center = bounding_box({key: val}, origin='center')
            self.writefield_centers[key] = -center

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
            ax.set_axisbelow(True)

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

            for v in verts:
                msp.add_lwpolyline(v, dxfattribs={'layer':l})

        dwg.saveas(self.filename[:-4]+'_edited.dxf')

    def process_files_for_npgs(self, layers = None, origin='ignore',
                                sort_type = None, sort_timeout = 0,
                                scaling_method=None, scaling_params = [], dose_params = []):
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
                sorting (str): 'tsp' for travelling salesman sorting
                               'typewriter' for left-right up-down sorting

            Returns:
                None """


        if layers is None:
            working_dict = self.poly_dict
            layers = self.layers
        else:
            if type(layers)==type(''):
                layers = [layers]
            elif type(layers)==type([]):
                pass
            else:
                print("Layers should be a string or list of strings")
            working_dict = {k: self.poly_dict[k] for k in layers}

        # create a unique set of colors for layers
        # this will be overwritten if dose scaling is used
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

            if 'ALIGN' in l:
                # always sort alignment marks like a typewriter
                sort_idx = sorting.typewriter_sort(verts, SMALLEST_SCALE)
                verts = verts[sort_idx]

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
                # sort vertices
                if sort_type is None:
                    sort_idx = np.arange(len(verts))
                elif sort_type == 'tsp':
                    sort_idx = sorting.travelling_ebeam_sort(verts, timeout = sort_timeout)
                elif sort_type == 'typewriter':
                    sort_idx = sorting.typewriter_sort(verts, SMALLEST_SCALE)
                else:
                    raise TypeError('Invalid sorting type specified.')
                verts = verts[sort_idx]

                if isinstance(scaling_method, str):
                    if scaling_method is 'width':
                        if len(scaling_params)==2 and len(dose_params)==2:
                            # overwrite layer color
                            c = proximity.scale_by_width(verts, *scaling_params, *dose_params)
                        else:
                            raise ValueError('scaling_params and dose_params must be length 2')
                    elif scaling_method is 'distance':

                        # overwrite layer color with list
                        if len(dose_params)==2:
                            # overwrite layer color
                            c = proximity.scale_by_center_dist(verts,*dose_params)
                        else:
                            raise ValueError('dose_params must be length 2')
                    else:
                        raise TypeError('Choose a valid scaling method')

                write_layer_dc2(f, i+1, verts, (255*c).astype('uint8'))

        if id != '':
            print('pattern output: ' + self.filename[:-4]+'_{0}.dc2'.format(id))
            f.close()

#     def process_files_for_raith(self):
#         do not currently have a Raith system to test
#         should look very similar to process_files_for_npgs
