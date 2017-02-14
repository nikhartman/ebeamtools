""" The module name is a bit misleading. this is not a real proximity correction. rather,
    it uses the width of lines in order to scale the dose each polygon in a pattern gets.
    
    Maybe someday this will turn into a real homemade proximity correction algorithm. """
    
    
from ebeamtools.polygons import polyArea, polyPerimeter, polyUtility
from matplotlib.colors import LinearSegmentedColormap


# first create a custom colormap
# this colormap is heplful in that it is linear in the green channel
# that channel will be used to hold dose information
# while still making sensible looking plots

# adjust this so red and blue extend up further
# the lower half of this colormap is kind of useless
# the upper half could use a lot more contrast

cdict = {'red':   ((0.0, 0.8, 0.8),
                   (0.66, 0.8, 0.8),
                   (1.0, 0.5, 0.5)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.8, 0.8),
                   (0.9, 0.8, 0.8),
                   (1.0, 0.5, 0.5))}

lin_green = LinearSegmentedColormap('LinearGreen', cdict)

def get_widths(verts):
    """ return the approximate width of all polygons defined in verts. """
    
    return 2*polyUtility(verts, polyArea)/polyUtility(verts, polyPerimeter)
    
def scale_by_width(verts, min_width, max_width, min_dose, max_dose):
    """ scale dose by inverse polygon width. widths are in microns. 
        dose values are in percentage of full dose. 
        
        returns a color for each polygon. the percent dose is given by the
        green channel. """
        
    vals = 1.0/get_widths(verts) # dose will be scale by the inverse width

    min_val = 1.0/max_width
    max_val = 1.0/min_width
    
    m = (max_dose-min_dose)/(max_val-min_val)
    b = max_dose - m*max_val
    
    scaling = np.clip(np.round(np.array([m*v + b for v in vals])/0.01)*0.01, 
                            min_dose, max_dose)
    
    return lin_green(scaling)
    
# add dose scaling into drawing.py