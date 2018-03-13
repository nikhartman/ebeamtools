""" The module name is a bit misleading. This is not a real proximity correction. Rather,
    it uses the width of lines in order to scale the dose each polygon gets.

    Maybe someday this will turn into a real homemade proximity correction algorithm. """

import numpy as np
from ebeamtools.polygons import polyArea, polyPerimeter, polyUtility, polyCOM
from matplotlib.colors import LinearSegmentedColormap

# first create a custom colormap
# this colormap is heplful in that it is linear in the green channel
# that channel will be used to hold dose information
# while still making sensible looking plots
# it is a little ugly, though

cdict = {'red':   ((0.0, 0.8, 0.8),
                   (0.4, 0.5, 0.5),
                   (1.0, 0.5, 0.5)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.8, 0.8),
                   (0.6, 0.8, 0.8),
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

    m = (1.0-(min_dose/max_dose))/(max_val-min_val)
    b = 1.0 - m*max_val

    scaling = np.clip(np.round(np.array([m*v + b for v in vals])/0.01)*0.01, min_dose/max_dose, 1.0)

    return lin_green(scaling)[:,0:3]

def get_center_dist(verts):
	""" Return the distance from the center of the
		pattern for each polygon."""

	center_vector = polyUtility(verts,polyCOM)
	center = np.zeros(len(center_vector))
	for i in range(len(center_vector)):
		center[i] = np.hypot(center_vector[i,0],center_vector[i,1])

	return np.around(center-np.min(center),decimals=3)

def get_low_green(min_dose,max_dose):
  """ Calculate the minimum RGB value
    for green (G), max is always 255."""

  if max_dose >= min_dose:
      alpha = max_dose/min_dose
  else:
      raise TypeError('max_dose must be bigger or equal to min_dose')

  return int(255/alpha)

def scale_by_center_dist(verts,min_dose,max_dose):
	""" Scale dose by distance from center of layer.
		Useful for big arrays of same size objects.
		Dose values are in percentage of the full dose.

		Returns a color for each polygon. Uses the green
		channel."""

	vals = get_center_dist(verts)
	low_green = get_low_green(min_dose,max_dose)

	m = 1.0/(np.max(vals)-np.min(vals))
	scaling = (np.clip(np.round(np.array([m*v for v in vals])/0.01)*0.01,
                        0.0, 1.0)*(255.0-low_green)+low_green)/255.0

	return lin_green(scaling)[:,0:3]
