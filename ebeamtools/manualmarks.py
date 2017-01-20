# a python implementation of manual markers alignment

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy
from skimage import io, img_as_float
from skimage import transform as tf
import os
from datetime import datetime

# sources:
# http://stackoverflow.com/questions/21654008/matplotlib-drag-overlapping-points-interactively
# http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# http://stackoverflow.com/questions/28758079
# http://matplotlib.org/users/event_handling.html

def order_points(points):
    """ order the points by angle phi measured from x-axis 
    
        returns: points in counterclockwise order """
    shifted = points - np.mean(points, axis=0) # center points around origin
    
    phi = np.arctan2(shifted[:,1],shifted[:,0])
    sumphi = abs(sum(phi)) #can be used to determine different square types
    
    points = points[np.argsort(phi)] # sort original counter clockwise
    
    if abs(sumphi-np.pi) < sumphi:
        # square vertices on axis
        if (points[-1,1]>points[0,1]) & (points[-1,0]<points[0,0]):
            return np.roll(points, 1, axis=0)
        else:  
            return points
    else:
        # square edges parallel to axis
        return points
        
def warp_transform(im, pnts, src, val=0.0):
    """ warps image according to pnts and src 
        
        im: image to warp
        pnts: location of points on im
        scr: locations pnts will be mapped to
        val: fill value for outside image edges """
        
    tform3 = tf.ProjectiveTransform()
    tform3.estimate(src, pnts)
    return tf.warp(im, tform3, mode='constant', cval=val, output_shape=im.shape)
        
class DraggableMarkers(object):
    def __init__(self, artists, template, tolerance=5):
        """ 
            artists (list) -- matplotlib.artist polygons
            template ((N,2) array) -- verticies defining the shape of the markers
                                      first point should be the center point """
                                      
        for artist in artists:
            artist.set_picker(tolerance)
        self.artists = artists
        self.template = template
        self.currently_dragging = False
        self.current_artist = None
        self.offset = (0, 0)

        for canvas in set(artist.figure.canvas for artist in self.artists):
            canvas.mpl_connect('button_press_event', self.on_press)
            canvas.mpl_connect('button_release_event', self.on_release)
            canvas.mpl_connect('pick_event', self.on_pick)
            canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        self.currently_dragging = True

    def on_release(self, event):
        self.currently_dragging = False
        self.current_artist = None

    def on_pick(self, event):
        # calculate how far from the center the mouse click was
    
        if self.current_artist is None:
            self.current_artist = event.artist
            x0, y0 = event.artist.get_xy()[0]
            x1, y1 = event.mouseevent.xdata, event.mouseevent.ydata
            
            self.offset = (x0 - x1), (y0 - y1) 

    def on_motion(self, event):
        if not self.currently_dragging:
            return
        if self.current_artist is None:
            return
        
        try:    
            dx, dy = self.offset # mouse click offset 
            new_center = (event.xdata + dx, event.ydata + dy)
            self.current_artist.set_xy(self.template
                                        + (event.xdata + dx, event.ydata + dy)) # actual move
        except TypeError as e:
            pass
        
        self.current_artist.figure.canvas.draw() # redraw the figure

def align_markers(im_file, real_pos):

    """ Do the actual alignment. Assumes CAD image import is set to 1unit = 1mm """
    
    im = mpl.image.imread(im_file) # import image to np array
    y_px, x_px = im.shape[0:2] # image extents
    
    real_pos = order_points(real_pos) # order alignment markers by phi
    real_center = np.mean(real_pos, axis=0) # get center position in drawing units
    
    # normalize marker positions
    # center at zero
    # scale to largest distance from center (* sqrt(2))
    norm_pos = real_pos - np.mean(real_pos, axis=0)
    norm_pos *= (np.sqrt(2)/np.sqrt(np.sum((norm_pos**2), axis = 1)).max())

    fig, ax = plt.subplots(1,1, figsize = (10, 10*y_px/x_px))
    
    if len(im.shape) == 3:
        ax.imshow(im)
    else:
        ax.imshow(im, cmap = plt.cm.gray)
    
    # guess where to position markers on image
    start_pos = 0.2*max(x_px, y_px)*norm_pos
    start_pos += 0.5*np.array([x_px, y_px])

    # create alignment crosses
    cross_scale = max(x_px,y_px)*0.025
    cross_verts = cross_scale*np.array([[0,0],[1,0],[1,-1],[0,-1],[0,1],[-1,1],[-1,0],[0,0]])
    
    # add crosses to image using DraggableMarkers objects
    crosses = []
    for center in start_pos:
        c = patches.Polygon(cross_verts+center, lw=1, ec='r', fc='none')
        crosses.append(c)
        ax.add_patch(c)

    dr = DraggableMarkers(crosses, cross_verts)
    plt.show() # script continues executing after figure is closed
    
    # get marker positions and center as located by user
    found_pos = np.array([c.get_xy()[0] for c in crosses]) 
    found_pos = order_points(found_pos)
    found_center = np.mean(found_pos, axis=0) # rotate/scale about this point
    
    # calculate the correct marker locations
    final_pos = norm_pos*(np.sqrt(np.sum(((found_pos-found_center)**2), axis = 1)).max()/np.sqrt(2))
    final_pos += found_center
    
    im_warped = warp_transform(im,found_pos,final_pos,val=1.0)
        
    # get the proper resolution for the output
    pixel_per_mm = np.sqrt(np.sum((final_pos[0]-final_pos[1])**2))/ \
                        np.sqrt(np.sum((real_pos[0]-real_pos[1])**2))
    
    scaling = 1.0   
    
    # find where the lower left corner of the image will be located in the drawing
    insert_pos = (real_center - 
                    np.array([found_center[0],y_px-found_center[1]])/pixel_per_mm)
    
    return scaling, insert_pos, im_warped

def save_image(output_file, im, pixel_per_mm):
    """ save image as png """
    mpl.image.imsave(output_file.format(datetime.now().strftime("%Y%m%d-%H%M%S")), 
                            im_warped, dpi = pixel_per_mm*25.4)
                            
def align_from_file(im_file, dxf_file, marker_layer):

    align_markers(im_file, real_pos)