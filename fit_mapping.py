#!/usr/bin/python
# Copyright 2011 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pylab import *
from fit_cone import *
from  scipy.optimize import leastsq
import sys
import itertools

#from sim_stereo import distance_from_disparity

import mpl_toolkits.mplot3d.axes3d as p3

from color_block import gucci_dict

import pdb


def fitfunc(u, M):
  Ned = (M.shape[1]-3)/2
  R = zeros(Ned+3)
  D = dot(u,M)**2
  R[:Ned] = D[0:-3:2]+D[1:-3:2]
  R[-3:] = D[-3:]
  return R

def devfunc(u, M):
  return 2*dot(u,M)

errfunc = lambda u, M, d_x: fitfunc(u, M) - d_x


def distance_from_disparity(d):
  z = zeros(d.shape, dtype=float)
  ## "identity" version
  #return 1/(d/1e3)
  # return 3e2-1./(d/5e1) ## for cone-00
  # return 2-1./(d/5e3) ## for trig-00
  # return 1000-1/(d/1e5)
  ## Correct version, inverse of the function from http://mathnathan.com/2011/02/03/depthvsdistance/
  return 348.0 / (1091.5 - d)
  # return d


class IntrinsicParameters:
  def __init__(self, f, center):
    self.f = f
    self.center = center

  def subsample(self, sub):
    self.f /= sub
    self.center /= sub

  def crop(self, bbox):
    self.center -= array([bbox[0], bbox[1]])

  ## The magical formula that gives distance form the disparity. This is the
  ## theoretical perfect model, a x**-1 expression.
  def distance_from_disparity(self, d):
    return distance_from_disparity(d)

  def coordinates_from_disparity(self, disparity):
    ## Calculate the world coordinates of each pixel.

    ## Initialize the output matrix with pixel coordinates over image plane, on
    ## camera reference frame.
    output = zeros((disparity.shape[0]*disparity.shape[1], 3))
    output[:,:2] = mgrid[:disparity.shape[1],:disparity.shape[0]].T.reshape(-1,2) - self.center
    output[:,2] = self.f

    ## Calculate z from disparity
    z = self.distance_from_disparity(disparity.ravel())

    #pdb.set_trace()
    output[:,0] *= z / self.f
    output[:,1] *= z / self.f
    output[:,2] = z
    return output

class SquareMesh:
  def __init__(self, disparity, intparam):
    self.disparity = disparity
    self.intparam = intparam

  def generate_xyz_mesh(self):
    ## Calculate the coordinate values.
    self.xyz = self.intparam.coordinates_from_disparity(self.disparity)

    ## Calculate the connections.
    Nl,Nk = self.disparity.shape
    Ncon = 4 * (Nk - 1) * (Nl - 1) + Nk + Nl - 2
    self.con = zeros((Ncon,2), dtype=uint16)

    ## Loop through every pixel. Add connections when possible. Just either the
    ## same-line pixel to the right, or any of the three 8-neighbours below.
    i=0
    for p in range(Nl*Nk):
      ## If it's not in the last column, connect to right.
      if (p + 1) % Nk:
        self.con[i,0] = p
        self.con[i,1] = p+1
        i += 1
      ## If it not in the last line
      if p <  Nk * (Nl - 1):
        ## Connect to the point below
        self.con[i,0] = p
        self.con[i,1] = p+Nk
        i += 1
        ## If it's not in the first column, connect to lower left.
        if p % Nk:
          self.con[i,0] = p
          self.con[i,1] = p+Nk-1
          i += 1
        ## If it's not in the last column, connect to lower right.
        if (p + 1) % Nk:
          self.con[i,0] = p
          self.con[i,1] = p+Nk+1
          i += 1

  def subsample(self, sub):
    self.disparity = self.disparity[::sub,::sub]
    self.intparam.subsample(sub)

  def crop(self, bbox):
    self.disparity = self.disparity[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    self.intparam.crop(bbox)

  def smash(self):
    ## Deal with outliers, just look for the maximum value outside of the maximum possible, then make the outliers the same.
    self.disparity[self.disparity==2047] = self.disparity[self.disparity<2047].max()





def run_optimization(sqmesh, u0):
  ## Start to set up optimization stuff
  Np = sqmesh.xyz.shape[0] #disparity.shape[0] * disparity.shape[1]
  Ned = sqmesh.con.shape[0]

  M = zeros((2*Np, 2*Ned+3))
  d_x = zeros(Ned+3)

  for i in range(Ned):
    a,b = sqmesh.con[i]

    M[a*2,2*i] = 1
    M[b*2,2*i] = -1
    M[a*2+1,2*i+1] = 1
    M[b*2+1,2*i+1] = -1
    #d_x[i] = sqrt( ((sqmesh.xyz[a] - sqmesh.xyz[b]) ** 2 ).sum() )
    d_x[i] = ( ((sqmesh.xyz[a] - sqmesh.xyz[b]) ** 2 ).sum() )


  ## Find the "middle" point to make it the origin
  mp = (sqmesh.disparity.shape[0]/2) * sqmesh.disparity.shape[1] + sqmesh.disparity.shape[1]/2
  M[2*mp,-3] = 1
  M[2*mp+1,-2] = 1
  M[2*mp+3,-1] = 1

  # M[0,-3] = 1
  # M[1,-2] = 1
  # M[3,-1] = 1

  mdist = d_x.mean()

  ## Fit this baby
  u_opt, success = scipy.optimize.leastsq(errfunc, u0, args=(M, d_x,))

  final_err = (errfunc(u_opt, M, d_x)**2).sum()

  return u_opt, success, final_err

###############################################################################
##
##
if __name__ == '__main__':

  ion() ## Turn on real-time plotting

  ## Plot stuff or not?
  # plot_wireframe = True
  # plot_scatter = False
  plot_wireframe = False
  plot_scatter = True
  plot_meshes = False

  register_cmap(name='guc', data=gucci_dict)
  rc('image', cmap='guc')
  # rc('image', cmap='RdBu')

  ## Check number of parameters
  if len(sys.argv)<2:
    raise Exception('''Incorrect number of parameters.

Usage: %s <data_path>'''%(sys.argv[0]))

  paul_data = True

  ## Get the name of directory that contains the data. It should contain two
  ## files named 'params.txt' and 'disparity.txt'.
  data_path = '%s/'%(sys.argv[1])

  if paul_data:
    ## Load the image with the disparity values. E.g., the range data produced by Kinect.
    disparity = loadtxt(data_path+'kinect.mat')

    optical_center = .5*(1+array([disparity.shape[1], disparity.shape[0]]))
    f = 640
  else:

    ## Load the image with the disparity values. E.g., the range data produced by Kinect.
    disparity = loadtxt(data_path+'disparity.txt')
    ## Load the file with the camera parameters used to render the scene
    ## The values are: [f, p[0], p[1], p[2], theta, phi, psi, k]
    params_file = loadtxt(data_path+'params.txt')
    ## The optical center is another important intrinsic parameter, but the
    ## current simulator just pretend this is not an issue. So the optical center
    ## is just the middle of the image, and there is also no radial lens
    ## distortion.
    optical_center = .5*(1+array([disparity.shape[1], disparity.shape[0]]))
    ## Focal distance
    f = params_file[0]

  ## Instantiate intrinsic parameters object.
  mypar = IntrinsicParameters(f, optical_center)

  ## Parameters to pre-process the image. First crop out the interest region,
  ## then downsample, then turn the outliers into more ammenable values.
  #bbox = (0, 0, disparity.shape[1], disparity.shape[0])
  bbox = (230, 125, 550, 375)
  sub = 1

  ## Instantiate mesh object, and calculate grid parameters in 3D from the
  ## disparity array and intrinsic parameters.
  sqmesh = SquareMesh(disparity, mypar)
  ## Cut the image (i.e. segment the book...)
  sqmesh.crop(bbox)
  ## resample down the image 'sub' times
  sqmesh.subsample(sub)
  sqmesh.smash()

  ## Generate the 3D point cloud and connection array
  sqmesh.generate_xyz_mesh()

  #######################################################
  ## Run the optimization

  ## Find the "middle" point to make it the origin, and make it.
  mp = (sqmesh.disparity.shape[0]/2) * sqmesh.disparity.shape[1] + sqmesh.disparity.shape[1]/2
  ## Set the initial estimate from the original xy coordinates.
  u0 = reshape(sqmesh.xyz[:,:2] - sqmesh.xyz[mp,:2] ,-1)
  #u0 -= u0[0] ## ..or set the first point as origin.

  #u_opt, success, final_err  = run_optimization(sqmesh,u0)
  #print 'final err:', final_err

  q0 = reshape(u0, (-1, 2)) # , reshape(u_opt,(-1,2)), final_err
  #q_opt = reshape(u_opt, (-1, 2)) # , reshape(u_opt,(-1,2)), final_err
  q_opt = reshape(u0, (-1, 2)) # , reshape(u_opt,(-1,2)), final_err

  #############################################################################
  ## Plot stuff
  if plot_wireframe:
    ## Plot disparity data as an image
    x,y,z = sqmesh.xyz.T
    x = -x.reshape(sqmesh.disparity.shape)
    y = y.reshape(sqmesh.disparity.shape)
    z = -z.reshape(sqmesh.disparity.shape)

    figure()
    title('Kinect data', fontsize=20, fontweight='bold')
    #fig.suptitle('Wireframe from reconstructed kinect data', fontsize=20, fontweight='bold')
    title('Kinect data (disparity)', fontsize=16)

    dmax = disparity[disparity<2047].max()
    dmin = disparity.min()

    cax = imshow(disparity, interpolation='nearest', vmin=dmin, vmax=dmax)
    colorbar(cax, shrink=.5)

    ## Plot wireframe
    fig = figure()
    ax = p3.Axes3D(fig, aspect='equal')
    title('Square mesh on 3D space', fontsize=20, fontweight='bold')

    ax.axis('equal')
    ax.plot_wireframe(x,y,z)

    mrang = max([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()])/2
    midx = (x.max()+x.min())/2
    midy = (y.max()+y.min())/2
    midz = (z.max()+z.min())/2
    ax.set_xlim3d(midx-mrang, midx+mrang)
    ax.set_ylim3d(midy-mrang, midy+mrang)
    ax.set_zlim3d(midz-mrang, midz+mrang)

  if plot_scatter:
    ## Plot disparity data as an image
    x,y,z = sqmesh.xyz[sqmesh.xyz[:,2]<sqmesh.xyz[:,2].max()].T
    x=-x
    z=-z

    ## Plot wireframe
    fig = figure(figsize=(10,8))
    ax = p3.Axes3D(fig, aspect='equal')
    title('Square mesh on 3D space', fontsize=20, fontweight='bold')

    ax.axis('equal')
    ax.scatter(x,y,z, c='b', marker='+')

    mrang = max([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()])/2
    midx = (x.max()+x.min())/2
    midy = (y.max()+y.min())/2
    midz = (z.max()+z.min())/2
    ax.set_xlim3d(midx-mrang, midx+mrang)
    ax.set_ylim3d(midy-mrang, midy+mrang)
    ax.set_zlim3d(midz-mrang, midz+mrang)



  if plot_meshes:
    figure()
    for p in sqmesh.con:
      q0 = reshape(u0,(-1,2))
      #plot(sqmesh.xyz[p,0], sqmesh.xyz[p,1], 'g-')
      plot(q0[p,0], q0[p,1], 'b-')
    axis('equal')
    yla,ylb = ylim()
    ylim(ylb,yla)

    figure(3)
    for p in sqmesh.con:
      plot(q_opt[p,0], q_opt[p,1], 'r-')

    axis('equal')
    yla,ylb = ylim()
    ylim(ylb,yla)
