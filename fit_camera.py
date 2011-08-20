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

import matplotlib
#matplotlib.use('WXAgg') ## Seems to be the only backend to work without complaints.
from pylab import *
from color_block import gucci_dict

from fit_mapping import IntrinsicParameters, ExtrinsicParameters, PinholeCamera, quaternion_to_matrix

###############################################################################
##
##
if __name__ == '__main__':

  ion() ## Turn on real-time plotting

  ## Image plotting colors
  register_cmap(name='guc', data=gucci_dict)
  rc('image', cmap='guc')
  # rc('image', cmap='RdBu')

  get_points = True
  #get_points = False

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
    #disparity = rot90(loadtxt(data_path+'kinect.mat'),2)

    k_f = 640
    k_oc = .5*(1+array([disparity.shape[1], disparity.shape[0]]))


  kin_int = IntrinsicParameters(k_f, k_oc)


  cam_shot = rot90(imread(data_path+'img.png'),3)
  c_f = 86/.009 # (Lens focal length divided by pixel size, in mm)
  c_oc = array([cam_shot.shape[1]/2., cam_shot.shape[0]/2.])
  cam_int = IntrinsicParameters(c_f, c_oc)



  if get_points:
    figure(1)

    print """
  Please select four points in the frist image. Click on a 5th point (that will be
  discarded) to finish. Remember the order you selected them. If you click on the
  right spot, the result will be bad, and it will not be a fault of the
  algorithm. It can only be attributable to human error.
  """

    imshow(disparity, vmin=420, vmax=560)
    k_pts = array(ginput(n=5, show_clicks=True)[:-1])

    print """
  Now select the corresponding points. _In the same order_.
  """
    imshow(cam_shot)
    c_pts = array(ginput(n=5, show_clicks=True)[:-1])
  else:

    k_pts = array([[ 194.33870968,   93.69354839],
                   [ 190.46774194,  374.98387097],
                   [ 360.79032258,  373.69354839],
                   [ 372.40322581,  119.5       ]])
    c_pts=array([[  390.5 ,  125.5],
                 [  390.5 , 3181. ],
                 [ 2543.  , 3170.5],
                 [ 2574.5 ,  125.5]])

  print k_pts
  print c_pts

  pp = array(k_pts, dtype=int)
  pp_dis = disparity[pp[:,1], pp[:,0]]
  xyz = kin_int.coordinates_from_xy_disparity(pp, pp_dis)

  print xyz

  T = array([0,0,0])
  R = quaternion_to_matrix(array([0,0,0]))
  cam_ext = ExtrinsicParameters(T, R)

  c_camera = PinholeCamera(cam_int, cam_ext)

  print c_camera.project_into_camera(xyz)

  c_camera.find_pose(xyz, c_pts)

  reproj = c_camera.project_into_camera(xyz)

  print c_camera.ext_param.T
  print c_camera.ext_param.Q

  savetxt(data_path+'params.txt', [c_camera.ext_param.T, c_camera.ext_param.Q ])

  figure(2)
  subplot(1,2,1)
  imshow(disparity, vmin=420, vmax=560)
  plot(k_pts[:,0], k_pts[:,1], 'b+', ms=10, mew=2)

  subplot(1,2,2)
  imshow(cam_shot)
  plot(c_pts[:,0], c_pts[:,1], 'b+', ms=10, mew=2)
  plot(reproj[:,0], reproj[:,1], 'rx', ms=10, mew=2)

  suptitle('Camera localization from user selected points', fontweight='bold', fontsize=20)
