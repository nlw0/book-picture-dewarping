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

ion()


#rc('image', cmap='RdBu')
#rc('image', cmap='RdYlBu')


def pcyl_funL(d, p, f=1e2, k=5e-2):
  '''Calculates distances from a parabolic cylinder, with perspectve distortion.'''
  # { (x,y,z) = tau * d + p
  # { z = k * x**2
  # Replacing z anx x gives us:
  # tau**2 (k d_x**2) + tau (2 k d_x p_x - d_z) + (k p_x**2 - p_z) = 0
  # Because d_x can be 0, we must take care to use an appropriate
  # method to calculate the roots to find tau.  From Numerical
  # Recipes, (sec 5.6 p.183) considering ax^2+bx+c=0, we first calculate:
  # q = -1/2*(b+sgn(b)sqrt(b^2-4ac))
  # Then x_1 = q/a and x_2 = c/q. Now x_1 can be inf, beautifuly.
  # in this case,
  a = k * d[:,0]**2
  b = 2*k*d[:,0]*p[0] - d[:,2]
  c = k*p[0]**2 - p[2]

  # We also calculate delta separately first, to know when we hit the surface or not.
  delta = b**2-4*a*c
  where = delta>0

  q = zeros((d.shape[0]))
  tau = zeros((d.shape[0],2))
  out = 1e6*ones(d.shape)

  q[where] = -0.5*( b[where] + sign(b[where]) * sqrt(delta[where]) )
  ## This assumes c is a number, not a vector.
  tau[where,0] = q[where] / a[where]
  tau[where,1] = c / q[where]
  taumin = tau.min(1)
  taumax = tau.max(1)

  wtaumin = ( taumin>=0 )
  wtaumax = ( taumin<0 ) * (taumax>0)

  out[wtaumin] = p + c_[taumin[wtaumin],taumin[wtaumin],taumin[wtaumin]]*d[wtaumin]
  out[wtaumax] = p + c_[taumax[wtaumax],taumax[wtaumax],taumax[wtaumax]]*d[wtaumax]

  return out


def parabola_length(x,k):
  return 0.5*(x*sqrt(4*k**2*x**2+1)+log(2*k*x+sqrt(4*k**2*x**2+1))*0.5/k)
def pcyl_get_texture_coordinates(verticesW,k):
  return c_[parabola_length(verticesW[:,0],k), verticesW[:,1]]


def cone_funL(d, p, f=1e2, k=5e-2):
  '''Calculates distances from a cone. Symmetri axis is over y direction.'''
  ## First try
  a = d[:,0]**2 - k*d[:,1]**2 + d[:,2]**2
  b = 2*(d[:,0]*p[0] - k*d[:,1]*p[1] + d[:,2]*p[2])
  c = p[0]**2-k*p[1]**2+p[2]**2
  ## Now pointing to y axis...

  # We also calculate delta separately first, to know when we hit the surface or not.
  delta = b**2-4*a*c
  where = delta>0

  q = zeros((d.shape[0]))
  tau = zeros((d.shape[0],2))
  out = 1e6*ones(d.shape)

  q[where] = -0.5*( b[where] + sign(b[where]) * sqrt(delta[where]) )
  ## This assumes c is a number, not a vector.
  tau[where,0] = q[where] / a[where]
  tau[where,1] = c / q[where]
  taumin = tau.min(1)
  taumax = tau.max(1)

  wtaumin = ( taumin>=0 )
  wtaumax = ( taumin<0 ) * (taumax>0)

  out[wtaumin] = p + c_[taumin[wtaumin],taumin[wtaumin],taumin[wtaumin]]*d[wtaumin]
  out[wtaumax] = p + c_[taumax[wtaumax],taumax[wtaumax],taumax[wtaumax]]*d[wtaumax]

  return out

def cone_get_texture_coordinates(verticesW,k):
  rho = verticesW[:,1]*sqrt(1+k)
  theta = arctan2(verticesW[:,0],verticesW[:,2])*sqrt(k)/sqrt(1+k)
  return c_[rho * sin(theta), rho * cos(theta)]




## PARAM
## The 'input parameters'.

## Choose either 'cone' for the cone model, or 'pcyl' for the
## parabolic cylinder model. This affects the functions used in
## calculations, and also the default scene parameters.
#model_type = 'cone'
model_type = 'pcyl'
ex_case = 0

## mysize: Image size in pixels
## f: Focal distance, in pixels

if model_type == 'cone':
  funL = cone_funL
  get_texture_coordinates = cone_get_texture_coordinates
elif model_type == 'pcyl':
  funL = pcyl_funL
  get_texture_coordinates = pcyl_get_texture_coordinates
else:
  raise TypeError

## Extrinsic parameters, camera pose.
if model_type == 'cone':
  if ex_case == 0:
  ## Looking straight into world origin
    mysize=(480,640)
    f = mysize[0]/3.
    p = array([0,100,0])
    theta = 0*pi/180
    phi = 90*pi/180
    psi = 0
    k = 1
  elif ex_case == 1:
    mysize=(480,640)
    f = mysize[0]/2.
    p = array([0,100,60])
    theta = 10*pi/180
    phi = 60*pi/180
    psi = 0*pi/180
    k = 1
elif model_type == 'pcyl':
  if ex_case == 0:
    mysize=(480,640)
    f = mysize[0]/3.
    p = array([80,0,-15])
    theta = 10*pi/180
    phi = 8*pi/180
    psi = 10*pi/180
    k = 1e-3
else:
  raise TypeError


## Initialize image array
pix = zeros((mysize[0],mysize[1],3))

pix[:,:,1],pix[:,:,0] = mgrid[-mysize[0]/2:mysize[0]/2,-mysize[1]/2:mysize[1]/2]+0.5
pix[:,:,2] = f

R1 = array([[+cos(theta), +sin(theta), 0],
            [-sin(theta), +cos(theta), 0],
            [0, 0, 1]])
R2 = array([[1, 0, 0],
            [0, +cos(phi), +sin(phi)],
            [0, -sin(phi), +cos(phi)],])
R3 = array([[+cos(psi),0,+sin(psi)],
            [0,1,0],
            [-sin(psi),0,+cos(psi)],])
R = dot(dot(R1,R2),R3)

## Reshape image into a list of 3D vectors. Apply rotation matrix.
d = dot(pix.reshape(mysize[0]*mysize[1],3),R)

## Calculate World coordinates of each pixel measurement.
verticesW = funL(d, p, f=f, k=k)

## Find (again...) the valid measurements.
where = verticesW[:,2]<1e6
## Calculate coordinates in the camera reference frame.
vertices = zeros(verticesW.shape)
vertices[where] = dot(verticesW[where]-p, inv(R))

## Max and min measurements, for plotting.
maxdist = vertices[where,2].max()
mindist = vertices[where,2].min()

## For making plot cute.
for kk in find(1-where):
  vertices[kk,2] = maxdist*1.05

## The range measurements. An image containing the z coordinates (relative to camera position)
I = reshape(vertices[:,2], mysize)

## Get texture coordinates from original model
uv = reshape( get_texture_coordinates(verticesW, k), (mysize[0],mysize[1],2) )



figure(1, figsize=(16,12))
suptitle('Parabolic cylinder ranging and mapping coords',
         fontweight='bold', fontsize=20)

subplot(2,2,1)
title('Range measurements')
imshow(I, cmap=cm.gray, interpolation='nearest', vmin=mindist, vmax=maxdist*1.001)
axis([0,mysize[1], mysize[0], 0])

subplot(2,2,3)
title('Contour plot of above')
contourf(I, list(mgrid[mindist:mindist+(maxdist-mindist)*11/10.:(maxdist-mindist)/10]))
axis('equal')
axis([0,mysize[1], mysize[0], 0])


## Plot the texture coordinates
ll = 200
subplot
subplot(2,2,2)
title('u coordinate (algebric)')
imshow(uv[:,:,0], interpolation='nearest', vmin=-ll, vmax=ll)
axis([0,mysize[1], mysize[0], 0])
subplot(2,2,4)
title('v coordinate (algebric)')
imshow(uv[:,:,1], interpolation='nearest', vmin=-ll, vmax=ll)
axis([0,mysize[1], mysize[0], 0])


figure(2)
title('UV mesh view', fontweight = 'bold', size=20)
VV = (mgrid[0:201:1.0]-100)*2.0
contour(uv[:,:,0],VV)
contour(uv[:,:,1],VV)
axis('equal')
axis([0,mysize[1], mysize[0], 0])
