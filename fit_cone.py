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

################################################################################
## This gets the tree last parameters from the quaternion and maps it into
## values that make sense. Either just to the unit sphere, or eliminating the
## symmetries...
def fix_quaternion_parameters(myx):
  xb,xc,xd = myx
  xnormsq = xb*xb+xc*xc+xd*xd

  if xnormsq < 1:
    ## If inside the unit sphere, these are the components themselves, and we
    ## just have to calculate a to normalize the quaternion.
    b,c,d = xb,xc,xd
    a = np.sqrt(1-xnormsq)
  else:
    ## Just to work gracefully if we have invalid inputs, we reflect the vectors
    ## outside the unit sphere to the other side, and use the inverse norm and
    ## negative values of a.
    b,c,d = -xb/xnormsq,-xc/xnormsq,-xd/xnormsq
    a = -np.sqrt(1-  1.0/xnormsq  )
    ## This should not be used, it defeats the whole concept that small (b,c,d)
    ## vector norms have small rotation angles. It's really just to let us work
    ## near the borders of the sphere. Any optimization algorithm should work
    ## with initalizations just inside the spere, and avoid wandering outside of
    ## it.

  assert a >= -1
  assert a <= 1

  return a,b,c,d
##
################################################################################

################################################################################
## Produces a rotation matrix from the 3 last components of a
## quaternion.
def quaternion_to_matrix(myx):
  '''Converts from a quaternion representation (last 3 values) to rotation
  matrix.'''

  a,b,c,d = fix_quaternion_parameters(myx)

    ## Notice we return a transpose matrix, because we work with line-vectors
  return np.array([ [(a*a+b*b-c*c-d*d), (2*b*c-2*a*d),     (2*b*d+2*a*c)      ],
                    [(2*b*c+2*a*d),     (a*a-b*b+c*c-d*d), (2*c*d-2*a*b)      ],
                    [(2*b*d-2*a*c),     (2*c*d+2*a*b),     (a*a-b*b-c*c+d*d)] ]
                  ).T / (a*a+b*b+c*c+d*d)
##
################################################################################


def estimate_normal(p):
  '''Estimate the normal direction of a plane fitted to a set of points. It
  asserts that the points are minimally planar. (Maybe it shouldn't.)'''
  pmean = p.mean(0)
  pnorm = p-pmean
  sig = sqrt((pnorm**2).sum())
  pnorm = (p-pmean)/sig
  MP = c_[pnorm, ones(p.shape[0])]
  U, s, Vh = svd( MP  )
  ## Asserts data fits well into a plane. Last singular value should be 'zero'
  ## for a plane. Of course you should be way more generous if you are fitting
  ## curved surfaces such as in this project.
  print s
  assert s[-1] < s[-2] * 2e-2

  ## Solution is taken from the last right singular vector. Its last value is
  ## also supposed to be close to zero due to subtracting mean form vectors, but
  ## that's not important to check. The three first values give the normal
  ## direction, and we then normalize it (again).
  sol = Vh[-1,:-1]
  sol = sol/linalg.norm(sol)
  return sol

def fit_cone(p):
  normal = estimate_normal(p)

  rho = mean(dot(p, normal) )

  if rho <0:
    rho = -rho
    normal = -normal

  return rho, normal


def distance_func(x,p):
  k = x[0]
  rho = x[1]
  n_elev = x[2]
  n_azim = x[3]
  a_zeni = x[4]
  a_azim = x[5]

  n = array([cos(n_elev)*sin(n_azim),
             sin(n_elev),
             cos(n_elev)*cos(n_azim)])
  a = array([sin(a_elev)*sin(a_azim),
             cos(a_zeni),
             sin(a_elev)*cos(a_azim)])

  p_hat = p-rho*n

  nXa2 = coss(n,a)**2
  d = ((k/2 * (nXa2 * dot(p_hat, p_hat) - dot(p_hat, a)**2)
        - dot(p_hat, n) * nXa2)
       / (k * dot(p_hat,a) * dot(n,a) + nXa2))

  return d


if __name__ == '__main__':
  ## Generate some data for testing norm estimation (not quite plane
  ## fitting, one parameter missing). Nine coplanar points in a
  ## square, rotated and translated.
  op = reshape(mgrid[-1:2,0:1,-1:2].T,(-1,3))
  trans = array([2,1,3])
  Q = array([0.1,0.02,0.02])
  R = quaternion_to_matrix(Q)
  p = dot(op,R)+trans
  n_ok = dot( array([0,1,0]), R)

  ## Estimate norm...
  sol = estimate_normal(p)
  assert np.abs(sign(sol[0])*sign(n_ok[0])*sol - n_ok).max()<1e-15

  print sol
  print n_ok

  ## Generate data for testing cone fitting.
  op = reshape(.0+mgrid[10:13,0:1,-20:-17].T,(-1,3))
  op[:,1] = sqrt(op[:,0]**2+op[:,2]**2)
  trans = array([2,1,3])
  Q = array([0.1,0.02,0.02])
  R = quaternion_to_matrix(Q)
  p = dot(op,R)+trans

  tt = arctan2(op[op.shape[0]/2,0], op[op.shape[0]/2,2])
  nR = quaternion_to_matrix(array([0,sin(tt/2),0]))
  n_ok = array([0,-1,1])*sqrt(.5)
  n_ok = dot(dot(n_ok, nR), R)
  ## Estimate
  sol = estimate_normal(p)

  assert np.abs(sign(sol[0])*sign(n_ok[0])*sol - n_ok).max()<2e-2

  print n_ok
  print sol


  rho, normal = fit_cone(p)

  print rho, normal
