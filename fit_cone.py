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
import pdb

from scipy.optimize import fmin
import scipy
import numpy


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
  '''Estimate the normal direction of a plane fitted to a set of points. Does
  not test the plane fit, just return the direction that looks good.'''
  pmean = p.mean(0)
  pnorm = p-pmean
  sig = sqrt((pnorm**2).sum())
  pnorm = (p-pmean)/sig
  MP = c_[pnorm, ones(p.shape[0])]
  U, s, Vh = svd( MP  )

  ## Solution is taken from the last right singular vector. Its last value is
  ## also supposed to be close to zero due to subtracting mean form vectors, but
  ## that's not important to check. The three first values give the normal
  ## direction, and we then normalize it (again).
  sol = Vh[-1,:-1]
  sol = sol/linalg.norm(sol)
  return sol

def fit_cone(p):
  n = estimate_normal(p)
  rho = mean(dot(p, n) )

  if rho <0:
    rho = -rho
    n = -n

  return rho, n


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
  a = array([sin(a_zeni)*sin(a_azim),
             cos(a_zeni),
             sin(a_zeni)*cos(a_azim)])

  p_hat = p-rho*n

  nXa2 = (cross(n,a)**2).sum()
  # pdb.set_trace()
  d = ((k/2 * (nXa2 * (p_hat**2).sum(1) - dot(p_hat, a)**2)
        - dot(p_hat, n) * nXa2)
       / (k * dot(p_hat,a) * dot(n,a) + nXa2))

  return d

def objective_func(s,p):
  ## Maximum absolute error
  #return numpy.abs(distance_func(s,p)).max()
  ## Mean squared error
  return (distance_func(s,p)**2).sum()



def generate_cone_points(L, T):
  op = reshape(.0+mgrid[-1:2,0:1,-L-1:-L+2].T,(-1,3))
  #op = reshape(.0+mgrid[-2:3,0:1,-L-2:-L+3].T,(-1,3))
  op[:,1] = sqrt(op[:,0]**2+op[:,2]**2)
  trans1 = array([0,-L,L])
  Q1 = array([sin(pi/8),0.,0.])
  R1 = quaternion_to_matrix(Q1)
  p = dot(op+trans1,R1)
  trans2 = array([0,0,T])
  Q2 = array([0.,0.,0.])
  R2 = quaternion_to_matrix(Q2)
  p = dot(p,R2)+trans2
  return p

def generate_cyl_points(k,tt):
  op = reshape(.0+mgrid[0:3,0:1,-1:2].T,(-1,3))/k
  Q1 = array([0.,sin(tt/2),0.])
  R1 = quaternion_to_matrix(Q1)
  p1 = dot(op,R1)
  p1[:,1] = sqrt(1-p1[:,0]**2)
  R2 = R1.T
  p = dot(p1,R2)*k
  return p

def test_normal():
  ##############################################################################
  ## Test normal estimation (not quite plane fitting, one parameter missing).

  ## Generate some data. Nine coplanar points in a square, rotated and
  ## translated.
  op = reshape(mgrid[-1:2,0:1,-1:2].T,(-1,3))
  trans = array([2,1,3])
  Q = array([0.1,0.02,0.02])
  R = quaternion_to_matrix(Q)
  p = dot(op,R)+trans

  ## Calculate answer by rotating y vector.
  n_ok = dot( array([0,1,0]), R)

  ## Estimate norm...
  sol = estimate_normal(p)
  assert np.abs(sign(sol[0])*sign(n_ok[0])*sol - n_ok).max()<1e-15

  print n_ok
  print sol

  ##############################################################################
  ## Test just surface fitting to a set of cone points...

  ## Generate data
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

if __name__ == '__main__':
  ##############################################################################
  ## Now test the cone fitting procedure.

  ## Generate data
  L = 10 ## Create points around (0,-L,-L)
  T = 10 ## Translate to make the points this far from

  p = generate_cone_points(L,T)
  rho, n = fit_cone(p)

  ## First approximation
  print '\n***'
  print 'First approximation'
  print 'rho:', rho
  print 'normal:', n

  ## Testing optimiztion
  print '\n***'
  print 'Testing Optimization...'

  # s0 = array([.0, L, 0,0,0,0])
  # n_elev = arctan2(n[1], sqrt(n[0]**2+n[2]**2))
  # n_azim = arctan2(n[2], n[0])
  # a_zeni = n_elev-.75*pi
  # a_azim = 0
  n_elev = 0
  n_azim = 0
  a_zeni = n_elev-.75*pi
  a_azim = 0
  #s0 = array([0, rho, n_elev, n_azim, a_zeni, a_azim])
  s0 = array([sqrt(.5)/L, L, n_elev, n_azim, a_zeni, a_azim])

  print
  print 's0: ', array_str(s0, precision=2)
  print array_str(sort(numpy.abs(distance_func(s0,p))[-1::-1]), precision=2)
  print '=>', objective_func(s0,p)

  sop = scipy.optimize.fmin(objective_func, s0, args=(p,), xtol=1e-15,ftol=1e-15, maxfun=10000, maxiter=10000)

  print
  print 'sop: ', array_str(sop, precision=2)
  print array_str(sort(numpy.abs(distance_func(sop,p)))[-1::-1], precision=2)
  print '=>', objective_func(sop,p)

  print
  print 'k err: %5.3f%%'%((1/(sqrt(2)*L*sop[0])-1)*100)
  print 'rho: %5.3f'%sop[1]
  print 'rhofromk: %5.3f'% (1/(sqrt(2)*sop[0])) 

  print 70*'='


 
  print 'Wild tests'
  print ' rho   rhoe             rhoe2   k ke'

  for L in range(5, 100, 5):
    T=L
    p = generate_cone_points(L,T)
    rho, n = fit_cone(p)
    s0 = array([.1, rho, 0,0,-.75*pi,0])
    #s0 = array([sqrt(.5)/L, L, 0,0,-.75*pi,0 ])
    sop = scipy.optimize.fmin(objective_func, s0, args=(p,), xtol=1e-10,ftol=1e-10, maxfun=10000, disp=False)
    rhoe=sop[1]
    ke = 1/sop[0]

    err = objective_func(sop , p)
    sopop = array([sqrt(.5)/L, L, 0,0,-.75*pi,0 ])
    myerr = objective_func(sopop , p)

    print '% 4d %6.2f %10.2f (%11.3f%%) %5.3f %5.3f %5.3f %5.3f '%(L,rhoe,ke*sqrt(.5),(1./(sqrt(2)*L*sop[0])-1)*100, sqrt(.5)/L, sop[0], err, myerr)
