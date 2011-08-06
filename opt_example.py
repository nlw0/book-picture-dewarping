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

from fit_cone import *

from  scipy.optimize import leastsq
ion()

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


def execute_test(k,tt):
  x = generate_cyl_points(k,tt)

  Np = x.shape[0]
  q = 4
  con=[]
  for a in range(Np):
    if a%q != (q-1):
      con.append((a, a+1))
    if a < (Np-q):
      con.append((a, a+q))
      if a%q != (q-1):
        con.append((a, a+q+1))

  con=array(con, dtype=uint8)

  print con

  Ned = con.shape[0]

  print 'Np', Np
  print 'Ned', Ned


  M = zeros((2*Np, 2*Ned+3))
  d_x = zeros(Ned+3)

  for i in range(Ned):
    a,b = con[i]

    M[a*2,2*i] = 1
    M[b*2,2*i] = -1
    M[a*2+1,2*i+1] = 1
    M[b*2+1,2*i+1] = -1
    d_x[i] = ((x[a]-x[b])**2).sum()

  M[0,-3] = 1
  M[1,-2] = 1
  M[3,-1] = 1

  print 'Ms', M.shape


  ## Start as a square mesh, with first point centered and second over x axis
  # u0 = reshape(x[:,[0,2]]-x[0,[0,2]],-1)
  # u0 = reshape(x[:,[0,2]]-x[0,[0,2]],-1)
  u0 = reshape(.0+mgrid[0:4,0:4].T,-1)
  # pdb.set_trace()

  ## Fit this baby
  u_opt, success = scipy.optimize.leastsq(errfunc, u0, args=(M, d_x,))

  final_err = (errfunc(u_opt, M, d_x)**2).sum()
  print 'final err:', final_err

  return reshape(u0,(-1,2)), reshape(u_opt,(-1,2)), con, final_err

if __name__ == '__main__':
  figure(1, figsize=[12,8])
  suptitle('Cylinder dewarping with simple distance model, different curvatures', fontsize=20, fontweight='bold')

  tt = pi/5
  for en,k in enumerate([ .5, 1, 2, 3, 7]):
    subplot(2,3,en+2)
    u0, ua, con, err = execute_test(k, tt)
    title('k = %d, err=%5.2f'%((100./k), err))
    Ned = con.shape[0]
    for i in range(Ned):
      plot( u0[con[i],0], u0[con[i],1], 'b-x')
      plot( ua[con[i],0], ua[con[i],1], 'r-x')
    grid()
    axis('equal')
    axis([-1, 4, -.75, 3.75])
    xlim(-.75,3.75)

  k = 2
  x = generate_cyl_points(k,tt)
  subplot(2,3,1)
  title('Original xz coords, k=%d'%(100./k))
  for i in range(Ned):
    plot( x[con[i],0], x[con[i],2], 'g-x')
  grid()
  axis('equal')
  axis([-1, 4, -1.75, 2.75])
