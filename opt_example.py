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

def fitfunc(u, M):
  Ned = M.shape[0]-3
  R = zeros((1,Ned))
  D = dot(u,M)**2
  print 'ds',D.shape
  print 'rs',R.shape
  R[0,:Ned] = D[0,0:Ned*2:2]+D[0,1:Ned*2:2]
  R[0,-3:] = D[0,-3:]
  return R

def devfunc(u, M):
  return 2*dot(u,M)

def errfunc(u, M, d_x):
  print d_x.shape
  print fitfunc(u, M).shape
  fitfunc(u, M) - d_x

if __name__ == '__main__':
  k = 5.0
  tt = pi/8
  x = generate_cyl_points(k,tt)


  Np = 9
  con = array([[0,1],[0,3],[1,2],[1,4],[2,5],[3,4],[3,6],
               [4,5],[4,7],[5,8],[6,7],[7,8],], dtype=uint8)
  Ned = con.shape[0]



  M = zeros((2*Np, 2*Ned+3))
  d_x = zeros(Ned+3)

  for i in range(Ned):
    a,b, = con[i]

    M[a*2,2*i] = 1
    M[b*2,2*i] = -1
    M[a*2+1,2*i+1] = 1
    M[b*2+1,2*i+1] = -1
    d_x[i] = linalg.norm(x[a]-x[b])

  M[1,-3] = 1
  M[2,-2] = 1
  M[4,-1] = 1


  ## Start as a square mesh, with first point centered and second over x axis
  u0 = reshape(x[:,[0,2]]-x[0,[0,2]], (1,-1))
  u0 = u0 - u0[0]

  ## Fit this baby
  u_opt, success = scipy.optimize.leastsq(errfunc, u0[:], args=(M, d_x,))
