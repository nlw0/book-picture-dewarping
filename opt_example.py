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


def coisa_o_trem(k,tt):
  x = generate_cyl_points(k,tt)


  Np = 9
  con = array([[0,1],[0,3],[1,2],[1,4],[2,5],[3,4],[3,6],
               [4,5],[4,7],[5,8],[6,7],[7,8],[0,4],[2,4],[4,6],[4,8]], dtype=uint8)
  Ned = con.shape[0]

  print 'Np', Np
  print 'Ned', Ned


  M = zeros((2*Np, 2*Ned+3))
  d_x = zeros(Ned+3)

  for i in range(Ned):
    a,b, = con[i]

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
  u0 = reshape(x[:,[0,2]]-x[0,[0,2]],-1)


  fitfunc(u0, M)

  ## Fit this baby
  u_opt, success = scipy.optimize.leastsq(errfunc, u0, args=(M, d_x,))

  final_err = (errfunc(u_opt, M, d_x)**2).sum()
  print 'final err:', final_err

  return (reshape(u_opt,(-1,2)), con)






if __name__ == '__main__':
  figure(1)
  for en,k in enumerate([ 100,4,3 ]):
    ua, con = coisa_o_trem(k, pi+pi/8)
    Ned = con.shape[0]
    for i in range(Ned-4):
      plot( ua[con[i],0], ua[con[i],1],     'bgrcmykw'[en])
  title('Cylinder dewarping with simple distance model, different curvatures')
  axis('equal')





