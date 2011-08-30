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

###############################################################################
## This program is to develop and test the new optimization target function,
## based on Lagrange multipliers.

from pylab import *

import pdb

from fit_cone import *
import mpl_toolkits.mplot3d.axes3d as p3

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

U = 0
V = 0

def sys_eqs(pl, q):
  global U
  global V

  assert (pl.shape[0] % 6) == 0

  ## Number of points
  N = pl.shape[0]/6

  ## Split pl into the p matrix with 3D coordinates of each point, and the l
  ## matrix with the 3 multipliers for the conditions over each point.

  ## First 3N values are the coordinates for the N points
  p = reshape(pl[:3*N], (N, -1))
  ## Last 3N values are the Lagrange multipliers ("lambdas"). Each point has 3
  ## corresponding restrictions, and each of these has one multiplier.
  l = reshape(pl[3*N:], (N, -1))

  ## Calculate the p derivatives (sum U_j^k p_k^w) into temporary arrays.
  p_u = dot(U, p)
  p_v = dot(V, p)

  ## Calculate "residue" function for each coordinate (p - q plus all the
  ## magical lambdas), and calculate the values of the restriction functions.

  ## The actual calculation of the "h" function, the decomposition of the
  ## gradient of the original objective function as a linear combination of the
  ## gradients of the "g" functions (constraints). The obscure l[:,3*[0]] is a
  ## matrix with the first column of l replicated 3 times.
  h = p - q + (dot(U.T, l[:,3*[0]] * p_u) +
               dot(V.T, l[:,3*[1]] * p_v) +
               (dot(U.T, l[:,3*[2]] * p_v) + dot(V.T, l[:,3*[2]] * p_u)) / 2)

  g = c_[(p_u**2).sum(1) - 1,
         (p_v**2).sum(1) - 1,
         (p_u*p_v).sum(1)]

  return r_[ravel(h), ravel(g)]

def sys_jacobian(pl, q):
  global U
  global V

  assert (pl.shape[0] % 6) == 0

  ## Number of points
  N = pl.shape[0]/6

  output = zeros((N,N))

  ## Split pl into the p matrix with 3D coordinates of each point, and the l
  ## matrix with the 3 multipliers for the conditions over each point.

  ## First 3N values are the coordinates for the N points
  p = reshape(pl[:3*N], (N, -1))
  ## Last 3N values are the Lagrange multipliers ("lambdas"). Each point has 3
  ## corresponding restrictions, and each of these has one multiplier.
  l = reshape(pl[3*N:], (N, -1))

  ## Calculate the p derivatives (sum U_j^k p_k^w) into temporary arrays.
  p_u = dot(U, p)
  p_v = dot(V, p)

  ## Calculate "residue" function for each coordinate (p - q plus all the
  ## magical lambdas), and calculate the values of the restriction functions.

  ## The actual calculation of the "h" function, the decomposition of the
  ## gradient of the original objective function as a linear combination of the
  ## gradients of the "g" functions (constraints). The obscure l[:,3*[0]] is a
  ## matrix with the first column of l replicated 3 times.
  dhdp = identity(3*Np) + (dot(U.T * U, l[:,3*[0]]) +
                           dot(V.T * V, l[:,3*[1]]) +
                           dot(U.T*V + V.T*U, l[:,3*[2]]) / 2)
  dhdlu = U.T * c_[N*[p_u]].T
  dhdlv = V.T * c_[N*[p_v]].T
  dhdluv = (U.T * c_[N*[p_v]].T + V.T * c_[3*N*[p_u]].T) / 2
  ## Interleave the lu lv and luvs
  dhdl = reshape(c_[ravel(dhdlu.T), ravel(dhdlv.T), ravel(dhdluv.T)], 3*N, 3*N)

  dgudp = 2 * U * c_[N*[p_u]]
  dgvdp = 2 * V * c_[N*[p_v]]
  dguvdp = U * c_[3*N*[p_v]] + V * c_[N*[p_u]]


  dgdl = zeros((3*N, 3*N))

  return r_[ravel(h), ravel(g)]


def calculate_U_and_V(Nl,Nk):
  global U
  global V

  U = zeros((Nl*Nk,Nl*Nk))
  V = zeros((Nl*Nk,Nl*Nk))

  eight_neighborhood = array([-1-Nk, -Nk, 1-Nk, -1, 0, 1, -1+Nk, Nk, 1+Nk], dtype=int16)
  for l in range(Nl):
    for k in range(Nk):
      ## The point around which we are taking the derivative. index calculated
      ## using normal row-major order.
      ind = l*Nk+k
      ## The reference point around which we set the values of the Sobel
      ## operator, or whatever filter is chosen to calculate the
      ## derivatives. This is for the magic in the next lines
      dind = ind

      # if l>1 and l < Nl-1 and k>1 and k < Nk-1 : NOP
      ## The derivatives in the borders are the same values as the derivatives
      ## right inside the rectangle. So we just modify dind accordingly.
      if k == 0:
        dind += 1
      elif k == Nk-1:
        dind -= 1
      if l == 0:
        dind += Nk
      elif l == Nl-1:
        dind -= Nk

      U[ind, dind + eight_neighborhood] = array([-1,0,1,-2,0,2,-1,0,1])/8.
      V[ind, dind + eight_neighborhood] = array([-1,-2,-1,0,0,0,1,2,1])/8.




def execute_test(k,tt):
  x = generate_cyl_points(k,tt)

  Np = x.shape[0]
  q = 4
  con = []
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

  Nk = 7
  Nl = 7

  calculate_U_and_V(Nl, Nk)

<<<<<<< HEAD
  k = 2
  tt = 0.5*pi/3
  q = generate_cyl_points(k,tt,Nk)

  q[:,1] *=1.5

=======
  k = 10
  tt = pi/6
  q = generate_cyl_points(k,tt,Nk)
>>>>>>> 13e03bb49ab90da0113f9b53c08291254b3754b9

  Np = Nl*Nk
  pl0 = zeros(6*Np)
  pl0[:3*Np] = 0
  #pl0[:3*Np] = q.ravel()
  #pl0[1:3*Np:3] = .6

<<<<<<< HEAD
  #print sys_eqs(pl0, q)
  pl_opt, success = scipy.optimize.leastsq(sys_eqs, pl0, args=(q,))

  p = pl_opt.reshape(-1,3)[:Np]

  lim = abs(p-q).max()
  subplot(3,3,1)
  imshow(reshape(q[:,0],(Nk,-1)), interpolation='nearest')
  subplot(3,3,2)
  imshow(reshape(q[:,1],(Nk,-1)), interpolation='nearest')
  subplot(3,3,3)
  imshow(reshape(q[:,2],(Nk,-1)), interpolation='nearest')
  subplot(3,3,4)
  imshow(reshape(p[:,0],(Nk,-1)), interpolation='nearest')
  subplot(3,3,5)
  imshow(reshape(p[:,1],(Nk,-1)), interpolation='nearest')
  subplot(3,3,6)
  imshow(reshape(p[:,2],(Nk,-1)), interpolation='nearest')
  subplot(3,3,7)
  imshow(reshape(p[:,0]-q[:,0],(Nk,-1)), interpolation='nearest', vmin=-lim, vmax=lim, cmap='RdBu')
  subplot(3,3,8)
  imshow(reshape(p[:,1]-q[:,1],(Nk,-1)), interpolation='nearest', vmin=-lim, vmax=lim, cmap='RdBu')
  subplot(3,3,9)
  imshow(reshape(p[:,2]-q[:,2],(Nk,-1)), interpolation='nearest', vmin=-lim, vmax=lim, cmap='RdBu')


  ## Plot wireframe
  fig = figure()
  ax = p3.Axes3D(fig, aspect='equal')
  title('Square mesh on 3D space', fontsize=20, fontweight='bold')
  ax.axis('equal')
  ax.plot_wireframe(q[:,0].reshape(Nl,Nk),q[:,1].reshape(Nl,Nk),q[:,2].reshape(Nl,Nk), color='b')
  ax.plot_wireframe(p[:,0].reshape(Nl,Nk),p[:,1].reshape(Nl,Nk),p[:,2].reshape(Nl,Nk), color='r')

  mrang = max([p[:,0].max()-p[:,0].min(), p[:,1].max()-p[:,1].min(), p[:,2].max()-p[:,2].min()])/2
  midx = (p[:,0].max()+p[:,0].min())/2
  midy = (p[:,1].max()+p[:,1].min())/2
  midz = (p[:,2].max()+p[:,2].min())/2
  ax.set_xlim3d(midx-mrang, midx+mrang)
  ax.set_ylim3d(midy-mrang, midy+mrang)
  ax.set_zlim3d(midz-mrang, midz+mrang)
=======
  pl0[:3*Np] = q.ravel()

  pl0[1*Np:3] = 0

  #print sys_eqs(pl, q)
  pl_opt, success = scipy.optimize.leastsq(sys_eqs, pl0, args=(q,))

  p = pl_opt.reshape(-1,3)[:Np]
  suptitle('Point cloud, fitted coordinates, and errors', fontsize=20, fontweight='bold')
  subplot(3,3,1)
  imshow(reshape(q[:,0],(Nk,-1)), interpolation='nearest', vmin=-2, vmax=10, cmap='RdBu')
  subplot(3,3,2)
  imshow(reshape(q[:,1],(Nk,-1)), interpolation='nearest', vmin=-2, vmax=10, cmap='RdBu')
  subplot(3,3,3)
  imshow(reshape(q[:,2],(Nk,-1)), interpolation='nearest', vmin=-2, vmax=10, cmap='RdBu')
  subplot(3,3,4)
  imshow(reshape(p[:,0],(Nk,-1)), interpolation='nearest', vmin=-2, vmax=10, cmap='RdBu')
  subplot(3,3,5)
  imshow(reshape(p[:,1],(Nk,-1)), interpolation='nearest', vmin=-2, vmax=10, cmap='RdBu')
  subplot(3,3,6)
  imshow(reshape(p[:,2],(Nk,-1)), interpolation='nearest', vmin=-2, vmax=10, cmap='RdBu')
  subplot(3,3,7)
  imshow(reshape(p[:,0]-q[:,0],(Nk,-1)), interpolation='nearest', vmin=-0.1, vmax=0.1, cmap='RdBu')
  subplot(3,3,8)
  imshow(reshape(p[:,1]-q[:,1],(Nk,-1)), interpolation='nearest', vmin=-0.1, vmax=0.1, cmap='RdBu')
  subplot(3,3,9)
  imshow(reshape(p[:,2]-q[:,2],(Nk,-1)), interpolation='nearest', vmin=-0.1, vmax=0.1, cmap='RdBu')
>>>>>>> 13e03bb49ab90da0113f9b53c08291254b3754b9

  if False:

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
