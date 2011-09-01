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

  ## These are "diagonal" matrices, with one xyz triplet every line. By
  ## multiplying U or V with these we get the products we need in the correct
  ## order. Of course you should not build this whole matrix in memory on a C
  ## implementation, it's just to look pretty in Python.
  p_u_diag = c_[p_u, zeros((N, 3*N))].reshape(-1,3*N)[:-1,:]
  p_v_diag = c_[p_v, zeros((N, 3*N))].reshape(-1,3*N)[:-1,:]

  ## Derivative of the objective function (plus Lagrange stuff) relative to
  ## coordinates. It is zero for any different dimensions, and is the same for
  ## every three coordinates. So we can calculate the matrix once and replicate.
  # dhdp_base = identity(Np) + (dot(U.T * U, l[:,0]) + dot(V.T * V, l[:,1]) +
  #                             dot(U.T*V + V.T*U, l[:,2]) / 2)
  dhdp_base = identity(Np) + (dot(dot(U.T, diag(l[:,0])), U) +
                              dot(dot(V.T, diag(l[:,1])), V) +
                              (dot(dot(U.T, diag(l[:,2])),V) +
                               dot(dot(V.T, diag(l[:,2])),U))/2)

  dhdp = zeros((3*N, 3*N))
  dhdp[0::3,0::3] = dhdp_base
  dhdp[1::3,1::3] = dhdp_base
  dhdp[2::3,2::3] = dhdp_base

  ## Calculate derivatives of restriction functions relative to
  ## coordinates. There is probably a smarter way to calculate all that... We
  ## need to figure out the really optimal ordering of all the variables.
  dhdl = zeros((3*N, 3*N))
  dhxdlu = dot(U.T, diag(p_u[:,0]))
  dhydlu = dot(U.T, diag(p_u[:,1]))
  dhzdlu = dot(U.T, diag(p_u[:,2]))
  dhxdlv = dot(V.T, diag(p_v[:,0]))
  dhydlv = dot(V.T, diag(p_v[:,1]))
  dhzdlv = dot(V.T, diag(p_v[:,2]))
  dhxdluv = dot(U.T, diag(p_v[:,0])) + dot(V.T, diag(p_u[:,0]))
  dhydluv = dot(U.T, diag(p_v[:,1])) + dot(V.T, diag(p_u[:,1]))
  dhzdluv = dot(U.T, diag(p_v[:,2])) + dot(V.T, diag(p_u[:,2]))
  dhdl[0::3,0::3] = dhxdlu
  dhdl[1::3,0::3] = dhydlu
  dhdl[2::3,0::3] = dhzdlu
  dhdl[0::3,1::3] = dhxdlv
  dhdl[1::3,1::3] = dhydlv
  dhdl[2::3,1::3] = dhzdlv
  dhdl[0::3,2::3] = dhxdluv
  dhdl[1::3,2::3] = dhydluv
  dhdl[2::3,2::3] = dhzdluv

  ## Calculate derivatives of restriction functions relative to coordinates.
  dgdp = zeros((3*N, 3*N))
  dgdp[0::3,:] = 2 * dot(U, p_u_diag)
  dgdp[1::3,:] = 2 * dot(V, p_v_diag)
  dgdp[2::3,:] = dot(U, p_v_diag) + dot(V, p_u_diag)

  ## Derivatives of restriction functions relative to multipliers are just 0.
  dgdl = zeros((3*N, 3*N))

  ## Assemble result witht he four blocks

  jacobian = zeros((6*N, 6*N))

  jacobian[:3*N,:3*N] = dhdp
  jacobian[:3*N,3*N:] = dhdl
  jacobian[3*N:,:3*N] = dgdp
  jacobian[3*N:,3*N:] = dgdl

  return jacobian

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

      ## Shigeru filter 3x3 doi://10.1109/34.841757
      U[ind, dind + eight_neighborhood] = array([-0.112737,0,0.112737,
                                                  -0.274526,0,0.274526,
                                                  -0.112737,0,0.112737])
      V[ind, dind + eight_neighborhood] = array([-0.112737,-0.274526,-0.112737,
                                                  0,0,0,
                                                  0.112737,0.274526,0.112737])
      ## Sobel filter
      #U[ind, dind + eight_neighborhood] = array([-1,0,1,-2,0,2,-1,0,1])/8.
      #V[ind, dind + eight_neighborhood] = array([-1,-2,-1,0,0,0,1,2,1])/8.
      ## Scharr filter
      #U[ind, dind + eight_neighborhood] = array([-3,0,3,-10,0,10,-3,0,3])/32.
      #V[ind, dind + eight_neighborhood] = array([-3,-10,-3,0,0,0,3,10,3])/32.




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

  Nk = 13
  Nl = 13

  calculate_U_and_V(Nl, Nk)

  k = 15
  tt = 0.5*pi/3
  q = generate_cyl_points(k,tt,Nk)

  #q[:,0] *= 1.5

  Np = Nl*Nk
  pl0 = zeros(6*Np)
  #pl0[:3*Np] = 0
  pl0[:3*Np] = q.ravel()
  #pl0[1:3*Np:3] = .6

  #print sys_eqs(pl0, q)
  import time
  a = time.clock()
  Niter = 1
  for kk in range(1):
    #pl_opt, success = scipy.optimize.leastsq(sys_eqs, pl0, args=(q,), Dfun=None)
    pl_opt, success = scipy.optimize.leastsq(sys_eqs, pl0, args=(q,), Dfun=sys_jacobian)
  a = time.clock()-a
  print 'Time: ', a/float(Niter)

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
