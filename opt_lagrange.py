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
###############################################################################

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

def sys_eqs(pl, q, U, V):
  '''This function outputs a vector with the values of the functions from the
  big non-linear system of equations that we need to solve in order to fit the
  inextensible surface. The solution is found by minimizing the sums of the
  squares of these, e.g. using Levenberg-Marquards least-quares fitting. The
  sys_jacobian function provides the jacobian matrix of this function to make
  that better.'''

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

  ## The restriction functions, pretty straight forward.
  g = c_[(p_u**2).sum(1) - 1,
         (p_v**2).sum(1) - 1,
         (p_u*p_v).sum(1)]

  return r_[ravel(h), ravel(g)]

def sys_jacobian(pl, q, U, V):
  '''This function returns the Jacobian martix of the target function to fit an
  inextensible surface to given data. It receives q (associated measurements) as
  a parameter because it is necessary in sys_eqs, but it is never used for this
  specific and simple distance function used.
  '''

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
  ## coordinates. There is probably a smarter way to calculate all that. We need
  ## to figure out the really optimal ordering of all the variables, etc.
  dgdp = zeros((3*N, 3*N))
  dgudpx = 2 * dot(diag(p_u[:,0]), U)
  dgudpy = 2 * dot(diag(p_u[:,1]), U)
  dgudpz = 2 * dot(diag(p_u[:,2]), U)
  dgvdpx = 2 * dot(diag(p_v[:,0]), V)
  dgvdpy = 2 * dot(diag(p_v[:,1]), V)
  dgvdpz = 2 * dot(diag(p_v[:,2]), V)
  dguvdpx = dot(diag(p_v[:,0]), U) + dot(diag(p_u[:,0]), V)
  dguvdpy = dot(diag(p_v[:,1]), U) + dot(diag(p_u[:,1]), V)
  dguvdpz = dot(diag(p_v[:,2]), U) + dot(diag(p_u[:,2]), V)
  dgdp[0::3,0::3] = dgudpx
  dgdp[0::3,1::3] = dgudpy
  dgdp[0::3,2::3] = dgudpz
  dgdp[1::3,0::3] = dgvdpx
  dgdp[1::3,1::3] = dgvdpy
  dgdp[1::3,2::3] = dgvdpz
  dgdp[2::3,0::3] = dguvdpx
  dgdp[2::3,1::3] = dguvdpy
  dgdp[2::3,2::3] = dguvdpz

  ## Calculate derivatives of objective function gradient relative to Lagrange
  ## multipliers. It turns out it's just the transpose of the dgdp, scaled. That
  ## probably means something good.
  dhdl = 0.5*dgdp.T

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

  return U, V

if __name__ == '__main__':


  ## Size of the model, lines and columns
  Nl = 7
  Nk = 7

  ## Calculates the U and V matrices. (Partial derivatives on u and v directions).
  U, V = calculate_U_and_V(Nl, Nk)

  ## Gnerate points over a cylinder for test.
  k = 20 # Curvature
  tt = 0.5*pi/3 # Angle between the mesh and cylinder axis
  oversample = 5
  Nko = Nk * oversample
  Nlo = Nl * oversample
  q_data = generate_cyl_points(k,tt,Nk*oversample)/5.8
  #q_data = generate_cyl_points(k,tt,Nk*oversample)/6.
  #q_data = generate_cyl_points(k,tt,Nk*oversample)/5
  #q_data = generate_cyl_points(k,tt,Nk*oversample)/3.5

  q = c_[q_data[:,0].reshape(Nlo,Nko)[oversample/2::oversample,oversample/2::oversample].ravel(),
         q_data[:,1].reshape(Nlo,Nko)[oversample/2::oversample,oversample/2::oversample].ravel(),
         q_data[:,2].reshape(Nlo,Nko)[oversample/2::oversample,oversample/2::oversample].ravel(),
         ]

  Np = Nl*Nk
  pl0 = zeros(6*Np)

  p0 = mgrid[:1,-2:Nk-2,-1:Nl-1].reshape(3,-1).T
  p0[:,0] = mean(q[:,0])

  pl0[:3*Np] = p0.ravel()

  # pl0[:3*Np] = 0
  # pl0[:3*Np] = q.ravel()

  ## Run optimization
  pl_opt, success = scipy.optimize.leastsq(sys_eqs, pl0, args=(q,U,V), Dfun=sys_jacobian)

  ## Get the estimated coordinates, organize (and dump multipliers)
  p = pl_opt.reshape(-1,3)[:Np]

  #############################################################################
  ## Plot stuff

  ## Plot wireframes of input and resulting model
  fig = figure()
  ax = p3.Axes3D(fig, aspect='equal')
  title('Square mesh on 3D space', fontsize=20, fontweight='bold')
  ax.axis('equal')
  ax.plot_wireframe(q_data[:,0].reshape(Nl*oversample,Nk*oversample),q_data[:,1].reshape(Nl*oversample,Nk*oversample),q_data[:,2].reshape(Nl*oversample,Nk*oversample), color='b')
  ax.plot_wireframe(p0[:,0].reshape(Nl,Nk),p0[:,1].reshape(Nl,Nk),p0[:,2].reshape(Nl,Nk), color='#880000')
  ax.plot_wireframe(q[:,0].reshape(Nl,Nk),q[:,1].reshape(Nl,Nk),q[:,2].reshape(Nl,Nk), color='g')
  ax.plot_wireframe(p[:,0].reshape(Nl,Nk),p[:,1].reshape(Nl,Nk),p[:,2].reshape(Nl,Nk), color='r')

  mrang = max([p[:,0].max()-p[:,0].min(), p[:,1].max()-p[:,1].min(), p[:,2].max()-p[:,2].min()])/2
  midx = (p[:,0].max()+p[:,0].min())/2
  midy = (p[:,1].max()+p[:,1].min())/2
  midz = (p[:,2].max()+p[:,2].min())/2
  ax.set_xlim3d(midx-mrang, midx+mrang)
  ax.set_ylim3d(midy-mrang, midy+mrang)
  ax.set_zlim3d(midz-mrang, midz+mrang)
