#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import argparse
import scipy.optimize

def normalize(v):
  norm = np.linalg.norm(v)
  assert norm > 0
  return v / norm

class Polynomial:
  def __init__(self, p):
    self.p = p

  def stretchtime(self, factor):
    recip = 1.0 / factor;
    scale = recip
    for i in range(1, len(self.p)):
      self.p[i] *= scale
      scale *= recip

  # evaluate a polynomial using horner's rule
  def eval(self, t):
    assert t >= 0
    x = 0.0
    for i in range(0, len(self.p)):
      x = x * t + self.p[len(self.p) - 1 - i]
    return x

  # compute and return derivative
  def derivative(self):
    return Polynomial([(i+1) * self.p[i+1] for i in range(0, len(self.p) - 1)])

class TrajectoryOutput:
  def __init__(self):
    self.pos = None   # position [m]
    self.vel = None   # velocity [m/s]
    self.acc = None   # acceleration [m/s^2]
    self.omega = None # angular velocity [rad/s]
    self.yaw = None   # yaw angle [rad]
    self.roll = None  # required roll angle [rad]
    self.pitch = None # required pitch angle [rad]

# 4d single polynomial piece for x-y-z-yaw, includes duration.
class Polynomial4D:
  def __init__(self, duration, px, py, pz, pyaw):
    self.duration = duration
    self.px = Polynomial(px)
    self.py = Polynomial(py)
    self.pz = Polynomial(pz)
    self.pyaw = Polynomial(pyaw)

  # compute and return derivative
  def derivative(self):
    return Polynomial4D(
      self.duration,
      self.px.derivative().p,
      self.py.derivative().p,
      self.pz.derivative().p,
      self.pyaw.derivative().p)

  def stretchtime(self, factor):
    self.duration *= factor
    self.px.stretchtime(factor)
    self.py.stretchtime(factor)
    self.pz.stretchtime(factor)
    self.pyaw.stretchtime(factor)

  # see Daniel Mellinger, Vijay Kumar:
  #     Minimum snap trajectory generation and control for quadrotors. ICRA 2011: 2520-2525
  #     section III. DIFFERENTIAL FLATNESS
  def eval(self, t):
    result = TrajectoryOutput()
    # flat variables
    result.pos = np.array([self.px.eval(t), self.py.eval(t), self.pz.eval(t)])
    result.yaw = self.pyaw.eval(t)

    # 1st derivative
    derivative = self.derivative()
    result.vel = np.array([derivative.px.eval(t), derivative.py.eval(t), derivative.pz.eval(t)])
    dyaw = derivative.pyaw.eval(t)

    # 2nd derivative
    derivative2 = derivative.derivative()
    result.acc = np.array([derivative2.px.eval(t), derivative2.py.eval(t), derivative2.pz.eval(t)])

    # 3rd derivative
    derivative3 = derivative2.derivative()
    jerk = np.array([derivative3.px.eval(t), derivative3.py.eval(t), derivative3.pz.eval(t)])

    thrust = result.acc + np.array([0, 0, 9.81]) # add gravity

    z_body = normalize(thrust)
    x_world = np.array([np.cos(result.yaw), np.sin(result.yaw), 0])
    y_body = normalize(np.cross(z_body, x_world))
    x_body = np.cross(y_body, z_body)

    jerk_orth_zbody = jerk - (np.dot(jerk, z_body) * z_body)
    h_w = jerk_orth_zbody / np.linalg.norm(thrust)

    result.omega = np.array([-np.dot(h_w, y_body), np.dot(h_w, x_body), z_body[2] * dyaw])

    # compute required roll/pitch angles
    result.pitch = np.arcsin(-x_body[2])
    result.roll = np.arctan2(y_body[2], z_body[2])

    return result

class Trajectory:
  
  def __init__(self):
    self.polynomials = None
    self.duration = None

  def loadcsv(self, filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=range(33), ndmin=2)
    self.polynomials = [Polynomial4D(row[0], row[1:9], row[9:17], row[17:25], row[25:33]) for row in data]
    self.duration = np.sum(data[:,0])

  def savecsv(self, filename):
    data = np.empty((len(self.polynomials), 8*4+1))
    for i, p in enumerate(self.polynomials):
      data[i,0] = p.duration
      data[i,1:9] = p.px.p
      data[i,9:17] = p.py.p
      data[i,17:25] = p.pz.p
      data[i,25:33] = p.pyaw.p
    np.savetxt(filename, data, fmt="%.6f", delimiter=",", header="duration,x^0,x^1,x^2,x^3,x^4,x^5,x^6,x^7,y^0,y^1,y^2,y^3,y^4,y^5,y^6,y^7,z^0,z^1,z^2,z^3,z^4,z^5,z^6,z^7,yaw^0,yaw^1,yaw^2,yaw^3,yaw^4,yaw^5,yaw^6,yaw^7")

  def to_msg(self,delay=0.0):
      data = np.empty((len(self.polynomials), 8*4+1))
      for i, p in enumerate(self.polynomials):
        data[i,0] = p.duration
        data[i,1:9] = p.px.p
        data[i,9:17] = p.py.p
        data[i,17:25] = p.pz.p
        data[i,25:33] = p.pyaw.p
      data = data.flatten()
      data = np.append(data,delay)
      data = data.flatten().tolist()
      return data
  
  
  def from_msg(self, data):
    data = data[:-1]
    data = np.array(data).reshape(-1, 33)
    self.polynomials = [Polynomial4D(row[0], row[1:9], row[9:17], row[17:25], row[25:33]) for row in data]
    self.duration = np.sum(data[:,0])


  def stretchtime(self, factor):
    for p in self.polynomials:
      p.stretchtime(factor)
    self.duration *= factor

  def eval(self, t):
    assert t >= 0
    assert t <= self.duration

    current_t = 0.0
    for p in self.polynomials:
      if t < current_t + p.duration:
        return p.eval(t - current_t)
      current_t = current_t + p.duration
      
def func(coefficients, times, values, piece_length):
  result = 0
  i = 0
  for t, value in zip(times, values):
    if t > (i+1) * piece_length:
      i = i + 1
    estimate = np.polyval(coefficients[i*8:(i+1)*8], t - i * piece_length)
    # print(coefficients[i*8:(i+1)*8], t - i * piece_length, estimate)
    result += (value - estimate) ** 2 #np.sum((values - estimates) ** 2)
  # print(coefficients, result)
  return result

# constraint to match values between spline pieces
# def func_eq_constraint_val(coefficients, i, piece_length):
#   result = 0
#   end_val = np.polyval(coefficients[(i-1)*8:i*8], piece_length)
#   start_val = np.polyval(coefficients[i*8:(i+1)*8], 0)
#   return end_val - start_val

def func_eq_constraint_der(coefficients, i, piece_length, order):
  result = 0
  last_der = np.polyder(coefficients[(i-1)*8:i*8], order)
  this_der = np.polyder(coefficients[i*8:(i+1)*8], order)

  end_val = np.polyval(last_der, piece_length)
  start_val = np.polyval(this_der, 0)
  return end_val - start_val

def func_eq_constraint_der_value(coefficients, i, t, desired_value, order):
  result = 0
  der = np.polyder(coefficients[i*8:(i+1)*8], order)

  value = np.polyval(der, t)
  return value - desired_value

# def func_eq_constraint(coefficients, tss, yawss):
#   result = 0
#   last_derivative = None
#   for ts, yaws, i in zip(tss, yawss, range(0, len(tss))):
#     derivative = np.polyder(coefficients[i*8:(i+1)*8])
#     if last_derivative is not None:
#       result += np.polyval(derivative, 0) - last_derivative
#     last_derivative = np.polyval(derivative, tss[-1])


  # # apply coefficients to trajectory
  # for i,p in enumerate(traj.polynomials):
  #   p.pyaw.p = coefficients[i*8:(i+1)*8]
  # # evaluate at each timestep and compute the sum of squared differences
  # result = 0
  # for t,yaw in zip(ts,yaws):
  #   e = traj.eval(t)
  #   result += (e.yaw - yaw) ** 2
  # return result

def generate_trajectory(data, num_pieces):
  piece_length = data[-1,0] / num_pieces

  x0 = np.zeros(num_pieces * 8)

  constraints = []
  # piecewise values and derivatives have to match
  for i in range(1, num_pieces):
    for order in range(0, 4):
      constraints.append({'type': 'eq', 'fun': func_eq_constraint_der, 'args': (i, piece_length, order)})

  # zero derivative at the beginning and end
  for order in range(1, 3):
    constraints.append({'type': 'eq', 'fun': func_eq_constraint_der_value, 'args': (0, 0, 0, order)})
    constraints.append({'type': 'eq', 'fun': func_eq_constraint_der_value, 'args': (num_pieces-1, piece_length, 0, order)})


  resX = scipy.optimize.minimize(func, x0, (data[:,0], data[:,1], piece_length), method="SLSQP", options={"maxiter": 100}, 
    constraints=constraints
    )
  resY = scipy.optimize.minimize(func, x0, (data[:,0], data[:,2], piece_length), method="SLSQP", options={"maxiter": 100}, 
    constraints=constraints
    )
  resZ = scipy.optimize.minimize(func, x0, (data[:,0], data[:,3], piece_length), method="SLSQP", options={"maxiter": 100}, 
    constraints=constraints
    )

  resYaw = scipy.optimize.minimize(func, x0, (data[:,0], data[:,4], piece_length), method="SLSQP", options={"maxiter": 100}, 
    constraints=constraints
     )

  # resYaw = np.zeros((8,))
  
  traj = Trajectory()
  traj.polynomials = [Polynomial4D(
    piece_length, 
    np.array(resX.x[i*8:(i+1)*8][::-1]),
    np.array(resY.x[i*8:(i+1)*8][::-1]),
    np.array(resZ.x[i*8:(i+1)*8][::-1]),
    np.array(resYaw.x[i*8:(i+1)*8][::-1])) for i in range(0, num_pieces)]
    
  traj.duration = data[-1,0]
  return traj
