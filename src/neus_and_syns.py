# -*- coding: utf-8 -*-

import brainpy as bp
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import torch
from torch.autograd.functional import _as_tuple, _tuple_postprocess
from torch.autograd.functional import _check_requires_grad, _autograd_grad
from torch.autograd.functional import _grad_preprocess, _grad_postprocess


def torch_max(x, y):
  if not isinstance(x, torch.Tensor):
    x = torch.ones_like(y) * x
  if not isinstance(y, torch.Tensor):
    y = torch.ones_like(x) * y
  return torch.max(x, y)


bp.ops.set_buffer('pytorch', sqrt=torch.sqrt, random=torch.rand, maximum=torch_max)
bp.ops.set_buffer('numpy', sqrt=np.sqrt, random=np.random.random, maximum=np.maximum)
bp.ops.set_buffer('numba', sqrt=np.sqrt, random=np.random.random, maximum=np.maximum)
bp.ops.set_buffer('numba-parallel', sqrt=np.sqrt, random=np.random.random, maximum=np.maximum)

T = 36.

# parameters of INa, IK
g_Na = 100.
E_Na = 50.
g_K = 10.
phi_m = phi_h = phi_n = 3 ** ((T - 36) / 10)

# parameters of IT
E_T = 120.
phi_p = 5 ** ((T - 24) / 10)
phi_q = 3 ** ((T - 24) / 10)
p_half, p_k = -52., 7.4
q_half, q_k = -80., -5.

# parameters of V
C, Vth, area = 1., 20., 1.43e-4
V_factor = 1e-3 / area


class OriginalTRN(bp.NeuGroup):
  """TRN neuron model which is inspired from [1, 2].

  References
  -------
  [1] Bazhenov, M., Timofeev, I., Steriade, M., & Sejnowski,
      T. J. (1999). Self–sustained rhythmic activity in the
      thalamic reticular nucleus mediated by depolarizing
      GABA A receptor potentials. Nature neuroscience, 2(2),
      168-174.
  [2] Bazhenov, Maxim, Igor Timofeev, Mircea Steriade, and
      Terrence J. Sejnowski. "Cellular and network models for
      intrathalamic augmenting responses during 10-Hz stimulation."
      Journal of Neurophysiology 79, no. 5 (1998): 2730-2748.

  """
  target_backend = 'general'

  def __init__(self, size, **kwargs):
    super(OriginalTRN, self).__init__(size=size, **kwargs)

    self.IT_th = -3.
    self.b = 0.5
    self.g_T = 2.0
    self.g_L = 0.02
    self.E_L = -70.
    self.g_KL = 0.005
    self.E_KL = -95.
    self.NaK_th = -55.

    self.V = bp.ops.zeros(self.num)
    self.m = bp.ops.zeros(self.num)
    self.h = bp.ops.zeros(self.num)
    self.n = bp.ops.zeros(self.num)
    self.p = bp.ops.zeros(self.num)
    self.q = bp.ops.zeros(self.num)
    self.spike = bp.ops.zeros(self.num, dtype=bool)
    self.input = bp.ops.zeros(self.num)

    self.int_m = bp.odeint(method='rk4', f=self.m_derivative)
    self.int_h = bp.odeint(method='rk4', f=self.h_derivative)
    self.int_n = bp.odeint(method='rk4', f=self.n_derivative)
    self.int_p = bp.odeint(method='rk4', f=self.p_derivative)
    self.int_q = bp.odeint(method='rk4', f=self.q_derivative)
    self.int_V = bp.odeint(method='rk4', f=self.V_derivative)

  @staticmethod
  def m_derivative(m, t, V, NaK_th):
    alpha = 0.32 * (V - NaK_th - 13.) / (1 - np.exp(-(V - NaK_th - 13.) / 4.))
    beta = -0.28 * (V - NaK_th - 40.) / (1 - np.exp((V - NaK_th - 40.) / 5.))
    tau = 1. / phi_m / (alpha + beta)
    inf = alpha / (alpha + beta)
    dmdt = (inf - m) / tau
    return dmdt

  @staticmethod
  def h_derivative(h, t, V, NaK_th):
    alpha = 0.128 * np.exp(-(V - NaK_th - 17.) / 18.)
    beta = 4. / (1. + np.exp(-(V - NaK_th - 40.) / 5.))
    tau = 1. / phi_h / (alpha + beta)
    inf = alpha / (alpha + beta)
    dhdt = (inf - h) / tau
    return dhdt

  @staticmethod
  def n_derivative(n, t, V, NaK_th, b):
    alpha = 0.032 * (V - NaK_th - 15.) / (1. - np.exp(-(V - NaK_th - 15.) / 5.))
    beta = b * np.exp(-(V - NaK_th - 10.) / 40.)
    tau = 1 / phi_n / (alpha + beta)
    inf = alpha / (alpha + beta)
    dndt = (inf - n) / tau
    return dndt

  @staticmethod
  def p_derivative(p, t, V, IT_th):
    inf = 1. / (1. + np.exp((-V + p_half + IT_th) / p_k))
    tau = 3. + 1. / (np.exp((V + 27. - IT_th) / 10.) +
                     np.exp(-(V + 102. - IT_th) / 15.))
    dpdt = phi_p * (inf - p) / tau
    return dpdt

  @staticmethod
  def q_derivative(q, t, V, IT_th):
    inf = 1. / (1. + np.exp(-(V - q_half - IT_th) / q_k))
    tau = 85. + 1. / (np.exp((V + 48. - IT_th) / 4.) +
                      np.exp(-(V + 407. - IT_th) / 50.))
    dqdt = phi_q * (inf - q) / tau
    return dqdt

  @staticmethod
  def V_derivative(V, t, m, h, n, p, q, Isyn, E_KL, E_L, g_KL, g_L, g_T):
    INa = g_Na * m ** 3 * h * (V - E_Na)
    IK = g_K * n ** 4 * (V - E_KL)
    IT = g_T * p ** 2 * q * (V - E_T)
    IL = g_L * (V - E_L)
    IKL = g_KL * (V - E_KL)
    Icur = INa + IK + IT + IL + IKL
    dvdt = (-Icur + Isyn * V_factor) / C
    return dvdt

  def update(self, _t, _i, _dt):
    self.m = self.int_m(self.m, _t, self.V, self.NaK_th)
    self.h = self.int_h(self.h, _t, self.V, self.NaK_th)
    self.n = self.int_n(self.n, _t, self.V, self.NaK_th, self.b)
    self.p = self.int_p(self.p, _t, self.V, self.IT_th)
    self.q = self.int_q(self.q, _t, self.V, self.IT_th)
    V = self.int_V(self.V, _t, m=self.m, h=self.h, n=self.n,
                   p=self.p, q=self.q, Isyn=self.input,
                   E_KL=self.E_KL, E_L=self.E_L, g_KL=self.g_KL,
                   g_L=self.g_L, g_T=self.g_T)
    self.spike = np.logical_and(V >= Vth, self.V < Vth)
    self.V = V
    self.input[:] = 0.

  def reset(self, Vr):
    self.V[:] = Vr

    alpha = 0.32 * (self.V - self.NaK_th - 13.) / (1 - np.exp(-(self.V - self.NaK_th - 13.) / 4.))
    beta = -0.28 * (self.V - self.NaK_th - 40.) / (1 - np.exp((self.V - self.NaK_th - 40.) / 5.))
    self.m[:] = alpha / (alpha + beta)

    alpha = 0.128 * np.exp(-(self.V - self.NaK_th - 17.) / 18.)
    beta = 4. / (1. + np.exp(-(self.V - self.NaK_th - 40.) / 5.))
    self.h[:] = alpha / (alpha + beta)

    alpha = 0.032 * (self.V - self.NaK_th - 15.) / (1. - np.exp(-(self.V - self.NaK_th - 15.) / 5.))
    beta = self.b * np.exp(-(self.V - self.NaK_th - 10.) / 40.)
    self.n[:] = alpha / (alpha + beta)

    self.p[:] = 1. / (1. + np.exp((-self.V - 52. + self.IT_th) / 7.4))
    self.q[:] = 1. / (1. + np.exp((self.V + 80. - self.IT_th) / 5.))

    self.spike[:] = False
    self.input[:] = 0.


class EqPotentialTRN(bp.NeuGroup):
  target_backend = 'general'

  def __init__(self, size, **kwargs):
    super(EqPotentialTRN, self).__init__(size=size, **kwargs)

    self.b = 0.5
    self.g_L = 0.02
    self.E_L = -70.
    self.g_KL = 0.005
    self.E_KL = -95.
    self.NaK_th = -55.
    self.g_T = 2.0
    self.IT_th = -3.

    self.V = bp.ops.zeros(self.num)
    self.vm = bp.ops.zeros(self.num)
    self.vh = bp.ops.zeros(self.num)
    self.vn = bp.ops.zeros(self.num)
    self.vp = bp.ops.zeros(self.num)
    self.vq = bp.ops.zeros(self.num)
    self.input = bp.ops.zeros(self.num)
    self.spike = bp.ops.zeros(self.num, dtype=bool)
    self.reset(Vr=-75.)

    self.integral = bp.odeint(method='rk4', f=self.derivative)

  @staticmethod
  def derivative(V, vm, vh, vn, vp, vq, t, Isyn, E_KL, E_L, g_KL, g_L, g_T,
                 NaK_th, b, IT_th):
    # vm
    vm_alpha_by_V = 0.32 * (V - NaK_th - 13.) / (1 - np.exp(-(V - NaK_th - 13.) / 4.))
    vm_alpha_by_vm = 0.32 * (vm - NaK_th - 13.) / (1 - np.exp(-(vm - NaK_th - 13.) / 4.))
    vm_beta_by_V = -0.28 * (V - NaK_th - 40.) / (1 - np.exp((V - NaK_th - 40.) / 5.))
    vm_beta_by_vm = -0.28 * (vm - NaK_th - 40.) / (1 - np.exp((vm - NaK_th - 40.) / 5.))
    vm_tau_by_V = 1. / phi_m / (vm_alpha_by_V + vm_beta_by_V)
    vm_inf_by_V = vm_alpha_by_V / (vm_alpha_by_V + vm_beta_by_V)
    vm_inf_by_vm = vm_alpha_by_vm / (vm_alpha_by_vm + vm_beta_by_vm)
    exp1 = bp.ops.exp(-(vm - 13. - NaK_th) / 4.)
    alpha_diff_by_vm = (0.32 * (1 - exp1) - 0.08 * (vm - 13 - NaK_th) * exp1) / (exp1 - 1) ** 2
    exp2 = bp.ops.exp((vm - 40. - NaK_th) / 5.)
    beta_diff_by_vm = (-0.28 * (1 - exp2) - 0.056 * (vm - 40. - NaK_th) * exp2) / (exp2 - 1) ** 2
    vm_inf_diff = (alpha_diff_by_vm * vm_beta_by_vm - vm_alpha_by_vm * beta_diff_by_vm) / \
                  (vm_alpha_by_vm + vm_beta_by_vm) ** 2
    dmdt = (vm_inf_by_V - vm_inf_by_vm) / vm_tau_by_V / vm_inf_diff

    # vh
    vh_alpha_by_V = 0.128 * np.exp((17. - V + NaK_th) / 18.)
    vh_beta_by_V = 4. / (1. + np.exp((40. - V + NaK_th) / 5.))
    vh_inf_by_V = vh_alpha_by_V / (vh_alpha_by_V + vh_beta_by_V)
    vh_tau_by_V = 1. / phi_h / (vh_alpha_by_V + vh_beta_by_V)
    vh_alpha_by_vh = 0.128 * np.exp((17. - vh + NaK_th) / 18.)
    vh_beta_by_vh = 4. / (1. + np.exp((40. - vh + NaK_th) / 5.))
    vh_inf_by_vh = vh_alpha_by_vh / (vh_alpha_by_vh + vh_beta_by_vh)
    alpha_diff_by_vh = -0.128 * bp.ops.exp((17 - vh + NaK_th) / 18) / 18
    beta_diff_by_vh = 0.8 * bp.ops.exp((40 - vh + NaK_th) / 5) / \
                      (1 + bp.ops.exp((40 - vh + NaK_th) / 5)) ** 2
    vh_inf_diff = (alpha_diff_by_vh * vh_beta_by_vh - vh_alpha_by_vh * beta_diff_by_vh) / \
                  (vh_alpha_by_vh + vh_beta_by_vh) ** 2
    dhdt = (vh_inf_by_V - vh_inf_by_vh) / vh_tau_by_V / vh_inf_diff

    # vn
    vn_alpha_by_V = 0.032 * (V - NaK_th - 15.) / (1. - np.exp((15. - V + NaK_th) / 5.))
    vn_beta_by_V = b * np.exp((10. - V + NaK_th) / 40.)
    vn_tau_by_V = 1 / phi_n / (vn_alpha_by_V + vn_beta_by_V)
    vn_inf_by_V = vn_alpha_by_V / (vn_alpha_by_V + vn_beta_by_V)
    alpha_vn = 0.032 * (vn - NaK_th - 15.) / (1. - np.exp(-(vn - NaK_th - 15.) / 5.))
    beta_vn = b * np.exp(-(vn - NaK_th - 10.) / 40.)
    vn_inf_by_vn = alpha_vn / (alpha_vn + beta_vn)
    alpha_diff_by_vn = (-0.032 * (bp.ops.exp((15 - vn + NaK_th) / 5) - 1) +
                        0.0064 * (15 - vn + NaK_th) * bp.ops.exp((15 - vn + NaK_th) / 5)) / \
                       (bp.ops.exp((15 - vn + NaK_th) / 5) - 1) ** 2
    beta_diff_by_vn = -b / 40 * bp.ops.exp((10 - vn + NaK_th) / 40)
    vn_inf_diff = (alpha_diff_by_vn * beta_vn - alpha_vn * beta_diff_by_vn) / \
                  (alpha_vn + beta_vn) ** 2
    dndt = (vn_inf_by_V - vn_inf_by_vn) / vn_tau_by_V / vn_inf_diff

    # vp
    exp10 = np.exp((- vp + p_half + IT_th) / p_k)
    vp_inf_by_vp = 1. / (1. + exp10)
    vp_inf_diff = exp10 / p_k / (1 + exp10) ** 2
    vp_inf_by_V = 1. / (1. + np.exp((- V + p_half + IT_th) / p_k))
    vp_tau_by_V = 3. + 1. / (np.exp((V + 27. - IT_th) / 10.) + np.exp(-(V + 102. - IT_th) / 15.))
    dpdt = phi_p * (vp_inf_by_V - vp_inf_by_vp) / vp_tau_by_V / vp_inf_diff

    # vq
    vq_inf_by_vq = 1. / (1. + np.exp((-80 - vq + IT_th) / -5))
    vq_inf_diff = bp.ops.exp((80 + vq - IT_th) / 5) / -5 / \
                  (1 + bp.ops.exp((80 + vq - IT_th) / 5)) ** 2
    vq_inf_by_V = 1. / (1. + np.exp((-80 - V + IT_th) / -5))
    vq_tau_by_V = 85. + 1. / (np.exp((V + 48. - IT_th) / 4.) +
                              np.exp(-(V + 407. - IT_th) / 50.))
    dqdt = phi_q * (vq_inf_by_V - vq_inf_by_vq) / vq_tau_by_V / vq_inf_diff

    # V
    INa = g_Na * vm_inf_by_vm ** 3 * vh_inf_by_vh * (V - E_Na)
    IK = g_K * vn_inf_by_vn ** 4 * (V - E_KL)
    IT = g_T * vp_inf_by_vp ** 2 * vq_inf_by_vq * (V - E_T)
    IL = g_L * (V - E_L)
    IKL = g_KL * (V - E_KL)
    Icur = INa + IK + IT + IL + IKL
    dvdt = (-Icur + Isyn * V_factor) / C
    return dvdt, dmdt, dhdt, dndt, dpdt, dqdt

  def update(self, _t, _i, _dt):
    V, self.vm, self.vh, self.vn, self.vp, self.vq = self.integral(
      V=self.V, vm=self.vm, vh=self.vh, vn=self.vn, vp=self.vp, vq=self.vq,
      t=_t, Isyn=self.input, E_KL=self.E_KL, E_L=self.E_L, g_KL=self.g_KL,
      g_L=self.g_L, g_T=self.g_T, NaK_th=self.NaK_th, b=self.b, IT_th=self.IT_th)
    self.spike = np.logical_and(V >= Vth, self.V < Vth)
    self.V = V
    self.input[:] = 0.

  def reset(self, Vr):
    self.V[:] = Vr
    self.vm[:] = Vr
    self.vh[:] = Vr
    self.vn[:] = Vr
    self.vp[:] = Vr
    self.vq[:] = Vr

    self.spike[:] = False
    self.input[:] = 0.


@nb.njit
def get_channel_currents(V, y, z, g_T, NaK_th, b, E_KL, IT_th):
  # m channel
  t1 = 13. - V + NaK_th
  t1_exp = np.exp(t1 / 4.)
  m_alpha_by_V = 0.32 * t1 / (t1_exp - 1.)  # \alpha_m(V)
  t2 = V - 40. - NaK_th
  t2_exp = np.exp(t2 / 5.)
  m_beta_by_V = 0.28 * t2 / (t2_exp - 1.)  # \beta_m(V)
  m_inf_by_V = m_alpha_by_V / (m_alpha_by_V + m_beta_by_V)  # \m_{\infty}(V)

  # h channel
  h_alpha_by_y = 0.128 * np.exp((17. - y + NaK_th) / 18.)  # \alpha_h(y)
  t3 = np.exp((40. - y + NaK_th) / 5.)
  h_beta_by_y = 4. / (t3 + 1.)  # \beta_h(y)
  h_inf_by_y = h_alpha_by_y / (h_alpha_by_y + h_beta_by_y)  # \h_{\infty}(y)

  # n channel
  t4 = 15. - y + NaK_th
  n_alpha_by_y = 0.032 * t4 / (np.exp(t4 / 5.) - 1.)  # \alpha_n(y)
  n_beta_by_y = b * np.exp((10. - y + NaK_th) / 40.)  # \beta_n(y)
  n_inf_by_y = n_alpha_by_y / (n_alpha_by_y + n_beta_by_y)  # n_{\infty}(y)

  # p channel
  p_inf_by_y = 1. / (1. + np.exp((p_half - y + IT_th) / p_k))  # p_{\infty}(y)
  q_inf_by_z = 1. / (1. + np.exp((q_half - z + IT_th) / q_k))  # q_{\infty}(z)

  # currents
  gK = g_K * n_inf_by_y ** 4  # gK
  gNa = g_Na * m_inf_by_V ** 3 * h_inf_by_y  # gNa
  gT = g_T * p_inf_by_y * p_inf_by_y * q_inf_by_z  # gT
  INa = gNa * (V - E_Na)
  IK = gK * (V - E_KL)
  IT = gT * (V - E_T)
  return INa, IK, IT


class ReducedTRN(bp.NeuGroup):
  @staticmethod
  def derivative(*args, **kwargs):
    raise NotImplementedError

  def __init__(self, size, **kwargs):
    super(ReducedTRN, self).__init__(size=size, **kwargs)

    self.IT_th = -3.
    self.b = 0.14
    self.g_T = 2.0
    self.g_L = 0.02
    self.E_L = -70.
    self.g_KL = 0.005
    self.E_KL = -95.
    self.NaK_th = -55.
    self.rho_p = 0.35

    num = bp.size2len(size)
    self.V = bp.ops.zeros(num)
    self.y = bp.ops.zeros(num)
    self.z = bp.ops.zeros(num)
    self.spike = bp.ops.zeros(num, dtype=bool)
    self.input = bp.ops.zeros(num)

  @staticmethod
  @nb.njit
  def f_optimize_v(V, g_T, Iext, b, NaK_th, g_L, E_L, g_KL, E_KL, IT_th):
    INa, IK, IT = get_channel_currents(V, y=V, z=V, g_T=g_T, NaK_th=NaK_th,
                                       b=b, E_KL=E_KL, IT_th=IT_th)
    IL = g_L * (V - E_L)
    IKL = g_KL * (V - E_KL)
    dxdt = -INa - IK - IT - IL - IKL + V_factor * Iext
    return dxdt

  @staticmethod
  def get_resting_potential(g_T, Iext, b, NaK_th, g_L, E_L, g_KL, E_KL, IT_th):
    vs = np.arange(-100, 55, 0.01)
    roots = bp.analysis.find_root_of_1d(
      ReducedTRN.f_optimize_v, f_points=vs,
      args=(g_T, Iext, b, NaK_th, g_L, E_L, g_KL, E_KL, IT_th))
    return roots

  @staticmethod
  def suggest_gL(Vr, g_T, Iext, b, NaK_th, E_L, g_KL, E_KL, IT_th):
    INa, IK, IT = get_channel_currents(V=Vr, y=Vr, z=Vr, g_T=g_T,
                                       NaK_th=NaK_th, b=b, E_KL=E_KL, IT_th=IT_th)
    IKL = g_KL * (Vr - E_KL)
    gL = (-INa - IK - IT - IKL + Iext) / (Vr - E_L)
    return gL

  @staticmethod
  def suggest_gKL(Vr, g_T, Iext, b, NaK_th, g_L, E_L, E_KL, IT_th):
    INa, IK, IT = get_channel_currents(V=Vr, y=Vr, z=Vr, g_T=g_T,
                                       NaK_th=NaK_th, b=b, E_KL=E_KL, IT_th=IT_th)
    IL = g_L * (Vr - E_L)
    gKL = (-INa - IK - IT - IL + Iext) / (Vr - E_KL)
    return gKL

  def reset(self, Vr):
    self.V[:] = Vr
    self.y[:] = Vr
    self.z[:] = Vr
    self.spike[:] = False
    self.input[:] = 0.


class ReducedTRNv1(ReducedTRN):
  """Reduced TRN version 1.

  In this reduced TRN neuron model, we make two reductions:

  1. group n, h, p channels.
  2. group V, and m channel.

  """
  target_backend = 'general'

  def suggest_gKL(self, Vr, Iext, g_T=None):
    return super(ReducedTRNv1, self).suggest_gKL(Vr, g_T=self.g_T if g_T is None else g_T, Iext=Iext,
                                                 b=self.b, NaK_th=self.NaK_th, g_L=self.g_L,
                                                 E_L=self.E_L, E_KL=self.E_KL, IT_th=self.IT_th)

  @staticmethod
  def coefficient(V, y, z, t, b, g_T, g_L, g_KL, E_KL, IT_th, NaK_th):
    # m channel
    t1 = 13. - V + NaK_th
    t1_exp = bp.ops.exp(t1 / 4.)
    m_alpha_by_V = 0.32 * t1 / (t1_exp - 1.)  # \alpha_m(V)
    m_alpha_by_V_diff = (-0.32 * (t1_exp - 1.) + 0.08 * t1 * t1_exp) / (t1_exp - 1.) ** 2  # \alpha_m'(V)
    t2 = V - 40. - NaK_th
    t2_exp = bp.ops.exp(t2 / 5.)
    m_beta_by_V = 0.28 * t2 / (t2_exp - 1.)  # \beta_m(V)
    m_beta_by_V_diff = (0.28 * (t2_exp - 1) - 0.056 * t2 * t2_exp) / (t2_exp - 1) ** 2  # \beta_m'(V)
    m_inf_by_V = m_alpha_by_V / (m_alpha_by_V + m_beta_by_V)  # \m_{\infty}(V)
    m_inf_by_V_diff = (m_alpha_by_V_diff * m_beta_by_V - m_alpha_by_V * m_beta_by_V_diff) / \
                      (m_alpha_by_V + m_beta_by_V) ** 2  # \m_{\infty}'(V)

    # h channel
    h_alpha_by_V = 0.128 * bp.ops.exp((17. - V + NaK_th) / 18.)  # \alpha_h(V)
    h_beta_by_V = 4. / (bp.ops.exp((40. - V + NaK_th) / 5.) + 1.)  # \beta_h(V)
    h_alpha_by_y = 0.128 * bp.ops.exp((17. - y + NaK_th) / 18.)  # \alpha_h(y)
    t3 = bp.ops.exp((40. - y + NaK_th) / 5.)
    h_beta_by_y = 4. / (t3 + 1.)  # \beta_h(y)
    h_beta_by_y_diff = 0.8 * t3 / (1 + t3) ** 2  # \beta_h'(y)
    h_inf_by_y = h_alpha_by_y / (h_alpha_by_y + h_beta_by_y)  # h_{\infty}(y)
    h_alpha_by_y_diff = - h_alpha_by_y / 18.  # \alpha_h'(y)
    h_inf_by_y_diff = (h_alpha_by_y_diff * h_beta_by_y - h_alpha_by_y * h_beta_by_y_diff) / \
                      (h_beta_by_y + h_alpha_by_y) ** 2  # h_{\infty}'(y)

    # n channel
    t4 = (15. - V + NaK_th)
    n_alpha_by_V = 0.032 * t4 / (bp.ops.exp(t4 / 5.) - 1.)  # \alpha_n(V)
    n_beta_by_V = b * bp.ops.exp((10. - V + NaK_th) / 40.)  # \beta_n(V)
    t5 = (15. - y + NaK_th)
    t5_exp = bp.ops.exp(t5 / 5.)
    n_alpha_by_y = 0.032 * t5 / (t5_exp - 1.)  # \alpha_n(y)
    t6 = bp.ops.exp((10. - y + NaK_th) / 40.)
    n_beta_y = b * t6  # \beta_n(y)
    n_inf_by_y = n_alpha_by_y / (n_alpha_by_y + n_beta_y)  # n_{\infty}(y)
    n_alpha_by_y_diff = (0.0064 * t5 * t5_exp - 0.032 * (t5_exp - 1.)) / (t5_exp - 1.) ** 2  # \alpha_n'(y)
    n_beta_by_y_diff = -n_beta_y / 40  # \beta_n'(y)
    n_inf_by_y_diff = (n_alpha_by_y_diff * n_beta_y - n_alpha_by_y * n_beta_by_y_diff) / \
                      (n_alpha_by_y + n_beta_y) ** 2  # n_{\infty}'(y)

    # p channel
    t7 = bp.ops.exp((p_half - y + IT_th) / p_k)
    p_inf_by_y = 1. / (1. + t7)  # p_{\infty}(y)
    p_inf_by_y_diff = t7 / p_k / (1. + t7) ** 2  # p_{\infty}'(y)

    # p channel
    t8 = bp.ops.exp((q_half - z + IT_th) / q_k)
    q_inf_by_z = 1. / (1. + t8)  # q_{\infty}(z)

    # ----
    #  x
    # ----

    gNa = g_Na * m_inf_by_V ** 3 * h_inf_by_y  # gNa
    gK = g_K * n_inf_by_y ** 4  # gK
    gT = g_T * p_inf_by_y * p_inf_by_y * q_inf_by_z  # gT
    FV = gNa + gK + gT + g_L + g_KL  # dF/dV
    Fvm = 3 * g_Na * h_inf_by_y * (V - E_Na) * m_inf_by_V * m_inf_by_V * m_inf_by_V_diff  # dF/dvm

    Fvh = g_Na * m_inf_by_V ** 3 * (V - E_Na) * h_inf_by_y_diff  # dF/dvh
    Fvn = 4 * g_K * (V - E_KL) * n_inf_by_y ** 3 * n_inf_by_y_diff  # dF/dvn
    Fvp = 2 * g_T * p_inf_by_y * q_inf_by_z * (V - E_T) * p_inf_by_y_diff

    # rho_h = Fvh / (Fvh + Fvn + Fvp)

    return Fvh, Fvn, Fvp, Fvm, FV

  @staticmethod
  def derivative(V, y, z, t, Isyn, b, rho_p, g_T, g_L, g_KL, E_L, E_KL, IT_th, NaK_th):
    # m channel
    t1 = 13. - V + NaK_th
    t1_exp = bp.ops.exp(t1 / 4.)
    m_alpha_by_V = 0.32 * t1 / (t1_exp - 1.)  # \alpha_m(V)
    m_alpha_by_V_diff = (-0.32 * (t1_exp - 1.) + 0.08 * t1 * t1_exp) / (t1_exp - 1.) ** 2  # \alpha_m'(V)
    t2 = V - 40. - NaK_th
    t2_exp = bp.ops.exp(t2 / 5.)
    m_beta_by_V = 0.28 * t2 / (t2_exp - 1.)  # \beta_m(V)
    m_beta_by_V_diff = (0.28 * (t2_exp - 1) - 0.056 * t2 * t2_exp) / (t2_exp - 1) ** 2  # \beta_m'(V)
    m_tau_by_V = 1. / phi_m / (m_alpha_by_V + m_beta_by_V)  # \tau_m(V)
    m_inf_by_V = m_alpha_by_V / (m_alpha_by_V + m_beta_by_V)  # \m_{\infty}(V)
    m_inf_by_V_diff = (m_alpha_by_V_diff * m_beta_by_V - m_alpha_by_V * m_beta_by_V_diff) / \
                      (m_alpha_by_V + m_beta_by_V) ** 2  # \m_{\infty}'(V)

    # h channel
    h_alpha_by_V = 0.128 * bp.ops.exp((17. - V + NaK_th) / 18.)  # \alpha_h(V)
    h_beta_by_V = 4. / (bp.ops.exp((40. - V + NaK_th) / 5.) + 1.)  # \beta_h(V)
    h_inf_by_V = h_alpha_by_V / (h_alpha_by_V + h_beta_by_V)  # h_{\infty}(V)
    h_tau_by_V = 1. / phi_h / (h_alpha_by_V + h_beta_by_V)  # \tau_h(V)
    h_alpha_by_y = 0.128 * bp.ops.exp((17. - y + NaK_th) / 18.)  # \alpha_h(y)
    t3 = bp.ops.exp((40. - y + NaK_th) / 5.)
    h_beta_by_y = 4. / (t3 + 1.)  # \beta_h(y)
    h_beta_by_y_diff = 0.8 * t3 / (1 + t3) ** 2  # \beta_h'(y)
    h_inf_by_y = h_alpha_by_y / (h_alpha_by_y + h_beta_by_y)  # h_{\infty}(y)
    h_alpha_by_y_diff = - h_alpha_by_y / 18.  # \alpha_h'(y)
    h_inf_by_y_diff = (h_alpha_by_y_diff * h_beta_by_y - h_alpha_by_y * h_beta_by_y_diff) / \
                      (h_beta_by_y + h_alpha_by_y) ** 2  # h_{\infty}'(y)

    # n channel
    t4 = (15. - V + NaK_th)
    n_alpha_by_V = 0.032 * t4 / (bp.ops.exp(t4 / 5.) - 1.)  # \alpha_n(V)
    n_beta_by_V = b * bp.ops.exp((10. - V + NaK_th) / 40.)  # \beta_n(V)
    n_tau_by_V = 1. / (n_alpha_by_V + n_beta_by_V) / phi_n  # \tau_n(V)
    n_inf_by_V = n_alpha_by_V / (n_alpha_by_V + n_beta_by_V)  # n_{\infty}(V)
    t5 = (15. - y + NaK_th)
    t5_exp = bp.ops.exp(t5 / 5.)
    n_alpha_by_y = 0.032 * t5 / (t5_exp - 1.)  # \alpha_n(y)
    t6 = bp.ops.exp((10. - y + NaK_th) / 40.)
    n_beta_y = b * t6  # \beta_n(y)
    n_inf_by_y = n_alpha_by_y / (n_alpha_by_y + n_beta_y)  # n_{\infty}(y)
    n_alpha_by_y_diff = (0.0064 * t5 * t5_exp - 0.032 * (t5_exp - 1.)) / (t5_exp - 1.) ** 2  # \alpha_n'(y)
    n_beta_by_y_diff = -n_beta_y / 40  # \beta_n'(y)
    n_inf_by_y_diff = (n_alpha_by_y_diff * n_beta_y - n_alpha_by_y * n_beta_by_y_diff) / \
                      (n_alpha_by_y + n_beta_y) ** 2  # n_{\infty}'(y)

    # p channel
    p_inf_by_V = 1. / (1. + bp.ops.exp((p_half - V + IT_th) / p_k))  # p_{\infty}(V)
    p_tau_by_V = (3 + 1. / (bp.ops.exp((V + 27. - IT_th) / 10.) +
                            bp.ops.exp(-(V + 102. - IT_th) / 15.))) / phi_p  # \tau_p(V)
    t7 = bp.ops.exp((p_half - y + IT_th) / p_k)
    p_inf_by_y = 1. / (1. + t7)  # p_{\infty}(y)
    p_inf_by_y_diff = t7 / p_k / (1. + t7) ** 2  # p_{\infty}'(y)

    # p channel
    q_inf_by_V = 1. / (1. + bp.ops.exp((q_half - V + IT_th) / q_k))  # q_{\infty}(V)
    t8 = bp.ops.exp((q_half - z + IT_th) / q_k)
    q_inf_by_z = 1. / (1. + t8)  # q_{\infty}(z)
    q_inf_diff_z = t8 / q_k / (1. + t8) ** 2  # q_{\infty}'(z)
    q_tau_by_V = (85. + 1 / (bp.ops.exp((V + 48. - IT_th) / 4.) +
                             bp.ops.exp(-(V + 407. - IT_th) / 50.))) / phi_q  # \tau_q(V)

    # ----
    #  x
    # ----

    gNa = g_Na * m_inf_by_V ** 3 * h_inf_by_y  # gNa
    gK = g_K * n_inf_by_y ** 4  # gK
    gT = g_T * p_inf_by_y * p_inf_by_y * q_inf_by_z  # gT
    FV = gNa + gK + gT + g_L + g_KL  # dF/dV
    Fm = 3 * g_Na * h_inf_by_y * (V - E_Na) * m_inf_by_V * m_inf_by_V * m_inf_by_V_diff  # dF/dvm
    t9 = C / m_tau_by_V
    t10 = FV + Fm
    t11 = t9 + FV
    rho_V = (t11 - bp.ops.sqrt(bp.ops.maximum(t11 ** 2 - 4 * t9 * t10, 0.))) / 2 / t10  # rho_V
    INa = gNa * (V - E_Na)
    IK = gK * (V - E_KL)
    IT = gT * (V - E_T)
    IL = g_L * (V - E_L)
    IKL = g_KL * (V - E_KL)
    Iext = V_factor * Isyn
    dVdt = rho_V * (-INa - IK - IT - IL - IKL + Iext) / C

    # ----
    #  y
    # ----

    Fvh = g_Na * m_inf_by_V ** 3 * (V - E_Na) * h_inf_by_y_diff  # dF/dvh
    Fvn = 4 * g_K * (V - E_KL) * n_inf_by_y ** 3 * n_inf_by_y_diff  # dF/dvn
    f4 = Fvh + Fvn
    rho_h = (1 - rho_p) * Fvh / f4
    rho_n = (1 - rho_p) * Fvn / f4
    fh = (h_inf_by_V - h_inf_by_y) / h_tau_by_V / h_inf_by_y_diff
    fn = (n_inf_by_V - n_inf_by_y) / n_tau_by_V / n_inf_by_y_diff
    fp = (p_inf_by_V - p_inf_by_y) / p_tau_by_V / p_inf_by_y_diff
    dydt = rho_h * fh + rho_n * fn + rho_p * fp

    # ----
    #  z
    # ----

    dzdt = (q_inf_by_V - q_inf_by_z) / q_tau_by_V / q_inf_diff_z

    return dVdt, dydt, dzdt

  def __init__(self, size, **kwargs):
    self.integral = bp.odeint(f=self.derivative)
    super(ReducedTRNv1, self).__init__(size, **kwargs)

  def update(self, _t):
    V, self.y, self.z = self.integral(self.V, self.y, self.z, _t, self.input,
                                      b=self.b, rho_p=self.rho_p, g_T=self.g_T,
                                      g_L=self.g_L, g_KL=self.g_KL, E_L=self.E_L,
                                      E_KL=self.E_KL, IT_th=self.IT_th, NaK_th=self.NaK_th)
    self.spike = (self.V < Vth) * (V >= Vth)
    self.V = V
    self.input[:] = 0.


def constant_inputs(model, Vr=-75., inputs=(0.01, 0.015, 0.02), rho_p=0.6, b=0.14,
                    g_L=0.05, E_L=-77., E_KL=-95, g_KL=None, g_T=2.0, NaK_th=-55,
                    IT_th=-3., init_vr=None, report=True, duration=5e3):
  # set parameters
  inputs = np.array(inputs)
  group = model(size=len(inputs), monitors=['V'])
  group.rho_p = rho_p
  group.b = b
  group.g_L = g_L
  group.E_L = E_L
  group.E_KL = E_KL
  group.g_T = g_T
  group.IT_th = IT_th
  group.NaK_th = NaK_th
  if g_KL is None:
    g_KL = group.suggest_gKL(Vr, g_T=g_T, Iext=0.)
  group.g_KL = g_KL

  # initialize neuron group
  if init_vr is None:
    init_vr = Vr
  group.reset(init_vr)
  group.run(duration=duration, inputs=['input', inputs], report=report)

  fig, gs = bp.visualize.get_figure(row_num=int(np.ceil(len(inputs) / 4)),
                                    col_num=min([4, len(inputs)]),
                                    row_len=3,
                                    col_len=4.5)
  for i in range(len(inputs)):
    fig.add_subplot(gs[i // 4, i % 4])
    plt.plot(group.mon.ts, group.mon.V[:, i])
    plt.title('I={:.4f}'.format(inputs[i]))
    plt.xlabel('time (ms)')
  title = r'b=%.2f,$\rho_p$=%.2f,$g_T$=%.2f,$g_L$=%.3f,$E_L$=%.1f,$g_{KL}$=%.5f,' \
          r'$E_{KL}$=%.1f,Vr=%.1f,init=%.1f' % (b, rho_p, g_T, g_L, E_L, g_KL, E_KL, Vr, init_vr)
  print(title.replace('$', '').replace('\\', '').replace('{', '').replace('}', ''))
  plt.suptitle(title)
  plt.show()


def jacobian(f, V, y, z, input, b, rho_p, g_T, g_L, g_KL, E_L, E_KL, IT_th, NaK_th):
  """Adopted from https://pytorch.org/docs/stable/_modules/torch/autograd/functional.html#jacobian
  """
  create_graph = False
  strict = False
  is_inputs_tuple = True
  inputs = (torch.tensor(V), torch.tensor(y), torch.tensor(z))
  inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

  outputs = f(*inputs, 0., input, b, rho_p, g_T, g_L, g_KL, E_L, E_KL, IT_th, NaK_th)
  is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "jacobian")
  _check_requires_grad(outputs, "outputs", strict=strict)

  jacobian = tuple()
  for i, out in enumerate(outputs):
    # mypy complains that expression and variable have different types due to the empty list
    jac_i: Tuple[List[torch.Tensor]] = tuple([] for _ in range(len(inputs)))  # type: ignore
    for j in range(out.nelement()):
      vj = _autograd_grad((out.reshape(-1)[j],), inputs, retain_graph=True, create_graph=create_graph)
      for el_idx, (jac_i_el, vj_el, inp_el) in enumerate(zip(jac_i, vj, inputs)):
        if vj_el is not None:
          if strict and create_graph and not vj_el.requires_grad:
            msg = ("The jacobian of the user-provided function is "
                   "independent of input {}. This is not allowed in "
                   "strict mode when create_graph=True.".format(i))
            raise RuntimeError(msg)
          jac_i_el.append(vj_el)
        else:
          if strict:
            msg = ("Output {} of the user-provided function is "
                   "independent of input {}. This is not allowed in "
                   "strict mode.".format(i, el_idx))
            raise RuntimeError(msg)
          jac_i_el.append(torch.zeros_like(inp_el))
    jacobian += (tuple(torch.stack(jac_i_el, dim=0).view(out.size() + inputs[el_idx].size())
                       for (el_idx, jac_i_el) in enumerate(jac_i)),)
  jacobian = _grad_postprocess(jacobian, create_graph)
  results = _tuple_postprocess(jacobian, (is_outputs_tuple, is_inputs_tuple))
  return torch.tensor(results).numpy()


def roots_stability(f, g_L, E_L, E_KL, Vr, b=0.14, rho_p=0.5, g_T=2.,
                    IT_th=-3., NaK_th=-55., Iext=0., returns=None):
  g_KL = ReducedTRN.suggest_gKL(Vr, g_T=g_T, Iext=0., b=b, NaK_th=NaK_th, g_L=g_L, E_L=E_L, E_KL=E_KL, IT_th=IT_th)
  roots = ReducedTRN.get_resting_potential(g_T=g_T, Iext=Iext, b=b, NaK_th=NaK_th, g_L=g_L,
                                           E_L=E_L, g_KL=g_KL, E_KL=E_KL, IT_th=IT_th)
  stabilises = []
  for root in roots:
    root = np.array(root, dtype=np.float32)
    jac = jacobian(f, root, root, root, input=Iext, b=b, rho_p=rho_p, g_T=g_T, g_L=g_L,
                   g_KL=g_KL, E_L=E_L, E_KL=E_KL, IT_th=IT_th, NaK_th=NaK_th)
    st = bp.analysis.stability_analysis(jac)
    print(st)
    stabilises.append(st)

  title = f'g_T={g_T:.2f},Vr={Vr:.3f},Iext={Iext:4f},g_KL={g_KL:5f}'
  if returns is not None:
    returns[title] = (roots, stabilises)
  return title, roots, stabilises
