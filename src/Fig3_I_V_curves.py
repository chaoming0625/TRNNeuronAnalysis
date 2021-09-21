# -*- coding: utf-8 -*-

import brainpy as bp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numba import njit

bp.backend.set('numba', dt=0.01)

import sys

sys.path.append('../')

from src import neus_and_syns


def V_nuclline(I=-0.05,  Vr=-66,):
  bp.backend.set('numba')
  derivative = njit(neus_and_syns.ReducedTRNv1.derivative)

  def V_value(V, z, Isyn, b, g_T, g_L, g_KL, E_L, E_KL, NaK_th, IT_th=-3):
    dvdt, dydt, dzdt = derivative(V=V, y=V, z=z, Isyn=Isyn, b=b, g_T=g_T, g_L=g_L,
                                  g_KL=g_KL, E_L=E_L, E_KL=E_KL, IT_th=IT_th,
                                  NaK_th=NaK_th, rho_p=0., t=0., )
    return dvdt

  @njit
  def solve_V(V, z, Isyn, b, g_T, g_L, g_KL, E_L, E_KL, NaK_th, IT_th=-3):
    dvdt, dydt, dzdt = derivative(V=V, y=V, z=z, Isyn=Isyn, b=b, g_T=g_T, g_L=g_L, g_KL=g_KL, E_L=E_L,
                                  E_KL=E_KL, IT_th=IT_th, NaK_th=NaK_th, rho_p=0., t=0., )
    return dvdt

  sns.set_theme(font_scale=1.5)
  analysis = True

  g_T, NaK_th, g_L, E_L, E_KL, b, rho_p, = \
    2.25, -55, 0.06, -70., -100, 0.5, 0.,

  trn = neus_and_syns.ReducedTRNv1(1, )
  trn.NaK_th = NaK_th
  trn.rho_p = rho_p
  trn.E_KL = E_KL
  trn.g_T = g_T
  trn.g_L = g_L
  trn.E_L = E_L
  trn.b = b
  trn.g_KL = trn.suggest_gKL(Vr=Vr, Iext=0., )

  print(trn.g_KL)

  fig, gs = bp.visualize.get_figure(1, 1, 4, 5)
  fig.add_subplot(gs[0, 0])

  # I-V relation
  vs = np.arange(-90, -15, 0.1)
  for z in [-75, -70, -65, -60,]:
    dv = V_value(V=vs, z=z, Isyn=I, b=b, g_T=g_T, g_L=g_L,
                 g_KL=trn.g_KL, E_L=E_L, E_KL=E_KL, NaK_th=NaK_th)
    # plt.plot(vs, dv, label=f'z = {z} mV', lw=3)
    plt.plot(vs, dv, lw=3)

  if analysis:
    # roots
    z2 = -65.
    roots = bp.analysis.find_root_of_1d(solve_V, f_points=vs,
                                        args=(z2, I, b, g_T, g_L,
                                              trn.g_KL, E_L, E_KL, NaK_th))
    bp.backend.set('pytorch')
    results = {}
    for root in roots:
      root = np.array(root)
      jac = neus_and_syns.jacobian(trn.derivative, root, root, np.array(z2), input=I, b=b,
                                   rho_p=rho_p, g_T=g_T, g_L=g_L, g_KL=trn.g_KL,
                                   E_L=E_L, E_KL=E_KL, IT_th=-3., NaK_th=NaK_th)
      res = bp.analysis.stability_analysis(jac)
      if res not in results: results[res] = []
      results[res].append(root[()])
    for type_, vals in results.items():
      plt.plot(vals, [0.] * len(vals), '.', label=type_, markersize=15,
               **bp.analysis.plot_scheme[type_])

  # visualize
  plt.axhline(0)
  plt.xlim(vs[0] - 0.1, vs[-1] + 0.1)
  plt.ylim(-2.5, 2.5)
  plt.ylabel(r'$I(V, z)$ ($\mu \mathrm{A/cm^2}$)')
  plt.xlabel('V (mV)')
  plt.title(r'$\mathrm{I_{syn}}=%.2f\, \mathrm{\mu A/cm^2}$' % I)
  lg = plt.legend(loc='best', fontsize=14)
  lg.get_frame().set_alpha(0.2)
  plt.show()


def V_z_nuclline():
  derivative = njit(neus_and_syns.ReducedTRNv1.derivative)

  @njit
  def solve_V(V, z, Isyn, b, g_T, g_L, g_KL, E_L, E_KL, NaK_th, IT_th=-3):
    dvdt, dydt, dzdt = derivative(V=V, y=V, z=z, Isyn=Isyn, b=b, g_T=g_T, g_L=g_L,
                                  g_KL=g_KL, E_L=E_L, E_KL=E_KL, IT_th=IT_th, NaK_th=NaK_th,
                                  rho_p=0., t=0., )
    return dvdt

  g_T, NaK_th, g_L, E_L, E_KL, b, rho_p, Vr, I = \
    2.25, -55, 0.06, -70., -100, 0.5, 0., -68, 0.

  trn = neus_and_syns.ReducedTRNv1(1, )
  trn.NaK_th = NaK_th
  trn.rho_p = rho_p
  trn.E_KL = E_KL
  trn.g_T = g_T
  trn.g_L = g_L
  trn.E_L = E_L
  trn.b = b
  trn.g_KL = trn.suggest_gKL(Vr=Vr, Iext=0., )

  zs = np.arange(-80, -20, 0.1)

  sns.set_theme(font_scale=1.5)
  fig, gs = bp.visualize.get_figure(1, 1, 5, 6)
  fig.add_subplot(gs[0, 0])
  for z in zs:
    vs = bp.analysis.find_root_of_1d(
      solve_V, zs, args=(z, I, b, g_T, g_L, trn.g_KL, E_L, E_KL, NaK_th))
    plt.plot([z] * len(vs), vs, 'r.')
  plt.plot(zs, zs, label='z nuclline')

  # plt.xlim(zs[0] - 0.1, zs[-1] + 0.1)
  # plt.ylim(-5, 5)
  # plt.ylabel(r'$I_V$ ($\mu \mathrm{A/cm^2}$)')
  # plt.xlabel('V (mV)')
  plt.legend(loc='best', fontsize=14)
  plt.show()


if __name__ == '__main__':
  V_nuclline()
  # V_z_nuclline()
