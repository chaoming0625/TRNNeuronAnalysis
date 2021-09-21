# -*- coding: utf-8 -*-

import sys

import brainpy as bp
import numpy as np

sys.path.append('../')

from src import neus_and_syns

bp.backend.set('pytorch', dt=0.005)
bp.integrators.set_default_odeint('rk4')


def effect_of_Iext(I=0.):
  Vr = -66
  P = bp.tools.DictPlus(g_Na=100., g_K=10., b=0.5, rho_p=0., IT_th=-3.,
                        NaK_th=-55., E_KL=-100, g_L=0.06, E_L=-70, g_T=2.25)
  trn = neus_and_syns.ReducedTRNv1(1, monitors=['V'])
  trn.g_Na = P['g_Na']
  trn.g_K = P['g_K']
  trn.b = P['b']
  trn.rho_p = P['rho_p']
  trn.IT_th = P['IT_th']
  trn.NaK_th = P['NaK_th']
  trn.E_KL = P['E_KL']
  trn.g_L = P['g_L']
  trn.E_L = P['E_L']
  trn.g_T = P['g_T']
  trn.g_KL = trn.suggest_gKL(Vr=Vr, Iext=0.)
  P['g_KL'] = trn.g_KL

  roots = trn.get_resting_potential(g_T=trn.g_T, Iext=I, b=trn.b, NaK_th=trn.NaK_th,
                                    g_L=trn.g_L, E_L=trn.E_L, g_KL=trn.g_KL,
                                    E_KL=trn.E_KL, IT_th=trn.IT_th)
  for root in roots:
    root = np.array(root)
    jac = neus_and_syns.jacobian(
      f=neus_and_syns.ReducedTRNv1.derivative,
      V=root, y=root, z=root, input=0., b=P.b, rho_p=P.rho_p, g_T=P.g_T, g_L=P.g_L,
      E_L=P.E_L, E_KL=P.E_KL, g_KL=trn.g_KL, IT_th=P.IT_th, NaK_th=P.NaK_th)
    eigenvalues = np.linalg.eigvals(np.array(jac))
    print('root: ', root)
    print('eigenvalues: ', eigenvalues)
    print('stability: ', bp.analysis.stability_analysis(jac))
    print('-' * 20)


if __name__ == '__main__':
  # effect_of_Iext(I=0.)
  # effect_of_Iext(I=-0.03)
  effect_of_Iext(I=-0.05)
  effect_of_Iext(I=-0.06)

