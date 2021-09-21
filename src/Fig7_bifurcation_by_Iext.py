# -*- coding: utf-8 -*-

import multiprocessing
import sys
import time

import brainpy as bp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append('../')
from src import neus_and_syns

bp.backend.set('pytorch', dt=0.005)
bp.integrators.set_default_odeint('rk4')


def plot_limit_cycle_by_sim(model_pars, fixed_points, xlabel='Iext',
                            bigger=None, smaller=None,
                            duration=1000, tol=0.001):
  print('plot limit cycle ...')
  bp.backend.set('numba')

  all_xs, all_ys, all_zs, all_ps = [], [], [], []

  # unstable node
  unstable_node = fixed_points[bp.analysis.UNSTABLE_NODE_2D]
  all_xs.extend(unstable_node['V'])
  all_ys.extend(unstable_node['V'])
  all_zs.extend(unstable_node['V'])
  all_ps.extend(unstable_node[xlabel])

  # unstable focus
  unstable_focus = fixed_points[bp.analysis.UNSTABLE_FOCUS_2D]
  all_xs.extend(unstable_focus['V'])
  all_ys.extend(unstable_focus['V'])
  all_zs.extend(unstable_focus['V'])
  all_ps.extend(unstable_focus[xlabel])

  # saddle node
  unstable_focus = fixed_points[bp.analysis.SADDLE_NODE]
  all_xs.extend(unstable_focus['V'])
  all_ys.extend(unstable_focus['V'])
  all_zs.extend(unstable_focus['V'])
  all_ps.extend(unstable_focus[xlabel])

  # format points
  all_xs = np.array(all_xs) + 0.1
  all_ys = np.array(all_ys) + 0.1
  all_zs = np.array(all_zs) + 0.1
  all_ps = np.array(all_ps)

  if bigger is not None:
    idx = all_ps >= bigger
    all_xs = all_xs[idx]
    all_ys = all_ys[idx]
    all_zs = all_zs[idx]
    all_ps = all_ps[idx]
  if smaller is not None:
    idx = all_ps <= smaller
    all_xs = all_xs[idx]
    all_ys = all_ys[idx]
    all_zs = all_zs[idx]
    all_ps = all_ps[idx]

  # initialize neuron group
  length = all_xs.shape[0]
  trn = neus_and_syns.ReducedTRNv1(length, monitors=['V'])
  trn.g_Na = model_pars['g_Na']
  trn.g_K = model_pars['g_K']
  trn.b = model_pars['b']
  trn.rho_p = model_pars['rho_p']
  trn.IT_th = model_pars['IT_th']
  trn.NaK_th = model_pars['NaK_th']
  trn.E_KL = model_pars['E_KL']
  trn.g_L = model_pars['g_L']
  trn.E_L = model_pars['E_L']
  trn.g_T = model_pars['g_T']
  trn.V[:] = all_xs
  trn.y[:] = all_ys
  trn.z[:] = all_zs
  if xlabel == 'Iext':
    inputs = ('input', all_ps)
  else:
    inputs = ()
    if hasattr(trn, xlabel):
      setattr(trn, xlabel, all_ps)
  trn.run(duration=duration, inputs=inputs, report=True)

  # find limit cycles
  limit_cycle_max = []
  limit_cycle_min = []
  p0_limit_cycle = []
  for i in range(length):
    data = trn.mon.V[:, i]
    max_index = bp.analysis.utils.find_indexes_of_limit_cycle_max(data, tol=tol)
    if max_index[0] != -1:
      x_cycle = data[max_index[0]: max_index[1]]
      limit_cycle_max.append(data[max_index[1]])
      limit_cycle_min.append(x_cycle.min())
      p0_limit_cycle.append(all_ps[i])

  limit_cycle_max = np.array(limit_cycle_max)
  limit_cycle_min = np.array(limit_cycle_min)
  p0_limit_cycle = np.array(p0_limit_cycle)
  idx = np.argsort(p0_limit_cycle)
  p0_limit_cycle = p0_limit_cycle[idx]
  limit_cycle_max = np.array(limit_cycle_max)[idx]
  limit_cycle_min = np.array(limit_cycle_min)[idx]

  plt.plot(p0_limit_cycle, limit_cycle_max, label='limit cycle (max)')
  plt.plot(p0_limit_cycle, limit_cycle_min, label='limit cycle (min)')
  plt.fill_between(p0_limit_cycle, limit_cycle_min, limit_cycle_max, alpha=0.1)

  # plt.plot(p0_limit_cycle, limit_cycle_max, '.', label='limit cycle (max)')
  # plt.plot(p0_limit_cycle, limit_cycle_min, '.', label='limit cycle (min)')


def visualize_one_par(stabilities, xlabel='Iext'):
  for s_type, s_val in stabilities.items():
    if len(s_val[xlabel]) > 0:
      plt.scatter(s_val[xlabel], s_val['V'], label=s_type, **bp.analysis.plot_scheme[s_type])


def single_run(P, Iext, duration=2000.):
  bp.backend.set('numba')
  single_trn = neus_and_syns.ReducedTRNv1(1, monitors=['V'])

  single_trn.b = P['b']
  single_trn.rho_p = P['rho_p']
  single_trn.IT_th = P['IT_th']
  single_trn.NaK_th = P['NaK_th']
  single_trn.E_KL = P['E_KL']
  single_trn.E_L = P['E_L']
  single_trn.g_Na = P['g_Na']
  single_trn.g_K = P['g_K']
  single_trn.g_T = P['g_T']
  single_trn.g_L = P['g_L']
  single_trn.g_KL = P['g_KL']
  single_trn.run(duration=duration, inputs=('input', Iext))

  plt.plot(single_trn.mon.ts, single_trn.mon.V[:, 0])

  half = single_trn.mon.V[int(duration * 3 / 4 / bp.backend.get_dt()):, 0]
  print(f'{Iext}, {half.min()}, {half.max()}')

  plt.show()


# %%
def bifurcation_trajectory():
  bp.backend.set('numba')
  g_T, NaK_th, g_L, E_L, E_KL, b, rho_p, Vr, I = \
    2.25, -55, 0.06, -70., -100, 0.5, 0., -66, 0.05

  settings = [
    (-0.025, [0., 0., 0.],),
    (-0.01, [0., 0., 0.],),
    (0.07, [0., 0., 0.], [-65., -65., -58.]),
    (0.14, [0., 0., 0.],),
  ]

  Iexts, inits = [], []
  for s in settings:
    Iexts.extend([s[0] for _ in range(len(s[1:]))])
    inits.extend(s[1:])
  Iexts, inits = np.array(Iexts), np.array(inits)
  trn = neus_and_syns.ReducedTRNv1(len(Iexts), monitors=['V', 'y', 'z'])
  trn.NaK_th = NaK_th
  trn.rho_p = rho_p
  trn.E_KL = E_KL
  trn.g_T = g_T
  trn.g_L = g_L
  trn.E_L = E_L
  trn.b = b
  trn.g_KL = trn.suggest_gKL(Vr=Vr, Iext=0., )
  trn.V[:] = inits[:, 0]
  trn.y[:] = inits[:, 1]
  trn.z[:] = inits[:, 2]
  print(trn.g_KL)
  trn.run(2e3, inputs=('input', Iexts))

  sns.set(font_scale=1.5)
  # sns.set_style("white")

  fig, gs = bp.visualize.get_figure(len(settings), 1, 2, 12)
  i = 0
  for j, I in enumerate(settings):
    fig.add_subplot(gs[j, 0])
    for k in range(len(I) - 1):
      plt.plot(trn.mon.ts, trn.mon.V[:, i],
               # label=f'traj{k}'
               )
      i += 1
    # lg = plt.legend(fontsize=14, loc='upper right')
    # lg.get_frame().set_alpha(0.2)
    plt.xlim(0, 2080)
    plt.ylabel('$\mathrm{I_{syn}}$=%.2f' % I[0] +
               '\n'
               r'$\mathrm{\mu A/cm^2}$')  # horizontalalignment='right'
    plt.annotate(f"{j + 1}", xy=(2040, -20), xytext=(2040, -20),
                 xycoords='data', textcoords='data', va="top",
                 ha="center", fontsize=18,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 bbox=dict(boxstyle='circle', fc="w", ec="k"))
    if j + 1 != len(settings):
      plt.xticks([])
  plt.xlabel('Time (ms)')
  plt.show()
  return trn


# trn = bifurcation_trajectory()


if __name__ == '__main__':
# def effect_of_Iext():

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
  print(trn.g_KL)
  P['g_KL'] = trn.g_KL

  # single_run(P, Iext=0.01, duration=3e3)
  # sys.exit(-1)

  bp.backend.set('pytorch')
  func = neus_and_syns.ReducedTRNv1.derivative

  t0 = time.time()
  manager = multiprocessing.Manager()
  return_dict = manager.dict()
  all_params = []
  I_range = np.arange(-0.15, 0.2001, 0.001)
  for Iext in I_range:
    pars = dict(f=func, g_L=P.g_L, E_L=P.E_L, E_KL=P.E_KL, Vr=Vr, b=P.b,
                rho_p=P.rho_p, g_T=P.g_T, IT_th=P.IT_th, NaK_th=P.NaK_th,
                Iext=Iext, returns=return_dict)
    all_params.append(pars)
  bp.running.process_pool(neus_and_syns.roots_stability, all_params, 7)
  print(f'Used {(time.time() - t0) / 60} min.')

  # 0.059
  # 0.1316

  bursting = [
    # irregular spiking
    (-0.002, -73.6994344890098, 47.764621296318325),
    (-0.003, -73.50591554952794, 47.762083021986776),
    (-0.004, -72.09563889205668, 47.75644596637665),
    (-0.005, -73.65296023863344, 47.77276005867762),
    (-0.0055, -72.22969324298677, 47.753033606944676),

    # regular bursting
    (-0.006, -72.0478407978535, 47.75753586913234),
    (-0.008, -72.08376787956751, 47.76335210895918),
    (-0.010, -72.10960117724784, 47.75521402188428),
    (-0.012, -72.53278513550828, 47.79312974161327),
    (-0.015, -73.86683297682674, 47.81067945913251),
    (-0.020, -73.41430591910961, 47.824143938757175),
    (-0.025, -73.07739971219802, 47.842210233041015),
    (-0.030, -73.59282719107472, 47.85879674969422),
    (-0.035, -74.3396673764637, 47.87953460964065),
    (-0.040, -75.12539682873016, 47.900606894211975),
    (-0.045, -75.90443818146694, 47.919299015128814),
    (-0.050, -76.55426860523566, 47.929777262483775),
    (-0.052, -76.83905471295581, 47.92868577444185),
    # (-0.055, -77.23998576878606, 47.903172010287435),
  ]

  # stability results
  xlabel, xlabel_id = 'Iext', 2
  stability_results = {k: {'V': [], xlabel: []} for k in bp.analysis.get_3d_stability_types()}
  for key, val in dict(return_dict).items():
    splits = key.split(',')
    x = float(splits[xlabel_id].split('=')[1])
    for root, stability in zip(*val):
      stability_results[stability][xlabel].append(x)
      stability_results[stability]['V'].append(root)

  # sns.set(font_scale=1.5)
  # sns.set_style("white")
  fig, gs = bp.visualize.get_figure(1, 1, row_len=6, col_len=9)
  fig.add_subplot(gs[0, 0])
  visualize_one_par(stabilities=stability_results, xlabel='Iext')
  plot_limit_cycle_by_sim(model_pars=P, fixed_points=stability_results,
                          bigger=0.02, duration=1000)
  # plot_limit_cycle_by_sim(model_pars=P, fixed_points=stability_results,
  #                         smaller=-0.1, duration=1000)

  p1_bursting = np.array([a[0] for a in bursting])
  p1_bursting_min = np.array([a[1] for a in bursting])
  p1_bursting_max = np.array([a[2] for a in bursting])
  plt.plot(p1_bursting, p1_bursting_max, linestyle='dotted', lw=3, label='bursting (max)')
  plt.plot(p1_bursting, p1_bursting_min, linestyle='dotted', lw=3, label='bursting (min)')
  plt.fill_between(p1_bursting, p1_bursting_min, p1_bursting_max, alpha=0.1)

  for i, x in enumerate([-0.025, -0.01, 0.07, 0.14]):
    plt.annotate(f"{i + 1}", xy=(x, -80), xytext=(x, -90),
                 xycoords='data', textcoords='data', va="top",
                 ha="center", fontsize=12,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 bbox=dict(boxstyle='circle', fc="w", ec="k"))

  plt.axvline(x=0)
  lg = plt.legend(fontsize=14, loc='best')
  lg.get_frame().set_alpha(0.1)
  plt.xticks(fontsize=16)
  plt.xlabel(r'$\mathrm{I_{syn}}\, (\mu \mathrm{A/cm^2})$', fontsize=16)
  plt.yticks(fontsize=16)
  plt.ylabel('V (mV)', fontsize=16)
  plt.xlim(I_range[0], I_range[-1])
  plt.ylim(-100, 55)
  plt.show()
