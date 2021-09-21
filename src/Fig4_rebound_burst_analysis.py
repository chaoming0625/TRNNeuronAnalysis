# -*- coding: utf-8 -*-

import brainpy as bp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

bp.backend.set('numba', dt=0.01)
bp.integrators.set_default_odeint('rk4')

import sys
sys.path.append('../')
from src import neus_and_syns

plot_scheme = {
  bp.analysis.STABLE_POINT_1D: {"color": 'tab:red'},
  bp.analysis.STABLE_NODE_2D: {"color": 'tab:red'},

  bp.analysis.UNSTABLE_POINT_1D: {"color": 'tab:olive'},
  bp.analysis.UNSTABLE_NODE_2D: {"color": 'tab:olive'},

  bp.analysis.STABLE_FOCUS_2D: {"color": 'tab:purple'},
  bp.analysis.UNSTABLE_FOCUS_2D: {"color": 'tab:cyan'},

  bp.analysis.SADDLE_NODE: {"color": 'tab:blue'},
  bp.analysis.CENTER_2D: {'color': 'lime'},
  # _2D_UNIFORM_MOTION: {'color': 'red'},

  bp.analysis.CENTER_MANIFOLD: {'color': 'orangered'},
  bp.analysis.UNSTABLE_LINE_2D: {'color': 'dodgerblue'},

  bp.analysis.UNSTABLE_STAR_2D: {'color': 'green'},
  bp.analysis.STABLE_STAR_2D: {'color': 'orange'},

  bp.analysis.UNSTABLE_DEGENERATE_2D: {'color': 'springgreen'},
  bp.analysis.STABLE_DEGENERATE_2D: {'color': 'blueviolet'},
}


class PhasePlaneAnalysis(bp.analysis._PhasePlane2D):
  def plot_nullcline(self, **kwargs):
    print('plot nullcline ...')

    # Null-cline of the y variable
    # ----
    y_style = dict(color='cornflowerblue', alpha=.7, )
    y_by_x = self.get_y_by_x_in_y_eq()
    xs = self.resolutions[self.x_var]
    y_values_in_y_eq = y_by_x['f'](xs)
    plt.plot(xs, y_values_in_y_eq, **y_style)

    # Null-cline of the x variable
    # ----
    x_style = dict(color='lightcoral', alpha=.7, )
    optimizer = self.get_f_optimize_x_nullcline()
    x_values_in_x_eq, y_values_in_x_eq = optimizer()

    cls = kwargs['cls']
    if cls == '1':
      x_th = kwargs['x_th']
      y_th = kwargs['y_th']
      # Data in line 1
      idx1 = y_values_in_x_eq >= y_th
      line1_xs = x_values_in_x_eq[idx1]
      line1_ys = y_values_in_x_eq[idx1]
      arg_sort = np.argsort(line1_xs)
      line1_xs = line1_xs[arg_sort]
      line1_ys = line1_ys[arg_sort]
      plt.plot(line1_xs, line1_ys, **x_style)

      # data in line 2
      idx2 = np.logical_and(y_values_in_x_eq < y_th, x_values_in_x_eq >= x_th)
      line2_xs = x_values_in_x_eq[idx2]
      line2_ys = y_values_in_x_eq[idx2]
      arg_sort = np.argsort(line2_ys)
      line2_xs = line2_xs[arg_sort]
      line2_ys = line2_ys[arg_sort]
      plt.plot(line2_xs, line2_ys, **x_style)

      # data in line 3
      idx3 = np.logical_and(y_values_in_x_eq < y_th, x_values_in_x_eq < x_th)
      line3_xs = x_values_in_x_eq[idx3]
      line3_ys = y_values_in_x_eq[idx3]
      arg_sort = np.argsort(line3_xs)
      line3_xs = line3_xs[arg_sort]
      line3_ys = line3_ys[arg_sort]
      plt.plot(line3_xs, line3_ys, **x_style)

    elif cls == '2':
      x1_th = kwargs['x1_th']
      idx2 = x_values_in_x_eq < x1_th
      line2_xs = x_values_in_x_eq[idx2]
      line2_ys = y_values_in_x_eq[idx2]
      arg_sort = np.argsort(line2_ys)
      line2_xs = line2_xs[arg_sort]
      line2_ys = line2_ys[arg_sort]
      plt.plot(line2_xs, line2_ys, **x_style)

      x2_th = kwargs['x2_th']
      idx1 = np.logical_and(x_values_in_x_eq >= x1_th, x_values_in_x_eq < x2_th)
      line1_xs = x_values_in_x_eq[idx1]
      line1_ys = y_values_in_x_eq[idx1]
      arg_sort = np.argsort(line1_ys)
      line1_xs = line1_xs[arg_sort]
      line1_ys = line1_ys[arg_sort]
      plt.plot(line1_xs, line1_ys, **x_style)

      x3_th = kwargs['x3_th']
      idx1 = np.logical_and(x_values_in_x_eq >= x2_th, x_values_in_x_eq < x3_th)
      line1_xs = x_values_in_x_eq[idx1]
      line1_ys = y_values_in_x_eq[idx1]
      arg_sort = np.argsort(line1_xs)
      line1_xs = line1_xs[arg_sort]
      line1_ys = line1_ys[arg_sort]
      plt.plot(line1_xs, line1_ys, **x_style)

      idx1 = x_values_in_x_eq >= x3_th
      line1_xs = x_values_in_x_eq[idx1]
      line1_ys = y_values_in_x_eq[idx1]
      arg_sort = np.argsort(line1_ys)
      line1_xs = line1_xs[arg_sort]
      line1_ys = line1_ys[arg_sort]
      plt.plot(line1_xs, line1_ys, **x_style)

    else:
      raise ValueError

  def plot_fixed_point(self, show=False):
    print('plot fixed point ...')

    # function for fixed point solving
    f_fixed_point = self.get_f_fixed_point()
    x_values, y_values = f_fixed_point()

    # function for jacobian matrix
    f_jacobian = self.get_f_jacobian()

    # stability analysis
    # ------------------
    container = {a: {'x': [], 'y': []} for a in bp.analysis.stability.get_2d_stability_types()}
    for i in range(len(x_values)):
      x = x_values[i]
      y = y_values[i]
      fp_type = bp.analysis.stability.stability_analysis(f_jacobian(x, y))
      print(f"Fixed point #{i + 1} at {self.x_var}={x}, {self.y_var}={y} is a {fp_type}.")
      container[fp_type]['x'].append(x)
      container[fp_type]['y'].append(y)

    # visualization
    # -------------
    for fp_type, points in container.items():
      if len(points['x']):
        plot_style = plot_scheme[fp_type]
        plt.plot(points['x'], points['y'], '.', markersize=10, **plot_style, label=fp_type)
    plt.legend()
    if show:
      plt.show()

  def plot_trajectory(self, initials, duration, plot_duration=None, axes='v-v', show=False):
    print('plot trajectory ...')

    if axes not in ['v-v', 't-v']:
      raise bp.errors.ModelUseError(f'Unknown axes "{axes}", only support "v-v" and "t-v".')

    # 1. format the initial values
    if isinstance(initials, dict):
      initials = [initials]
    elif isinstance(initials, (list, tuple)):
      if isinstance(initials[0], (int, float)):
        initials = [{self.dvar_names[i]: v for i, v in enumerate(initials)}]
      elif isinstance(initials[0], dict):
        initials = initials
      elif isinstance(initials[0], (tuple, list)) and isinstance(initials[0][0], (int, float)):
        initials = [{self.dvar_names[i]: v for i, v in enumerate(init)} for init in initials]
      else:
        raise ValueError
    else:
      raise ValueError

    # 2. format the running duration
    if isinstance(duration, (int, float)):
      duration = [(0, duration) for _ in range(len(initials))]
    elif isinstance(duration[0], (int, float)):
      duration = [duration for _ in range(len(initials))]
    else:
      assert len(duration) == len(initials)

    # 3. format the plot duration
    if plot_duration is None:
      plot_duration = duration
    if isinstance(plot_duration[0], (int, float)):
      plot_duration = [plot_duration for _ in range(len(initials))]
    else:
      assert len(plot_duration) == len(initials)

    # 5. run the network
    for init_i, initial in enumerate(initials):
      traj_group = bp.analysis.Trajectory(size=1,
                                          integrals=self.model.integrals,
                                          target_vars=initial,
                                          fixed_vars=self.fixed_vars,
                                          pars_update=self.pars_update,
                                          scope=self.model.scopes)

      #   5.2 run the model
      traj_group.run(duration=duration[init_i], report=False, )

      #   5.4 trajectory
      start = int(plot_duration[init_i][0] / bp.backend.get_dt())
      end = int(plot_duration[init_i][1] / bp.backend.get_dt())

      #   5.5 visualization
      lines = plt.plot(traj_group.mon[self.x_var][start: end, 0],
                       traj_group.mon[self.y_var][start: end, 0],
                       label='trajectory')
      bp.analysis.utils.add_arrow(lines[0])

  def plot_limit_cycle_by_sim(self, initials, duration, tol=0.001, show=False):
    print('plot limit cycle ...')

    # 1. format the initial values
    if isinstance(initials, dict):
      initials = [initials]
    elif isinstance(initials, (list, tuple)):
      if isinstance(initials[0], (int, float)):
        initials = [{self.dvar_names[i]: v for i, v in enumerate(initials)}]
      elif isinstance(initials[0], dict):
        initials = initials
      elif isinstance(initials[0], (tuple, list)) and isinstance(initials[0][0], (int, float)):
        initials = [{self.dvar_names[i]: v for i, v in enumerate(init)} for init in initials]
      else:
        raise ValueError
    else:
      raise ValueError

    # 2. format the running duration
    if isinstance(duration, (int, float)):
      duration = [(0, duration) for _ in range(len(initials))]
    elif isinstance(duration[0], (int, float)):
      duration = [duration for _ in range(len(initials))]
    else:
      assert len(duration) == len(initials)

    # 5. run the network
    for init_i, initial in enumerate(initials):
      traj_group = bp.analysis.Trajectory(size=1,
                                          integrals=self.model.integrals,
                                          target_vars=initial,
                                          fixed_vars=self.fixed_vars,
                                          pars_update=self.pars_update,
                                          scope=self.model.scopes)

      #   5.2 run the model
      traj_group.run(duration=duration[init_i], report=False, )
      x_data = traj_group.mon[self.x_var][:, 0]
      y_data = traj_group.mon[self.y_var][:, 0]
      max_index = bp.analysis.utils.find_indexes_of_limit_cycle_max(x_data, tol=tol)
      if max_index[0] != -1:
        x_cycle = x_data[max_index[0]: max_index[1]]
        y_cycle = y_data[max_index[0]: max_index[1]]
        # 5.5 visualization
        lines = plt.plot(x_cycle, y_cycle, label='limit cycle')
        bp.analysis.utils.add_arrow(lines[0])
      else:
        print(f'No limit cycle found for initial value {initial}')

  def plot_vector_field(self, plot_method='streamplot', plot_style=None, show=False):
    print('plot vector field ...')

    if plot_style is None:
      plot_style = dict()

    xs = self.resolutions[self.x_var]
    ys = self.resolutions[self.y_var]
    X, Y = np.meshgrid(xs, ys)

    # dx
    try:
      dx = self.get_f_dx()(X, Y)
    except TypeError:
      raise bp.errors.ModelUseError('Missing variables. Please check and set missing '
                                    'variables to "fixed_vars".')

    # dy
    try:
      dy = self.get_f_dy()(X, Y)
    except TypeError:
      raise bp.errors.ModelUseError('Missing variables. Please check and set missing '
                                    'variables to "fixed_vars".')

    # vector field
    styles = dict()
    styles['arrowsize'] = plot_style.get('arrowsize', 1.2)
    styles['density'] = plot_style.get('density', 1)
    styles['color'] = plot_style.get('color', 'thistle')
    linewidth = plot_style.get('linewidth', None)
    if (linewidth is None) and (not np.isnan(dx).any()) and (not np.isnan(dy).any()):
      min_width = plot_style.get('min_width', 0.5)
      max_width = plot_style.get('min_width', 5.5)
      speed = np.sqrt(dx ** 2 + dy ** 2)
      linewidth = min_width + max_width * speed / speed.max()
    plt.streamplot(X, Y, dx, dy, linewidth=linewidth, **styles)


inputs_set1 = [
  -0.06, -0.05, -0.04, -0.03,
  -0.020, -0.010, -0.008, -0.004,
  -0.0020, -0.0010, -0.0008, -0.0004,
  -0.0002, 0.00, 0.0002, 0.0004,
  0.0008, 0.0010, 0.0020, 0.0040,
  0.0060, 0.0080, 0.0100, 0.0150,
  0.0200, 0.0210, 0.0220, 0.0230,
  0.0240, 0.0250, 0.0300, 0.0350,
  0.0400, 0.0500, 0.0600, 0.0700,
  0.0800, 0.0900, 0.10, 0.11,
]

inputs_set2 = [
  -0.06, -0.05, -0.04, -0.03,
  -0.020, -0.010, -0.008, -0.004,
  -0.0020, -0.0010, -0.0008, -0.0004,
  -0.0002, 0.00, 0.0002, 0.0004,
  0.0008, 0.0010, 0.0020, 0.0040,
  0.0060, 0.0080, 0.0100, 0.0200,
  0.0300, 0.040, 0.0600, 0.0800,
  0.100, 0.1200, 0.14, 0.16,
  0.18, 0.20, 0.22, 0.24,
]


def try_constant_inputs(model, inputs, Vr=-70., init_vr=None):
  neus_and_syns.constant_inputs(type(model), Vr=Vr,
                                inputs=inputs, rho_p=model.rho_p, b=model.b,
                                g_L=model.g_L, E_L=model.E_L, E_KL=model.E_KL,
                                g_KL=model.g_KL, g_T=model.g_T, NaK_th=model.NaK_th,
                                IT_th=model.IT_th, init_vr=init_vr,
                                report=True, duration=5e3)


g_T, NaK_th, g_L, E_L, E_KL, b, rho_p, Vr, I = \
  2.25, -55, 0.06, -70., -100, 0.5, 0., -66, -0.06

settings = [
  (50, Vr, 0., ('limit_cycle', (-22.7, -22.7), 50), dict(cls='1', x_th=20., y_th=-40)),

  (100, -70.94, I, ('limit_cycle', (-22.7, -22.7), 50), dict(cls='1', x_th=20., y_th=-40)),

  (250, -75.43, 0., ('limit_cycle', (-22.7, -22.7), 50), dict(cls='1', x_th=20., y_th=-40)),

  (315, -65., 0., ('limit_cycle', (-22.7, -22.7), 50), dict(cls='1', x_th=20., y_th=-40)),

  (328, -63.4, 0., ('limit_cycle', (-22.0, -22.0), 100), dict(cls='1', x_th=20., y_th=-40)),

  (337, -61.59, 0., ('traj', (-56.65, -56.65), 250), dict(cls='1', x_th=20., y_th=-46)),
]


def simulation():
  # TRN neuron
  trn = neus_and_syns.ReducedTRNv1(1, monitors=['V', 'y', 'z'])
  trn.NaK_th = NaK_th
  trn.rho_p = rho_p
  trn.E_KL = E_KL
  trn.g_T = g_T
  trn.g_L = g_L
  trn.E_L = E_L
  trn.b = b
  trn.g_KL = trn.suggest_gKL(Vr=Vr, Iext=0., )

  print(trn.g_KL)

  # simulation
  # -------------------
  trn.reset(Vr=Vr)
  currents, duration = bp.inputs.period_input(values=[0, I, 0],
                                              durations=[50, 200, 150],
                                              return_length=True)
  trn.run(duration, inputs=('input', currents))
  fig, gs = bp.visualize.get_figure(row_num=10, col_num=1, col_len=14, row_len=0.5)
  fig.add_subplot(gs[0:9, 0])
  plt.plot(trn.mon.ts, trn.mon.V[:, 0], label='V', lw=3)
  plt.plot(trn.mon.ts, trn.mon.z[:, 0], label='z', lw=3)
  plt.ylim((-96, 50))
  plt.xlim((-1, duration + 1))
  for i, x in enumerate(settings):
    x = x[0]
    plt.annotate(f"{i + 1}", xy=(x, -74.5), xytext=(x, -88), xycoords='data',
                 textcoords='data', va="top", ha="center", fontsize=14,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 bbox=dict(boxstyle='circle', fc="w", ec="k"))
  lg = plt.legend(loc='best', fontsize=14)
  lg.get_frame().set_alpha(0.4)
  plt.axis('off')
  fig.add_subplot(gs[9, 0])
  plt.plot(trn.mon.ts, currents, color='orangered', label='External Current', lw=3)
  plt.xlim((-1, duration + 1))
  lg = plt.legend(loc='best', fontsize=14)
  lg.get_frame().set_alpha(0.4)
  plt.axis('off')
  plt.tight_layout()
  plt.show()
  return trn


def phase_plane_analysis():
  trn = neus_and_syns.ReducedTRNv1(1, monitors=['V', 'y', 'z'])
  trn.NaK_th = NaK_th
  trn.rho_p = rho_p
  trn.E_KL = E_KL
  trn.g_T = g_T
  trn.g_L = g_L
  trn.E_L = E_L
  trn.b = b
  trn.g_KL = trn.suggest_gKL(Vr=Vr, Iext=0., )

  sns.set(font_scale=1.2)
  sns.set_style("white")

  for i, (time, z, Iext, ops, nuclline) in enumerate(settings):
    V_scale = (-90., 55.)
    y_scale = (-90., 55.)
    analyzer = PhasePlaneAnalysis(
      model_or_integrals=trn.integral,
      target_vars=dict(V=V_scale, y=y_scale),
      fixed_vars=dict(z=z),
      pars_update=dict(Isyn=Iext, b=b, rho_p=rho_p, g_T=g_T, g_L=g_L,
                       g_KL=trn.g_KL, E_L=E_L, E_KL=E_KL, IT_th=trn.IT_th,
                       NaK_th=trn.NaK_th),
      options={'y_by_x_in_y_eq': 'y = V',
               'escape_sympy_solver': True,
               'perturbation': -1e-10},
      numerical_resolution=0.02,
    )
    plt.figure(figsize=(4, 4))
    if ops[0] == 'limit_cycle':
      analyzer.plot_limit_cycle_by_sim(initials=[{'V': ops[1][0], 'y': ops[1][1]}],
                                       duration=ops[2],
                                       tol=0.005)
    else:
      analyzer.plot_trajectory(initials=[{'V': ops[1][0], 'y': ops[1][1]}],
                               duration=ops[2])
    analyzer.plot_fixed_point()
    analyzer.plot_nullcline(**nuclline)
    analyzer.plot_vector_field()
    plt.xlim(*bp.analysis.utils.rescale(V_scale, scale=0.025))
    plt.ylim(*bp.analysis.utils.rescale(y_scale, scale=0.025))
    lg = plt.legend(fontsize=12)
    lg.get_frame().set_alpha(0.4)
    plt.annotate(f"{i + 1}", xy=(20, -80), xytext=(20, -80), xycoords='data',
                 textcoords='data', arrowprops=dict(arrowstyle="->"),
                 bbox=dict(boxstyle='circle', fc="w", ec="k"))
    plt.title(r'Time = %.1f ms, I = %.3f $\mathrm{\mu A / cm^2}$' % (time, Iext))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
  # simulation()
  phase_plane_analysis()
