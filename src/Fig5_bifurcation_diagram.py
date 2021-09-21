# -*- coding: utf-8 -*-

# %%
import gc
from collections import OrderedDict

import brainpy as bp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

bp.backend.set('numba', dt=0.01)
bp.integrators.set_default_odeint('rk4')

# %%
import sys
sys.path.append('../')
from src import neus_and_syns

# %%
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


# %%
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


# %%
class FastSlowAnalysis(bp.analysis._Bifurcation2D):
  def __init__(self,
               model_or_integrals,
               fast_vars,
               slow_vars,
               fixed_vars=None,
               pars_update=None,
               numerical_resolution=0.1,
               options=None):
    model = bp.analysis.utils.transform_integrals_to_model(model_or_integrals)

    if fixed_vars is None: fixed_vars = dict()
    if pars_update is None: pars_update = dict()

    super(FastSlowAnalysis, self).__init__(model_or_integrals=model,
                                           target_pars=slow_vars,
                                           target_vars=fast_vars,
                                           fixed_vars=fixed_vars,
                                           pars_update=pars_update,
                                           numerical_resolution=numerical_resolution,
                                           options=options)

    # fast variables
    if isinstance(fast_vars, OrderedDict):
      self.fast_var_names = list(fast_vars.keys())
    else:
      self.fast_var_names = list(sorted(fast_vars.keys()))

    # slow variables
    if isinstance(slow_vars, OrderedDict):
      self.slow_var_names = list(slow_vars.keys())
    else:
      self.slow_var_names = list(sorted(slow_vars.keys()))

  def plot_trajectory(self, initials, duration, plot_duration=None, legend='trajectory'):
    print('plot trajectory ...')

    # 1. format the initial values
    all_vars = self.fast_var_names + self.slow_var_names
    if isinstance(initials, dict):
      initials = [initials]
    elif isinstance(initials, (list, tuple)):
      if isinstance(initials[0], (int, float)):
        initials = [{all_vars[i]: v for i, v in enumerate(initials)}]
      elif isinstance(initials[0], dict):
        initials = initials
      elif isinstance(initials[0], (tuple, list)) and isinstance(initials[0][0], (int, float)):
        initials = [{all_vars[i]: v for i, v in enumerate(init)} for init in initials]
      else:
        raise ValueError
    else:
      raise ValueError
    for initial in initials:
      if len(initial) != len(all_vars):
        raise bp.errors.AnalyzerError(f'Should provide all fast-slow variables ({all_vars}) '
                                      f' initial values, but we only get initial values for '
                                      f'variables {list(initial.keys())}.')

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
      traj_group.run(duration=duration[init_i], report=False)

      #   5.4 trajectory
      start = int(plot_duration[init_i][0] / bp.backend.get_dt())
      end = int(plot_duration[init_i][1] / bp.backend.get_dt())

      #   5.5 visualization
      var_name = 'V'
      s0 = traj_group.mon[self.slow_var_names[0]][start: end, 0]
      fast = traj_group.mon[var_name][start: end, 0]

      lines = plt.plot(s0, fast, label=legend)
      bp.analysis.utils.add_arrow(lines[0])
      # middle = int(s0.shape[0] / 2)
      # plt.arrow(s0[middle], fast[middle],
      #           s0[middle + 1] - s0[middle], fast[middle + 1] - fast[middle],
      #           shape='full')

  def plot_bifurcation(self, show=False):
    print('plot bifurcation ...')

    # functions
    f_fixed_point = self.get_f_fixed_point()
    f_jacobian = self.get_f_jacobian()

    # bifurcation analysis of co-dimension 1
    container = {c: {'p': [], self.x_var: [], self.y_var: []}
                 for c in bp.analysis.stability.get_2d_stability_types()}

    # fixed point
    for p in self.resolutions[self.dpar_names[0]]:
      xs, ys = f_fixed_point(p)
      for x, y in zip(xs, ys):
        dfdx = f_jacobian(x, y, p)
        fp_type = bp.analysis.stability.stability_analysis(dfdx)
        container[fp_type]['p'].append(p)
        container[fp_type][self.x_var].append(x)
        container[fp_type][self.y_var].append(y)

    # visualization
    # for var in self.dvar_names:
    # plt.figure(var)
    var = 'V'
    for fp_type, points in container.items():
      if len(points['p']):
        plot_style = plot_scheme[fp_type]
        plt.plot(points['p'], points[var], '.', **plot_style, label=fp_type)

    self.fixed_points = container
    return container

  def plot_limit_cycle_by_sim(self, var, duration=100, tol=0.001):
    print('plot limit cycle ...')

    if self.fixed_points is None:
      raise bp.errors.AnalyzerError('Please call "plot_bifurcation()" before "plot_limit_cycle_by_sim()".')

    if var not in [self.x_var, self.y_var]:
      raise bp.errors.AnalyzerError()

    all_xs, all_ys, all_p0, all_p1 = [], [], [], []

    # unstable node
    unstable_node = self.fixed_points[bp.analysis.UNSTABLE_NODE_2D]
    all_xs.extend(unstable_node[self.x_var])
    all_ys.extend(unstable_node[self.y_var])
    all_p0.extend(unstable_node['p'])

    # unstable focus
    unstable_focus = self.fixed_points[bp.analysis.UNSTABLE_FOCUS_2D]
    all_xs.extend(unstable_focus[self.x_var])
    all_ys.extend(unstable_focus[self.y_var])
    all_p0.extend(unstable_focus['p'])

    # format points
    all_xs = np.array(all_xs)
    all_ys = np.array(all_ys)
    all_p0 = np.array(all_p0)
    all_p1 = np.array(all_p1)

    # fixed variables
    fixed_vars = dict()
    for key, val in self.fixed_vars.items():
      fixed_vars[key] = val
    fixed_vars[self.dpar_names[0]] = all_p0

    # initialize neuron group
    length = all_xs.shape[0]
    traj_group = bp.analysis.Trajectory(size=length,
                                        integrals=self.model.integrals,
                                        target_vars={self.x_var: all_xs, self.y_var: all_ys},
                                        fixed_vars=fixed_vars,
                                        pars_update=self.pars_update,
                                        scope=self.model.scopes)
    traj_group.run(duration=duration)

    # find limit cycles
    limit_cycle_max = []
    limit_cycle_min = []
    # limit_cycle = []
    p0_limit_cycle = []
    for i in range(length):
      data = traj_group.mon[var][:, i]
      max_index = bp.analysis.utils.find_indexes_of_limit_cycle_max(data, tol=tol)
      if max_index[0] != -1:
        x_cycle = data[max_index[0]: max_index[1]]
        limit_cycle_max.append(data[max_index[1]])
        limit_cycle_min.append(x_cycle.min())
        # limit_cycle.append(x_cycle)
        p0_limit_cycle.append(all_p0[i])
    self.fixed_points['limit_cycle'] = {var: {'max': limit_cycle_max,
                                              'min': limit_cycle_min,
                                              # 'cycle': limit_cycle
                                              }}
    p0_limit_cycle = np.array(p0_limit_cycle)
    idx = np.argsort(p0_limit_cycle)
    p0_limit_cycle = p0_limit_cycle[idx]
    limit_cycle_max = np.array(limit_cycle_max)[idx]
    limit_cycle_min = np.array(limit_cycle_min)[idx]

    # visualization
    self.fixed_points['limit_cycle'] = {'p': p0_limit_cycle}
    if len(limit_cycle_max):
      # plt.figure(var)
      plt.plot(p0_limit_cycle, limit_cycle_max, label='limit cycle (max)')
      plt.plot(p0_limit_cycle, limit_cycle_min, label='limit cycle (min)')
      plt.fill_between(p0_limit_cycle, limit_cycle_min, limit_cycle_max, alpha=0.1)

    del traj_group
    gc.collect()


# %%
def bifurcation_diagram():
  # ----------------
  # parameter 1
  # ----------------

  g_T, NaK_th, g_L, E_L, E_KL, b, rho_p, Vr, I = \
    2.25, -55, 0.06, -70., -100, 0.5, 0., -66, 0.05
  settings = [
    (0., 400., None, 'trajectory'),
    (-0.025, 1000., (750, 1000), 'periodic burst'),
    (-0.06, 400., None, 'trajectory'),
    (0.10, 400., (300, 400), 'tonic spike'),
  ]
  skips = []
  numerical_resolution = 0.01
  limit_cycle = dict(duration=3000, tol=0.005)

  # ---------------
  # Main code
  # ---------------

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

  sns.set(font_scale=1.2)
  sns.set_style("white")

  for i, (I, dur, plot_dur, leg) in enumerate(settings):
    if i in skips: continue
    bp.backend.set('numba')

    analyzer = FastSlowAnalysis(
      model_or_integrals=trn.integral,
      fast_vars={'V': [-90., 55], 'y': [-90., 55.]},
      slow_vars={'z': [-80., -45]},
      pars_update=dict(Isyn=I, b=b, rho_p=rho_p, g_T=g_T, g_L=g_L,
                       g_KL=trn.g_KL, E_L=E_L, E_KL=E_KL,
                       IT_th=trn.IT_th, NaK_th=NaK_th),
      options={'y_by_x_in_y_eq': 'y = V',
               'escape_sympy_solver': True,
               'perturbation': -1e-10},
      numerical_resolution=numerical_resolution,
    )
    fig, gs = bp.visualize.get_figure(1, 1, 5, 6)
    fig.add_subplot(gs[0, 0])

    # bifurcation analysis
    # -------
    analyzer.plot_bifurcation()
    analyzer.plot_trajectory(initials=[{'V': -63, 'y': -63, 'z': -70}],
                             duration=dur, plot_duration=plot_dur, legend=leg)
    analyzer.plot_limit_cycle_by_sim(var='V', **limit_cycle)
    nullcline = np.arange(-80., -45, 0.1)
    plt.plot(nullcline, nullcline, label='z nullcline')

    # show global fixed points
    # -----
    bp.backend.set('pytorch')
    func = neus_and_syns.ReducedTRNv1.derivative
    _, roots, stabilises = neus_and_syns.roots_stability(
      f=func, g_L=g_L, E_L=E_L, E_KL=E_KL, Vr=Vr, b=b, rho_p=rho_p,
      g_T=g_T, IT_th=trn.IT_th, NaK_th=NaK_th, Iext=I)
    stability_results = {k: [] for k in bp.analysis.get_3d_stability_types()}
    for root, stability in zip(roots, stabilises):
      stability_results[stability].append(root)
    for key, values in stability_results.items():
      values = np.array(values)
      values = values[values <= -45.]
      if len(values):
        plt.scatter(values, values, label=key, **bp.analysis.plot_scheme[key],
                    marker="x", s=100)

    print(roots)
    print(stabilises)

    plt.xlabel('z (mV)')
    plt.ylabel('V (mV)')
    plt.xlim((-80, -45))
    plt.ylim((-90, 55))
    plt.title(r'$\mathrm{I_{syn}} = %.2f \mathrm{\mu A / cm^2}$' % I)
    lg = plt.legend(fontsize=12)
    lg.get_frame().set_alpha(0.2)
    plt.show()


if __name__ == '__main__':
  bifurcation_diagram()

