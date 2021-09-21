# -*- coding: utf-8 -*-

# %% [markdown]
# # Comparison between the reduced and the original

# %%
import sys
sys.path.append('../')

# %%
import brainpy as bp
import matplotlib.pyplot as plt
import seaborn as sns

bp.backend.set('numba', dt=0.01)
bp.set_default_odeint('rk4')

# %%
from src import neus_and_syns

# %%
import warnings

warnings.filterwarnings("ignore")


# %%
origin = neus_and_syns.OriginalTRN(size=1, monitors=['V', 'spike'])
reduced = neus_and_syns.ReducedTRNv1(size=1, monitors=['V', 'spike'])


# %%
def compare3(b=0.5, rho_p=0., NaK_th=-55., Vr=-71., g_T=2.25, Iexts=(0.2, 0.2, -0.05)):
    sns.set(font_scale=1.5)
    sns.set_style("white")

    reduced.NaK_th = origin.NaK_th = NaK_th
    reduced.g_T = origin.g_T = g_T
    reduced.g_L = origin.g_L = 0.06
    reduced.E_L = origin.E_L = -70
    reduced.E_KL = origin.E_KL = -100.
    reduced.b = b
    reduced.rho_p = rho_p

    reduced.g_KL = origin.g_KL = reduced.suggest_gKL(Vr=Vr, Iext=0., g_T=g_T)
    print(reduced.g_KL)


    n_row = 5
    fig, gs = bp.visualize.get_figure(row_num=n_row, col_num=3,
                                      row_len=1.5, col_len=5)

    ax1_ylim = (-85, 50)

    # Input 1: [(0.1, 750.)]
    inputs, duration = bp.inputs.period_input(values=[Iexts[0]],
                                              durations=[750.],
                                              return_length=True)

    origin.reset(Vr=Vr)
    origin.run(duration=duration, inputs=['input', inputs])

    reduced.reset(Vr=Vr)
    reduced.run(duration=duration, inputs=['input', inputs])

    start, end = 0, duration
    ax = fig.add_subplot(gs[0:n_row - 1, 0])
    plt.plot(origin.mon.V_t, origin.mon.V[:, 0], label='original')
    plt.plot(origin.mon.V_t, reduced.mon.V[:, 0], label='reduced')
    # plt.legend()
    plt.xlim(start, end)
    plt.ylim(ax1_ylim)
    plt.ylabel('Membrane Potential (mV)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_ticks([])

    ax = fig.add_subplot(gs[n_row - 1, 0])
    plt.plot(origin.mon.ts, inputs)
    plt.xlim(start, end)
    plt.ylim(-0.02, Iexts[0] + 0.02)
    # plt.ylabel(r'$I_{ext}$ (nA)')
    plt.ylabel(r'$I$ ($\mathrm{\mu A / cm^2}$)')
    plt.xlabel('Time (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    # Input 2: [(0., 50.), (0.08, 200.), (0., 500.)]
    inputs, duration = bp.inputs.period_input(values=[0., Iexts[1], 0.],
                                              durations=[50., 200., 500.],
                                              return_length=True)

    origin.reset(Vr=Vr)
    origin.run(duration=duration, inputs=['input', inputs])

    reduced.reset(Vr=Vr)
    reduced.run(duration=duration, inputs=['input', inputs])

    # start = origin.mon.ts[int(0. / bp.backend.get_dt())]
    # end = origin.mon.ts[int(350. / bp.backend.get_dt())]
    start, end = 0, duration
    ax = fig.add_subplot(gs[0:n_row - 1, 1])
    plt.plot(origin.mon.ts, origin.mon.V[:, 0], label='original')
    plt.plot(origin.mon.ts, reduced.mon.V[:, 0], label='reduced')
    # plt.legend()
    plt.xlim(start, end)
    plt.ylim(ax1_ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])

    ax = fig.add_subplot(gs[n_row - 1, 1])
    plt.plot(origin.mon.ts, inputs)
    plt.xlim(start, end)
    plt.ylim(-0.02, Iexts[1] + 0.02)
    plt.xlabel('Time (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    # Input 3: [(0., 50.), (-0.04, 200.), (0., 500.)]
    inputs, duration = bp.inputs.period_input(values=[0., Iexts[2], 0.],
                                              durations=[50., 200., 500.],
                                              return_length=True)

    origin.reset(Vr=Vr)
    origin.run(duration=duration, inputs=['input', inputs])
    reduced.reset(Vr=Vr)
    reduced.run(duration=duration, inputs=['input', inputs])

    start, end = 0, duration
    ax = fig.add_subplot(gs[0:n_row - 1, 2])
    plt.plot(origin.mon.ts, origin.mon.V[:, 0], label='original')
    plt.plot(origin.mon.ts, reduced.mon.V[:, 0], label='reduced')
    plt.xlim(start, end)
    plt.ylim(ax1_ylim)
    plt.legend(fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])

    ax = fig.add_subplot(gs[n_row - 1, 2])
    plt.plot(origin.mon.ts, inputs)
    plt.xlim(start, end)
    plt.ylim(Iexts[2] - 0.01, 0.01)
    plt.xlabel('Time (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax.spines['left'].set_visible(False)
    # ax.get_xaxis().set_ticks([])

    plt.tight_layout()
    plt.show()


# %%
if __name__ == '__main__':
    compare3(Vr=-71.)
