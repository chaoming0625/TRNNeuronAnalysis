# -*- coding: utf-8 -*-

# %% [markdown]
# # Equivalent Potentials of TRN neuron model


# %%
import seaborn as sns
import brainpy as bp

bp.backend.set('numba', dt=0.01)

# %%
import sys
sys.path.append('../')

from src import neus_and_syns


# %% code_folding=[0]
def try1():
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=1.5)

    group = neus_and_syns.EqPotentialTRN(1, monitors=['V', 'vm', 'vh', 'vn', 'vp', 'vq'])
    group.g_L = 0.05
    group.E_L = -77.
    group.g_KL = 0.00792954
    group.E_KL = -95.
    group.b = 0.5
    group.NaK_th = -55.
    group.g_T = 2.
    group.IT_th = -3.
    group.reset(Vr=-75.)

    Iext, duration = bp.inputs.period_input(values=[-0.03, 0], durations=[200, 400],
                                            return_length=True)
    group.run(duration, inputs=['input', Iext])

    fig, gs = bp.visualize.get_figure(row_num=2, col_num=3, row_len=4, col_len=5)
    fig.add_subplot(gs[0, 0])
    bp.visualize.line_plot(group.mon.ts, group.mon.V, title='V',
                           xlim=(99., 501), ylabel='Potential (mV)',
                           xlabel=None, xticks=[])

    fig.add_subplot(gs[0, 1])
    bp.visualize.line_plot(group.mon.ts, group.mon.vm, title='vm',
                           xlim=(99., 501), xlabel=None, xticks=[])

    fig.add_subplot(gs[0, 2])
    bp.visualize.line_plot(group.mon.ts, group.mon.vh, title='vh',
                           xlim=(99., 501), xlabel=None, xticks=[])

    fig.add_subplot(gs[1, 0])
    bp.visualize.line_plot(group.mon.ts, group.mon.vn, title='vn',
                           ylabel='Potential (mV)', xlim=(99., 501))

    fig.add_subplot(gs[1, 1])
    bp.visualize.line_plot(group.mon.ts, group.mon.vp, title='vp',
                           xlim=(99., 501))

    fig.add_subplot(gs[1, 2])
    bp.visualize.line_plot(group.mon.ts, group.mon.vq, title='vq',
                           xlim=(99., 501), show=True)


    fig, gs = bp.visualize.get_figure(row_num=1, col_num=3, row_len=3, col_len=5)
    fig.add_subplot(gs[0, 0])
    bp.visualize.line_plot(group.mon.ts, group.mon.vh - group.mon.vn,
                           title='vh-vn', xlim=(99., 501))

    fig.add_subplot(gs[0, 1])
    bp.visualize.line_plot(group.mon.ts, group.mon.vh - group.mon.vp,
                           title='vh-vp', xlim=(99., 501))

    fig.add_subplot(gs[0, 2])
    bp.visualize.line_plot(group.mon.ts, group.mon.vn - group.mon.vp,
                           title='vn-vp', xlim=(99., 501), show=True)



# %% code_folding=[0]
if __name__ == '__main__':
    try1()

