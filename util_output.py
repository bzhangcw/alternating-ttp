"""
output utilities
"""

from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import *

DEBUGGING = True
if DEBUGGING:
    pd.set_option("display.max_columns", None)
    np.set_printoptions(linewidth=200, precision=3)

###########################
# matplotlib options
###########################
plt.rcParams['figure.figsize'] = (18.0, 9.0)
# plt.rcParams["font.family"] = 'Times'
plt.rcParams["font.size"] = 9
fig = plt.figure(dpi=200)
color_value = {
    '0': 'midnightblue',
    '1': 'mediumblue',
    '2': 'c',
    '3': 'orangered',
    '4': 'm',
    '5': 'fuchsia',
    '6': 'olive'
}

###########################
# output format
###########################
COLUMNS_CHS_ENG = {
    '车次': 'id',
    '站序': 'station#',
    '站名': 'station_name',
    '到点': 'str_time_arr',
    '发点': 'str_time_dep'
}


def read_timetable_csv(fpath, st=None, station_name_map=None):
    """

    read standard timetable csv

    Args:
        fpath: file path
        st:    start-time of the timetable.
            if not defined, infer from the data.

    Returns:
    """
    df = pd.read_excel(fpath).rename(
        columns=COLUMNS_CHS_ENG
    ).assign(
        station_id=lambda df: df['station_name'].apply(station_name_map.get),
        station_i=lambda df: df['station_id'].apply(lambda x: f"_{x}"),
        station_o=lambda df: df['station_id'].apply(lambda x: f"{x}_"),
        time_arr=lambda df: pd.to_datetime(df['str_time_arr']).apply(lambda x: x.hour * 60 + x.minute - st),
        time_dep=lambda df: pd.to_datetime(df['str_time_dep']).apply(lambda x: x.hour * 60 + x.minute - st)
    )
    train_paths = df.groupby('id').apply(
        lambda grp: sorted(
            list(zip(grp['station_i'], grp['time_arr'])) + list(zip(grp['station_o'], grp['time_dep'])),
            key=lambda x: x[-1]
        )
    )
    return df, train_paths


def plot_timetables_h5(train_list, miles, station_list, param_sys: SysParams, param_subgrad: SubgradParam,
                       selective: List = None):
    import plotly.graph_objects as go

    fig = go.Figure(layout=go.Layout(
        title=f"Best primal solution of # trains, station, periods: ({len(train_list)}, {param_sys.station_size}, {param_sys.time_span})\n"
              f"Number of trains {param_subgrad.max_number}")
    )
    for i in range(len(train_list)):
        train = train_list[i]
        if selective is not None and train.traNo not in selective:
            continue
        xlist = []
        ylist = []
        if not train.is_best_feasible:
            continue
        for sta_id in range(len(train.staList)):
            try:
                sta = train.staList[sta_id]
                if sta_id != 0:  # 不为首站, 有到达
                    if "_" + sta in train.v_staList:
                        xlist.append(train.timetable["_" + sta])
                        ylist.append(miles[station_list.index(sta)])
                if sta_id != len(train.staList) - 1:  # 不为末站，有出发
                    if sta + "_" in train.v_staList:
                        xlist.append(train.timetable[sta + "_"])
                        ylist.append(miles[station_list.index(sta)])
            except:
                # todo, in later development, this should not be allowed.
                pass
        fig.add_scatter(
            mode='lines+markers',
            x=xlist, y=ylist,
            line={"dash": "solid"},
            name=f"train-{train.traNo}:{train.speed}",
        )
    fig.update_xaxes(title="minutes",
                     tickvals=np.arange(0, param_sys.time_span, param_sys.time_span / 30).round(2))
    fig.update_yaxes(title="miles",
                     tickvals=miles)

    fig.write_html(
        f"{param_sys.fdir_result}/{param_subgrad.dual_method}.{param_subgrad.feasible_provider}@{param_subgrad.iter}-{param_sys.train_size}.{param_sys.station_size}.{param_sys.time_span}.html",
    )

    # PUBLICATION
    fig.update_layout(
        font_family="Latin Modern Roman",
        font_color="rgb(0,0,0)",
        xaxis=dict(
            title=f"epoch",
            color='black',
        ),
        font=dict(family="Latin Modern Roman", size=12),
        showlegend=False,
        title=None
    )
    fig.write_image(
        f"{param_sys.fdir_result}/{param_subgrad.dual_method}.{param_subgrad.feasible_provider}@{param_subgrad.iter}-{param_sys.train_size}.{param_sys.station_size}.{param_sys.time_span}.png",
        height=1400,
        width=2000,
        scale=3
    )
    fig.write_image(
        f"{param_sys.fdir_result}/{param_subgrad.dual_method}.{param_subgrad.feasible_provider}@{param_subgrad.iter}-{param_sys.train_size}.{param_sys.station_size}.{param_sys.time_span}.pdf",
        height=1400,
        width=2000,
        scale=3
    )




def plot_convergence(param_sys: SysParams, param_subgrad: SubgradParam):
    ## plot the bound updates
    font_dic = {
        "family": "Times",
        "style": "oblique",
        "weight": "normal",
        "color": "green",
        "size": 20
    }

    x_cor = range(0, param_subgrad.iter)
    plt.figure(0)
    plt.plot(x_cor, param_subgrad.lb_arr, label=f'LB_{param_subgrad.gamma:.4f}_{param_subgrad.alpha:.4f}')
    plt.legend()
    plt.xlabel('Iteration', fontdict=font_dic)
    plt.ylabel('Bounds update', fontdict=font_dic)
    plt.title('LR: Bounds updates \n', fontsize=23)
    plt.savefig(
        f"{param_sys.fdir_result}/{param_subgrad.dual_method}.{param_subgrad.primal_heuristic_method}-{param_sys.train_size}.{param_sys.station_size}.{param_sys.time_span}.convergence.png",
        dpi=500)
    # plt.clf()

    plt.figure(1)
    plt.plot(x_cor, param_subgrad.norms[0], label=f'norm_1_{param_subgrad.gamma:.4f}_{param_subgrad.alpha:.4f}')
    plt.hlines(0, xmin=x_cor[0], xmax=x_cor[-1])
    plt.legend()
    plt.xlabel('Iteration', fontdict=font_dic)
    plt.ylabel('norm', fontdict=font_dic)
    plt.title('LR: Primal Infeasibility \n', fontsize=23)
    plt.savefig(
        f"{param_sys.fdir_result}/{param_subgrad.dual_method}.{param_subgrad.primal_heuristic_method}-{param_sys.train_size}.{param_sys.station_size}.{param_sys.time_span}.norm_1.png",
        dpi=500)

    plt.figure(2)
    plt.plot(x_cor, param_subgrad.norms[1], label=f'norm_2_{param_subgrad.gamma:.4f}_{param_subgrad.alpha:.4f}')
    plt.hlines(0, xmin=x_cor[0], xmax=x_cor[-1])
    plt.legend()
    plt.xlabel('Iteration', fontdict=font_dic)
    plt.ylabel('norm', fontdict=font_dic)
    plt.title('LR: Primal Infeasibility \n', fontsize=23)
    plt.savefig(
        f"{param_sys.fdir_result}/{param_subgrad.dual_method}.{param_subgrad.primal_heuristic_method}-{param_sys.train_size}.{param_sys.station_size}.{param_sys.time_span}.norm_2.png",
        dpi=500)

    plt.figure(3)
    plt.plot(x_cor, param_subgrad.norms[2], label=f'norm_inf_{param_subgrad.gamma:.4f}_{param_subgrad.alpha:.4f}')
    plt.hlines(0, xmin=x_cor[0], xmax=x_cor[-1])
    plt.legend()
    plt.xlabel('Iteration', fontdict=font_dic)
    plt.ylabel('norm', fontdict=font_dic)
    plt.title('LR: Primal Infeasibility \n', fontsize=23)
    plt.savefig(
        f"{param_sys.fdir_result}/{param_subgrad.dual_method}.{param_subgrad.primal_heuristic_method}-{param_sys.train_size}.{param_sys.station_size}.{param_sys.time_span}.norm_inf.png",
        dpi=500)

    plt.figure(4)
    plt.plot(x_cor, param_subgrad.multipliers[0], label=f'multiplier_norm_1_{param_subgrad.gamma:.4f}_{param_subgrad.alpha:.4f}')
    plt.hlines(0, xmin=x_cor[0], xmax=x_cor[-1])
    plt.legend()
    plt.xlabel('Iteration', fontdict=font_dic)
    plt.ylabel('norm', fontdict=font_dic)
    plt.title('LR: Dual Excess \n', fontsize=23)
    plt.savefig(
        f"{param_sys.fdir_result}/{param_subgrad.dual_method}.{param_subgrad.primal_heuristic_method}-{param_sys.train_size}.{param_sys.station_size}.{param_sys.time_span}.multiplier_norm_1.png",
        dpi=500)

    plt.figure(5)
    plt.plot(x_cor, param_subgrad.multipliers[1], label=f'multiplier_norm_2_{param_subgrad.gamma:.4f}_{param_subgrad.alpha:.4f}')
    plt.hlines(0, xmin=x_cor[0], xmax=x_cor[-1])
    plt.legend()
    plt.xlabel('Iteration', fontdict=font_dic)
    plt.ylabel('norm', fontdict=font_dic)
    plt.title('LR: Primal Excess \n', fontsize=23)
    plt.savefig(
        f"{param_sys.fdir_result}/{param_subgrad.dual_method}.{param_subgrad.primal_heuristic_method}-{param_sys.train_size}.{param_sys.station_size}.{param_sys.time_span}.multiplier_norm_2.png",
        dpi=500)

    plt.figure(6)
    plt.plot(x_cor, param_subgrad.multipliers[2], label=f'multiplier_norm_inf_{param_subgrad.gamma:.4f}_{param_subgrad.alpha:.4f}')
    plt.hlines(0, xmin=x_cor[0], xmax=x_cor[-1])
    plt.legend()
    plt.xlabel('Iteration', fontdict=font_dic)
    plt.ylabel('norm', fontdict=font_dic)
    plt.title('LR: Primal Excess \n', fontsize=23)
    plt.savefig(
        f"{param_sys.fdir_result}/{param_subgrad.dual_method}.{param_subgrad.primal_heuristic_method}-{param_sys.train_size}.{param_sys.station_size}.{param_sys.time_span}.multiplier_norm_inf.png",
        dpi=500)
