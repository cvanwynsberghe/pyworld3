# -*- coding: utf-8 -*-

# © Copyright Charles Vanwynsberghe (2021)

# Pyworld3 is a computer program whose purpose is to run configurable
# simulations of the World3 model as described in the book "Dynamics
# of Growth in a Finite World".

# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".

# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.

# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import inspect
import re
from collections import defaultdict, deque

import numpy as np
from numpy import isnan

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from matplotlib.image import imread

def step(reads=None, writes=None):
    """
    Declare the variables read and written by an _update_* method, with the
    time index at which each access happens.

    Parameters
    ----------
    reads : dict, optional
        variable name -> index it is read at. "k"/"kl" mean the current step,
        "j"/"jk" the previous one. A variable may map to a list of indices.
    writes : dict, optional
        variable name -> index it is written at.

    The dependency graph of the model is built from this metadata only (see
    build_execution_order): a consumer depends on the producer of a variable
    only when it reads it at the current step; reads at the previous step
    create no edge.

    """

    def step_decorator(fn):
        fn._step_info = (reads or {}, writes or {})
        return fn

    return step_decorator

_EXECUTION_ORDERS = {}

def compile_steps(obj, order):
    """
    Bind the (method name, argument names) pairs of an execution order into
    callables taking (k, j, jk, kl), so the simulation loop has no per-step
    argument dispatch overhead.

    """
    steps = []
    for name, argnames in order:
        fn = getattr(obj, name)
        a = tuple(argnames)
        if a == ("k",):
            steps.append(lambda k, j, jk, kl, fn=fn: fn(k))
        elif a == ("k", "j"):
            steps.append(lambda k, j, jk, kl, fn=fn: fn(k, j))
        elif a == ("k", "jk"):
            steps.append(lambda k, j, jk, kl, fn=fn: fn(k, jk))
        elif a == ("k", "kl"):
            steps.append(lambda k, j, jk, kl, fn=fn: fn(k, kl))
        elif a == ("k", "j", "jk"):
            steps.append(lambda k, j, jk, kl, fn=fn: fn(k, j, jk))
        elif a == ("k", "j", "jk", "kl"):
            steps.append(lambda k, j, jk, kl, fn=fn: fn(k, j, jk, kl))
        else:
            def make(fn=fn, a=a):
                def step(k, j, jk, kl):
                    argmap = {"k": k, "j": j, "jk": jk, "kl": kl}
                    return fn(*[argmap[x] for x in a])
                return step
            steps.append(make())
    return steps

def run_init_loops(obj, loop0_fns):
    """
    Run the k=0 initialization of a World3 (or sector) instance.

    The loop0_* initializers apply the initial conditions (and the
    exogenous inputs in standalone sector mode); the _update_* methods
    are then executed once per pass in the topological order of the @step
    dependency graph at k=0 (build_execution_order with init=True, where
    all time indices collapse to 0 and the variables directly assigned in
    the loop0_* bodies act as roots). Passes are rescheduled until every
    variable read at k=0 is known, which resolves the circular
    dependencies of the original model at k=0.

    Parameters
    ----------
    obj : World3 or sector instance
        Initialized instance (constants, variables, tables, delays set).
    loop0_fns : list of callables
        The loop0_* methods applying the initial conditions at each pass.

    """
    cls = type(obj)
    leaves = build_execution_order(cls, init=True)
    steps = compile_steps(obj, [(name, argnames) for name, argnames in leaves
                                if not name.startswith("loopk_exogenous")])
    obj.redo_loop = True
    while obj.redo_loop:
        obj.redo_loop = False
        for fn in loop0_fns:
            fn()
        for fn in steps:
            fn(0, 0, 0, 0)
        for name, _argnames in leaves:
            for var in getattr(cls, name)._step_info[0]:
                arr = getattr(obj, var, None)
                if (isinstance(arr, np.ndarray) and arr.shape == (obj.n,)
                        and isnan(arr[0])):
                    obj.redo_loop = True


def build_execution_order(cls, init=False):
    """
    Build the topological execution order of all _update_* methods of a
    class, from their @step metadata. The result is cached per class.

    Parameters
    ----------
    cls : class
        The World3 class (or a sector class).
    init : bool, optional
        Order for the initialization step (k=0), where all time indices
        collapse to 0 and every read creates an edge (state leaves are
        excluded, as they are never called at k=0). If False (default),
        only reads at the current step ("k"/"kl") create an edge; reads
        at the previous step ("j"/"jk") refer to the computed step.

    Returns
    -------
    list of (str, list of str)
        (method name, argument names) in execution order.

    Raises
    ------
    ValueError
        If a circular same-step dependency is found.

    """
    key = (cls, init)
    if key in _EXECUTION_ORDERS:
        return _EXECUTION_ORDERS[key]

    methods = {}
    for klass in cls.__mro__:
        for name, attr in vars(klass).items():
            if hasattr(attr, "_step_info"):
                # state leaves are never called at k=0; excluding them keeps
                # the init graph acyclic
                if init and name.startswith("_update_state_"):
                    continue
                methods[name] = attr

    # in the full model, the loopk_exogenous methods are unused: their
    # variables are then produced by _update_* methods. In a standalone
    # sector they provide the exogenous inputs and must stay in the graph.
    update_writes = {v for m, a in methods.items()
                     if m.startswith("_update_") for v in a._step_info[1]}
    for name in [n for n in methods if n.startswith("loopk_exogenous")]:
        exo_writes = methods[name]._step_info[1]
        if any(v in update_writes for v in exo_writes):
            del methods[name]

    # at k=0, variables directly assigned in the loop0_* methods are initial
    # conditions: they are roots of the init graph (no incoming edge). This
    # is what breaks the circular dependencies of the original model at k=0.
    init_roots = set()
    if init:
        for klass in cls.__mro__:
            for name, attr in vars(klass).items():
                if name.startswith("loop0_") and "exogenous" not in name:
                    for var in re.findall(r"self\.([a-z0-9_]+)\[0\] =",
                                          inspect.getsource(attr)):
                        init_roots.add(var)

    producers = defaultdict(list)
    for name, attr in methods.items():
        for var in attr._step_info[1]:
            producers[var].append(name)

    preds = {}
    for name, attr in methods.items():
        reads = attr._step_info[0]
        p = set()
        for var, idx in reads.items():
            if var in init_roots:
                continue
            indices = idx if isinstance(idx, (list, tuple)) else (idx,)
            # "k0" marks the input of a smooth/delay callable: read at the
            # current step only at k=0 (at k>0 it reads the previous step)
            if init or any(i in ("k", "kl") for i in indices):
                for prod in producers.get(var, []):
                    if prod != name:
                        p.add(prod)
        preds[name] = p

    indeg = {m: len(p) for m, p in preds.items()}
    succ = defaultdict(set)
    for m, p in preds.items():
        for x in p:
            succ[x].add(m)

    ready = sorted(m for m in methods if indeg[m] == 0)
    order = []
    while ready:
        m = ready.pop(0)
        order.append(m)
        for c in sorted(succ[m]):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
        ready.sort()

    if len(order) != len(methods):
        cyclic = sorted(set(methods) - set(order))
        raise ValueError("circular dependency among: "
                         + ", ".join(cyclic))

    result = [(m, [a for a in inspect.signature(methods[m]).parameters
                   if a != "self"]) for m in order]
    _EXECUTION_ORDERS[key] = result
    return result

def plot_world_variables(time, var_data, var_names, var_lims,
                         img_background=None,
                         title=None,
                         figsize=None,
                         dist_spines=0.09,
                         grid=False):
    """
    Plots world state from an instance of World3 or any single sector.

    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    var_number = len(var_data)

    fig, host = plt.subplots(figsize=figsize)
    axs = [host, ]
    for i in range(var_number-1):
        axs.append(host.twinx())

    fig.subplots_adjust(left=dist_spines*2)
    for i, ax in enumerate(axs[1:]):
        ax.spines["left"].set_position(("axes", -(i + 1)*dist_spines))
        ax.spines["left"].set_visible(True)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')

    if img_background is not None:
        im = imread(img_background)
        axs[0].imshow(im, aspect="auto",
                      extent=[time[0], time[-1],
                              var_lims[0][0], var_lims[0][1]], cmap="gray")

    ps = []
    for ax, label, ydata, color in zip(axs, var_names, var_data, colors):
        ps.append(ax.plot(time, ydata, label=label, color=color)[0])
    axs[0].grid(grid)
    axs[0].set_xlim(time[0], time[-1])

    for ax, lim in zip(axs, var_lims):
        ax.set_ylim(lim[0], lim[1])

    for ax_ in axs:
        formatter_ = EngFormatter(places=0, sep="\N{THIN SPACE}")
        ax_.tick_params(axis='y', rotation=90)
        ax_.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax_.yaxis.set_major_formatter(formatter_)

    tkw = dict(size=4, width=1.5)
    axs[0].set_xlabel("time [years]")
    axs[0].tick_params(axis='x', **tkw)
    for i, (ax, p) in enumerate(zip(axs, ps)):
        ax.set_ylabel(p.get_label(), rotation="horizontal")
        ax.yaxis.label.set_color(p.get_color())
        ax.tick_params(axis='y', colors=p.get_color(), **tkw)
        ax.yaxis.set_label_coords(-i*dist_spines, 1.01)

    if title is not None:
        fig.suptitle(title, x=0.95, ha="right", fontsize=10)

    plt.tight_layout()

