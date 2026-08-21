"""Smoke tests for the plotting layer.

These deliberately assert almost nothing about the *content* of a figure --
only that each entry point runs to completion and writes a non-empty file.
That is enough to catch the class of breakage that actually bites: an upstream
matplotlib API removal.  `matplotlib.cm.get_cmap` disappeared in 3.11 and took
out every plot call in htpolynet, and because this module sat at 6.7% coverage
nothing noticed until a cluster run died after densification.

Every function below routes through ``_get_cmap``, so these run against
whatever matplotlib the environment happens to resolve.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import shutil

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

from htpolynet.analysis.plot import (
    _get_cmap, scatter, multi_trace, global_trace, network_graph, draw_reaction_dag,
)

needs_dot = pytest.mark.skipif(shutil.which('dot') is None, reason='graphviz dot not on PATH')


@pytest.fixture
def df():
    t = np.linspace(0.0, 100.0, 50)
    return pd.DataFrame({
        'time(ps)': t,
        'Density': 1000.0 + 10.0 * np.sin(t / 10.0),
        'Temperature': 300.0 + t / 10.0,
        'Potential': -5.0e4 + t,
    })


def _written(path):
    return path.is_file() and path.stat().st_size > 0


# --- the colormap accessor itself -----------------------------------------

@pytest.mark.parametrize('name', ['plasma', 'seismic', 'viridis'])
def test_get_cmap_returns_a_callable_colormap(name):
    cmap = _get_cmap(name)
    assert callable(cmap)
    assert len(cmap(0.5)) == 4      # RGBA


def test_get_cmap_rejects_an_unknown_name():
    with pytest.raises(Exception):
        _get_cmap('not-a-colormap')


# --- figure-producing entry points ----------------------------------------

def test_scatter_writes_a_figure(df, tmp_path):
    out = tmp_path / 'scatter.png'
    scatter(df, 'time(ps)', columns=['Density', 'Temperature'], outfile=str(out))
    assert _written(out)


def test_scatter_with_no_y_columns_still_writes(df, tmp_path):
    out = tmp_path / 'empty.png'
    scatter(df, 'time(ps)', columns=[], outfile=str(out))
    assert _written(out)


def test_multi_trace_writes_a_figure(df, tmp_path):
    out = tmp_path / 'multi.png'
    multi_trace([df, df], ['time(ps)'] * 2, ['Density'] * 2,
                labels=['a', 'b'], outfile=str(out))
    assert _written(out)


def test_multi_trace_rejects_mismatched_x_and_y_lists(df, tmp_path):
    with pytest.raises(AssertionError):
        multi_trace([df], ['time(ps)', 'time(ps)'], ['Density'],
                    labels=['a'], outfile=str(tmp_path / 'x.png'))


def test_global_trace_writes_a_figure(df, tmp_path):
    out = tmp_path / 'global.png'
    global_trace(df, ['Density', 'Temperature'], outfile=str(out))
    assert _written(out)


def test_global_trace_with_transitions_and_labels(df, tmp_path):
    """interval_labels must be one shorter than transition_times."""
    out = tmp_path / 'global2.png'
    global_trace(df, ['Density'], outfile=str(out),
                 transition_times=[0.0, 50.0, 100.0],
                 interval_labels=['heat', 'cool'])
    assert _written(out)


def test_global_trace_with_a_secondary_axis(df, tmp_path):
    out = tmp_path / 'global3.png'
    global_trace(df, ['Density'], y2names=['Temperature'], outfile=str(out))
    assert _written(out)


def test_network_graph_writes_a_figure(tmp_path):
    G = nx.Graph()
    for i in range(6):
        G.add_node(i, molecule_name='BPA' if i % 2 else 'TAZ')
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    out = tmp_path / 'graph.png'
    network_graph(G, str(out), figsize=(4, 4))
    assert _written(out)


def test_network_graph_handles_nodes_without_molecule_names(tmp_path):
    """Nodes lacking 'molecule_name' fall back to 'anonymous'."""
    G = nx.path_graph(4)
    out = tmp_path / 'anon.png'
    network_graph(G, str(out), figsize=(4, 4))
    assert _written(out)


@needs_dot
def test_draw_reaction_dag_writes_a_figure(tmp_path):
    """Regression guard for the container that shipped without graphviz."""
    from htpolynet.cure.reaction import Reaction
    r = Reaction({
        'name': 'etherify',
        'stage': 'cure',
        'reactants': {1: 'BPA', 2: 'TAZ'},
        'product': 'BPA~O1-C1~TAZ',
        'probability': 1.0,
        'atoms': {'A': {'reactant': 1, 'resid': 1, 'atom': 'O1', 'z': 1},
                  'B': {'reactant': 2, 'resid': 1, 'atom': 'C1', 'z': 1}},
        'bonds': [{'atoms': ['A', 'B'], 'order': 1}],
    })
    out = tmp_path / 'dag.png'
    draw_reaction_dag([r], str(out))
    assert _written(out)
