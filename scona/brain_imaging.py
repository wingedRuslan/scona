import warnings

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from scona.visualisations_helpers import setup_color_list
from scona.visualisations_helpers import save_fig
from scona.visualisations_helpers import add_colorbar


def axial_layout(x, y, z):
    """
    Axial (horizontal)  plane, the plane that is horizontal and parallel to the
    axial plane of the body. It contains the lateral and the medial axes of the
    brain.

    Parameters
    ----------
    x, y, z : float
        Node Coordinates

    Returns
    -------
    numpy array
        The node coordinates excluding z-axis. `array([x, y])`

    """

    return np.array([x, y])


def sagittal_layout(x, y, z):
    """
    Sagittal plane, a vertical plane that passes from between the cerebral
    hemispheres, dividing the brain into left and right halves.

    Parameters
    ----------
    x, y, z : float
        Node Coordinates

    Returns
    -------
    numpy array
        The node coordinates excluding y-axis. `array([x, z])`

    """

    return np.array([x, z])


def coronal_layout(x, y, z):
    """
    Coronal (frontal) plane, a vertical plane that passes through both ears,
    and contains the lateral and dorsoventral axes.

    Parameters
    ----------
    x, y, z : float
        Node Coordinates

    Returns
    -------
    numpy array
        The node coordinates excluding y-axis. `array([y, z])`

    """

    return np.array([y, z])


def anatomical_layout(x, y, z, orientation="sagittal"):
    """
    This function extracts the required coordinates of a node based on the given
     anatomical layout.

    Parameters
    ----------
    x, y, z : float
        Node Coordinates

    orientation: str, (optional, default="sagittal)
        The name of the plane: `sagittal` or `axial` or `coronal`.

    Returns
    -------
    numpy array
        The node coordinates for the given anatomical layout.
    """

    if orientation == "sagittal":
        return sagittal_layout(x, y, z)
    if orientation == "axial":
        return axial_layout(x, y, z)
    if orientation == "coronal":
        return coronal_layout(x, y, z)
    else:
        raise ValueError(
            "{} is not recognised as an anatomical layout. orientation values "
            "should be one of 'sagittal', 'axial' or 'coronal'.".format(orientation))


def plot_anatomical_network(G10, G02, measure="module", orientation="sagittal",
                            node_list=None, node_shape='o', node_size=300,
                            rc_node=None, rc_node_shape='s',
                            edge_list=None, edge_color='lightgrey', edge_width=1,   # noqa
                            cmap_name="tab10", sns_palette=None, continuous=False,  # noqa
                            vmin=None, vmax=None, figure_name=None):
    """
    Plots each node in the graph in one of three orientations
    (sagittal, axial or coronal).
    The nodes are sorted according to the orientation given
    (default value: sagittal) and then plotted in that order. The color for each
    node is determined by the measure value (default measure: module).

    Parameters
    ----------
    G10 : :class:`BrainNetwork`
        A binary graph with 10% as many edges as the complete graph G

    G02 : :class:`BrainNetwork`
        A binary graph with 2% as many edges as the complete graph G

    measure: str, (optional, default="module")
        A nodal measure of a Graph.

    orientation : str, (optional, default="sagittal")
        The anatomical plane of the brain, i.g. sagittal, coronal, axial.

    node_list: list, optional
        List of nodes to display. By default, all nodes of graph will be
        displayed.

    node_shape : string
        The shape of the node.  Specification is as matplotlib.scatter marker,
        one of 'so^>v<dph8' (default='o').

    node_size : scalar or array
        Size of nodes (default=300).
        If an array is specified it must be the same length as node_list.
        The number in node_size at index <i> - is the size of the node at index
        <i> in node_list. For example, node_list=[5,6,7] and node_size=[100,200,
        300], the sizes for nodes will be {5:100, 6:200, 7:300}
        (100=node_size[0], node_list[0]=5 -> 5:100). If we plot all nodes
        (node_list=None), each element in node_size at index <i> is the size of
        <i> node in Graph.

    rc_node: list, optional
        List of nodes in the Graph that are rich club.

    rc_node_shape: string
        The shape of rich club nodes. Specification is as matplotlib.scatter
        marker, one of 'so^>v<dph8' (default='s').

    edge_list : list of tuples, optional
        List of edges, where each edge is represented as a tuple, e.g. (0,1).
        Edges of a graph can be accessed by executing `Graph.edges()`.

    edge_color: color string, or array of floats
        Edge color. Can be a single color format string (default='lightgrey'),
        or a sequence of colors with the same length as edgelist.

    edge_width: float, or array of floats
        Line width of edges (default=1.0)

    cmap_name:

    sns_palette: seaborn palette, (optional, default=None)
        Discrete color palette only for discrete data. List of colors defining
        a color palette (list of RGB tuples from seaborn color palettes).

    continuous: str, (optional, default=False)

    vmin :


    vmax :


    figure_name : str, optional
        path to the file to store the created figure in (e.g. "/home/Desktop/name")   # noqa
        or to store in the current directory include just a name ("fig_name").

    Returns
    -------
        Plot the Figure and if figure_name provided, save it in a figure_name
        file.

    """

    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)

    # get the graph's nodes if not provided
    if node_list is None:
        node_list = G10.nodes()
        node_list = sorted(node_list)

    # store node:size_node as a dict
    node_size_list = dict()

    # handle different types of node_size (scalar or array)
    if isinstance(node_size, int) or isinstance(node_size, float):
        for i in node_list:
            node_size_list[i] = node_size
    elif (isinstance(node_size, list) or isinstance(node_size, np.ndarray)) \
            and len(node_size) == len(node_list):
        for i, node in enumerate(node_list):
            node_size_list[node] = node_size[i]
    else:
        raise ValueError("node_size parameter must be either scalar or array. "
              "If an array is specified it must be the same length as nodelist")

    # set shape of nodes
    node_shape_list = dict()
    for i in node_list:
        node_shape_list[i] = node_shape

    # set shape of rich_club nodes if rc_node given
    if rc_node:
        for i in rc_node:
            node_shape_list[i] = rc_node_shape

    # report the attributes of each node in BrainNetwork Graph
    nodal_measures = G10.report_nodal_measures()

    # no centroids -> no plot of graph
    if not G10.graph["centroids"]:
        raise TypeError("There are no centroids (nodes coordinates) in the "
                        "Graph. Please initialise BrainNetwork with centroids.")

    # get the appropriate positions for each node based on the orientation
    pos = dict()

    for node in G10.nodes:
        pos[node] = anatomical_layout(G10.node[node]['x'], G10.node[node]['y'],
                                      G10.node[node]['z'],
                                          orientation=orientation)

    # We're going to figure out the best way to plot these nodes
    # so that they're sensibly on top of each other
    sort_dict = {}
    sort_dict['axial'] = 'z'
    sort_dict['coronal'] = 'y'
    sort_dict['sagittal'] = 'x'

    # sort nodes [from min_coordinate to max_coordinate]
    # where coordinate determined by the provided orientation
    # this is the order of displaying nodes based on the orientation
    node_order = np.argsort(nodal_measures[sort_dict[orientation]].values)

    # Now remove all the nodes that are not in the node_list
    node_order = [x for x in node_order if x in node_list]

    # get the color for each node based on the nodal measure
    if measure in nodal_measures.columns:
        colors_list = setup_color_list(df=nodal_measures, cmap_name=cmap_name,
                                       sns_palette=sns_palette, measure=measure,
                                       continuous=continuous, vmin=vmin, vmax=vmax)
    else:
        warnings.warn("Measure \"{}\" does not exist in the nodal attributes of Graph. "
                      "The default color will be used for all nodes.".format(measure))
        colors_list = ["blue"] * len(node_order)

    # Create figure
    fig_size_dict = {}
    fig_size_dict['axial'] = (9, 12)
    fig_size_dict['sagittal'] = (12, 9)
    fig_size_dict['coronal'] = (11, 9)

    # create the figure
    fig = plt.figure(figsize=fig_size_dict[orientation])

    # Height rations of the rows and no space between brain plot and colorbar plot
    grid = mpl.gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[50, 1],
                                 hspace=0)

    # left=0, right=1, bottom=0, top=1
    grid.update(bottom=0.1, top=1)

    # Add an axis to the big_fig
    ax_brain = plt.Subplot(fig, grid[0])
    fig.add_subplot(ax_brain)

    # loop through each node and plot one after each other
    for node in node_order:
        nx.draw_networkx_nodes(G10,
                               pos=pos,
                               node_color=colors_list[node],
                               node_size=node_size_list[node],
                               node_shape = node_shape_list[node],
                               nodelist=[node],
                               ax=ax_brain)

    # get the graph's edges if not provided
    if edge_list is None:
        # plot edges of Graph02 (thresholded at cost 2) - (nodes important, not edges)
        edge_list = list(G02.edges())

    # plot edges
    nx.draw_networkx_edges(G10,
                           pos=pos,
                           edgelist=edge_list,
                           edge_color=edge_color,
                           width=edge_width,
                           ax=ax_brain)

    # calculate vmin and vmax of data values if not passed
    if vmin is None:
        vmin = min(nodal_measures[measure].values)
    if vmax is None:
        vmax = max(nodal_measures[measure].values)

    # add the colorbar to the plot if plotting continuous data
    if continuous:
        add_colorbar(fig, grid[1], cb_name=measure, cmap_name=cmap_name,
                     vmin=vmin, vmax=vmax)

    # remove all spines from plot
    sns.despine(top=True, right=True, left=True, bottom=True)

    # display the figure
    plt.show()

    # save the figure if the location-to-save is provided
    if figure_name:
        # use the helper-function from module helpers to save the figure
        save_fig(fig, figure_name)
        # close the file after saving to a file
        plt.close(fig)
