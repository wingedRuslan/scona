from nilearn import plotting
import networkx as nx
import numpy as np


def graph_to_nilearn_array(
        G,
        node_colour_att=None,
        node_size_att=None,
        edge_attribute="weight"):
    """
    Derive from G (BrainNetwork Graph) the necessary inputs for the `nilearn`
    graph plotting functions.

    G : :class:`networkx.Graph`
        G should have nodal locations in MNI space indexed by nodal
        attribute "centroids"

    node_colour_att : str, optional
        index a nodal attribute to scale node colour by

    node_size_att : str, optional
        index a nodal attribute to scale node size by

    edge_attribute : str (optional, default = 'weight')
        The edge attribute that holds the numerical value used for the edge
        weight. If an edge does not have that attribute, then the value 1 is
        used instead.

    Returns
    -------
    (adjacency_matrix, node_coords, node_colour_att, node_size_att)
        adjacency_matrix - represents the link strengths of the graph;
        node_coords - 3d coordinates of the graph nodes in world space;
        node_colour_att - list of nodes colors if there is nodal attribute for it;        # noqa
        node_size_att - list of nodes sizes if there is nodal attribute for it.
    """

    # make ordered nodes to produce ordered rows and columns in adjacency matrix
    node_order = sorted(list(G.nodes()))

    # return the graph adjacency matrix as a NumPy matrix
    adjacency_matrix = nx.convert_matrix.to_numpy_matrix(G, nodelist=node_order,
                                                         weight=edge_attribute)

    # store nodes coordinates in NumPy array if nodal coordinates exist
    try:
        node_coords = np.array([G._node[node]["centroids"] for node in node_order])       # noqa
    except KeyError:
        raise ValueError("Graph does not contain nodal centroids")

    # create array to store color of each node if there is nodal attribure for colors     # noqa
    if node_colour_att is not None:
        try:
            node_colour_att = [G._node[node][node_colour_att] for node in node_order]     # noqa
        except KeyError:
            raise ValueError(
                "There is no nodal attribute - {}".format(node_size_att))

    if node_size_att is not None:
        try:
            node_size_att = [G._node[node][node_size_att] for node in node_order]          # noqa
        except KeyError:
            raise ValueError(
                "There is no nodal attribute - {}".format(node_size_att))

    return adjacency_matrix, node_coords, node_colour_att, node_size_att


def view_nodes_3d(
        G,
        node_size=5.,
        node_color='black'):
    """
    Plot nodes of a BrainNetwork using
    :func:`nilearn.plotting.view_markers()` tool.

    Insert a 3d plot of markers in a brain into an HTML page.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        G should have nodal locations in MNI space indexed by nodal
        attribute "centroids"

    node_size : float or array-like, optional (default=5.)
        Size of the nodes showing the seeds in pixels.

    node_colors : str or list of str (default 'black')
        node_colour determines the colour given to each node.
        If a single string is given, this string will be interpreted as a
        a colour, and all nodes will be rendered in this colour.
        If a list of colours is given, it must be the same length as the length
        of nodes coordinates.
    """

    # get the nodes coordinates
    a, node_coords, colour_list, z = graph_to_nilearn_array(G)

    #
    if isinstance(node_color, str):
        node_color = [node_color for i in range(len(node_coords))]


    # plot nodes
    ConnectomeView = plotting.view_markers(node_coords, marker_color=node_color,
                                           marker_size=node_size)

    return ConnectomeView
