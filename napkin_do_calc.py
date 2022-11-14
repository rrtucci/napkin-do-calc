from potentials.DiscreteCondPot import *
from graphs.BayesNet import *
from pprint import pprint


def create_random_bnet(nodes,
                       arrows,
                       nd_to_size):
    """
    This method returns a BayesNet object whose structure is given by
    'nodes' and 'arrows'. The TPM (transition probability matrix) for
    each node is created at random, with the only constraint being that
    the number of states of each node be as specified by the input
    'nd_to_size'.

    Parameters
    ----------
    nodes: list[str]
        example: ['a', 'b', 'c']
    arrows: list[tuple[str, str]]
        example: [('a', 'b'), ('a', 'c')]
    nd_to_size: dict[str, int]
        dictionary mapping node name to its size (i.e., the number of
        values or states)

    Returns
    -------
    BayesNet

    """
    bnet_nodes = []
    for k, node_name in enumerate(nodes):
        nd = BayesNode(k, name=node_name)
        nd.size = nd_to_size[node_name]
        bnet_nodes.append(nd)
    bnet = BayesNet(set(bnet_nodes))
    for arrow in arrows:
        pa_nd = bnet.get_node_named(arrow[0])
        child_nd = bnet.get_node_named(arrow[1])
        child_nd.add_parent(pa_nd)

    # print("ccvv", bnet.nodes)
    for nd in bnet_nodes:
        ord_nodes = list(nd.parents) + [nd]
        # print("llkjh", ord_nodes)
        nd.potential = DiscreteCondPot(False, ord_nodes)
        nd.potential.set_to_random()
        nd.potential.normalize_self()
    # print("aadf", bnet)
    return bnet


def run(draw=True, jupyter=False):
    """
    This method creates a random Napkin bnet, and evaluates the widely
    accepted adjustment formula (AF) for P(y|do(x)) for all z.

    Parameters
    ----------
    draw: bool
        Use True iff you want to draw the Napkin bnet
    jupyter: bool
        Use True if drawing to a jupyter notebook and False if drawing to
        the console.

    Returns
    -------
    None

    """
    nodes, arrows = DotTool.read_dot_file("dot_atlas/napkin.dot")
    nd_to_size = {}
    for nd in nodes:
        nd_to_size[nd] = 2
    nd_to_size["z"] = 3
    bnet = create_random_bnet(
        nodes,
        arrows,
        nd_to_size)
    print("Random Napkin bnet:")
    if draw:
        bnet.gv_draw(jupyter)
    print(bnet)

    node_list = list(bnet.nodes)
    pot_wzxy = node_list[0].potential
    for k in range(1, len(node_list)):
        pot_wzxy = pot_wzxy*node_list[k].potential

    nd_w = bnet.get_node_named('w')
    nd_z = bnet.get_node_named('z')
    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')

    pot_wz = pot_wzxy.get_new_marginal([nd_w, nd_z])
    pot_w = pot_wz.get_new_marginal([nd_w])

    unnormalized_pot = pot_wzxy*pot_w/pot_wz

    numer_pot = unnormalized_pot.get_new_marginal([nd_z, nd_x, nd_y])
    denom_pot = numer_pot.get_new_marginal([nd_z, nd_x])
    final_pot = numer_pot/denom_pot
    final_pot.set_to_transpose([nd_z, nd_x, nd_y])
    arr_zxy = final_pot.pot_arr

    arr_y_bar_x = final_pot.get_new_marginal([nd_x, nd_y]).pot_arr
    pot_y_bar_x = DiscreteCondPot(False, [nd_x, nd_y], arr_y_bar_x)
    pot_y_bar_x.normalize_self()
    prob_y_bar_x = pot_y_bar_x.pot_arr


    print("P(y|x):")
    pprint(prob_y_bar_x)

    print("\nP(y| do(x)) arrays for each value of z:")
    for z_val in range(nd_to_size['z']):
        print("\nz=", z_val, ", numpy array indices= ['x', 'y']")
        pprint(arr_zxy[z_val, :, :])
    print("\nSurprise!!!: P(y|do(x)) = P(y|x) for all z")


if __name__ == "__main__":
    run(draw=True, jupyter=False)
