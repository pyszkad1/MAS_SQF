import contextlib
import pygambit
import sys

import gurobipy as gp
from gurobipy import GRB


def sqf(I, Sigma, A, seq, g):
    """The Sequence form linear program defined in terms of the parameters
    I, Sigma, A, seq, g."""

    # Implement this first. The function should be completely general. Do not
    # traverse the graph at this point! Simply use the gurobi modeling interface
    # and formulate the SQF. If you do this correctly, you should be able to
    # compute the payoff of the miners by switching the parameters.

    m = gp.Model("SQF")

    # Decision variables
    v = m.addVars(I[1], name="v", lb=-GRB.INFINITY, ub=GRB.INFINITY)  # Payoff variables for Player 2
    r = m.addVars(seq, name="r", lb=0)  # Reach probabilities for Player 1's sequences

    # Objective: maximize the payoff for Player 1
    m.setObjective(g.prod(r), GRB.MAXIMIZE)

    # Constraints
    # (1) Flow conservation constraints
    m.addConstr(r[()] == 1, name="Root sequence")  # Root sequence reaches with probability 1
    for sigma in Sigma[0]:  # Player 1 sequences
        if sigma not in A[0]:
            m.addConstr(r[sigma] == gp.quicksum(r[tau] for tau in seq if A[0][tau] == sigma), name=f"Flow-{sigma}")

    # (2) Payoff constraints for Player 2
    for i in I[1]:
        m.addConstr(v[i] >= gp.quicksum(g[(sigma, i)] * r[sigma] for sigma in seq), name=f"Payoff-{i}")

    m.optimize()

    return m.ObjVal


def extract_parameters(efg):
    """Converts an extensive form game into the SQF parameters:
    I, Sigma, A, seq, g."""

    # Implement this second. It does not matter how you implement the
    # parameters -- functions, classes, or dictionaries, anything will work.

    I = [[], []]  # Information sets for Player 1 and Player 2
    Sigma = [set(), set()]  # Sequence sets for Player 1 and Player 2
    A = [{}, {}]  # Maps sequences to available actions
    seq = set()  # All sequences
    g = {}  # Payoff matrix mapping (sequence, information set) to payoffs

    for player in range(2):
        I[player] = [infoset for infoset in efg.players[player].infosets]
        Sigma[player] = {seq for seq in efg.players[player].sequences}

        for infoset in I[player]:
            for action in infoset.actions:
                parent_seq = infoset.sequence
                child_seq = parent_seq + (action,)
                A[player][parent_seq] = action
                seq.add(child_seq)

    for sigma in Sigma[0]:
        for infoset in I[1]:
            g[(sigma, infoset)] = efg.utility(sigma, infoset)  # Get the utility of a sequence-infoset pair

    return I, Sigma, A, seq, g


def payoff(efg):
    """Computes the value of the extensive form game"""

    I, Sigma, A, seq, g = extract_parameters(efg)
    with contextlib.redirect_stdout(sys.stderr):
        p = sqf(I, Sigma, A, seq, g)

    return p


if __name__ == "__main__":
    efg = sys.stdin.read()
    game = pygambit.Game.parse_game(efg)
    root = import_efg.efg_to_nodes(game)

    print(payoff(root))
