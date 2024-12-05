import contextlib
import pygambit
import sys
from cgt_bandits import import_efg
from cgt_bandits.nodes import ChanceNode, TerminalNode, PersonalNode

import gurobipy as gp

from gurobipy import GRB

def sqf(I, S, seq, g):
    """
    Solve the Sequence Form Linear Program (SQF) for the given parameters.

    Parameters:
        I: Information sets for each player.
        S: Sequences for each player.
        seq: Function mapping information sets to sequences.
        g: Payoff matrix.

    Returns:
        The value of the game for the first player.
    """
    m = gp.Model("SQF")
    m.setParam("OutputFlag", 0)  # Suppress solver output

    # Decision variables
    r = m.addVars(S[0], name="r", lb=0, ub=1)  # Reach probabilities for Player 1
    v = m.addVars(I[1], name="v", lb=-GRB.INFINITY)  # Payoff variables for Player 2

    # Initial sequence probability constraint
    m.addConstr(r[()] == 1, name="RootSequence")

    # Objective function: Maximize Player 1's payoff
    # for s0 in S[0]:
    #     print("propability ", float(g.get((s0, ()), ([0.0, 0.0], 1.0))[1]))
    m.setObjective(
        gp.quicksum(
            float(g.get((s0, ()), ([0.0, 0.0], 0.0))[0][0]) *  # Player 1 payoff
            float(g.get((s0, ()), ([0.0, 0.0], 0.0))[1]) *     # Probability
            r[s0] for s0 in S[0]
        ) +
        gp.quicksum(v[j] for j in I[1] if seq[1][j] == ()),
        GRB.MAXIMIZE
    )

    # Payoff constraints for Player 2
    for i in I[1]:  # Iterate over Player 2's information sets
        sigma_1 = seq[1][i]  # Get the sequence for information set i
        actions = I[1][i]  # Get the available actions for information set i

        for a in actions:  # Iterate over actions in A_1(i)
            sigma_1_a = sigma_1 + (a,)  # Extend sequence by action a

            # Calculate payoff sum for Player 2
            # for s0 in S[0]:
            #     print("propability ", float(g.get((s0, sigma_1_a), ([0.0, 0.0], 1.0))[1]))
            payoff_sum = gp.quicksum(
                float(g.get((s0, sigma_1_a), ([0.0, 0.0], 0.0))[0][0]) *  # Player 2 payoff
                float(g.get((s0, sigma_1_a), ([0.0, 0.0], 0.0))[1]) *     # Probability
                r[s0] for s0 in S[0]
            )

            # Calculate continuation value for Player 2
            continue_value = gp.quicksum(v[j] for j in I[1] if seq[1][j] == sigma_1_a)

            # Add the payoff constraint
            m.addConstr(
                continue_value + payoff_sum >= v[i],
                name=f"Payoff-{i}-{a}"
            )

    # Flow conservation constraints for Player 1
    for i in I[0]:  # Iterate over Player 1's information sets
        sigma_0 = seq[0][i]
        actions = I[0][i]
        m.addConstr(
            gp.quicksum(r[sigma_0 + (a,)] for a in actions) == r[sigma_0],
            name=f"Flow-{i}"
        )

    m.update()

    # print("CONSTRAINTS:")
    # for constr in m.getConstrs():
    #     lhs_expr = m.getRow(constr)  # Get the linear expression (left-hand side) of the constraint
    #     rhs_value = constr.RHS           # Right-hand side value of the constraint
    #     sense = constr.Sense             # Constraint sense (<=, >=, =)
    #
    #     # Convert sense code to human-readable format
    #     if sense == gp.GRB.LESS_EQUAL:
    #         sense_str = "≤"
    #     elif sense == gp.GRB.GREATER_EQUAL:
    #         sense_str = "≥"
    #     elif sense == gp.GRB.EQUAL:
    #         sense_str = "="
    #
    #     # Build the left-hand side expression as a readable string
    #     lhs_terms = []
    #     for j in range(lhs_expr.size()):
    #         coeff = lhs_expr.getCoeff(j)
    #         var = lhs_expr.getVar(j)
    #
    #         # Formatting coefficient and variable
    #         if coeff == 1:  # Skip printing "1 *"
    #             term_str = f"{var.VarName}"
    #         elif coeff == -1:  # Handle "-1 *" as just "-"
    #             term_str = f"- {var.VarName}"
    #         else:
    #             term_str = f"{coeff} * {var.VarName}" if coeff > 0 else f"- {-coeff} * {var.VarName}"
    #
    #         lhs_terms.append(term_str)
    #
    #     # Combine terms with proper spacing for a clean look
    #     lhs_str = " + ".join(lhs_terms).replace("+ -", "- ")
    #
    #     # Print the complete constraint in a clear format
    #     print(f"{constr.ConstrName}: {lhs_str} {sense_str} {rhs_value}")

    # Optimize the model
    m.optimize()

    return m.ObjVal



def extract_parameters(efg):
    """
    Extract the Sequence Form Linear Program (SQF) parameters from a parsed
    game tree (efg root), with action labels including indices.

    Parameters:
        efg: The root node of the game tree after parsing with pygambit.

    Returns:
        I: A dictionary for each player where keys are information sets,
           and values are lists of available actions in those sets.
        S: Sequences for each player.
        seq: A dictionary where seq[0][i] maps Player 0's information set to its sequence,
             and seq[1][i] maps Player 1's information set to its sequence.
        g: Payoff matrix at terminal nodes with labeled actions.
    """
    from collections import defaultdict

    I = {0: {}, 1: {}}  # Information sets and available actions for each player
    S = defaultdict(set)  # Sequences per player
    g = {}  # Payoff matrix (sequence-pair to terminal payoffs)
    seq = {0: {}, 1: {}}  # Separate mappings for each player's info sets to sequences

    def label_action(action, infoset):
        """Generate a labeled action with its player and index."""
        return f"{action}_{infoset}"

    def traverse(node, sequences, probability=1):
        """
        Recursive function to traverse the game tree and extract parameters.
        """
        if isinstance(node, ChanceNode):
            # Chance node: propagate to children
            for prob, child in zip(node.action_probs, node.children):
                traverse(child, sequences, probability * prob)

        elif isinstance(node, PersonalNode):
            # Personal node: record information sets and sequences
            player = node.player
            infoset = node.infoset
            actions = node.action_names

            # Map infoset to sequence for the current player
            current_sequence = sequences[player]
            seq[player][infoset] = current_sequence

            # Add available actions for this infoset
            if infoset not in I[player]:
                I[player][infoset] = [label_action(action, infoset) for action in actions]

            # Generate labeled actions
            labeled_actions = I[player][infoset]

            # Traverse children with updated sequences
            for labeled_action, child in zip(labeled_actions, node.children):
                sequences[player] = current_sequence + (labeled_action,)
                S[player].add(sequences[player])

                traverse(child, sequences, probability)
            # Restore sequence after recursion
            sequences[player] = current_sequence

        elif isinstance(node, TerminalNode):
            # Terminal node: record payoffs with probabilities
            sequence_pair = tuple(sequences[p] for p in range(len(sequences)))
            g[sequence_pair] = tuple((node.payoffs, probability))

    # Initialize traversal
    initial_sequences = {0: (), 1: ()}
    S[0].add(())
    S[1].add(())
    traverse(efg, initial_sequences)

    # Debug prints for verification
    # print("I[0]: ", I[0])
    # print("I[1]: ", I[1])
    # print("S[0]: ", S[0])
    # print("S[1]: ", S[1])
    # print("seq[0]: ", seq[0])
    # print("seq[1]: ", seq[1])
    # print("g: ", g)

    return I, S, seq, g


def payoff(efg):
    """Computes the value of the extensive form game"""
    I, Sigma, seq, g = extract_parameters(efg)
    with contextlib.redirect_stdout(sys.stderr):
        p = sqf(I, Sigma, seq, g)

    return p


if __name__ == "__main__":
    efg = sys.stdin.read()
    game = pygambit.Game.parse_game(efg)
    root = import_efg.efg_to_nodes(game)

    print(payoff(root))
