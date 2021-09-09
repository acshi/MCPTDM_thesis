#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain

dy = -12
initial_dx = 30  # to the left and the right
example_dx = initial_dx * 2
branching_factor = 2
max_depth = 4
circle_r = 6.4

zero_prior_std_dev = 10.0
nominal_std_dev = 10.0

plt.gcf().set_figwidth(6.4 * 3)
plt.gcf().set_figheight(6.4 / 2 * 3)
plt.gca().set_aspect(1)

def prior_discounting(mean, std_dev):
    var = std_dev**2
    prior_var = zero_prior_std_dev**2
    return mean * prior_var / (prior_var + var)

def mean_or_zero(vals):
    if len(vals) == 0:
        return 0
    return np.sum(vals) / len(vals)

def std_dev_or_nominal(vals):
    if len(vals) == 0:
        return 0.0
    if len(vals) == 1:
        return nominal_std_dev
    return np.std(vals, ddof=1) / np.sqrt(len(vals))

class Node:
    def __init__(self, true_marginal_cost, observed_marginal_costs, children=[]):
        self.depth = 0
        self.true_marginal_cost = true_marginal_cost
        self.true_intermediate_cost = 0
        self.marginal_costs = observed_marginal_costs
        self.marginal_cost = mean_or_zero(observed_marginal_costs)
        self.marginal_cost_std_dev = std_dev_or_nominal(observed_marginal_costs)
        self.corrected_marginal_cost = prior_discounting(self.marginal_cost, self.marginal_cost_std_dev)
        self.intermediate_costs = self.marginal_costs
        self.intermediate_cost = self.marginal_cost
        self.expected_cost = None
        self.chosen_child_i = None
        self.children = children

        self.calculate_all()

    def is_leaf(self):
        return len(self.children) == 0

    def calculate_all(self, parent=None, parent_intermediate_costs=[]):
        if parent:
            self.depth = parent.depth + 1

            self.intermediate_costs = [parent_intermediate_costs[i] +
                                       m for (i, m) in enumerate(self.marginal_costs)]
            self.intermediate_cost = mean_or_zero(self.intermediate_costs)
            # self.marginal_cost = self.intermediate_cost - parent.intermediate_cost
            self.true_intermediate_cost = parent.true_intermediate_cost + self.true_marginal_cost
        else:
            self.depth = 0
            self.intermediate_costs = self.marginal_costs
            self.intermediate_cost = self.marginal_cost
            # self.marginal_cost = self.intermediate_cost
            self.true_intermediate_cost = self.marginal_cost

        parent_costs_i = 0
        for child in self.children:
            child_costs_n = len(child.marginal_costs)
            start_i = parent_costs_i
            end_i = start_i + child_costs_n
            child.calculate_all(self, self.intermediate_costs[start_i:end_i])
            parent_costs_i = end_i

        if len(self.children) == 0:
            self.costs = [self.intermediate_cost]
        else:
            self.costs = list(chain.from_iterable(c.costs for c in self.children))

    def mean_cost(self):
        return np.sum(self.costs) / len(self.costs)

    def min_child_expected_cost(self):
        if len(self.children) == 0:
            return None
        else:
            best_child_i = np.argmin([c.expected_cost for c in self.children])
            expected_cost = self.children[best_child_i].expected_cost
            return (best_child_i, expected_cost)


tree = Node(99.0, [0.0] * 6, children=[
    Node(99.0, [10.0, 11.0, 12.0], children=[
        Node(99.0, [20.0]),
        Node(99.0, [0.0, 40.0]),
    ]),
    Node(99.0, [20.0, 21.0, 22.0], children=[
        Node(99.0, [9.0]),
        Node(99.0, [0.0, 22.0]),
    ])
])


def draw_level(node, depth, start_x, start_y, in_best_path):
    plt.gca().add_artist(plt.Circle((start_x, start_y), circle_r, fill=True,
                                    zorder=100, edgecolor="black", facecolor="white", clip_on=False))
    display_str = str(node.display)
    fontsize = 13.5 if len(display_str) > 4 else 20.0
    plt.text(start_x, start_y, display_str, fontsize=fontsize,
             zorder=101, horizontalalignment='center', verticalalignment='center')

    dx = initial_dx / branching_factor**depth
    for (child_i, child) in enumerate(node.children):
        end_x = start_x + dx / (branching_factor - 1) * child_i - 0.5 * dx
        end_y = start_y + dy

        is_best_child = child_i == node.chosen_child_i
        child_best_path = in_best_path and is_best_child
        linewidth = 8 if child_best_path else 1

        # if child_best_path:
        #     import pdb; pdb.set_trace()

        plt.plot([start_x, end_x], [start_y, end_y], '-', color="black", linewidth=linewidth)

        draw_level(child, depth + 1, end_x, end_y, child_best_path)

def short_num(n):
    num = f"{float(n):.1f}"
    if num.endswith(".0"):
        num = num.replace(".0", "")
    return num

def push_down_expected_costs_and_display(name, node, cost):
    if name == "MAC w/ prior (MACP)":
        cost += node.corrected_marginal_cost
    elif name == "Marginal action costs (MAC)":
        cost += node.marginal_cost

    for child in node.children:
        push_down_expected_costs_and_display(name, child, cost)

    if node.is_leaf():
        node.display += f"\n{short_num(cost)}"

def calculate_expected_costs_and_display(name, node):
    for child in node.children:
        calculate_expected_costs_and_display(name, child)
    if name == "MAC w/ prior (MACP)":
        (node.chosen_child_i, node.expected_cost) = node.min_child_expected_cost() or (None, 0)
        node.expected_cost += node.corrected_marginal_cost

        if node.depth == 0:
            node.display = f"σ_p = {short_num(zero_prior_std_dev)}\nσ_n = {short_num(nominal_std_dev)}"
        else:
            node.display = "mac = "
            if len(node.marginal_costs) > 2:
                node.display += "\n"
            for i in range(0, len(node.marginal_costs), 3):
                if i > 0:
                    node.display += ",\n"
                node.display += ", ".join(short_num(c) for c in node.marginal_costs[i:i+3])
            node.display += f"\nm̂ = {short_num(node.corrected_marginal_cost)}"
            node.display += f"\nσ = {short_num(node.marginal_cost_std_dev)}"
            # print(node.display)
    elif name == "Marginal action costs (MAC)":
        (node.chosen_child_i, node.expected_cost) = node.min_child_expected_cost() or (None, 0)
        node.expected_cost += node.marginal_cost

        if node.depth == 0:
            node.display = ""
        else:
            node.display = "mac = "
            if len(node.marginal_costs) > 2:
                node.display += "\n"
            for i in range(0, len(node.marginal_costs), 3):
                if i > 0:
                    node.display += ",\n"
                node.display += ", ".join(short_num(c) for c in node.marginal_costs[i:i+3])
            node.display += f"\nm̄ = {short_num(node.marginal_cost)}"

            # if node.is_leaf():
            #     node.display += f"\nx̄ = {short_num(node.intermediate_cost)}"

            # print(node.display)
    else:
        node.display = 42


def draw_example(name, start_x, start_y, letter):
    plt.text(start_x - initial_dx * 0.8, start_y, letter, fontsize=24.0,
             zorder=101, horizontalalignment='center', verticalalignment='center')
    plt.text(start_x, start_y + 8, name, fontsize=18.0,
             zorder=101, horizontalalignment='center', verticalalignment='center')
    calculate_expected_costs_and_display(name, tree)
    push_down_expected_costs_and_display(name, tree, 0)
    draw_level(tree, 0, start_x, start_y, True)

set_dy = dy * 2 - 20

start_x = 0
start_y = 0
letter = "A"
draw_example("Marginal action costs (MAC)", start_x, start_y, letter)
start_x += example_dx
letter = chr(ord(letter) + 1)

start_x = 0
start_y += set_dy
draw_example("MAC w/ prior (MACP)", start_x, start_y, letter)
start_x += example_dx
letter = chr(ord(letter) + 1)


plt.axis("off")
plt.tight_layout()
# plt.show()
plt.savefig(f"figures/pdf/macp_example.pdf",
            bbox_inches="tight", pad_inches=0)
plt.savefig(f"figures/macp_example.png")
