#!/usr/bin/python3
import matplotlib.pyplot as plt

dy = -10
initial_dx = 100
branching_factor = 1
max_depth = 4

plt.gcf().set_figwidth(6.4)
plt.gcf().set_figheight(6.4 / 3)


def draw_level(f, depth, start_x, start_y, nums, best_path):
    dx = initial_dx / branching_factor**depth
    for (child_i, child_w) in enumerate(nums):
        end_x = start_x + dx / (branching_factor - 1) * child_i - 0.5 * dx
        end_y = start_y + dy
        on_best_path = best_path[:1] == [child_i]
        color = "red" if on_best_path else "black" if child_w > 1 else "#bbbbbb"
        # if child_w > 1:
        plt.plot([start_x, end_x], [start_y, end_y], '-o',
                 color=color, linewidth=child_w / 2, ms=child_w / 1.5)
        if child_w > 0 and depth + 1 < max_depth:
            vals = read_vals(f)
            actual_child_i = vals[0]
            if actual_child_i != child_i:
                print(f"At depth {depth} actual_child_i {actual_child_i} != child_i {child_i}")
                print(
                    f"Parent line had nums {' '.join(str(n) for n in nums)}, this child has {' '.join(str(v) for v in vals)}")
                quit()
            child_best_path = best_path[1:] if on_best_path else []
            draw_level(f, depth + 1, end_x, end_y, vals[1:], child_best_path)


def read_vals(f):
    line = f.readline()
    nums = [int(num) for num in line.split() if len(num) > 0]
    return nums


with open("tree_exploration_report", "r") as f:
    best_path = read_vals(f)
    nums = read_vals(f)
    branching_factor = len(nums)
    draw_level(f, 0, 0, 0, nums, best_path)
    for line in f:
        print(f"still have '{line}'")

plt.axis("off")
plt.tight_layout()
# plt.show()
plt.savefig(f"figures/pdf/tree_exploration_report.pdf",
            bbox_inches="tight", pad_inches=0)
plt.savefig(f"figures/tree_exploration_report.png")
