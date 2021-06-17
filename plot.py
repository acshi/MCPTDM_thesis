#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 12})
plt.rcParams['pdf.fonttype'] = 42

show_only = True

envs = [""]
modes = [""]
formats = {"": "+-"}
labels = {"": "normal"}


class FigureKind:
    def __init__(self, param, ticks, names=None, locs=None):
        self.param = param
        self.ticks = ticks
        if names is None:
            self.names = [str(val) for val in ticks]
        else:
            self.names = names
        if locs is None:
            self.locs = [i for i in range(len(ticks))]
        else:
            self.locs = locs

        self.vals = dict()
        for env in envs:
            for mode in modes:
                self.vals[env + mode] = [list() for _ in range(len(ticks))]

    def check_file(self, filename, env, mode, values):
        for i in range(len(self.names)):
            if f"_{self.param}_{self.names[i]}_" in filename:
                self.vals[env + mode][i] += [values]
                break

    def _collect_vals(self, val_name, data):
        return [[values[val_name] for values in values_arr] for values_arr in data]

    def _set_show_save(self, title, xlabel, ylabel, file_suffix=""):
        plt.title(title)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if show_only:
            plt.show()
        else:
            ax = plt.gca()
            ax.set_aspect(1.0 / ax.get_data_ratio() * 0.4)
            plt.tight_layout()
            plt.savefig(f"by_{self.param}{file_suffix}.pdf",
                        bbox_inches="tight", pad_inches=0)
            plt.savefig(f"by_{self.param}{file_suffix}.png")

    def _plot(self, val_name, title, xlabel, ylabel, data, file_suffix=""):
        for mode in modes:
            plt.clf()
            has_any = False
            for env in envs:
                env_vals = self._collect_vals(val_name, data[env + mode])
                # print(f"{env + mode} has {len(env_vals[0])} values: {env_vals}")
                if len(env_vals[0]) == 0:
                    continue
                has_any = True
                means = [np.mean(vals) for vals in env_vals]
                stdev_mean = [np.std(vals) / np.sqrt(len(vals))
                              for vals in env_vals]
                plt.errorbar(self.locs, means, fmt=formats[env],
                             yerr=np.array(stdev_mean), label=labels[env])  # + "_" + mode)
            if has_any:
                plt.xticks(self.locs, self.ticks)
                self._set_show_save(title, xlabel, ylabel,
                                    file_suffix=f"_{file_suffix}{mode}")

    def plot(self, val_name, title, xlabel, ylabel):
        print(f"{self.param} {val_name}")
        self._plot(val_name, title, xlabel, ylabel, self.vals)

    def violinplot(self, title, xlabel, ylabel):
        plt.clf()
        for env in envs:
            for mode in modes:
                plt.violinplot(self.vals[env + mode],
                               self.locs, showmeans=True)
        plt.xticks(self.locs, self.ticks)
        self._set_show_save(title, xlabel, ylabel)

    def hist(self, title, xlabel, ylabel):
        plt.clf()
        for env in envs:
            for mode in modes:
                all_vals = [x for y in self.vals[env + mode] for x in y]
                plt.hist(all_vals)
        self._set_show_save(title, xlabel, ylabel)


method_kind = FigureKind("method", ["tree", "mpdm", "eudm", "mcts"])
discount_kind = FigureKind("discount", [0.6, 0.7, 0.8, 0.9, 1])

all_kinds = [method_kind,
             discount_kind]

with open("results.cache", "r") as f:
    for line in f:
        parts = line.split()
        if len(parts) > 4:
            name = parts[0]
            values = dict()
            values["efficiency"] = float(parts[1])
            values["safety"] = float(parts[2])
            values["smoothness"] = float(parts[3])
            values["seconds"] = float(parts[4])
            values["reward"] = values["efficiency"] + \
                values["safety"] + values["smoothness"]
        else:
            continue

        env = ""
        mode = ""

        for kind in all_kinds:
            kind.check_file(name, env, mode, values)

method_kind.plot("reward", "By method", "method", "reward")
method_kind.plot("efficiency", "Efficiency by method", "method", "efficiency")
method_kind.plot("safety", "Safety by method", "method", "safety")
method_kind.plot("smoothness", "Smoothness by method", "method", "smoothness")
method_kind.plot("seconds", "Compute time by method", "method", "time (s)")
discount_kind.plot("reward", "By discount", "discount", "reward")
# discount_kind.plot("efficiency", "By discount", "discount", "efficiency")
# discount_kind.plot("safety", "By discount", "discount", "safety")
# discount_kind.plot("smoothness", "By discount", "discount", "smoothness")
