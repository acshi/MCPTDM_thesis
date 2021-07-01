#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
import localreg

plt.rcParams.update({'font.size': 12})
plt.rcParams['pdf.fonttype'] = 42

show_only = False
make_pdf_also = False

save_dpi = 200

formats = {"": "+-"}
labels = {"": "normal"}

translations = dict()
translations["discount_factor"] = "Discount Factor"
translations["safety"] = "Safety"
translations["cost"] = "Cost"
translations["efficiency"] = "Efficiency"
translations["ud"] = "Uncomfortable decelerations"
translations["cc"] = "Curvature change"
translations["tree"] = "Tree"
translations["mpdm"] = "MPDM"
translations["eudm"] = "EUDM"
translations["mcts"] = "MCTS"
translations["method"] = "Method"
translations["false"] = "Normal"
translations["true"] = "CFB"
translations["seconds"] = "Computation time (s)"
translations["search_depth"] = "Search depth"
translations["samples_n"] = "# Samples"
translations[None] = "Average"


def translate(name):
    if name in translations:
        return translations[name]
    return name


class FigureMode:
    def __init__(self, param, values):
        self.param = param
        self.values = values

    def filter(self, results, value):
        return [entry for entry in results if f"_{self.param}_{value}_" in entry["name"]]


def filter_extra(results, filters):
    return [entry for entry in results if all([f in entry["name"] for f in filters])]


class FigureKind:
    def __init__(self, param, ticks=None, val_names=None, locs=None, xlim=None):
        self.param = param
        self.ticks = ticks
        self.xlim = xlim
        if val_names is None and ticks is not None:
            self.val_names = [str(val) for val in ticks]
        else:
            self.val_names = val_names
        if locs is None and ticks is not None:
            self.locs = [i for i in range(len(ticks))]
        else:
            self.locs = locs

    def collect_vals(self, results, result_name):
        if self.val_names is None:
            print(
                f"OOPS! Tried to directly plot continuous variable {self.param} as discrete")
            return []
        else:
            return [[entry[result_name] for entry in results if f"_{self.param}_{val_name}_" in entry["name"]] for val_name in self.val_names]

    def _set_show_save(self, title, xlabel, ylabel, result_name, mode, filters):
        self.ax.set_title(title)
        self.ax.legend()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        if show_only:
            plt.show()
        else:
            # self.fig.set_figwidth(12)
            # self.ax.set_aspect(1.0 / self.ax.get_data_ratio() * 0.4)

            mode_string = f"_{mode.param}" if mode is not None else ""
            filters_string = "_" + \
                "_".join([f.strip("_") for f in filters]
                         ) if len(filters) > 0 else ""
            file_suffix = f"_{result_name}{mode_string}{filters_string}"

            self.fig.tight_layout()
            if make_pdf_also:
                self.fig.savefig(f"figures/by_{self.param}{file_suffix}.pdf",
                                 bbox_inches="tight", pad_inches=0)
            self.fig.savefig(f"figures/by_{self.param}{file_suffix}.png")

    def _plot(self, results, result_name, title, xlabel, ylabel, mode, filters):
        self.fig, self.ax = plt.subplots(dpi=100 if show_only else save_dpi)

        has_any = False
        for mode_val in mode.values if mode else [None]:
            if mode_val is None:
                sub_results = results
            else:
                sub_results = mode.filter(results, mode_val)
            sub_results = filter_extra(sub_results, filters)
            value_sets = self.collect_vals(sub_results, result_name)
            if len(value_sets) == 0:
                print(
                    f"Data completely missing for {result_name} with {filters}")
                continue
            for i, vals in enumerate(value_sets):
                if len(vals) == 0:
                    print(
                        f"{mode_val} has 0 data points for {self.param} '{self.ticks[i]}'")
                    vals.append(np.nan)
            has_any = True
            means = [np.mean(vals) for vals in value_sets]
            stdev_mean = [np.std(vals) / np.sqrt(len(vals))
                          for vals in value_sets]
            self.ax.errorbar(self.locs, means,
                             yerr=np.array(stdev_mean), label=translate(mode_val))
        if has_any:
            self.ax.set_xticks(self.locs)
            self.ax.set_xticklabels(self.ticks)
            self._set_show_save(title, xlabel, ylabel,
                                result_name, mode, filters)

    def scatter(self, results, result_name, title, xlabel, ylabel, mode, filters):
        self.fig, self.ax = plt.subplots(dpi=100 if show_only else save_dpi)

        has_any = False
        for mode_val in mode.values:
            sub_results = filter_extra(
                mode.filter(results, mode_val), filters)
            if len(sub_results) == 0:
                continue
            has_any = True
            all_xs = [entry[self.param] for entry in sub_results]
            all_ys = [entry[result_name] for entry in sub_results]
            self.ax.scatter(all_xs, all_ys, label=translate(mode_val))
        if has_any:
            self._set_show_save(title, xlabel, ylabel,
                                result_name, mode, filters)

    def localreg_estimate(self, results, result_name, title, xlabel, ylabel, mode, filters):
        self.fig, self.ax = plt.subplots(dpi=100 if show_only else save_dpi)

        has_any = False
        for mode_val in mode.values:
            sub_results = filter_extra(
                mode.filter(results, mode_val), filters)
            if len(sub_results) == 0:
                continue
            has_any = True

            all_xs = np.array([entry[self.param] for entry in sub_results])
            all_ys = np.array([entry[result_name] for entry in sub_results])

            sorted_is = np.argsort(all_xs)
            all_xs = all_xs[sorted_is]
            all_ys = all_ys[sorted_is]

            reg_ys = localreg.localreg(
                all_xs, all_ys, degree=0, kernel=localreg.rbf.gaussian, width=0.02)

            self.ax.scatter(all_xs, reg_ys, label=translate(mode_val))
        if has_any:
            self._set_show_save(title, xlabel, ylabel,
                                result_name, mode, filters)

    def plot(self, results, result_name, title=None, xlabel=None, ylabel=None, mode=None, filters=[]):
        xlabel = xlabel or translate(self.param)
        ylabel = ylabel or translate(result_name)
        title = title or f"{translate(result_name)} by {translate(self.param).lower()}"
        print(f"{self.param} {result_name}")

        if self.ticks is None:
            self.localreg_estimate(results, result_name, title, xlabel, ylabel,
                                   mode, filters)
        else:
            self._plot(results, result_name, title, xlabel, ylabel,
                       mode, filters)


def evaluate_conditions(results, metrics, conditions):
    results = [entry for entry in results if all(
        [f"_{c[0]}_{c[1]}_" in entry["name"] for c in conditions])]
    conditions_string = "_".join([f"{c[0]}_{c[1]}" for c in conditions])

    print(f"{conditions_string}:")

    for metric in metrics:
        vals = [entry[metric] for entry in results]
        mean = np.mean(vals)
        stdev_mean = np.std(vals) / np.sqrt(len(vals))
        print(f"  {metric} has mean: {mean:6.4} and mean std dev: {stdev_mean:6.4}")
    print()


results = []
with open("results.cache", "r") as f:
    for line in f:
        parts = line.split()
        if len(parts) > 10:
            entry = dict()
            entry["name"] = parts[0]
            entry["efficiency"] = float(parts[6])
            entry["safety"] = float(parts[7])
            entry["ud"] = float(parts[8])
            entry["cc"] = float(parts[9])
            entry["seconds"] = float(parts[10])

            cost_efficiency = float(parts[1])
            cost_safety = float(parts[2])
            cost_smoothness = float(parts[3])
            cost_ud = float(parts[4])
            cost_cc = float(parts[5])
            cost = cost_efficiency + cost_safety + cost_smoothness + cost_ud + cost_cc
            entry["cost"] = cost

            results.append(entry)
        else:
            continue


method_kind = FigureKind("method", ["tree", "mpdm", "eudm", "mcts"])
discount_kind = FigureKind("discount_factor", [0.6, 0.7, 0.8, 0.9, 1])
search_depth_kind = FigureKind("search_depth", [3, 4, 5, 6, 7])
cfb_kind = FigureKind("use_cfb", ["false", "true"])
seconds_kind = FigureKind("seconds", None, xlim=(0, 1.0))

samples_n_kind = FigureKind("samples_n", [8, 16, 32, 64, 128, 256, 512])

# extra_accdec_kind = FigureKind("extra_ego_accdec_policies", [
#                                "-1", "1", "-2", "2", "-1,1", "-2,2", "1,2", "-1,-2", "-1,-2,-3,1,2,3"])

extra_accdec_kind = FigureKind("extra_ego_accdec_policies", [
                               "", "-1,-2,1,2", "-1,-2,-3,1,2,3"])

method_mode = FigureMode("method", ["fixed", "tree", "mpdm", "eudm", "mcts"])
cfb_mode = FigureMode("use_cfb", ["false", "true"])

metrics = ["efficiency", "safety", "seconds"]

print(max(results, key=lambda entry: entry["safety"]))

# for use_cfb in ["false", "true"]:
#     for samples_n in [128, 256, 512]:
#         evaluate_conditions(results, metrics, [
#                             ("method", "mcts"), ("search_depth", 8), ("layer_t", "1"), ("samples_n", samples_n), ("use_cfb", use_cfb)])
# print("layer_t 2")
# for use_cfb in ["false", "true"]:
#     for search_depth in [4, 5, 6, 7]:
#         evaluate_conditions(results, metrics, [
#                             ("method", "mcts"), ("search_depth", search_depth), ("layer_t", "2"), ("samples_n", 128), ("use_cfb", use_cfb)])

# print("eudm")
# for use_cfb in ["false", "true"]:
#     for search_depth in [4, 5, 6, 7]:
#         evaluate_conditions(results, metrics, [
#                             ("method", "eudm"), ("search_depth", search_depth), ("use_cfb", use_cfb)])


# samples_filters = ["_method_mcts_", "_use_cfb_true_"]
# samples_n_kind.plot(results, "efficiency", filters=samples_filters)
# samples_n_kind.plot(results, "cost", filters=samples_filters)
# samples_n_kind.plot(results, "safety", filters=samples_filters)
# samples_n_kind.plot(results, "ud", filters=samples_filters)
# samples_n_kind.plot(results, "cc", filters=samples_filters)
# samples_n_kind.plot(results, "seconds", filters=samples_filters)

# cargo run --release rng_seed 0-63 :: method mpdm mcts eudm :: mcts.search_depth 4-7 :: mcts.samples_n 8 16 32 64 128 256 512 :: use_cfb false true :: thread_limit 24 && cargo run --release rng_seed 0-15 :: method tree :: tree.samples_n 1 2 4 8 :: use_cfb false true :: thread_limit 24
# cargo run --release rng_seed 64-127 :: method mpdm mcts eudm :: mcts.search_depth 4-7 :: mcts.samples_n 8 16 32 64 128 256 512 :: use_cfb false true :: thread_limit 24 && cargo run --release rng_seed 16-31 :: method tree :: tree.samples_n 1 2 4 8 :: use_cfb false true :: thread_limit 24

# seconds_kind.plot(results, "efficiency", mode=method_mode)
# seconds_kind.plot(results, "cost", mode=method_mode)
# seconds_kind.plot(results, "safety", mode=method_mode)
# seconds_kind.plot(results, "ud", mode=method_mode)
# seconds_kind.plot(results, "cc", mode=method_mode)

# seconds_kind.plot(results, "efficiency", mode=cfb_mode)
# seconds_kind.plot(results, "cost", mode=cfb_mode)
# seconds_kind.plot(results, "safety", mode=cfb_mode)
# seconds_kind.plot(results, "ud", mode=cfb_mode)
# seconds_kind.plot(results, "cc", mode=cfb_mode)

# mcts_filter = ["_method_mcts_", "_use_cfb_true_"]
# samples_mode = None  # FigureMode("samples_n", ["8", "16", "32", "64", "128", "256", "512"])
# mcts_search_depth_kind = FigureKind("search_depth", [4, 5, 6, 7])
# mcts_search_depth_kind.plot(results, "efficiency", mode=samples_mode, filters=mcts_filter)
# mcts_search_depth_kind.plot(results, "cost", mode=samples_mode, filters=mcts_filter)
# mcts_search_depth_kind.plot(results, "safety", mode=samples_mode, filters=mcts_filter)
# mcts_search_depth_kind.plot(results, "ud", mode=samples_mode, filters=mcts_filter)
# mcts_search_depth_kind.plot(results, "cc", mode=samples_mode, filters=mcts_filter)

# search_depth_kind.plot(results, "efficiency", mode=method_mode)
# search_depth_kind.plot(results, "cost", mode=method_mode)
# search_depth_kind.plot(results, "safety", mode=method_mode)
# search_depth_kind.plot(results, "ud", mode=method_mode)
# search_depth_kind.plot(results, "cc", mode=method_mode)

# search_depth_kind.plot(results, "efficiency", mode=cfb_mode)
# search_depth_kind.plot(results, "cost", mode=cfb_mode)
# search_depth_kind.plot(results, "safety", mode=cfb_mode)
# search_depth_kind.plot(results, "ud", mode=cfb_mode)
# search_depth_kind.plot(results, "cc", mode=cfb_mode)

# for fs in ["_use_cfb_false_", "_use_cfb_true_"]:
#     search_depth_kind.plot(results, "efficiency", mode=method_mode, filters=[fs])
#     search_depth_kind.plot(results, "cost", mode=method_mode, filters=[fs])
#     search_depth_kind.plot(results, "safety", mode=method_mode, filters=[fs])
#     search_depth_kind.plot(results, "ud", mode=method_mode, filters=[fs])
#     search_depth_kind.plot(results, "cc", mode=method_mode, filters=[fs])

# extra_accdec_kind.plot(results, "efficiency", mode=method_mode)
# extra_accdec_kind.plot(results, "cost", mode=method_mode)
# extra_accdec_kind.plot(results, "safety", mode=method_mode)
# extra_accdec_kind.plot(results, "ud", mode=method_mode)
# extra_accdec_kind.plot(results, "cc", mode=method_mode)

# extra_accdec_kind.plot(results, "efficiency", mode=cfb_mode)
# extra_accdec_kind.plot(results, "cost", mode=cfb_mode)
# extra_accdec_kind.plot(results, "safety", mode=cfb_mode)
# extra_accdec_kind.plot(results, "ud", mode=cfb_mode)
# extra_accdec_kind.plot(results, "cc", mode=cfb_mode)

# discount_kind.plot(results, "cost", mode=method_mode)
# discount_kind.plot(results, "efficiency", mode=method_mode)
# discount_kind.plot(results, "safety", mode=method_mode)
# discount_kind.plot(results, "ud",
#                    ylabel="Uncomfortable deceleration per km", mode=method_mode)
# discount_kind.plot(results, "cc",
#                    ylabel="Curvature changes per km", mode=method_mode)

# method_kind.plot(results, "cost", mode=cfb_mode)
# method_kind.plot(results, "efficiency", mode=cfb_mode)
# method_kind.plot(results, "safety", mode=cfb_mode)
# method_kind.plot(
#     results, "ud", ylabel="Uncomfortable deceleration per km", mode=cfb_mode)
# method_kind.plot(results, "cc", "Curvature change by method",
#                  ylabel="Curvature changes per km", mode=cfb_mode)
# method_kind.plot(results, "seconds", title="Compute time by method",
#                  ylabel="time (s)", mode=cfb_mode)
