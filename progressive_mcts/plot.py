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
translations["regret"] = "Regret"
translations["estimation_error"] = "Estimation error"
translations["samples_n"] = "# Samples"
translations["bound_mode"] = "Mode to estimate cost"
translations["normal"] = "Normal"
translations["lower_bound"] = "Using lower bound"
translations["marginal"] = "Using marginal action costs"


def translate(name, mode=None):
    if mode is not None:
        long_name = f"{mode.param}_{name}"
        if long_name in translations:
            return translations[long_name]

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
                             yerr=np.array(stdev_mean), label=translate(mode_val, mode))
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
            self.ax.scatter(all_xs, all_ys, label=translate(mode_val, mode))
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

            self.ax.scatter(all_xs, reg_ys, label=translate(mode_val, mode))
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
        if len(parts) > 3:
            entry = dict()
            entry["name"] = parts[0]
            entry["chosen_cost"] = float(parts[1])
            entry["chosen_true_cost"] = float(parts[2])
            entry["true_best_cost"] = float(parts[3])

            entry["regret"] = entry["chosen_true_cost"] - entry["true_best_cost"]
            entry["estimation_error"] = abs(entry["chosen_true_cost"] - entry["chosen_cost"])

            results.append(entry)
        else:
            continue


all_metrics = ["regret", "estimation_error"]

# print(max([r for r in results if all(f in r["name"] for f in ["_method_mcts_", "_use_cfb_true_",
#                                                               "_smoothness_0_", "_safety_100_", "_ud_5_"])], key=lambda entry: entry["safety"]))

if False:
    for use_cfb in ["true"]:
        for samples_n in [8, 16, 32, 64, 128, 256, 512]:
            evaluate_conditions(results, all_metrics, [
                                ("method", "mcts"), ("search_depth", 4), ("layer_t", "2"),
                                ("smoothness", 0), ("safety", 100), ("ud", 5),
                                ("samples_n", samples_n), ("use_cfb", use_cfb)])
    # print("layer_t 2")
    # for use_cfb in ["false", "true"]:
    #     for search_depth in [4, 5, 6, 7]:
    #         evaluate_conditions(results, all_metrics, [
    #                             ("method", "mcts"), ("search_depth", search_depth), ("layer_t", "2"), ("samples_n", 128), ("use_cfb", use_cfb)])

    print("eudm")
    for use_cfb in ["true"]:
        for search_depth in [4]:
            evaluate_conditions(results, all_metrics, [
                                ("method", "eudm"),
                                ("smoothness", 0), ("safety", 100), ("ud", 5),
                                ("search_depth", search_depth), ("use_cfb", use_cfb)])


samples_n_kind = FigureKind("samples_n", [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])

use_low_bound_mode = FigureMode("bound_mode", ["normal", "lower_bound", "marginal"])

# cargo run --release rng_seed 0-127 :: samples_n 8 16 32 64 128 256 512 :: use_low_bound false true :: thread_limit 8
if True:
    for metric in all_metrics:
        samples_n_kind.plot(results, metric, filters=[], mode=use_low_bound_mode)
