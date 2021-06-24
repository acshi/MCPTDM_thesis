#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
import localreg

plt.rcParams.update({'font.size': 12})
plt.rcParams['pdf.fonttype'] = 42

show_only = False
make_pdf_also = False

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


class FigureKind:
    def __init__(self, param, ticks=None, val_names=None, locs=None):
        self.param = param
        self.ticks = ticks
        if val_names is None and ticks is not None:
            self.val_names = [str(val) for val in ticks]
        else:
            self.val_names = val_names
        if locs is None and ticks is not None:
            self.locs = [i for i in range(len(ticks))]
        else:
            self.locs = locs

    def filter_extra(self, results, extra_filters):
        return [entry for entry in results if all([f in entry["name"] for f in extra_filters])]

    def collect_vals(self, results, result_name):
        if self.val_names is None:
            print(
                f"OOPS! Tried to directly plot continuous variable {self.param} as discrete")
            return []
        else:
            return [[entry[result_name] for entry in results if f"_{self.param}_{val_name}_" in entry["name"]] for val_name in self.val_names]

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
            gcf = plt.gcf()
            gcf.set_figwidth(12)

            plt.tight_layout()
            if make_pdf_also:
                plt.savefig(f"figures/by_{self.param}{file_suffix}.pdf",
                            bbox_inches="tight", pad_inches=0)
            plt.savefig(f"figures/by_{self.param}{file_suffix}.png")

    def _plot(self, results, result_name, title, xlabel, ylabel, mode, extra_filters):
        plt.clf()
        has_any = False
        for mode_val in mode.values if mode else [None]:
            if mode_val is None:
                sub_results = results
            else:
                sub_results = mode.filter(results, mode_val)
            sub_results = self.filter_extra(sub_results, extra_filters)
            value_sets = self.collect_vals(sub_results, result_name)
            if len(value_sets) == 0:
                print(
                    f"Data completely missing for {result_name} with {extra_filters}")
                continue
            for i, vals in enumerate(value_sets):
                if len(vals) == 0:
                    print(
                        f"{mode_val} has 0 data points for '{self.ticks[i]}'")
                    vals.append(np.nan)
            has_any = True
            means = [np.mean(vals) for vals in value_sets]
            stdev_mean = [np.std(vals) / np.sqrt(len(vals))
                          for vals in value_sets]
            plt.errorbar(self.locs, means,
                         yerr=np.array(stdev_mean), label=translate(mode_val))
        if has_any:
            plt.xticks(self.locs, self.ticks)
            self._set_show_save(title, xlabel, ylabel,
                                file_suffix=f"_{result_name}{''.join(extra_filters)}")

    def scatter(self, results, result_name, title, xlabel, ylabel, mode, extra_filters):
        plt.clf()
        has_any = False
        for mode_val in mode.values:
            sub_results = self.filter_extra(
                mode.filter(results, mode_val), extra_filters)
            if len(sub_results) == 0:
                continue
            has_any = True
            all_xs = [entry[self.param] for entry in sub_results]
            all_ys = [entry[result_name] for entry in sub_results]
            plt.scatter(all_xs, all_ys, label=translate(mode_val))
        if has_any:
            self._set_show_save(
                title, xlabel, ylabel, file_suffix=f"_{result_name}{''.join(extra_filters)}")

    def localreg_estimate(self, results, result_name, title, xlabel, ylabel, mode, extra_filters):
        plt.clf()
        has_any = False
        for mode_val in mode.values:
            sub_results = self.filter_extra(
                mode.filter(results, mode_val), extra_filters)
            if len(sub_results) == 0:
                continue
            has_any = True

            all_xs = np.array([entry[self.param] for entry in sub_results])
            all_ys = np.array([entry[result_name] for entry in sub_results])

            sorted_is = np.argsort(all_xs)
            all_xs = all_xs[sorted_is]
            all_ys = all_ys[sorted_is]

            reg_ys = localreg.localreg(
                all_xs, all_ys, degree=0, kernel=localreg.rbf.gaussian, width=2)

            plt.scatter(all_xs, reg_ys, label=translate(mode_val))
        if has_any:
            self._set_show_save(
                title, xlabel, ylabel, file_suffix=f"_{result_name}{''.join(extra_filters)}")

    def plot(self, results, result_name, title=None, xlabel=None, ylabel=None, mode=None, extra_filters=[]):
        xlabel = xlabel or translate(self.param)
        ylabel = ylabel or translate(result_name)
        title = title or f"{translate(result_name)} by {translate(self.param).lower()}"
        print(f"{self.param} {result_name}")

        if self.ticks is None:
            self.localreg_estimate(results, result_name, title, xlabel, ylabel,
                                   mode, extra_filters)
        else:
            self._plot(results, result_name, title, xlabel, ylabel,
                       mode, extra_filters)


method_kind = FigureKind("method", ["tree", "mpdm", "eudm", "mcts"])
discount_kind = FigureKind("discount_factor", [0.6, 0.7, 0.8, 0.9, 1])
cfb_kind = FigureKind("use_cfb", ["false", "true"])
seconds_kind = FigureKind("seconds", None)

extra_accdec_kind = FigureKind("extra_ego_accdec_policies", [
                               "-1", "1", "-2", "2", "-1,1", "-2,2", "1,2", "-1,-2", "-1,-2,-5,1,2,5"])

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


method_mode = FigureMode("method", ["tree", "mpdm", "eudm", "mcts"])
cfb_mode = FigureMode("use_cfb", ["false", "true"])

seconds_kind.plot(results, "efficiency", mode=method_mode)
seconds_kind.plot(results, "cost", mode=method_mode)
seconds_kind.plot(results, "safety", mode=method_mode)
seconds_kind.plot(results, "ud", mode=method_mode)
seconds_kind.plot(results, "cc", mode=method_mode)

# extra_accdec_kind.plot(results, "efficiency", mode=method_mode)
# extra_accdec_kind.plot(results, "cost", mode=method_mode)
# extra_accdec_kind.plot(results, "safety", mode=method_mode)
# extra_accdec_kind.plot(results, "ud", mode=method_mode)
# extra_accdec_kind.plot(results, "cc", mode=method_mode)

# extra_accdec_kind.plot(results, "efficiency")
# extra_accdec_kind.plot(results, "cost")
# extra_accdec_kind.plot(results, "safety")
# extra_accdec_kind.plot(results, "ud")
# extra_accdec_kind.plot(results, "cc")

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
