#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
import localreg
import numbers

plt.rcParams.update({'font.size': 12})
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628"]) + plt.cycler(marker=['+', 'o', 'x', '^', 'v'])


show_only = False
make_pdf_also = False

save_dpi = 300
figure_zoom = 1.75


class FigureMode:
    def __init__(self, param, values):
        self.param = param
        self.values = values

    def filter(self, results, value):
        return [entry for entry in results if f",{self.param}={value}," in entry["name"]]


def filter_match(name, filter):
    param = filter[0]
    param_value = filter[1]
    param_split = param.split(".")

    if len(param_split) == 2:
        if param_split[0] == "max":
            name_value = float(name.split(f"{param_split[1]}=")[1].split(",")[0])
            return name_value <= param_value
        elif not f",method={param_split[0]}," in name:
            return True
        param = param_split[1]

    return f",{param}={param_value}," in name


def filter_extra(results, filters):
    return [entry for entry in results if all(filter_match(entry["name"], f) for f in filters)]


def short_num_string(val):
    scientific = f"{val:.1e}".replace(".0e", "e").replace(
        "+", "").replace("e0", "e").replace("e-0", "e-")
    normal = str(val)
    return scientific if len(scientific) < len(normal) else normal


class FigureKind:
    def __init__(self, param, ticks=None, val_names=None, locs=None, xlim=None, translations={}):
        self.param = param
        self.ticks = ticks
        self.xlim = xlim
        if val_names is None and ticks is not None:
            self.val_names = [str(val) for val in ticks]
            self.tick_labels = self.val_names

            if any(isinstance(val, numbers.Number) and abs(val) >= 1e5 for val in ticks):
                self.tick_labels = [short_num_string(val) for val in ticks]
        else:
            self.val_names = val_names
        if locs is None and ticks is not None:
            self.locs = [i for i in range(len(ticks))]
        else:
            self.locs = locs
        self.translations = translations

    def translate(self, name):
        if name in self.translations:
            return self.translations[name]
        return name

    def collect_vals(self, results, result_name):
        if self.val_names is None:
            print(
                f"OOPS! Tried to directly plot continuous variable {self.param} as discrete")
            return []
        else:
            return [[entry[result_name] for entry in results if f",{self.param}={val_name}," in entry["name"]] for val_name in self.val_names]

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
            self.fig.set_figwidth(6.4 * figure_zoom)
            self.fig.set_figheight(4.8 * figure_zoom)

            mode_string = f"_{mode.param}" if mode is not None else ""
            filters_string = "_" + \
                "_".join(f"{f[0]}_{f[1]}" for f in filters
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
            n_vals_in_set = len(value_sets[0])
            for i, vals in enumerate(value_sets):
                if len(vals) == 0:
                    print(
                        f"{mode_val} has 0 data points for {self.param} '{self.ticks[i]}'")
                    vals.append(np.nan)
                if len(vals) != n_vals_in_set:
                    print(f"{len(vals)} != {n_vals_in_set} for {self.ticks[i]}")
            has_any = True
            means = [np.mean(vals) for vals in value_sets]
            stdev_mean = [np.std(vals) / np.sqrt(len(vals))
                          for vals in value_sets]
            self.ax.errorbar(self.locs, means,
                             yerr=np.array(stdev_mean), label=self.translate(mode_val))
        if has_any:
            self.ax.set_xticks(self.locs)
            self.ax.set_xticklabels(self.tick_labels)
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
            self.ax.scatter(all_xs, all_ys, label=self.translate(mode_val))
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

            self.ax.scatter(all_xs, reg_ys, label=self.translate(mode_val))
        if has_any:
            self._set_show_save(title, xlabel, ylabel,
                                result_name, mode, filters)

    def plot(self, results, result_name, title=None, xlabel=None, ylabel=None, mode=None, filters=[]):
        xlabel = xlabel or self.translate(self.param)
        ylabel = ylabel or self.translate(result_name)
        title = title or f"{self.translate(result_name)} by {self.translate(self.param).lower()}"
        print(f"{self.param} {result_name}")

        if self.ticks is None:
            self.localreg_estimate(results, result_name, title, xlabel, ylabel,
                                   mode, filters)
        else:
            self._plot(results, result_name, title, xlabel, ylabel,
                       mode, filters)


def evaluate_conditions(results, metrics, filters):
    results = filter_extra(results, filters)
    filters_string = ",".join([f"{f[0]}={f[1]}" for f in filters])

    print(f"{filters_string}:")

    for metric in metrics:
        vals = [entry[metric] for entry in results]
        mean = np.mean(vals)
        stdev_mean = np.std(vals) / np.sqrt(len(vals))
        print(f"  {metric} has mean: {mean:6.4} and mean std dev: {stdev_mean:6.4}")
    print()


def print_all_parameter_values_used(results, filters):
    param_sets = {}
    for result in filter_extra(results, filters):
        name = result["name"]
        for pair in name.split(","):
            if len(pair) == 0:
                continue
            pair_parts = pair.split("=")
            param_name = pair_parts[0]
            param_value = pair_parts[1]
            if param_name not in param_sets:
                param_sets[param_name] = {}
            param_set = param_sets[param_name]
            if param_value not in param_set:
                param_set[param_value] = 0
            param_set[param_value] += 1
    for param_name in param_sets:
        param_set = param_sets[param_name]

        if param_name == "rng_seed":
            max_seed = max(int(val) for val in param_set)
            print(f"maximum rng_seed: {max_seed}")
            continue

        print(f"{param_name} has values: " +
              ", ".join(f"({param_value}: {param_set[param_value]})" for param_value in param_set))
