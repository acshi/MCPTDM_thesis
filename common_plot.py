#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
import localreg
import numbers

plt.rcParams.update({'font.size': 12})
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999"]) + plt.cycler(marker=["+", "o", "x", "^", "v", "s", "*"])


show_only = False
make_pdf_also = True

save_dpi = 300
figure_zoom = 1.25


class FigureMode:
    def __init__(self, param, values):
        self.param = param
        self.values = values

    def matches(self, params, value):
        return self.param in params and params[self.param] == str(value)


def filter_match(params, filter):
    param = filter[0]
    param_value = filter[1]
    param_split = param.split(".")

    if len(param_split) == 2:
        if param_split[0] == "max":
            name_value = float(params[param_split[1]])
            return name_value <= param_value
        elif params["method"] != param_split[0]:
            return True
        param = param_split[1]

    return param in params and params[param] == str(param_value)


def filter_extra(results, filters):
    return [entry for entry in results if all(filter_match(entry["params"], f) for f in filters)]


def short_num_string(val):
    scientific = f"{val:.1e}".replace(".0e", "e").replace(
        "+", "").replace("e0", "e").replace("e-0", "e-")
    normal = str(val)
    # same-length means scientific is shorter in most fonts because of "." being short
    return scientific if len(scientific) <= len(normal) else normal


def decapitalize_word(word):
    if len(word) == 0:
        return word
    if word.isupper():
        return word
    return word[0].lower() + word[1:]


def decapitalize(title):
    words = title.split(" ")
    return " ".join(decapitalize_word(word) for word in words)


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
            self.val_names = [str(val) for val in val_names]
            self.tick_labels = self.ticks
        if locs is None and ticks is not None:
            self.locs = [i for i in range(len(ticks))]
        else:
            self.locs = locs
        self.translations = translations

    def translate(self, name):
        name = str(name)
        if name in self.translations:
            return self.translations[name]
        return name

    def filter_entry(self, entry, filters, mode=None, mode_val=None):
        params = entry["params"]
        return self.param in params and all(filter_match(params, f) for f in filters) and (mode is None or mode.matches(params, mode_val))

    def collect_vals(self, results, result_name, filters, mode=None, mode_val=None):
        if self.val_names is None:
            print(
                f"OOPS! Tried to directly plot continuous variable {self.param} as discrete")
            return []
        else:
            val_sets = [list() for _ in range(len(self.val_names))]
            for entry in results:
                if not self.filter_entry(entry, filters, mode, mode_val):
                    continue
                for (i, val_name) in enumerate(self.val_names):
                    if entry["params"][self.param] == val_name:
                        val_sets[i].append(entry[result_name])
            return val_sets

    def _set_show_save(self, title, xlabel, ylabel, result_name, mode, filters):
        self.ax.set_title(title)
        if mode is not None:
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
                "_".join(f"{f[0]}_{f[1]}" for f in filters if not f[0].startswith("max.")
                         ) if len(filters) > 0 else ""
            file_suffix = f"_{result_name}{mode_string}{filters_string}"

            self.fig.tight_layout()
            if make_pdf_also:
                self.fig.savefig(f"figures/pdf/by_{self.param}{file_suffix}.pdf",
                                 bbox_inches="tight", pad_inches=0)
            self.fig.savefig(f"figures/by_{self.param}{file_suffix}.png")

    def _plot(self, results, result_name, title, xlabel, ylabel, mode, filters, extra_lines, extra_modes):
        self.fig, self.ax = plt.subplots(dpi=100 if show_only else save_dpi)

        has_any = False
        for mode_val in mode.values if mode else [None]:
            import time
            start_time = time.time()
            value_sets = self.collect_vals(results, result_name, filters, mode, mode_val)
            print(f"collect_vals took {time.time() - start_time:.2} seconds")
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

        for extra_line in extra_lines:
            line_label = extra_line[0]
            line_filters = extra_line[1]
            vals = [entry[result_name] for entry in filter_extra(results, line_filters)]
            mean = np.mean(vals)
            stdev_mean = np.std(vals) / np.sqrt(len(vals))
            self.ax.errorbar([self.locs[0], self.locs[-1]], [mean, mean],
                             yerr=[stdev_mean, stdev_mean], label=line_label)

        for extra_mode in extra_modes:
            mode_label = extra_mode[0]
            mode_filters = extra_mode[1]

            import time
            start_time = time.time()
            value_sets = self.collect_vals(results, result_name, mode_filters, None, None)
            print(f"collect_vals took {time.time() - start_time:.2} seconds")

            means = [np.mean(vals) for vals in value_sets]
            stdev_mean = [np.std(vals) / np.sqrt(len(vals))
                          for vals in value_sets]
            self.ax.errorbar(self.locs, means, yerr=stdev_mean, label=mode_label)

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

    def plot(self, results, result_name, title=None, xlabel=None, ylabel=None, mode=None, filters=[], extra_lines=[], extra_modes=[]):
        xlabel = xlabel or self.translate(self.param)
        ylabel = ylabel or self.translate(result_name)
        if title is None:
            title = f"{self.translate(result_name)} by {decapitalize(self.translate(self.param))}"
            if mode is not None:
                title = f"{title} and {decapitalize(self.translate(mode.param))}"
        print(f"{self.param} {result_name}")

        if self.ticks is None:
            self.localreg_estimate(results, result_name, title, xlabel, ylabel,
                                   mode, filters)
        else:
            self._plot(results, result_name, title, xlabel, ylabel,
                       mode, filters, extra_lines, extra_modes)


class FigureBuilder:
    def __init__(self, results, x_param, y_param, translations={}):
        self.results = results
        self.x_param = x_param
        self.defacto_x_param = x_param
        self.y_param = y_param
        self.all_modes = []
        self.translations = translations
        self.min_x = None
        self.max_x = None
        self.x_locs = []
        self.axins = None

        self.fig, self.ax = plt.subplots(dpi=100 if show_only else save_dpi)

    def translate(self, name):
        if name in self.translations:
            return self.translations[name]
        return name

    def filter_entry(self, entry, filters, modes=[]):
        params = entry["params"]
        return self.y_param in entry and all(filter_match(params, f) for f in filters) and all(mode_val[0].matches(params, mode_val[1]) for mode_val in modes)

    def collect_vals(self, x_mode, filters, legend_mode, legend_mode_val):
        x_val_sets = [list() for _ in range(len(x_mode.values))]
        y_val_sets = [list() for _ in range(len(x_mode.values))]
        modes = [(legend_mode, legend_mode_val)] if legend_mode else []
        for entry in self.results:
            if not self.filter_entry(entry, filters, modes):
                continue
            for (i, val_name) in enumerate(x_mode.values):
                if entry["params"][x_mode.param] == str(val_name):
                    if self.x_param is not None:
                        x_val_sets[i].append((entry[self.x_param])
                                             if self.x_param in entry else float(entry["params"][self.x_param]))
                    y_val_sets[i].append((entry[self.y_param]))
        return (x_val_sets, y_val_sets)

    def plot(self, x_mode, filters=[], legend_mode=None, label=None):
        if self.defacto_x_param is None:
            self.defacto_x_param = x_mode.param

        if legend_mode:
            if not any(legend_mode.param == mode.param for mode in self.all_modes):
                self.all_modes += [legend_mode]

        for legend_mode_val in legend_mode.values if legend_mode else [None]:
            import time
            start_time = time.time()
            (x_val_sets, y_val_sets) = self.collect_vals(
                x_mode, filters, legend_mode, legend_mode_val)
            print(f"collect_vals took {time.time() - start_time:.2} seconds")
            if len(y_val_sets) == 0:
                label_str = f"{label}: " if label else ""
                print(
                    f"{label_str}Data completely missing for {self.y_param} by {x_mode.param} with {filters}")
                if legend_mode:
                    print(f"And with {legend_mode.param} = {legend_mode_val}")
                continue
            n_vals_in_set = len(y_val_sets[0])
            for i, vals in enumerate(y_val_sets):
                if len(vals) == 0:
                    label_str = f"{label}: " if label else ""
                    legend_str = f"and with {legend_mode.param} = {legend_mode_val}" if legend_mode else ""
                    print(
                        f"{label_str}{x_mode.param} = {x_mode.values[i]} has 0 data points for {self.y_param} with {filters} {legend_str}")
                    vals.append(np.nan)
                if len(vals) != n_vals_in_set:
                    label_str = f"{label}: " if label else ""
                    legend_str = f"and with {legend_mode.param} = {legend_mode_val}" if legend_mode else ""
                    print(
                        f"{label_str}{len(vals)} != {n_vals_in_set} for {x_mode.param} = {x_mode.values[i]} {legend_str}")

            means = [np.mean(vals) for vals in y_val_sets]
            stdev_mean = [np.std(vals) / np.sqrt(len(vals))
                          for vals in y_val_sets]

            if self.x_param is None:
                x_means = [i for i in range(len(x_val_sets))]
            else:
                x_means = [np.mean(vals) for vals in x_val_sets]
            self.x_locs = x_means

            full_label = label
            if legend_mode:
                if full_label:
                    full_label = f"{label} {self.translate(legend_mode_val)}"
                else:
                    full_label = f"{self.translate(legend_mode_val)}"

            self.ax.errorbar(x_means, means,
                             yerr=np.array(stdev_mean), label=full_label)

            if self.axins:
                self.axins.errorbar(x_means, means, yerr=np.array(stdev_mean))

            x_mean_min = np.min(x_means)
            if self.min_x is None:
                self.min_x = x_mean_min
            else:
                self.min_x = min(self.min_x, x_mean_min)

            x_mean_max = np.max(x_means)
            if self.max_x is None:
                self.max_x = x_mean_max
            else:
                self.max_x = max(self.max_x, x_mean_max)

    def line(self, filters, label):
        vals = [entry[self.y_param] for entry in filter_extra(self.results, filters)]
        mean = np.mean(vals)
        stdev_mean = np.std(vals) / np.sqrt(len(vals))
        self.ax.errorbar([self.min_x, self.max_x], [mean, mean],
                         yerr=[stdev_mean, stdev_mean], label=label)
        if self.axins:
            self.axins.errorbar([self.min_x, self.max_x], [mean, mean],
                                yerr=[stdev_mean, stdev_mean])

    def _set_show_save(self, title, xlabel, ylabel, file_suffix):
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if show_only:
            plt.show()
        else:
            self.fig.set_figwidth(6.4 * figure_zoom)
            self.fig.set_figheight(4.8 * figure_zoom)

            modes_description = "_".join([""] + [mode.param for mode in self.all_modes])
            file_desc = f"{self.defacto_x_param}_{self.y_param}{modes_description}{file_suffix}"

            self.fig.tight_layout()
            if make_pdf_also:
                self.fig.savefig(f"figures/pdf/by_{file_desc}.pdf",
                                 bbox_inches="tight", pad_inches=0)
            self.fig.savefig(f"figures/by_{file_desc}.png")

    def show(self, title=None, xlabel=None, ylabel=None, file_suffix=""):
        xlabel = xlabel or self.translate(self.defacto_x_param)
        ylabel = ylabel or self.translate(self.y_param)

        if title is None:
            title = f"{self.translate(self.y_param)} by {decapitalize(self.translate(self.defacto_x_param))}"

            modes_str = " and ".join([""] + [decapitalize(self.translate(mode.param))
                                             for mode in self.all_modes])
            title += modes_str

        modes_str = " and ".join([""] + [mode.param for mode in self.all_modes])
        print(f"{self.y_param} by {self.defacto_x_param}{modes_str}")

        self._set_show_save(title, xlabel, ylabel, file_suffix)

    def legend(self, loc=None):
        self.ax.legend(loc=loc)

    def xlim(self, xlim):
        self.ax.set_xlim(xlim)

    def ylim(self, ylim):
        self.ax.set_ylim(ylim)

    def ticks(self, labels, locs=None):
        if locs is None:
            locs = self.x_locs

        self.ax.set_xticks(locs)
        self.ax.set_xticklabels(labels)

    def ax(self):
        return self.ax

    def fig(self):
        return self.fig

    # Call this before any plotting to also have things plot in the inset
    def inset_plot(self, xlim, ylim, bounds=[0.5, 0.5, 0.47, 0.47]):
        self.axins = self.ax.inset_axes(bounds)

        self.axins.set_xlim(xlim)
        self.axins.set_ylim(ylim)
        self.axins.set_xticklabels('')
        self.axins.set_yticklabels('')

        return self.ax.indicate_inset_zoom(self.axins, edgecolor="black")


def evaluate_conditions(results, metrics, filters):
    results = filter_extra(results, filters)
    filters_string = ",".join([f"{f[0]}={f[1]}" for f in filters])

    print(f"{filters_string}:")

    return_results = []

    for metric in metrics:
        vals = [entry[metric] for entry in results]
        mean = np.mean(vals)
        stdev_mean = np.std(vals) / np.sqrt(len(vals))
        print(f"  {metric} has mean: {mean:6.4} and mean std dev: {stdev_mean:6.4} and a total of {len(vals)} samples")
        return_results.append(mean)
    print()

    return return_results


def print_all_parameter_values_used(results, filters):
    param_sets = {}
    for result in filter_extra(results, filters):
        for param_name in result["params"]:
            param_value = result["params"][param_name]
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


def parse_parameters(parameters_string):
    parsed_params = {}
    for param in parameters_string.split(","):
        if len(param) == 0:
            continue
        param_split = param.split("=")
        param_name = param_split[0]
        param_value = param_split[1]
        parsed_params[param_name] = param_value
    return parsed_params
