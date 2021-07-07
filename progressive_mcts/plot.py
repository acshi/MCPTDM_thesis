#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
import localreg

plt.rcParams.update({"font.size": 12})
plt.rcParams["pdf.fonttype"] = 42

# plt.cycler(color=["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00"])
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628"]) + plt.cycler(marker=['+', 'o', 'x', '^', 'v'])

show_only = False
make_pdf_also = False

save_dpi = 300
figure_zoom = 1.5

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
translations["portion_bernoulli"] = "% cost Bernoulli (instead of Gaussian)"
translations["ucb_const"] = "UCB constant factor"


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
            self.fig.set_figwidth(6.4 * figure_zoom)
            self.fig.set_figheight(4.8 * figure_zoom)
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
    for samples_n in [64, 128]:
        for bound_mode in ["normal", "lower_bound", "marginal"]:
            evaluate_conditions(results, all_metrics, [
                                ("bound_mode", bound_mode),
                                ("samples_n", samples_n),
                                ("portion_bernoulli", 1)])

samples_n_kind = FigureKind("samples_n", [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
bound_mode = FigureMode("bound_mode", ["normal", "lower_bound", "marginal"])

# cargo run --release rng_seed 0-2047 :: samples_n 4 8 16 32 64 128 256 512 1024 2048 :: portion_bernoulli 0 1 :: bound_mode normal lower_bound marginal :: normal.ucb_const -1000 :: lower_bound.ucb_const -3000 :: marginal.ucb_const -3000 :: thread_limit 24
if False:
    for metric in all_metrics:
        for portion_bernoulli in [0, 1]:
            samples_n_kind.plot(results, metric, filters=[
                                f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)


selection_mode = FigureMode("selection_mode", ["ucb", "ucbv", "ucbd", "klucb", "klucb+"])
# cargo run --release rng_seed 0-4095 :: samples_n 4 8 16 32 64 128 256 512 1024 2048 :: portion_bernoulli 0 1 :: bound_mode marginal :: selection_mode ucb ucbd ucbv klucb klucb+ :: ucb_const -3000 :: ucbd.ucb_const -1000 :: klucb.ucb_const -1  :: klucb+.ucb_const -1 :: ucbv.ucbv_const 0 :: thread_limit 24
if False:
    samples_n_kind = FigureKind("samples_n", [16, 32, 64, 128, 256, 512, 1024, 2048])
    for metric in all_metrics:
        for portion_bernoulli in [0, 1]:
            samples_n_kind.plot(results, metric, filters=[
                                f"_portion_bernoulli_{portion_bernoulli}_"], mode=selection_mode)

portion_bernoulli_kind = FigureKind(
    "portion_bernoulli", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# cargo run --release rng_seed 0-8191 :: portion_bernoulli 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 :: samples_n 64 :: bound_mode normal lower_bound marginal :: normal.ucb_const -1000 :: lower_bound.ucb_const -3000 :: marginal.ucb_const -3000 :: thread_limit 24
if False:
    for metric in all_metrics:
        portion_bernoulli_kind.plot(results, metric, filters=[
                                    "_samples_n_64_"], mode=bound_mode)

ucb_const_kind = FigureKind(
    "ucb_const", [-10, -30, -100, -300, -1000, -3000, -10000, -30000])
# cargo run --release rng_seed 0-4095 :: selection_mode ucb :: ucb_const -10 -30 -100 -300 -1000 -3000 -10000 -30000 :: samples_n 64 :: portion_bernoulli 0 1 :: bound_mode normal lower_bound marginal :: thread_limit 24
if False:
    # for metric in all_metrics:
    #     for portion_bernoulli in [0, 1]:
    #         ucb_const_kind.plot(results, metric, filters=[
    #             "_selection_mode_ucb_", "_samples_n_64_", f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)

    for portion_bernoulli in [0, 1]:
        evaluate_conditions(results, all_metrics, [
            ("selection_mode", "ucb"),
            ("ucb_const", -3000),
            ("bound_mode", "marginal"),
            ("samples_n", 64),
            ("portion_bernoulli", portion_bernoulli)])

    # selection_mode_ucb_ucb_const_-3000_bound_mode_marginal_samples_n_64_portion_bernoulli_0:
    #   regret has mean:  45.36 and mean std dev:  1.716
    #   estimation_error has mean:  119.0 and mean std dev:   1.66

    # selection_mode_ucb_ucb_const_-3000_bound_mode_marginal_samples_n_64_portion_bernoulli_1:
    #   regret has mean:  77.03 and mean std dev:  2.445
    #   estimation_error has mean:  207.3 and mean std dev:  2.377

ucbv_const_kind = FigureKind(
    "ucbv_const", [0, 0.0001, 0.001, 0.01, 0.1, 1, 10])
# cargo run --release rng_seed 0-1023 :: selection_mode ucbv :: ucbv.ucbv_const 0 0.0001 0.001 0.01 0.1 1 10 :: ucb_const -10 -30 -100 -300 -1000 -3000 -10000 -30000 :: samples_n 64 :: portion_bernoulli 0 1 :: bound_mode normal lower_bound marginal :: thread_limit 24
if False:
    for metric in all_metrics:
        for portion_bernoulli in [0, 1]:
            ucb_const_kind.plot(results, metric, filters=[
                                "_selection_mode_ucbv_", "_samples_n_64_", f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)
            ucbv_const_kind.plot(results, metric, filters=[
                "_selection_mode_ucbv_", "_samples_n_64_", f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)

ucbd_const_kind = FigureKind(
    "ucbd_const", ["0.000001", "0.00001", 0.0001, 0.001, 0.01, 0.1, 1])
# cargo run --release rng_seed 0-1023 :: selection_mode ucbd :: ucbd.ucbd_const 0.0001 0.001 0.01 0.1 1 :: ucb_const -10 -30 -100 -300 -1000 -3000 -10000 -30000 :: samples_n 64 :: portion_bernoulli 0 1 :: bound_mode normal lower_bound marginal :: thread_limit 24
# cargo run --release rng_seed 0-16383 :: selection_mode ucbd :: ucbd.ucbd_const 0.000001 0.0001 0.001 0.01 0.1 1 :: ucb_const -1000 :: samples_n 64 :: portion_bernoulli 0 1 :: bound_mode normal lower_bound marginal :: thread_limit 24
if False:
    for metric in all_metrics:
        for portion_bernoulli in [0, 1]:
            ucb_const_kind.plot(results, metric, filters=[
                                "_selection_mode_ucbd_", "_samples_n_64_", f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)
            ucbd_const_kind.plot(results, metric, filters=[
                "_ucb_const_-1000_", "_selection_mode_ucbd_", "_samples_n_64_", f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)
    evaluate_conditions(results, all_metrics, [
        ("selection_mode", "ucbd"),
        ("ucb_const", -1000),
        ("ucbd_const", 1),
        ("bound_mode", "lower_bound"),
        ("samples_n", 64),
        ("portion_bernoulli", 0)])

    evaluate_conditions(results, all_metrics, [
        ("selection_mode", "ucbd"),
        ("ucb_const", -1000),
        ("ucbd_const", "0.00001"),
        ("bound_mode", "marginal"),
        ("samples_n", 64),
        ("portion_bernoulli", 1)])

    # selection_mode_ucbd_ucb_const_-1000_ucbd_const_1_bound_mode_lower_bound_samples_n_64_portion_bernoulli_0:
    #   regret has mean:  71.89 and mean std dev:  1.157
    #   estimation_error has mean:  166.9 and mean std dev:  1.059

    # selection_mode_ucbd_ucb_const_-1000_ucbd_const_0.00001_bound_mode_marginal_samples_n_64_portion_bernoulli_1:
    #   regret has mean:  74.68 and mean std dev:  1.689
    #   estimation_error has mean:  222.0 and mean std dev:  1.737


ucb_const_kind = FigureKind(
    "ucb_const", [-0.01, -0.03, -0.1, -0.3, -1, -3, -10, -30, -100, -300, -1000])
klucb_max_cost_kind = FigureKind(
    "klucb_max_cost", [1000, 2000, 4000, 8000, 16000])
# cargo run --release rng_seed 0-1023 :: selection_mode klucb :: klucb.klucb_max_cost 1000 2000 4000 8000 16000 :: klucb.ucb_const -0.01 -0.03 -0.1 -0.3 -1 -3 -10 -30 -100 -300 -1000 -3000 -10000 -30000 :: samples_n 64 :: portion_bernoulli 0 1 :: bound_mode normal lower_bound marginal :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: selection_mode klucb :: klucb.ucb_const -0.01 -0.03 -0.1 -0.3 -1 -3 -10 -30 -100 -300 -1000 -3000 -10000 -30000 :: samples_n 64 :: portion_bernoulli 0 1 :: bound_mode normal lower_bound marginal :: thread_limit 24
if True:
    for metric in all_metrics:
        for portion_bernoulli in [0, 1]:
            ucb_const_kind.plot(results, metric, filters=[
                                "_selection_mode_klucb_", "_samples_n_64_", f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)
    #         klucb_max_cost_kind.plot(results, metric, filters=[
    #             "_selection_mode_klucb_", "_samples_n_64_", f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)

    for portion_bernoulli in [0, 1]:
        evaluate_conditions(results, all_metrics, [
            ("selection_mode", "klucb"),
            ("ucb_const", -1),
            ("klucb_max_cost", 4000),
            ("bound_mode", "marginal"),
            ("samples_n", 64),
            ("portion_bernoulli", portion_bernoulli)])

    # selection_mode_klucb_ucb_const_-1_klucb_max_cost_4000_bound_mode_marginal_samples_n_64_portion_bernoulli_0:
    #   regret has mean:  39.36 and mean std dev:  3.119
    #   estimation_error has mean:  74.06 and mean std dev:  2.456

    # selection_mode_klucb_ucb_const_-1_klucb_max_cost_4000_bound_mode_marginal_samples_n_64_portion_bernoulli_1:
    #   regret has mean:  63.18 and mean std dev:  4.132
    #   estimation_error has mean:  142.0 and mean std dev:   3.93

# cargo run --release rng_seed 0-4095 :: selection_mode klucb+ :: klucb+.ucb_const -0.01 -0.03 -0.1 -0.3 -1 -3 -10 -30 -100 -300 -1000 -3000 -10000 -30000 :: samples_n 64 :: portion_bernoulli 0 1 :: bound_mode normal lower_bound marginal :: thread_limit 24
if False:
    # for metric in all_metrics:
    #     for portion_bernoulli in [0, 1]:
    #         ucb_const_kind.plot(results, metric, filters=[
    #                             "_selection_mode_klucb+_", "_samples_n_64_", f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)
    for portion_bernoulli in [0, 1]:
        evaluate_conditions(results, all_metrics, [
            ("selection_mode", "klucb+"),
            ("ucb_const", -1),
            ("klucb_max_cost", 4000),
            ("bound_mode", "marginal"),
            ("samples_n", 64),
            ("portion_bernoulli", portion_bernoulli)])

    # selection_mode_klucb+_ucb_const_-1_klucb_max_cost_4000_bound_mode_marginal_samples_n_64_portion_bernoulli_0:
    #   regret has mean:  39.55 and mean std dev:  1.642
    #   estimation_error has mean:  82.54 and mean std dev:  1.338

    # selection_mode_klucb+_ucb_const_-1_klucb_max_cost_4000_bound_mode_marginal_samples_n_64_portion_bernoulli_1:
    #   regret has mean:  71.85 and mean std dev:  2.317
    #   estimation_error has mean:  144.7 and mean std dev:   1.89
