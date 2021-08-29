#!/usr/bin/python3
import pdb
import time
import sys
import math
import sqlite3
from common_plot import SqliteFigureBuilder, FigureBuilder, parse_parameters, FigureMode, print_all_parameter_values_used, evaluate_conditions

conn = sqlite3.connect("results.db")
db_cursor = conn.cursor()

show_only = False
make_pdf_also = False


t10s = dict()
t10s["regret"] = "Regret"
t10s["estimation_error"] = "Estimation error"
t10s["samples_n"] = "# Samples"
t10s["bound_mode"] = "UCB expected-cost rule"
t10s["final_choice_mode"] = "Final choice expected-cost rule"
t10s["selection_mode"] = "UCB variation"
t10s["normal"] = "Normal"
t10s["lower_bound"] = "Lower bound"
t10s["bubble_best"] = "Bubble-best"
t10s["marginal"] = "Marginal action costs (proposed)"
t10s["marginal_prior"] = "Using MAC w/ zero-cost prior"
t10s["portion_bernoulli"] = "% cost Bernoulli (instead of Gaussian)"
t10s["ucb_const"] = "UCB constant factor"
t10s["prioritize_worst_particles_z"] = "Particle repetition z"
t10s["ucb"] = "UCB"
t10s["ucbv"] = "UCB-V"
t10s["ucbd"] = "UCB-delta"
t10s["klucb"] = "KL-UCB"
t10s["klucb+"] = "KL-UCB+"
t10s["random"] = "Random"
t10s["uniform"] = "Uniform"
t10s["1000"] = "No particle repeating"
t10s["-1000"] = "W/ particle repeating"
t10s["zero_mean_prior_std_dev"] = "Zero-mean prior std dev"
t10s["unknown_prior_std_dev_scalar"] = "Nominal std dev scalar"
t10s["bootstrap_confidence_z"] = "Top-level bootstrapping z-score"

figure_cmd_line_options = []
def should_make_figure(fig_name):
    figure_cmd_line_options.append(fig_name)
    return fig_name in sys.argv

# if len(sys.argv) > 1:
#     start_time = time.time()
#     results = []
#     with open("results.cache", "r") as f:
#         line_num = 0
#         for line in f:
#             parts = line.split()
#             if len(parts) > 3:
#                 entry = dict()
#                 entry["params"] = parse_parameters(parts[0])
#                 entry["steps_taken"] = float(parts[1])
#                 entry["chosen_cost"] = float(parts[2])
#                 entry["chosen_true_cost"] = float(parts[3])
#                 entry["true_best_cost"] = float(parts[4])

#                 entry["regret"] = entry["chosen_true_cost"] - entry["true_best_cost"]
#                 entry["estimation_error"] = abs(entry["chosen_true_cost"] - entry["chosen_cost"])

#                 results.append(entry)
#             else:
#                 continue
#             line_num += 1
#             # if line_num > 5000:
#             #     break
#     print(f"took {time.time() - start_time:.2f} seconds to load data")

all_metrics = ["regret"]  # , "estimation_error"]

# print(max([r for r in results if all(f in r["name"] for f in ["_method_mcts_", "_use_cfb_true_",
#                                                               "_smoothness_0_", "_safety_100_", "_ud_5_"])], key=lambda entry: entry["safety"]))

bound_mode = FigureMode("bound_mode", ["normal", "bubble_best", "lower_bound", "marginal", "marginal_prior"])
ucb_const_vals = [0, -68, -100, -150, -220, -330, -470, -680, -1000, -1500, -2200, -3300, -4700, -6800]
ucb_const_mode = FigureMode("ucb_const", ucb_const_vals)
ucb_const_ticknames = [val / 10 for val in ucb_const_vals]
samples_n_mode = FigureMode("samples_n", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

# Fig. 1a.
# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode marginal_prior :: final_choice_mode same :: unknown_prior_std_dev_scalar 1 1.2 1.4 1.6 1.8 2 :: zero_mean_prior_std_dev 22 33 47 68 100 150 220 330 470 1000000000 :: ucb_const -150 :: thread_limit 24
# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode marginal_prior :: final_choice_mode same :: single_trial_discount_factor 0.33 :: ucb_const -150 :: thread_limit 24
# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode marginal_prior :: final_choice_mode same :: preload_zeros 3 :: ucb_const -150 :: thread_limit 24
# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode marginal :: final_choice_mode same :: ucb_const -470 :: thread_limit 24
if should_make_figure("1a"):
    for metric in all_metrics:
        zero_mean_prior_std_dev_vals = [22, 33, 47, 68, 100, 150, 220, 330, 470, 1000000000]
        zero_mean_prior_std_dev_mode = FigureMode(
            "zero_mean_prior_std_dev", zero_mean_prior_std_dev_vals)
        unknown_prior_std_dev_scalar_mode = FigureMode(
            "unknown_prior_std_dev_scalar", [1, 1.2, 1.4, 1.6, 1.8, 2])

        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.plot(zero_mean_prior_std_dev_mode, [
                ("max.rng_seed", 511),
                ("bound_mode", "marginal_prior"),
                ("final_choice_mode", "same"),
                ("ucb_const", -150),
                ("selection_mode", "ucb"),
                ("single_trial_discount_factor", -1),
                ("preload_zeros", -1)
        ], unknown_prior_std_dev_scalar_mode)

        fig.line_from([
                ("max.rng_seed", 511),
                ("bound_mode", "marginal"),
                ("final_choice_mode", "same"),
                ("ucb_const", -470),
                ("selection_mode", "ucb"),
        ], "w/o zero-mean prior")

        fig.line_from([
                ("max.rng_seed", 511),
                ("bound_mode", "marginal_prior"),
                ("final_choice_mode", "same"),
                ("ucb_const", -150),
                ("selection_mode", "ucb"),
                ("single_trial_discount_factor", 0.33)
        ], "single_trial_discount_factor = 0.33")

        fig.line_from([
                ("max.rng_seed", 511),
                ("bound_mode", "marginal_prior"),
                ("final_choice_mode", "same"),
                ("ucb_const", -150),
                ("selection_mode", "ucb"),
                ("preload_zeros", 3)
        ], "preload_zeros = 3")

        # fig.axhline(1, color="black")
        fig.ticks(zero_mean_prior_std_dev_vals)
        # fig.ylim([0.5, 1.3])
        fig.legend()
        fig.show()

# Fig. 1b.
# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode marginal_prior :: final_choice_mode same :: single_trial_discount_factor 0 0.2 0.33 0.4 0.5 0.6 0.8 1 :: ucb_const -150 :: thread_limit 24
# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode marginal :: final_choice_mode same :: ucb_const -470 :: thread_limit 24
if should_make_figure("1b"):
    for metric in all_metrics:
        single_trial_discount_factor_vals = [0, 0.2, 0.33, 0.4, 0.5, 0.6, 0.8, 1]
        single_trial_discount_factor_mode = FigureMode(
            "single_trial_discount_factor", single_trial_discount_factor_vals)

        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.plot(single_trial_discount_factor_mode, [
                ("max.rng_seed", 511),
                ("bound_mode", "marginal_prior"),
                ("final_choice_mode", "same"),
                ("ucb_const", -150),
                ("selection_mode", "ucb"),
        ])

        fig.line_from([
                ("max.rng_seed", 511),
                ("bound_mode", "marginal"),
                ("final_choice_mode", "same"),
                ("ucb_const", -470),
                ("selection_mode", "ucb"),
        ], "w/o")

        fig.ticks(single_trial_discount_factor_vals)
        fig.legend()
        fig.show()

# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode marginal_prior :: final_choice_mode same :: preload_zeros 0 1 2 3 4 8 16 :: ucb_const 0 -100 -150 -220 -330 -470 -680 -1000 :: thread_limit 24
# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode marginal :: final_choice_mode same :: ucb_const -470 :: thread_limit 24
if should_make_figure("1c"):
    for metric in all_metrics:
        preload_zeros_vals = [0, 1, 2, 3, 4, 8, 16]
        preload_zeros_mode = FigureMode(
            "preload_zeros", preload_zeros_vals)

        ucb_const_vals = [0, -100, -150, -220, -330, -470, -680, -1000]
        ucb_const_mode = FigureMode("ucb_const", ucb_const_vals)

        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.plot(preload_zeros_mode, [
                ("max.rng_seed", 511),
                ("bound_mode", "marginal_prior"),
                ("final_choice_mode", "same"),
                # ("ucb_const", -150),
                ("selection_mode", "ucb"),
        ], ucb_const_mode)

        fig.line_from([
                ("max.rng_seed", 511),
                ("bound_mode", "marginal"),
                ("final_choice_mode", "same"),
                ("ucb_const", -470),
                ("selection_mode", "ucb"),
        ], "w/o")

        fig.ticks(preload_zeros_vals)
        fig.ylim([14, 25])
        fig.legend()
        fig.show()

# Fig. 1.
# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal marginal_prior :: final_choice_mode same :: ucb_const 0 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 :: thread_limit 24
if should_make_figure("1"):
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor,
            None, metric, translations=t10s)
        # samples_n = 512
        fig.plot(ucb_const_mode, [
                ("max.rng_seed", 511),
                # ("samples_n", samples_n),
                ("selection_mode", "ucb"),
                ("zero_mean_prior_std_dev", 220),
                ("unknown_prior_std_dev_scalar", 1.4),
                ("final_choice_mode", "same")], bound_mode)
        fig.ticks(ucb_const_ticknames)
        fig.legend()
        fig.show(title="Regret by UCB constant factor and expected-cost rule",
                 xlabel="UCB constant factor * 0.1",
                #  file_suffix=f"_samples_n_{samples_n}_final_choice_mode_same")
                file_suffix=f"_final_choice_mode_same")

# Fig. 2.
# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal marginal_prior :: final_choice_mode marginal_prior :: ucb_const 0 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 :: thread_limit 24
# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode uniform :: final_choice_mode marginal_prior :: thread_limit 24
if should_make_figure("2"):
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        # samples_n = 128
        common_filters = [
            ("max.rng_seed", 511),
            ("final_choice_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 220),
            ("unknown_prior_std_dev_scalar", 1.4)
        ]
        uniform_filters = common_filters + [("selection_mode", "uniform")]
        fig.plot(ucb_const_mode,
                 common_filters + [("selection_mode", "ucb")], bound_mode)

        fig.line_from(uniform_filters, "Uniform")

        fig.ylim([12, 52])
        fig.ticks(ucb_const_ticknames)
        fig.legend()

        fig.show(xlabel="UCB constant factor * 0.1",
                 file_suffix=f"_final_choice_mode_marginal_prior")

# Fig. 3.
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal marginal_prior :: final_choice_mode marginal_prior :: normal.ucb_const -3300 :: bubble_best.ucb_const -1500 :: lower_bound.ucb_const -1500 :: marginal.ucb_const -680 :: marginal_prior.ucb_const -150 :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode uniform :: final_choice_mode marginal_prior :: thread_limit 24
if should_make_figure("3"):
    for metric in all_metrics:
        common_filters = [
            ("max.rng_seed", 4095),
            ("final_choice_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 220),
            ("unknown_prior_std_dev_scalar", 1.4)
        ]
        filters = [("selection_mode", "ucb"),
                   ("normal.ucb_const", -3300),
                   ("bubble_best.ucb_const", -1500),
                   ("lower_bound.ucb_const", -1500),
                   ("marginal.ucb_const", -680),
                   ("marginal_prior.ucb_const", -150),
                   ] + common_filters
        uniform_filters = [("selection_mode", "uniform")] + common_filters

        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.inset_plot([5.8, 9.2], [-1, 9], [0.4, 0.4, 0.57, 0.57])

        fig.plot(samples_n_mode, filters, bound_mode)
        fig.plot(samples_n_mode, uniform_filters, label="Uniform")

        fig.legend("lower left")
        fig.ticks(["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
        fig.show(xlabel="log2(# of samples)")

# Fig. 4.
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal_prior :: selection_mode ucb ucbv ucbd klucb klucb+ uniform :: ucb.ucb_const -150 :: ucbv.ucb_const -150 :: ucbv.ucbv_const 0.0001 :: ucbd.ucb_const -68 :: ucbd.ucbd_const 0.01 :: klucb.ucb_const -0.047 :: klucb.klucb_max_cost 3300 :: klucb+.ucb_const -0.022 :: klucb+.klucb_max_cost 3300 :: thread_limit 24
if should_make_figure("4"):
    for metric in all_metrics:
        selection_mode = FigureMode(
            "selection_mode", ["ucb", "ucbv", "ucbd", "klucb", "klucb+", "uniform"])
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)
        filters = [
            ("max.rng_seed", 4095),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 220),
            ("unknown_prior_std_dev_scalar", 1.4),
            ("ucb.ucb_const", -150),
            ("ucbv.ucb_const", -150),
            ("ucbv.ucbv_const", 0.0001),
            ("ucbd.ucb_const", -68),
            ("ucbd.ucbd_const", 0.01),
            ("klucb.ucb_const", -0.047),
            ("klucb.klucb_max_cost", 3300),
            ("klucb+.ucb_const", -0.022),
            ("klucb+.klucb_max_cost", 3300),
        ]

        fig.inset_plot([5.8, 9.2], [-0.2, 1.6], [0.4, 0.4, 0.57, 0.57])

        fig.plot(samples_n_mode, filters, selection_mode)

        # fig.ylim([-20, 380])
        fig.legend("lower left")
        fig.ticks(["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
        fig.show(xlabel="log2(# of samples)")

# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 64 128 256 512 :: bound_mode marginal_prior :: bootstrap_confidence_z 0 0.01 0.022 0.047 0.1 0.22 0.47 1 2.2 :: thread_limit 24
# cargo run --release rng_seed 0-16383 :: samples_n 8 16 32 64 128 256 512 :: bound_mode marginal_prior :: bootstrap_confidence_z 0 0.01 0.022 0.047 0.1 0.22 0.47 1 2.2 :: thread_limit 24
# cargo run --release rng_seed 0-32767 :: samples_n 8 16 32 64 128 256 512 :: bound_mode marginal_prior :: bootstrap_confidence_z 0 0.01 0.022 0.047 0.1 0.22 0.47 1 2.2 :: thread_limit 24
if should_make_figure("bootstrap"):
    for metric in all_metrics:
        bootstrap_confidence_z_vals = [0, 0.01, 0.022, 0.047, 0.1, 0.22, 0.47, 1, 2.2]
        bootstrap_confidence_z_mode = FigureMode(
            "bootstrap_confidence_z", bootstrap_confidence_z_vals)
        samples_n_mode = FigureMode(
            "samples_n", [8, 16, 32, 64, 128, 256, 512])
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        filters = [
            ("max.rng_seed", 32767),
            ("selection_mode", "klucb"),
            ("ucb_const", -0.15),
            ("klucb_max_cost", 15000),
            ("bound_mode", "marginal_prior"),
            ("final_choice_mode", "same"),
            ("zero_mean_prior_std_dev", 220),
            ("unknown_prior_std_dev_scalar", 1.4),
            ("repeat_const", -1),
            ("prioritize_worst_particles_z", 1000),
            ("worst_particles_z_abs", "false"),
            ("bootstrap_n", -1)
        ]

        fig.plot(bootstrap_confidence_z_mode, filters, samples_n_mode, normalize="first")
        fig.axhline(1, color="black")
        fig.ticks(bootstrap_confidence_z_vals)
        fig.ylim([0.66, 1.12])
        fig.legend()
        fig.show("Relative regret by top-level bootstrapping z-score and # samples", ylabel="Relative regret")

# cargo run --release rng_seed 0-32767 :: samples_n 8 16 32 64 128 256 512 :: bound_mode marginal_prior :: bootstrap_n 0 1 2 3 4 5 6 7 :: thread_limit 24
if should_make_figure("bootstrap_n"):
    for metric in all_metrics:
        bootstrap_n_vals = [0, 1, 2, 3, 4, 5, 6, 7]
        bootstrap_n_mode = FigureMode(
            "bootstrap_n", bootstrap_n_vals)
        samples_n_mode = FigureMode(
            "samples_n", [8, 16, 32, 64, 128, 256, 512])
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        filters = [
            ("max.rng_seed", 32767),
            ("selection_mode", "klucb"),
            ("ucb_const", -0.15),
            ("klucb_max_cost", 15000),
            ("bound_mode", "marginal_prior"),
            ("final_choice_mode", "same"),
            ("zero_mean_prior_std_dev", 220),
            ("unknown_prior_std_dev_scalar", 1.4),
            ("repeat_const", -1),
            ("prioritize_worst_particles_z", 1000),
            ("worst_particles_z_abs", "false"),
        ]

        fig.plot(bootstrap_n_mode, filters, samples_n_mode, normalize="first")
        fig.axhline(1, color="black")
        fig.ticks(bootstrap_n_vals)
        # fig.ylim([0.38, 1.12])
        fig.legend()
        fig.show("Relative regret by bootstrap_n and # samples", ylabel="Relative regret")


# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: prioritize_worst_particles_z -1000 -2 -1 -0.5 0 0.5 1 2 1000 :: thread_limit 24
if should_make_figure("repeat_worst_z"):
    prioritize_z_vals = [-1000, -2, -1, -.5, 0, 0.5, 1, 2, 1000]
    prioritize_worst_particles_z_mode = FigureMode("prioritize_worst_particles_z", prioritize_z_vals)
    # repeat_at_all_levels_mode = FigureMode("repeat_at_all_levels", ["false", "true"])
    samples_n_mode = FigureMode(
        "samples_n", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        filters = [
            ("max.rng_seed", 16383),
            ("selection_mode", "klucb"),
            ("ucb_const", -0.15),
            ("klucb_max_cost", 15000),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 220),
            ("unknown_prior_std_dev_scalar", 1.4),
            ("worst_particles_z_abs", "false"),
            ("repeat_const", -1),
        ]

        prioritize_z_ticks = [str(v) for v in prioritize_z_vals]
        prioritize_z_ticks[0] = "Always"
        prioritize_z_ticks[-1] = "Never"


        if True:
            fig.plot(prioritize_worst_particles_z_mode, filters, samples_n_mode, normalize="last")
            fig.axhline(1, color="black")
            fig.ticks(prioritize_z_ticks)
            # fig.ylim([0.68, 1.18])
            fig.legend()
            fig.show(file_suffix="_separate")
        else:
            fig.plot(prioritize_worst_particles_z_mode, filters)
            fig.height_scale(0.5)
            fig.ticks(prioritize_z_ticks)
            fig.show(file_suffix="_combined")

# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 :: prioritize_worst_particles_z 0 0.5 1 2 1000 :: worst_particles_z_abs true :: thread_limit 24
if should_make_figure("repeat_worst_z_abs"):
    prioritize_z_vals = [0, 0.5, 1, 2, 1000]
    prioritize_worst_particles_z_mode = FigureMode("prioritize_worst_particles_z", prioritize_z_vals)
    # repeat_at_all_levels_mode = FigureMode("repeat_at_all_levels", ["false", "true"])
    samples_n_mode = FigureMode(
        "samples_n", [8, 16, 32, 64, 128, 256, 512])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        filters = [
            ("max.rng_seed", 16383),
            ("selection_mode", "klucb"),
            ("ucb_const", -0.15),
            ("klucb_max_cost", 15000),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 220),
            ("unknown_prior_std_dev_scalar", 1.4),
            ("worst_particles_z_abs", "true"),
            ("repeat_const", -1),
        ]

        fig.plot(prioritize_worst_particles_z_mode, filters, samples_n_mode, normalize="last")
        fig.axhline(1, color="black")
        fig.ticks(prioritize_z_vals)
        fig.ylim([0.68, 1.18])
        fig.legend()
        fig.show(file_suffix="_worst_particles_z_abs_true")

# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 64 128 256 512 1024 :: repeat_const 0 8 16 32 64 128 256 512 1024 :: thread_limit 24
# cargo run --release rng_seed 0-16383 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: repeat_const 0 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 :: thread_limit 24
# cargo run --release rng_seed 0-16383 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: repeat_const 0 65536 131072 262144 524288 1048576 2097152 :: thread_limit 24
if should_make_figure("repeat_const"):
    # repeat_const_vals = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
    repeat_const_vals = [0, 65536, 131072, 262144, 524288, 1048576, 2097152]
    repeat_const_mode = FigureMode("repeat_const", repeat_const_vals)
    samples_n_mode = FigureMode("samples_n", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        filters = [
            ("max.rng_seed", 16383),
            ("selection_mode", "klucb"),
            ("ucb_const", -0.15),
            ("klucb_max_cost", 15000),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 220),
            ("unknown_prior_std_dev_scalar", 1.4)
        ]

        fig.plot(repeat_const_mode, filters, samples_n_mode, normalize="first")
        fig.axhline(1, color="black")
        repeat_const_ticks = [math.log2(v) if v > 0 else "w/o" for v in repeat_const_vals]
        fig.ticks(repeat_const_ticks)
        # fig.ylim([0.68, 1.18])
        fig.legend()
        fig.show(xlabel="log2(repeat const)")

# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal :: ucb_const -4700 :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: ucb_const -1500 :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: repeat_const 256 :: bootstrap_confidence_z 0.1 :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: repeat_const 256 :: bootstrap_confidence_z 0.1 :: thread_limit 24
if should_make_figure("final"):
    repeat_const_vals = [0, 8, 16, 64, 128, 256, 512, 1024]
    repeat_const_mode = FigureMode("repeat_const", repeat_const_vals)
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        common_filters = [
            ("max.rng_seed", 16383),
            ("prioritize_worst_particles_z", 1000),
            ("worst_particles_z_abs", "false"),
        ]

        normal_filters = common_filters + [
            ("selection_mode", "ucb"),
            ("ucb_const", -4700),
            ("bound_mode", "normal"),
            ("repeat_const", -1),
            ("bootstrap_confidence_z", 0),
        ]
        macp_filters = common_filters + [
            ("selection_mode", "ucb"),
            ("ucb_const", -1500),
            ("bound_mode", "marginal_prior"),
        ]
        final_filters = common_filters + [
            ("selection_mode", "klucb"),
            ("ucb_const", -0.15),
            ("klucb_max_cost", 15000),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 220),
            ("unknown_prior_std_dev_scalar", 1.4),
            ("repeat_const", 256),
            ("bootstrap_confidence_z", 0.1),
        ]

        fig.plot(samples_n_mode, normal_filters, label="Normal")
        fig.plot(samples_n_mode, macp_filters, label="MACP")
        fig.plot(samples_n_mode, final_filters, label="MACP + Bootstrap + Repeat (proposed)")

        fig.legend()
        fig.ticks(["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
        fig.show(xlabel="log2(# of samples)", file_suffix="_final_comparison")

# cargo run --release rng_seed 0-127 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal_prior :: selection_mode ucbv :: ucbv.ucbv_const 0 0.0001 0.001 0.01 0.1 1 10 :: ucb_const -10 -15 -22 -33 -47 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000 :: thread_limit 24
if should_make_figure("ucbv"):
    ucb_const_vals = [-10, -15, -22, -33, -47, -68, -100, -150, -220, -330, -470, -680, -1000, -1500, -2200, -3300, -4700, -6800, -10000]
    ucb_const_mode = FigureMode(
        "ucb_const", ucb_const_vals)
    ucbv_const_mode = FigureMode(
        "ucbv_const", [0, 0.0001, 0.001, 0.01]) #, 0.1, 1, 10])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.plot(ucb_const_mode, [
            ("max.rng_seed", 127),
            ("selection_mode", "ucbv"),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 220),
            ("unknown_prior_std_dev_scalar", 1.4),
        ], ucbv_const_mode)

        fig.ticks(ucb_const_vals)
        fig.legend()
        fig.zoom(0.5)
        fig.show()

# cargo run --release rng_seed 0-127 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal_prior :: selection_mode ucbd :: ucbd.ucbd_const 0.000001 0.00001 0.0001 0.001 0.01 0.1 1 :: ucb_const -10 -15 -22 -33 -47 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000 :: thread_limit 24
if should_make_figure("ucbd"):
    ucb_const_vals = [-10, -15, -22, -33, -47, -68, -100, -150, -220, -330, -470, -680, -1000, -1500, -2200, -3300, -4700, -6800, -10000]
    ucb_const_mode = FigureMode(
        "ucb_const", ucb_const_vals)
    ucbd_const_mode = FigureMode("ucbd_const", ["0.000001", "0.00001", 0.0001, 0.001, 0.01, 0.1, 1])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.plot(ucb_const_mode, [
            ("max.rng_seed", 4095),
            ("selection_mode", "ucbd"),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 220),
            ("unknown_prior_std_dev_scalar", 1.4),
        ], ucbd_const_mode)

        fig.ticks(ucb_const_vals)
        fig.legend()
        fig.zoom(0.5)
        fig.show()


# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal_prior :: selection_mode klucb :: klucb.klucb_max_cost 330 470 680 1000 1500 2200 3300 4700 :: ucb_const -0.001 -0.0022 -0.0047 -0.01 -0.022 -0.047 -0.1 -0.22 -0.47 -1 :: thread_limit 24
if should_make_figure("klucb"):
    ucb_const_vals = [-0.001, -0.0022, -0.0047, -0.01, -0.022, -0.047, -0.1, -0.22, -0.47, -1]
    ucb_const_mode = FigureMode(
        "ucb_const", ucb_const_vals)
    klucb_max_cost_mode = FigureMode("klucb_max_cost", [330, 470, 680, 1000, 1500, 2200, 3300, 4700])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.plot(ucb_const_mode, [
            ("max.rng_seed", 511),
            ("selection_mode", "klucb"),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 220),
            ("unknown_prior_std_dev_scalar", 1.4),
        ], klucb_max_cost_mode)

        fig.ticks(ucb_const_vals)
        fig.zoom(0.5)
        fig.legend()
        fig.show(file_suffix="_selection_mode_klucb")

# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal_prior :: selection_mode klucb+ :: klucb+.klucb_max_cost 330 470 680 1000 1500 2200 3300 4700 :: ucb_const -0.001 -0.0022 -0.0047 -0.01 -0.022 -0.047 -0.1 -0.22 -0.47 -1 :: thread_limit 24
if should_make_figure("klucb+"):
    ucb_const_vals = [-0.001, -0.0022, -0.0047, -0.01, -0.022, -0.047, -0.1, -0.22, -0.47, -1]
    ucb_const_mode = FigureMode(
        "ucb_const", ucb_const_vals)
    klucb_max_cost_mode = FigureMode("klucb_max_cost", [330, 470, 680, 1000, 1500, 2200, 3300, 4700])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.plot(ucb_const_mode, [
            ("max.rng_seed", 511),
            ("selection_mode", "klucb+"),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 220),
            ("unknown_prior_std_dev_scalar", 1.4),
        ], klucb_max_cost_mode)

        fig.ticks(ucb_const_vals)
        fig.zoom(0.5)
        fig.legend()
        fig.show(file_suffix="_selection_mode_klucb+")

# cargo run --release rng_seed 0-8191 :: repeat_particle_sign -1 0 1 :: n_actions 5 6 7 8 9 10 :: samples_n 512 :: repeat_const 0 64 128 256 512 1024 2048 :: thread_limit 24
if False:
    repeat_const_kind = FigureKind(
        "repeat_const", [0, 64, 128, 256, 512, 1024, 2048], translations=t10s, ylim=[10, 35])
    n_actions_mode = FigureMode(
        "n_actions", [5, 6, 7, 8, 9, 10])
    for repeat_particle_sign in [0, 1, -1]:
        for metric in all_metrics:
            filters = [("selection_mode", "klucb+"),
                       ("samples_n", 512),
                       ("repeat_at_all_levels", "false"),
                       ("repeat_particle_sign", repeat_particle_sign)]
            repeat_const_kind.plot(
                results, metric, filters=filters, mode=n_actions_mode)

# cargo run --release rng_seed 0-4095 :: throwout_extreme_costs_z 1 1.5 2 2.5 3 1000 :: samples_n 256 384 512 1024 :: thread_limit 24
# cargo run --release rng_seed 0-8191 :: throwout_extreme_costs_z 1 1.5 2 2.5 3 1000 :: samples_n 256 384 512 :: thread_limit 24
if False:
    throwout_extreme_costs_z_kind = FigureKind(
        "throwout_extreme_costs_z", [1, 1.5, 2, 2.5, 3, 1000], translations=t10s)
    samples_n_mode = FigureMode(
        "samples_n", [256, 384, 512, 1024])
    for metric in all_metrics:
        filters = [("selection_mode", "klucb+")]
        throwout_extreme_costs_z_kind.plot(
            results, metric, filters=filters, mode=samples_n_mode)

# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 :: prioritize_worst_particles_z -1000 1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: bound_mode marginal :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: samples_n 64 128 256 512 1024 :: prioritize_worst_particles_z -1000 1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: bound_mode marginal :: thread_limit 24
# cargo run --release rng_seed 0-1023 :: samples_n 2048 4096 8192 16384 32768 :: prioritize_worst_particles_z -1000 1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: bound_mode marginal :: thread_limit 24
if False:
    samples_n_kind = FigureKind(
        "samples_n", ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"], [
            8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        translations=t10s)
    repeat_particles_mode = FigureMode(
        "prioritize_worst_particles_z", [1000, -1000])
    for metric in all_metrics:
        filters = [("selection_mode", "klucb+")]
        samples_n_kind.plot(results, metric, filters=filters,
                            mode=repeat_particles_mode, xlabel="log2(# of samples)")

# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 :: repeat_at_all_levels false true :: prioritize_worst_particles_z -1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: samples_n 64 128 256 512 1024 :: repeat_at_all_levels false true :: prioritize_worst_particles_z -1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
# cargo run --release rng_seed 0-1023 :: samples_n 2048 4096 8192 16384 32768 :: repeat_at_all_levels false true :: prioritize_worst_particles_z -1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
#
# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 :: repeat_at_all_levels false true :: prioritize_worst_particles_z -1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: samples_n 64 128 256 :: repeat_at_all_levels false true :: prioritize_worst_particles_z -1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
# cargo run --release rng_seed 0-511 :: samples_n 512 1024 2048 :: repeat_at_all_levels false true :: prioritize_worst_particles_z -1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
# cargo run --release rng_seed 0-127 :: samples_n 4096 8192 16384 32768 :: repeat_at_all_levels false true :: prioritize_worst_particles_z -1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
if False:
    samples_n_kind = FigureKind(
        "samples_n", ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"], [
            8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        translations=t10s)
    repeat_at_all_levels_mode = FigureMode(
        "repeat_at_all_levels", ["false", "true"])
    for metric in all_metrics:
        for portion_bernoulli in [0.5]:
            filters = [("selection_mode", "klucb+"),
                       ("prioritize_worst_particles_z", -1000),
                       ("portion_bernoulli", portion_bernoulli)]
            samples_n_kind.plot(results, metric, filters=filters,
                                mode=repeat_at_all_levels_mode, xlabel="log2(# of samples)")


# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: consider_repeats_after_portion 0.2 :: repeat_confidence_interval 0 0.5 3 1000 :: repeat_const 1000000 :: thread_limit 24
if False:
    for metric in all_metrics:
        for correct_future_std_dev_mean in ["false"]:
            repeat_confidence_interval_mode = FigureMode(
                "repeat_confidence_interval", [0, 0.5, 3, 1000])
            fig = FigureBuilder(results, None, metric, translations=t10s)
            samples_n_mode = FigureMode(
                "samples_n", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
            filters = [("max.rng_seed", 4095), ("consider_repeats_after_portion", 0.2),
                       ("correct_future_std_dev_mean", correct_future_std_dev_mean)]

            fig.plot(samples_n_mode, filters, repeat_confidence_interval_mode)

            fig.legend("lower left")
            fig.ticks(["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
            fig.show(xlabel="log2(# of samples)",
                     file_suffix=f"_correct_future_std_dev_mean_{correct_future_std_dev_mean}")

# print_all_parameter_values_used(
#     results, [("repeat_const", 1000000), ("consider_repeats_after_portion", 0.5), ("samples_n", 128), ("max.rng_seed", 8191)])
# quit()

# cargo run --release rng_seed 0-16383 :: samples_n 128 :: consider_repeats_after_portion 0 0.05 0.1 0.2 0.3 0.4 0.5 0.7 0.9 :: repeat_confidence_interval 0 0.5 1 2 3 1000 :: repeat_const 1000000 :: thread_limit 24
# cargo run --release rng_seed 0-32767 :: samples_n 128 :: consider_repeats_after_portion 0 0.05 0.1 0.2 0.3 0.4 0.5 0.7 0.9 :: repeat_confidence_interval 1 2 3 :: repeat_const 1000000 :: thread_limit 24
if False:
    for metric in all_metrics:
        for samples_n in [128]:
            consider_repeats_after_portion_mode = FigureMode(
                "consider_repeats_after_portion", [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9])
            repeat_confidence_interval_mode = FigureMode(
                "repeat_confidence_interval", [0, 0.5, 1, 2, 3, 1000])
            fig = FigureBuilder(results, "consider_repeats_after_portion",
                                metric, translations=t10s)
            filters = [("samples_n", samples_n), ("repeat_const", 1000000)]

            fig.plot(consider_repeats_after_portion_mode, filters, repeat_confidence_interval_mode)
            fig.legend()
            fig.show(file_suffix=f"_samples_n_{samples_n}")

# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: consider_repeats_after_portion 0 0.05 0.1 0.2 0.3 :: repeat_confidence_interval 1 :: repeat_const 1000000 :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: consider_repeats_after_portion 0 0.05 0.1 0.2 0.3 :: repeat_confidence_interval 1 :: repeat_const 1000000 :: repeat_at_all_levels false true :: thread_limit 24
if False:
    for metric in all_metrics:
        for correct_future_std_dev_mean in ["false"]:
            consider_repeats_after_portion_mode = FigureMode(
                "consider_repeats_after_portion", [0, 0.05, 0.1, 0.2, 0.3])
            fig = FigureBuilder(results, "steps_taken", metric, translations=t10s)
            for repeat_at_all_levels in ["false", "true"]:
                samples_n_mode = FigureMode(
                    "samples_n", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
                filters = [("repeat_confidence_interval", 1), ("correct_future_std_dev_mean", correct_future_std_dev_mean), ("repeat_at_all_levels", repeat_at_all_levels)]

                fig.plot(samples_n_mode, filters, consider_repeats_after_portion_mode, label="Repeats " if repeat_at_all_levels == "true" else "")

            fig.xscale("log")
            fig.legend("lower left")
            # fig.ticks(["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
            #fig.show(xlabel="log2(# of samples)", file_suffix=f"_correct_future_std_dev_mean_{correct_future_std_dev_mean}")
            fig.show(file_suffix=f"_correct_future_std_dev_mean_{correct_future_std_dev_mean}")

# cargo run --release rng_seed 0-2047 :: samples_n 8 16 32 64 128 256 512 :: bound_mode marginal_prior :: unknown_prior_std_dev 1000 :: zero_mean_prior_std_dev 220 330 470 680 1000 1500 2200 3300 4700 6800 10000 1000000000 :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: samples_n 8 32 128 512 2048 :: bound_mode marginal_prior :: unknown_prior_std_dev 1000 :: zero_mean_prior_std_dev 220 330 470 680 1000 1500 2200 3300 4700 6800 10000 1000000000 :: thread_limit 24
if False:
    for metric in all_metrics:
        zero_mean_prior_std_dev_vals = [220, 330, 470, 680, 1000, 1500, 2200, 3300, 4700, 6800, 10000, 1000000000]
        zero_mean_prior_std_dev_mode = FigureMode(
            "zero_mean_prior_std_dev", zero_mean_prior_std_dev_vals)
        samples_n_mode = FigureMode(
            "samples_n", [8, 32, 128, 512, 2048])
        fig = FigureBuilder(results, None,
                            metric, translations=t10s)
        filters = []

        fig.plot(zero_mean_prior_std_dev_mode, filters, samples_n_mode, normalize="last")
        fig.axhline(1, color="black")
        fig.ticks(zero_mean_prior_std_dev_vals)
        fig.ylim([0.5, 1.3])
        fig.legend()
        fig.show("Relative regret by optimistic zero-mean prior std dev and # samples", ylabel="Relative regret")


if len(sys.argv) == 1 or "help" in sys.argv:
    print("Valid figure options:")
    for option in figure_cmd_line_options:
        print(option)
