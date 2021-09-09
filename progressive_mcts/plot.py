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
t10s["samples_n"] = "# Monte Carlo trials"
t10s["steps_taken"] = t10s["samples_n"] # as long as we do the proper rescaling!!!
t10s["bound_mode"] = "UCB expected-cost rule"
t10s["final_choice_mode"] = "Final choice expected-cost rule"
t10s["selection_mode"] = "UCB variation"
t10s["normal"] = "Normal"
t10s["lower_bound"] = "Lower bound"
t10s["bubble_best"] = "Bubble-best"
t10s["marginal"] = "MAC (proposed)"
t10s["marginal_prior"] = "MACP (proposed)"
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
t10s["repeat_const"] = "Repetition constant"
t10s["zero_mean_prior_std_dev"] = "Zero-cost prior std dev"
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
ucb_const_vals = [0, -68, -100, -150, -220, -330, -470, -680, -1000, -1500, -2200, -3300, -4700, -6800, -10000]
ucb_const_mode = FigureMode("ucb_const", ucb_const_vals)
ucb_const_ticknames = [val / 10 for val in ucb_const_vals]
samples_n_mode = FigureMode("samples_n", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

# cargo run --release rng_seed 0-1023 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: ucb_const -150 :: bound_mode marginal_prior :: final_choice_mode same :: unknown_prior_std_dev_scalar 1.2 1.4 1.6 1.8 2 :: zero_mean_prior_std_dev 33 47 68 100 150 220 330 470 680 1000000000
# cargo run --release rng_seed 0-1023 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: ucb_const -470 :: bound_mode marginal :: final_choice_mode same
if should_make_figure("zero_prior"):
    for metric in all_metrics:
        zero_mean_prior_std_dev_vals = [33, 47, 68, 100, 150, 220, 330, 470, 680, 1000000000]
        zero_mean_prior_std_dev_mode = FigureMode(
            "zero_mean_prior_std_dev", zero_mean_prior_std_dev_vals)
        unknown_prior_std_dev_scalar_mode = FigureMode(
            "unknown_prior_std_dev_scalar", [1.2, 1.4, 1.6, 1.8, 2])

        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.plot(zero_mean_prior_std_dev_mode, [
                ("max.rng_seed", 1023),
                ("bound_mode", "marginal_prior"),
                ("final_choice_mode", "same"),
                ("ucb_const", -150),
                ("selection_mode", "ucb"),
                ("single_trial_discount_factor", -1),
                ("preload_zeros", -1),
                ("repeat_const", -1),
                ("klucb_max_cost", 2200)
        ], unknown_prior_std_dev_scalar_mode)

        # fig.line_from([
        #         ("max.rng_seed", 1023),
        #         ("bound_mode", "marginal"),
        #         ("final_choice_mode", "same"),
        #         ("ucb_const", -470),
        #         ("selection_mode", "ucb"),
        # ], "w/o zero-mean prior")

        # fig.axhline(1, color="black")
        fig.ticks(zero_mean_prior_std_dev_vals)
        # fig.ylim([0.5, 1.3])
        fig.legend()
        fig.show()

# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal marginal_prior :: final_choice_mode same :: ucb_const 0 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000
if should_make_figure("1"):
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)
        fig.plot(ucb_const_mode, [
                ("max.rng_seed", 511),
                ("selection_mode", "ucb"),
                ("zero_mean_prior_std_dev", 330),
                ("unknown_prior_std_dev_scalar", 1.8),
                ("repeat_const", -1),
                ("final_choice_mode", "same"),
        ], bound_mode)
        fig.ticks(ucb_const_ticknames)
        fig.legend()
        fig.show(title="Regret by UCB constant factor and expected-cost rule",
                 xlabel="UCB constant factor * 0.1",
                 file_suffix=f"_final_choice_mode_same")

# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode uniform :: final_choice_mode marginal_prior
# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal marginal_prior :: final_choice_mode marginal_prior :: ucb_const 0 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000
if should_make_figure("2"):
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        # samples_n = 128
        common_filters = [
            ("max.rng_seed", 511),
            ("final_choice_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8)
        ]
        uniform_filters = common_filters + [("selection_mode", "uniform")]
        fig.plot(ucb_const_mode,
                 common_filters + [("selection_mode", "ucb")], bound_mode)

        fig.line_from(uniform_filters, "Uniform")

        fig.ylim([10, 44])
        fig.ticks(ucb_const_ticknames)
        fig.legend()

        fig.show(xlabel="UCB constant factor * 0.1",
                 file_suffix=f"_final_choice_mode_marginal_prior")

# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode uniform :: final_choice_mode marginal_prior
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal marginal_prior :: final_choice_mode marginal_prior :: normal.ucb_const -3300 :: bubble_best.ucb_const -4700 :: lower_bound.ucb_const -6800 :: marginal.ucb_const -6800 :: marginal_prior.ucb_const -100
if should_make_figure("3"):
    for metric in all_metrics:
        common_filters = [
            ("max.rng_seed", 4095),
            ("final_choice_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8),
        ]
        filters = [("selection_mode", "ucb"),
                   ("normal.ucb_const", -3300),
                   ("bubble_best.ucb_const", -4700),
                   ("lower_bound.ucb_const", -6800),
                   ("marginal.ucb_const", -6800),
                   ("marginal_prior.ucb_const", -100),
                   ] + common_filters
        uniform_filters = [("selection_mode", "uniform")] + common_filters

        fig = SqliteFigureBuilder(db_cursor, "steps_taken", metric, translations=t10s, x_param_scalar=0.25, x_param_log=True)

        fig.inset_plot([8.8, 12.2], [-1, 9], [0.4, 0.4, 0.57, 0.57])

        fig.plot(samples_n_mode, filters, bound_mode)
        fig.plot(samples_n_mode, uniform_filters, label="Uniform")

        fig.xlim([2.8, 12.4])
        fig.ticks(range(3, 12 + 1))
        fig.legend("lower left")
        fig.show(xlabel="log2(# of trials)")

# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal_prior :: selection_mode ucb ucbv ucbd klucb klucb+ uniform :: ucb.ucb_const -100 :: ucbv.ucb_const -22 :: ucbv.ucbv_const 0 :: ucbd.ucb_const -33 :: ucbd.ucbd_const 0.0001 :: klucb.ucb_const -0.0047 :: klucb.klucb_max_cost 4700 :: klucb+.ucb_const -0.01 :: klucb+.klucb_max_cost 680
if should_make_figure("4"):
    for metric in all_metrics:
        selection_mode = FigureMode(
            "selection_mode", ["ucb", "ucbv", "ucbd", "klucb", "klucb+", "uniform"])
        fig = SqliteFigureBuilder(db_cursor, "steps_taken", metric, translations=t10s, x_param_scalar=0.25, x_param_log=True)
        filters = [
            ("max.rng_seed", 4095),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8),
            ("ucb.ucb_const", -100),
            ("ucbv.ucb_const", -22),
            ("ucbv.ucbv_const", 0),
            ("ucbd.ucb_const", -33),
            ("ucbd.ucbd_const", 0.0001),
            ("klucb.ucb_const", -0.0047),
            ("klucb.klucb_max_cost", 4700),
            ("klucb+.ucb_const", -0.01),
            ("klucb+.klucb_max_cost", 680),
        ]

        fig.inset_plot([8.8, 12.2], [0.2, 1.9], [0.4, 0.4, 0.57, 0.57])

        fig.plot(samples_n_mode, filters, selection_mode)

        # fig.ylim([-20, 380])
        fig.xlim([2.8, 12.4])
        fig.ticks(range(3, 12 + 1))
        fig.legend("lower left")
        fig.show(xlabel="log2(# of trials)")

# cargo run --release rng_seed 0-16383 :: samples_n 8 16 32 64 128 256 512 1024 :: repeat_const 0 1024 2048 4096 8192 16384 32768 65536
if should_make_figure("repeat_const"):
    # repeat_const_vals = [0, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
    repeat_const_vals = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    repeat_const_mode = FigureMode("repeat_const", repeat_const_vals)
    samples_n_mode = FigureMode("samples_n", [8, 16, 32, 64, 128, 256, 512, 1024]) #, 2048, 4096])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        filters = [
            ("max.rng_seed", 16383),
            ("selection_mode", "klucb"),
            ("ucb_const", -0.0047),
            ("klucb_max_cost", 4700),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8)
        ]

        fig.plot(repeat_const_mode, filters, samples_n_mode, normalize="first")
        fig.axhline(1, color="black")
        repeat_const_ticks = [math.log2(v) if v > 0 else "w/o" for v in repeat_const_vals]
        fig.ticks(repeat_const_ticks)
        fig.ylim([0.76, 1.18])
        fig.legend("upper left")
        fig.show("Relative regret by repetition constant and # monte carlo trials", xlabel="log2(repetition constant)", ylabel="Relative regret")

# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode normal :: selection_mode ucb :: ucb_const -470
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal :: selection_mode ucb :: ucb_const -470
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: repeat_const 16384
if should_make_figure("final"):
    repeat_const_vals = [0, 8, 16, 64, 128, 256, 512, 1024]
    repeat_const_mode = FigureMode("repeat_const", repeat_const_vals)
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, "steps_taken", metric, translations=t10s, x_param_scalar=0.25, x_param_log=True)

        common_filters = [
            ("max.rng_seed", 4095),
            ("prioritize_worst_particles_z", 1000),
            ("bootstrap_confidence_z", 0),
            ("worst_particles_z_abs", "false"),
        ]

        normal_filters = common_filters + [
            ("selection_mode", "ucb"),
            ("ucb_const", -470),
            ("bound_mode", "normal"),
            ("repeat_const", -1),

        ]
        mac_filters = common_filters + [
            ("selection_mode", "ucb"),
            ("ucb_const", -470),
            ("bound_mode", "marginal"),
            ("repeat_const", -1),
        ]
        macp_filters = common_filters + [
            ("selection_mode", "klucb"),
            ("ucb_const", -0.0047),
            ("klucb_max_cost", 4700),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8),
            ("repeat_const", -1),
        ]
        final_filters = common_filters + [
            ("selection_mode", "klucb"),
            ("ucb_const", -0.0047),
            ("klucb_max_cost", 4700),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8),
            ("repeat_const", 16384),
        ]

        fig.plot(samples_n_mode, normal_filters, label="Normal")
        fig.plot(samples_n_mode, mac_filters, label="MAC (proposed)")
        fig.plot(samples_n_mode, macp_filters, label="MACP (proposed)")
        fig.plot(samples_n_mode, final_filters, label="MACP + repetition (proposed)")

        fig.legend()
        fig.ticks(range(3, 12 + 1))
        fig.show(xlabel="log2(# of trials)", file_suffix="_final_comparison")

# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 64 128 256 512 :: bound_mode marginal_prior :: bootstrap_confidence_z 0 0.01 0.022 0.047 0.1 0.22 0.47 1 2.2
# cargo run --release rng_seed 0-16383 :: samples_n 8 16 32 64 128 256 512 :: bound_mode marginal_prior :: bootstrap_confidence_z 0 0.01 0.022 0.047 0.1 0.22 0.47 1 2.2
# cargo run --release rng_seed 0-32767 :: samples_n 8 16 32 64 128 256 512 :: bound_mode marginal_prior :: bootstrap_confidence_z 0 0.01 0.022 0.047 0.1 0.22 0.47 1 2.2
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
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8),
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
        fig.show("Relative regret by top-level bootstrapping z-score and # trials", ylabel="Relative regret")

# cargo run --release rng_seed 0-32767 :: samples_n 8 16 32 64 128 256 512 :: bound_mode marginal_prior :: bootstrap_n 0 1 2 3 4 5 6 7
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
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8),
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

# time cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 64 128 256 512 :: most_visited_best_cost_consistency false true :: repeat_const 256 :: bootstrap_confidence_z 0.1
if should_make_figure("consistency"):
    for metric in all_metrics:
        most_visited_best_cost_consistency_mode = FigureMode("most_visited_best_cost_consistency", ["false", "true"])
        samples_n_mode = FigureMode(
            "samples_n", [8, 16, 32, 64, 128, 256, 512])
        fig = SqliteFigureBuilder(db_cursor, "steps_taken", metric, translations=t10s)

        filters = [
            ("max.rng_seed", 8191),
            ("selection_mode", "klucb"),
            ("ucb_const", -0.15),
            ("klucb_max_cost", 15000),
            ("bound_mode", "marginal_prior"),
            ("final_choice_mode", "same"),
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8),
            ("repeat_const", 256),
            ("bootstrap_confidence_z", 0.1),
            ("prioritize_worst_particles_z", 1000),
            ("worst_particles_z_abs", "false"),
            ("bootstrap_n", -1)
        ]

        fig.plot(samples_n_mode, filters, most_visited_best_cost_consistency_mode)
        fig.legend()
        fig.show()

# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: prioritize_worst_particles_z -1000 -2 -1 -0.5 0 0.5 1 2 1000
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
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8),
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

# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 :: prioritize_worst_particles_z 0 0.5 1 2 1000 :: worst_particles_z_abs true
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
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8),
            ("worst_particles_z_abs", "true"),
            ("repeat_const", -1),
        ]

        fig.plot(prioritize_worst_particles_z_mode, filters, samples_n_mode, normalize="last")
        fig.axhline(1, color="black")
        fig.ticks(prioritize_z_vals)
        fig.ylim([0.68, 1.18])
        fig.legend()
        fig.show(file_suffix="_worst_particles_z_abs_true")

# cargo run --release rng_seed 0-127 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal_prior :: selection_mode ucbv :: ucbv.ucbv_const 0 0.0001 0.001 0.01 0.1 :: ucb_const -10 -15 -22 -33 -47 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000
if should_make_figure("ucbv"):
    ucb_const_vals = [-10, -15, -22, -33, -47, -68, -100, -150, -220, -330, -470, -680, -1000, -1500, -2200, -3300, -4700, -6800, -10000]
    ucb_const_mode = FigureMode(
        "ucb_const", ucb_const_vals)
    ucbv_const_mode = FigureMode(
        "ucbv_const", [0, 0.0001, 0.001, 0.01, 0.1])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.plot(ucb_const_mode, [
            ("max.rng_seed", 127),
            ("selection_mode", "ucbv"),
            ("bound_mode", "marginal_prior"),
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8),
        ], ucbv_const_mode)

        fig.ticks(ucb_const_vals)
        fig.legend()
        fig.zoom(0.5)
        fig.show()

# cargo run --release rng_seed 0-127 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal_prior :: selection_mode ucbd :: ucbd.ucbd_const 0.000001 0.00001 0.0001 0.001 0.01 0.1 1 :: ucb_const -10 -15 -22 -33 -47 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000
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
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8),
        ], ucbd_const_mode)

        fig.ticks(ucb_const_vals)
        fig.legend()
        fig.zoom(0.5)
        fig.show()

# cargo run --release rng_seed 0-127 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal_prior :: selection_mode klucb :: klucb.klucb_max_cost 330 470 680 1000 1500 2200 3300 4700 :: ucb_const -0.001 -0.0022 -0.0047 -0.01 -0.022 -0.047 -0.1 -0.22 -0.47 -1
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
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8),
        ], klucb_max_cost_mode)

        fig.ticks(ucb_const_vals)
        fig.zoom(0.5)
        fig.legend()
        fig.show(file_suffix="_selection_mode_klucb")

# cargo run --release rng_seed 0-127 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal_prior :: selection_mode klucb+ :: klucb+.klucb_max_cost 330 470 680 1000 1500 2200 3300 4700 :: ucb_const -0.001 -0.0022 -0.0047 -0.01 -0.022 -0.047 -0.1 -0.22 -0.47 -1
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
            ("zero_mean_prior_std_dev", 330),
            ("unknown_prior_std_dev_scalar", 1.8),
        ], klucb_max_cost_mode)

        fig.ticks(ucb_const_vals)
        fig.zoom(0.5)
        fig.legend()
        fig.show(file_suffix="_selection_mode_klucb+")

if len(sys.argv) == 1 or "help" in sys.argv:
    print("Valid figure options:")
    for option in figure_cmd_line_options:
        print(option)
