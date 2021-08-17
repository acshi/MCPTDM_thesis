#!/usr/bin/python3
import pdb
import time
from common_plot import FigureBuilder, parse_parameters, FigureKind, FigureMode, print_all_parameter_values_used, evaluate_conditions

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
t10s["lower_bound"] = "Using lower bound"
t10s["bubble_best"] = "Using bubble-best"
t10s["marginal"] = "Using marginal action costs"
t10s["portion_bernoulli"] = "% cost Bernoulli (instead of Gaussian)"
t10s["ucb_const"] = "UCB constant factor"
t10s["prioritize_worst_particles_z"] = "Particle repeating"
t10s["ucb"] = "UCB"
t10s["ucbv"] = "UCB-V"
t10s["ucbd"] = "UCB-delta"
t10s["klucb"] = "KL-UCB"
t10s["klucb+"] = "KL-UCB+"
t10s["random"] = "Random"
t10s["uniform"] = "Uniform"
t10s["1000"] = "No particle repeating"
t10s["-1000"] = "W/ particle repeating"

start_time = time.time()
results = []
with open("results.cache", "r") as f:
    line_num = 0
    for line in f:
        parts = line.split()
        if len(parts) > 3:
            entry = dict()
            entry["params"] = parse_parameters(parts[0])
            entry["chosen_cost"] = float(parts[1])
            entry["chosen_true_cost"] = float(parts[2])
            entry["true_best_cost"] = float(parts[3])

            entry["regret"] = entry["chosen_true_cost"] - entry["true_best_cost"]
            entry["estimation_error"] = abs(entry["chosen_true_cost"] - entry["chosen_cost"])

            results.append(entry)
        else:
            continue
        line_num += 1
        # if line_num > 5000:
        #     break
print(f"took {time.time() - start_time:.2f} seconds to load data")

all_metrics = ["regret"]  # , "estimation_error"]

# print(max([r for r in results if all(f in r["name"] for f in ["_method_mcts_", "_use_cfb_true_",
#                                                               "_smoothness_0_", "_safety_100_", "_ud_5_"])], key=lambda entry: entry["safety"]))

# Fig. 1.
# cargo run --release rng_seed 0-32767 :: samples_n 256 :: portion_bernoulli 0.5 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: final_choice_mode same :: ucb_const -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000 -15000 -22000 -33000 -47000 -68000 :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: samples_n 256 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: final_choice_mode same :: ucb_const -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000 -15000 -22000 -33000 -47000 -68000 :: thread_limit 24
# results.cache_ucb_const_bound_mode
bound_mode = FigureMode("bound_mode", ["normal", "bubble_best", "lower_bound", "marginal"])
ucb_const_vals = [-680, -1000, -1500, -2200, -3300, -4700, -
                  6800, -10000, -15000, -22000, -33000, -47000, -68000]
ucb_const_ticknames = [str(val / 100) for val in ucb_const_vals]
ucb_const_ticknames = [name.replace(".0", "") if name.endswith(
    ".0") else name for name in ucb_const_ticknames]
ucb_const_kind = FigureKind(
    "ucb_const", ucb_const_ticknames, ucb_const_vals, translations=t10s)
if False:
    for metric in all_metrics:
        ucb_const_kind.plot(results, metric, filters=[
                            ("samples_n", 256),
                            ("selection_mode", "ucb")], mode=bound_mode, title="Regret by UCB constant factor and expected-cost rule", xlabel="UCB constant factor * 10^-2")

# print_all_parameter_values_used(results, [])
# print_all_parameter_values_used(
#     results, [("final_choice_mode", "marginal"),
#   ("portion_bernoulli", "0.5"), ("samples_n", 512), ("max.rng_seed", 511)])
# quit()

# Fig. 2.
# cargo run --release rng_seed 0-32767 :: samples_n 256 :: portion_bernoulli 0.5 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: final_choice_mode marginal :: ucb_const -1000 -1500 -2200 -3300 -4700 -6800 -10000 -15000 -22000 -33000 -47000 -68000 :: thread_limit 7
# cargo run --release rng_seed 0-32767 :: samples_n 256 :: portion_bernoulli 0.5 :: selection_mode uniform :: final_choice_mode marginal :: thread_limit 7
# cargo run --release rng_seed 0-4095 :: samples_n 256 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: final_choice_mode marginal :: ucb_const -1000 -1500 -2200 -3300 -4700 -6800 -10000 -15000 -22000 -33000 -47000 -68000 :: thread_limit 7
# cargo run --release rng_seed 0-4095 :: samples_n 256 :: selection_mode uniform :: final_choice_mode marginal :: thread_limit 7
# results.cache_bound_mode_final_choice_marginal
if False:
    for metric in all_metrics:
        fig = FigureBuilder(results, None, metric, translations=t10s)

        common_filters = [("final_choice_mode", "marginal"),
                          ("samples_n", 256), ("max.rng_seed", 32767)]
        uniform_filters = common_filters + [("selection_mode", "uniform")]
        fig.plot(FigureMode("ucb_const", ucb_const_vals[1:]),
                 common_filters + [("selection_mode", "ucb")], bound_mode)

        fig.line(uniform_filters, "Uniform")

        # fig.ylim([75, 110])
        fig.ticks(ucb_const_ticknames[1:])
        fig.legend()

        fig.show(xlabel="UCB constant factor * 10^-2",
                 file_suffix="_samples_n_256_final_choice_mode_marginal")

# print_all_parameter_values_used(
#     results, [("final_choice_mode", "marginal"),
#               ("portion_bernoulli", "0.5"), ("samples_n", 512), ("max.rng_seed", 511)])
# quit()

# Fig. 3.
# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 :: portion_bernoulli 0.5 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: final_choice_mode marginal :: normal.ucb_const -10000 :: bubble_best.ucb_const -6800 :: lower_bound.ucb_const -15000 :: marginal.ucb_const -6800 :: thread_limit 24
# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 :: portion_bernoulli 0.5 :: selection_mode uniform :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: final_choice_mode marginal :: normal.ucb_const -4700 :: bubble_best.ucb_const -3300 :: lower_bound.ucb_const -3300 :: marginal.ucb_const -3300 :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode uniform :: thread_limit 24
# results.cache_bound_mode_samples_n
if False:
    for metric in all_metrics:
        common_filters = []
        filters = [("selection_mode", "ucb"),
                   ("final_choice_mode", "marginal")] + common_filters
        uniform_filters = [("selection_mode", "uniform")] + common_filters

        fig = FigureBuilder(results, None, metric, translations=t10s)
        samples_n_mode = FigureMode(
            "samples_n", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

        # (rect, connections) = fig.inset_plot([8.8, 12.2], [-1, 48], [0.4, 0.4, 0.57, 0.57])
        (rect, connections) = fig.inset_plot([5.8, 9.2], [-1, 35], [0.4, 0.4, 0.57, 0.57])
        for connection in connections:
            connection.set_visible(False)
        rect.set_label(None)

        fig.plot(samples_n_mode, filters, bound_mode)
        fig.plot(samples_n_mode, uniform_filters, label="Uniform")

        fig.legend("lower left")
        fig.ticks(["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
        fig.show(xlabel="log2(# of samples)")


# Fig. 4.
# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 :: portion_bernoulli 0.5 :: bound_mode marginal :: selection_mode ucb ucbv ucbd klucb klucb+ uniform :: ucb_const -6800 :: ucbv.ucb_const -4700 :: ucbv.ucbv_const 0 :: ucbd.ucb_const -22000 :: ucbd.ucbd_const 1 :: klucb.ucb_const -1.5 :: klucb.klucb_max_cost 10000 :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: thread_limit 24
# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 :: bound_mode marginal :: selection_mode ucb ucbv ucbd klucb klucb+ uniform :: ucb_const -6800 :: ucbv.ucb_const -4700 :: ucbv.ucbv_const 0 :: ucbd.ucb_const -22000 :: ucbd.ucbd_const 1 :: klucb.ucb_const -1.5 :: klucb.klucb_max_cost 10000 :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: thread_limit 24
# results.cache_selection_mode
if False:
    for metric in all_metrics:
        selection_mode = FigureMode(
            "selection_mode", ["ucb", "ucbv", "ucbd", "klucb", "klucb+", "uniform"])
        fig = FigureBuilder(results, None, metric, translations=t10s)
        samples_n_mode = FigureMode(
            "samples_n", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
        filters = [("portion_bernoulli", 0.5), ("max.rng_seed", 8191)]

        (rect, connections) = fig.inset_plot([7.8, 12.2], [-1, 45], [0.4, 0.4, 0.57, 0.57])
        for connection in connections:
            connection.set_visible(False)
        rect.set_label(None)

        fig.plot(samples_n_mode, filters, selection_mode)

        fig.legend("lower left")
        fig.ticks(["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
        fig.show(xlabel="log2(# of samples)")

ucb_const_kind = FigureKind(
    "ucb_const", [-10, -15, -22, -33, -47, -68, -100, -150, -220, -330, -470, -680, -1000, -1500, -2200, -3300, -4700, -6800, -10000, -15000, -22000, -33000, -47000, -68000, -100000], translations=t10s)
ucbv_const_kind = FigureKind(
    "ucbv_const", [0, 0.0001, 0.001, 0.01, 0.1, 1, 10], translations=t10s)
# cargo run --release rng_seed 0-4095 :: selection_mode ucbv :: ucbv.ucbv_const 0 0.0001 0.001 0.01 0.1 1 10 :: ucb_const -10 -15 -22 -33 -47 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000 -15000 -22000 -33000 -47000 -68000 -100000 :: samples_n 64 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
if False:
    for metric in all_metrics:
        for portion_bernoulli in [0.5]:
            filters = [("selection_mode", "ucbv"), ("samples_n", 64),
                       ("portion_bernoulli", portion_bernoulli)]
            ucb_const_kind.plot(results, metric, filters=filters)
            ucbv_const_kind.plot(results, metric, filters=filters + [("ucb_const", -4700)])
            # Best is ucb of -4700 and ucbv of 0...
            evaluate_conditions(results, all_metrics, filters +
                                [("ucb_const", -4700), ("ucbv_const", 0)])

ucbd_const_kind = FigureKind(
    "ucbd_const", ["0.000001", "0.00001", 0.0001, 0.001, 0.01, 0.1, 1], translations=t10s)
# cargo run --release rng_seed 0-4095 :: selection_mode ucbd :: ucbd.ucbd_const 0.000001 0.00001 0.0001 0.001 0.01 0.1 1 :: ucb_const -10 -15 -22 -33 -47 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000 -15000 -22000 -33000 -47000 -68000 -100000 :: samples_n 64 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
if False:
    for metric in all_metrics:
        for portion_bernoulli in [0.5]:
            filters = [("selection_mode", "ucbd"), ("samples_n", 64),
                       ("portion_bernoulli", portion_bernoulli)]
            ucb_const_kind.plot(results, metric, filters=filters)
            ucbd_const_kind.plot(results, metric, filters=filters + [("ucb_const", -22000)])
            # Best is ucb of -22000 and ucbd of 1...
            evaluate_conditions(results, all_metrics, filters +
                                [("ucb_const", -22000), ("ucbd_const", 1)])
            # regret has mean:  133.2 and mean std dev:  3.173


ucb_const_kind = FigureKind(
    "ucb_const", [-0.01, -0.015, -0.022, -0.033, -0.047, -0.068, -0.1, -0.15, -0.22, -0.33, -0.47, -0.68, -1, -1.5, -2.2, -3.3, -4.7, -6.8, -10, -15, -22, -33, -47, -68, -100], translations=t10s)
klucb_max_cost_kind = FigureKind(
    "klucb_max_cost", [1000, 1500, 2200, 3300, 4700, 6800, 10000, 15000], translations=t10s)
# cargo run --release rng_seed 0-4095 :: selection_mode klucb :: klucb.klucb_max_cost 1000 1500 2200 3300 4700 6800 10000 15000 :: ucb_const -0.01 -0.015 -0.022 -0.033 -0.047 -0.068 -0.1 -0.15 -0.22 -0.33 -0.47 -0.68 -1 -1.5 -2.2 -3.3 -4.7 -6.8 -10 -15 -22 -33 -47 -68 -100 :: samples_n 64 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
if False:
    for metric in all_metrics:
        for portion_bernoulli in [0.5]:
            filters = [("selection_mode", "klucb"), ("samples_n", 64),
                       ("portion_bernoulli", portion_bernoulli)]
            # ucb_const_kind.plot(results, metric, filters=filters)
            # klucb_max_cost_kind.plot(results, metric, filters=filters + [("ucb_const", -1.5)])

        evaluate_conditions(results, all_metrics, filters + [
            ("ucb_const", -1.5),
            ("klucb_max_cost", 10000)])
        # regret has mean:  137.2 and mean std dev:  3.147

# cargo run --release rng_seed 0-4095 :: selection_mode klucb+ :: klucb+.klucb_max_cost 1000 1500 2200 3300 4700 6800 10000 15000 :: ucb_const -0.01 -0.015 -0.022 -0.033 -0.047 -0.068 -0.1 -0.15 -0.22 -0.33 -0.47 -0.68 -1 -1.5 -2.2 -3.3 -4.7 -6.8 -10 -15 -22 -33 -47 -68 -100 :: samples_n 64 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
if False:
    for metric in all_metrics:
        for portion_bernoulli in [0.5]:
            filters = [("selection_mode", "klucb+"), ("samples_n", 64),
                       ("portion_bernoulli", portion_bernoulli)]
            ucb_const_kind.plot(results, metric, filters=filters)
            klucb_max_cost_kind.plot(results, metric, filters=filters + [("ucb_const", -2.2)])

        evaluate_conditions(results, all_metrics, filters + [
            ("ucb_const", -2.2),
            ("klucb_max_cost", 10000)])
        # regret has mean:  133.1 and mean std dev:  3.175


prioritize_worst_particles_n_kind = FigureKind(
    "prioritize_worst_particles_n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], translations=t10s)
# cargo run --release rng_seed 0-8191 :: prioritize_worst_particles_n 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 :: selection_mode klucb :: klucb.ucb_const -1 :: samples_n 64 :: portion_bernoulli 0 1 :: bound_mode normal bubble_best lower_bound marginal :: thread_limit 24
if False:
    for metric in all_metrics:
        for portion_bernoulli in [0, 1]:
            prioritize_worst_particles_n_kind.plot(results, metric, filters=[
                "_selection_mode_klucb_", "_ucb_const_-1_", "_samples_n_64_", f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)

prioritize_worst_particles_z_kind = FigureKind(
    "prioritize_worst_particles_z", [-1000, -2, -1, -.5, 0, 0.5, 1, 2, 1000], translations=t10s)
# cargo run --release rng_seed 0-4095 :: samples_n 128 256 512 :: prioritize_worst_particles_z -1000 -2 -1 -0.5 0 0.5 1 2 1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: bound_mode marginal :: thread_limit 24
if False:
    # repeat_at_all_levels_mode = FigureMode(
    #     "repeat_at_all_levels", ["false", "true"])
    samples_n_mode = FigureMode(
        "samples_n", [512, 1024, 2048])
    for metric in all_metrics:
        filters = [("selection_mode", "klucb+"), ("repeat_at_all_levels", "false")]
        prioritize_worst_particles_z_kind.plot(
            results, metric, filters=filters, mode=samples_n_mode)

# cargo run --release rng_seed 0-4095 :: samples_n 512 1024 2048 :: repeat_const 0 16 64 256 1024 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: bound_mode marginal :: thread_limit 24
if False:
    repeat_const_kind = FigureKind(
        "repeat_const", [0, 16, 64, 256, 1024], translations=t10s)
    samples_n_mode = FigureMode(
        "samples_n", [512, 1024, 2048])
    for metric in all_metrics:
        filters = [("selection_mode", "klucb+"),
                   ("repeat_at_all_levels", "false")]
        repeat_const_kind.plot(
            results, metric, filters=filters, mode=samples_n_mode)

# cargo run --release rng_seed 0-8191 :: repeat_particle_sign -1 0 1 :: n_actions 5 6 7 8 9 10 :: samples_n 512 :: repeat_const 0 64 128 256 512 1024 2048 :: thread_limit 24
if True:
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

# cargo run --release rng_seed 0-8191 :: samples_n 8 16 32 :: prioritize_worst_particles_z -1000 1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: samples_n 64 128 256 512 1024 :: prioritize_worst_particles_z -1000 1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
# cargo run --release rng_seed 0-1023 :: samples_n 2048 4096 8192 16384 32768 :: prioritize_worst_particles_z -1000 1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
if False:
    samples_n_kind = FigureKind(
        "samples_n", ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"], [
            8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        translations=t10s)
    repeat_particles_mode = FigureMode(
        "prioritize_worst_particles_z", [-1000, 1000])
    for metric in all_metrics:
        for portion_bernoulli in [0.5]:
            filters = [("selection_mode", "klucb+"),
                       ("portion_bernoulli", portion_bernoulli)]
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
