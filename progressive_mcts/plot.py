#!/usr/bin/python3
import pdb
import time
from common_plot import parse_parameters, FigureKind, FigureMode, print_all_parameter_values_used, evaluate_conditions

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
t10s["prioritize_worst_particles_z"] = "Prioritize worst particles with z-scores above"
t10s["ucb"] = "UCB"
t10s["ucbv"] = "UCB-V"
t10s["ucbd"] = "UCB-delta"
t10s["klucb"] = "KL-UCB"
t10s["klucb+"] = "KL-UCB+"
t10s["random"] = "Random"
t10s["uniform"] = "Uniform"

start_time = time.time()
results = []
with open("results.cache", "r") as f:
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
print(f"took {time.time() - start_time:.2f} seconds to load data")

all_metrics = ["regret"]  # , "estimation_error"]

# print(max([r for r in results if all(f in r["name"] for f in ["_method_mcts_", "_use_cfb_true_",
#                                                               "_smoothness_0_", "_safety_100_", "_ud_5_"])], key=lambda entry: entry["safety"]))

if False:
    for samples_n in [64, 128]:
        for bound_mode in ["normal", "lower_bound", "marginal"]:
            evaluate_conditions(results, all_metrics, [
                                ("bound_mode", bound_mode),
                                ("samples_n", samples_n),
                                ("portion_bernoulli", 1)])

bound_mode = FigureMode("bound_mode", ["normal", "bubble_best", "lower_bound", "marginal"])

ucb_const_kind = FigureKind(
    "ucb_const", [-680, -1000, -1500, -2200, -3300, -4700, -6800, -10000, -15000, -22000, -33000, -47000, -68000], translations=t10s)

# cargo run --release rng_seed 0-32767 :: portion_bernoulli 0.5 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: final_choice_mode same :: ucb_const -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000 -15000 -22000 -33000 -47000 -68000 :: thread_limit 24
if False:
    for metric in all_metrics:
        ucb_const_kind.plot(results, metric, filters=[
                            ("selection_mode", "ucb"),
                            ("portion_bernoulli", "0.5")], mode=bound_mode, title="Regret by UCB constant factor and expected-cost rule")


ucb_const_kind = FigureKind(
    "ucb_const", [-680, -1000, -1500, -2200, -3300, -4700, -6800, -10000, -15000, -22000, -33000], translations=t10s)

# print_all_parameter_values_used(results, [])
# print_all_parameter_values_used(
#     results, [("final_choice_mode", "marginal"),
#   ("portion_bernoulli", "0.5"), ("samples_n", 512), ("max.rng_seed", 511)])
# quit()

# cargo run --release rng_seed 0-32767 :: portion_bernoulli 0.5 :: selection_mode ucb :: final_choice_mode normal bubble_best lower_bound marginal :: bound_mode marginal :: ucb_const -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000 -15000 -22000 -33000 :: thread_limit 24
#
# cargo run --release rng_seed 0-16383 :: portion_bernoulli 0.5 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: final_choice_mode marginal :: ucb_const -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000 -15000 -22000 -33000 :: thread_limit 24
# cargo run --release rng_seed 0-16383 :: portion_bernoulli 0.5 :: selection_mode random :: final_choice_mode marginal :: thread_limit 24
# cargo run --release rng_seed 0-8191 :: samples_n 512 :: portion_bernoulli 0.5 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: final_choice_mode marginal :: ucb_const -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000 -15000 -22000 -33000 :: thread_limit 24
# cargo run --release rng_seed 0-8191 :: samples_n 512 :: portion_bernoulli 0.5 :: selection_mode random uniform :: final_choice_mode marginal :: thread_limit 24
final_choice_mode = FigureMode(
    "final_choice_mode", ["normal", "bubble_best", "lower_bound", "marginal"])
# results.cache_final_choice_mode64, results.cache_final_choice_mode512
if False:
    for metric in all_metrics:
        # ucb_const_kind.plot(results, metric, filters=[
        #                     ("selection_mode", "ucb"),
        #                     ("bound_mode", "marginal"),
        #                     ("portion_bernoulli", "0.5")], mode=final_choice_mode)
        for samples_n in [64, 512]:
            common_filters = [("final_choice_mode", "marginal"),
                              ("portion_bernoulli", "0.5"), ("samples_n", samples_n), ("max.rng_seed", 16383)]
            random_filters = common_filters + [("selection_mode", "random")]
            uniform_filters = common_filters + [("selection_mode", "uniform")]
            ucb_const_kind.plot(results, metric, filters=common_filters +
                                [("selection_mode", "ucb")], mode=bound_mode, extra_lines=[("Random", random_filters), ("Uniform", uniform_filters)])

samples_n_kind = FigureKind(
    "samples_n", [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], translations=t10s)

# cargo run --release rng_seed 0-4095 :: samples_n 4 8 16 32 64 128 256 512 1024 2048 :: portion_bernoulli 0.5 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: final_choice_mode marginal :: ucb_const -6800 :: thread_limit 24
if False:
    for metric in all_metrics:
        samples_n_kind.plot(results, metric, filters=[
                            ("selection_mode", "ucb"),
                            ("final_choice_mode", "marginal"),
                            ("ucb_const", "-6800"),
                            ("portion_bernoulli", "0.5")], mode=bound_mode)


samples_n_kind = FigureKind(
    "samples_n", [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], translations=t10s)

# print_all_parameter_values_used(
#     results, [("final_choice_mode", "marginal"),
#               ("portion_bernoulli", "0.5"), ("samples_n", 512), ("max.rng_seed", 511)])
# quit()

# results.cache_bound_mode and results.cache_bound_mode_final_choice_same (as a mistake)
# cargo run --release rng_seed 0-8191 :: samples_n 16 32 64 128 256 512 1024 2048 4096 8192 16384 :: portion_bernoulli 0.5 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: final_choice_mode marginal :: normal.ucb_const -15000 :: bubble_best.ucb_const -15000 :: lower_bound.ucb_const -15000 :: marginal.ucb_const -6800 :: thread_limit 24
if False:
    for metric in all_metrics:
        for portion_bernoulli in [0.5]:
            common_filters = [
                ("final_choice_mode", "marginal"),
                ("portion_bernoulli", portion_bernoulli)]
            filters = [("selection_mode", "ucb")] + common_filters
            uniform_filters = [("selection_mode", "uniform")] + common_filters
            samples_n_kind.plot(results, metric, filters=filters,
                                mode=bound_mode, extra_modes=[("Uniform", uniform_filters)])


selection_mode = FigureMode(
    "selection_mode", ["ucb", "ucbv", "ucbd", "klucb", "klucb+", "uniform"])
# results.cache_selection_mode
# cargo run --release rng_seed 0-8191 :: samples_n 16 32 64 128 256 512 1024 2048 4096 8192 16384 :: portion_bernoulli 0.5 :: bound_mode marginal :: selection_mode ucb ucbv ucbd klucb klucb+ random uniform :: ucb_const -6800 :: ucbv.ucb_const -4700 :: ucbv.ucbv_const 0 :: ucbd.ucb_const -22000 :: ucbd.ucbd_const 1 :: klucb.ucb_const -1.5 :: klucb.klucb_max_cost 10000 :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: thread_limit 24
if False:
    samples_n_kind = FigureKind(
        "samples_n", [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], translations=t10s)
    for metric in all_metrics:
        for portion_bernoulli in [0.5]:
            filters = [("portion_bernoulli", portion_bernoulli), ("max.rng_seed", 8191)]
            samples_n_kind.plot(results, metric, filters=filters, mode=selection_mode)

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
    "prioritize_worst_particles_z", [-1000, -3.5, -3, -2.5, -2, -1.5, -1, -.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 1000], translations=t10s)
# cargo run --release rng_seed 0-16383 :: prioritize_worst_particles_z -1000 -3.5 -3 -2.5 -2 -1.5 -1 -.5 0 0.5 1 1.5 2 2.5 3 3.5 1000 :: selection_mode klucb+ :: klucb+.ucb_const -2.2 :: klucb+.klucb_max_cost 10000 :: samples_n 512 :: portion_bernoulli 0.5 :: bound_mode marginal :: thread_limit 24
if True:
    bound_mode = FigureMode("bound_mode", ["normal", "bubble_best", "lower_bound", "marginal"])
    for metric in all_metrics:
        for portion_bernoulli in [0.5]:
            filters = [("selection_mode", "klucb+"), ("samples_n", 512),
                       ("portion_bernoulli", portion_bernoulli)]
            prioritize_worst_particles_z_kind.plot(results, metric, filters=filters)
