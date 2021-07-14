#!/usr/bin/python3
from common_plot import FigureKind, FigureMode, print_all_parameter_values_used

show_only = False
make_pdf_also = False


t10s = dict()
t10s["regret"] = "Regret"
t10s["estimation_error"] = "Estimation error"
t10s["samples_n"] = "# Samples"
t10s["bound_mode"] = "Mode to estimate cost"
t10s["normal"] = "Normal"
t10s["lower_bound"] = "Using lower bound"
t10s["bubble_best"] = "Using bubble-best"
t10s["marginal"] = "Using marginal action costs"
t10s["portion_bernoulli"] = "% cost Bernoulli (instead of Gaussian)"
t10s["ucb_const"] = "UCB constant factor"

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

bound_mode = FigureMode("bound_mode", ["normal", "bubble_best", "lower_bound", "marginal"])

ucb_const_kind = FigureKind(
    "ucb_const", [-220, -330, -470, -680, -1000, -1500, -2200, -3300, -4700, -6800, -10000, -15000, -22000, -33000, -47000, -68000, -100000], translations=t10s)

# cargo run --release rng_seed 0-8191 :: portion_bernoulli 0.5 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: ucb_const -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000 -15000 -22000 -33000 -47000 -68000 -100000 -150000 -220000 -330000 :: thread_limit 24
if False:
    for metric in all_metrics:
        ucb_const_kind.plot(results, metric, filters=[
                            "_selection_mode_ucb_",
                            "_portion_bernoulli_0.5_"], mode=bound_mode)

samples_n_kind = FigureKind(
    "samples_n", [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], translations=t10s)
# Gave panic!!! Check!!!
# cargo run --release rng_seed 0-4095 :: samples_n 4 8 16 32 64 128 256 512 1024 2048 :: portion_bernoulli 0 0.5 1 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: normal.ucb_const -15000 :: bubble_best.ucb_const -1500 :: lower_bound.ucb_const -3300 :: marginal.ucb_const -47000 :: thread_limit 24
if False:
    for metric in all_metrics:
        for portion_bernoulli in [0, 0.5, 1]:
            samples_n_kind.plot(results, metric, filters=[
                                "_selection_mode_ucb_",
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
    "portion_bernoulli", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], translations=t10s)

# cargo run --release rng_seed 0-8191 :: portion_bernoulli 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 :: samples_n 64 :: bound_mode normal lower_bound marginal :: normal.ucb_const -1000 :: lower_bound.ucb_const -3000 :: marginal.ucb_const -3000 :: thread_limit 24
if False:
    for metric in all_metrics:
        portion_bernoulli_kind.plot(results, metric, filters=[
                                    "_samples_n_64_"], mode=bound_mode)

ucb_const_kind = FigureKind(
    "ucb_const", [-10, -30, -100, -300, -1000, -3000, -10000, -30000], translations=t10s)
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
    "ucbv_const", [0, 0.0001, 0.001, 0.01, 0.1, 1, 10], translations=t10s)
# cargo run --release rng_seed 0-1023 :: selection_mode ucbv :: ucbv.ucbv_const 0 0.0001 0.001 0.01 0.1 1 10 :: ucb_const -10 -30 -100 -300 -1000 -3000 -10000 -30000 :: samples_n 64 :: portion_bernoulli 0 1 :: bound_mode normal lower_bound marginal :: thread_limit 24
if False:
    for metric in all_metrics:
        for portion_bernoulli in [0, 1]:
            ucb_const_kind.plot(results, metric, filters=[
                                "_selection_mode_ucbv_", "_samples_n_64_", f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)
            ucbv_const_kind.plot(results, metric, filters=[
                "_selection_mode_ucbv_", "_samples_n_64_", f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)

ucbd_const_kind = FigureKind(
    "ucbd_const", ["0.000001", "0.00001", 0.0001, 0.001, 0.01, 0.1, 1], translations=t10s)
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
    "ucb_const", [-0.01, -0.03, -0.1, -0.3, -1, -3, -10, -30, -100, -300, -1000], translations=t10s)
klucb_max_cost_kind = FigureKind(
    "klucb_max_cost", [1000, 2000, 4000, 8000, 16000], translations=t10s)
# cargo run --release rng_seed 0-1023 :: selection_mode klucb :: klucb.klucb_max_cost 1000 2000 4000 8000 16000 :: klucb.ucb_const -0.01 -0.03 -0.1 -0.3 -1 -3 -10 -30 -100 -300 -1000 -3000 -10000 -30000 :: samples_n 64 :: portion_bernoulli 0 1 :: bound_mode normal lower_bound marginal :: thread_limit 24
# cargo run --release rng_seed 0-4095 :: selection_mode klucb :: klucb.ucb_const -0.01 -0.03 -0.1 -0.3 -1 -3 -10 -30 -100 -300 -1000 -3000 -10000 -30000 :: samples_n 64 :: portion_bernoulli 0 1 :: bound_mode normal lower_bound marginal :: thread_limit 24
if False:
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

prioritize_worst_particles_n_kind = FigureKind(
    "prioritize_worst_particles_n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], translations=t10s)
# cargo run --release rng_seed 0-8191 :: prioritize_worst_particles_n 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 :: selection_mode klucb :: klucb.ucb_const -1 :: samples_n 64 :: portion_bernoulli 0 1 :: bound_mode normal bubble_best lower_bound marginal :: thread_limit 24
if False:
    for metric in all_metrics:
        for portion_bernoulli in [0, 1]:
            prioritize_worst_particles_n_kind.plot(results, metric, filters=[
                "_selection_mode_klucb_", "_ucb_const_-1_", "_samples_n_64_", f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)

prioritize_worst_particles_z_kind = FigureKind(
    "prioritize_worst_particles_z", [-3.5, -3, -2.5, -2, -1.5, -1, -.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], translations=t10s)
# cargo run --release rng_seed 0-8191 :: prioritize_worst_particles_z -3.5 -3 -2.5 -2 -1.5 -1 -.5 0 0.5 1 1.5 2 2.5 3 3.5 :: selection_mode klucb :: klucb.ucb_const -1 :: samples_n 64 :: portion_bernoulli 0 0.5 1 :: bound_mode normal bubble_best lower_bound marginal :: thread_limit 24
if False:
    bound_mode = FigureMode("bound_mode", ["normal", "bubble_best", "lower_bound", "marginal"])
    for metric in all_metrics:
        for portion_bernoulli in [0, 0.5, 1]:
            prioritize_worst_particles_z_kind.plot(results, metric, filters=[
                "_selection_mode_klucb_", "_ucb_const_-1_", "_samples_n_64_", f"_portion_bernoulli_{portion_bernoulli}_"], mode=bound_mode)
