#!/usr/bin/python3
from common_plot import parse_parameters, FigureBuilder, FigureMode, print_all_parameter_values_used, evaluate_conditions, filter_extra
import time
import sys

t10s = dict()
t10s["discount_factor"] = "Discount Factor"
t10s["safety"] = "Proportion unsafe"
t10s["cost.safety"] = "Safety cost"
t10s["cost.efficiency"] = "Efficiency cost"
t10s["cost"] = "Cost"
t10s["efficiency"] = "Efficiency"
t10s["ud"] = "Uncomfortable decelerations"
t10s["cc"] = "Curvature change"
t10s["tree"] = "Tree"
t10s["mpdm"] = "MPDM"
t10s["eudm"] = "EUDM"
t10s["mcts"] = "MCPTDM (proposed)"
t10s["method"] = "Method"
t10s["false"] = "w/o CFB"
t10s["true"] = "CFB"
t10s["use_cfb"] = "CFB"
t10s["seconds"] = "Computation time (s)"
t10s["997_ts"] = "99.7% Computation time (s)"
t10s["95_ts"] = "95% Computation time (s)"
t10s["mean_ts"] = "Mean computation time (s)"
t10s["search_depth"] = "Search depth"
t10s["samples_n"] = "# Samples"
t10s["bound_mode"] = "UCB expected-cost rule"
t10s["final_choice_mode"] = "Final choice expected-cost rule"
t10s["selection_mode"] = "UCB variation"
t10s["normal"] = "Normal"
t10s["lower_bound"] = "Using lower bound"
t10s["bubble_best"] = "Using bubble-best"
t10s["marginal"] = "Using marginal action costs"
t10s["ucb_const"] = "UCB constant factor"
t10s["prioritize_worst_particles_z"] = "Prioritize worst particles with z-scores above"
t10s[None] = "Average"

figure_cmd_line_options = []
def should_make_figure(fig_name):
    figure_cmd_line_options.append(fig_name)
    return fig_name in sys.argv

cache_file = sys.argv[2] if len(sys.argv) > 2 and ".cache" in sys.argv[2] else "results.cache"

start_time = time.time()
results = []
with open(cache_file, "r") as f:
    for line in f:
        parts = line.split()
        if len(parts) > 16:
            entry = dict()
            entry["params"] = parse_parameters(parts[0], skip=["extra_ego_accdec_policies", "search_depth", "total_forward_t", "single_trial_discount_factor", "bootstrap_confidence_z", "max_steps", "safety_margin_low", "safety_margin_high", "accel", "steer", "plan_change"])
            entry["crashed"] = float(parts[5])
            entry["end_t"] = float(parts[6])
            entry["dist_travelled"] = float(parts[7])
            entry["efficiency"] = float(parts[8])
            entry["safety"] = float(parts[9])
            entry["ud"] = float(parts[10])
            entry["cc"] = float(parts[11])
            entry["mean_ts"] = float(parts[12])
            entry["95_ts"] = float(parts[13])
            entry["997_ts"] = float(parts[14])
            entry["max_ts"] = float(parts[15])
            entry["stddev_ts"] = float(parts[16])

            entry["cost.efficiency"] = float(parts[1])
            entry["cost.safety"] = float(parts[2])
            entry["cost.accel"] = float(parts[3])
            entry["cost.steer"] = float(parts[4])
            entry["cost"] = entry["cost.efficiency"] + entry["cost.safety"] + \
                entry["cost.accel"] + entry["cost.steer"]
            entry["binary_safety"] = 1 if entry["safety"] > 0 else 0

            results.append(entry)
        else:
            continue
print(f"took {time.time() - start_time:.2f} seconds to load data")

method_mode = FigureMode("method", ["fixed", "tree", "mpdm", "eudm", "mcts"])
discount_mode = FigureMode("discount_factor", [0.6, 0.7, 0.8, 0.9, 1])
cfb_mode = FigureMode("use_cfb", ["false", "true"])

# extra_accdec_mode = FigureMode("extra_ego_accdec_policies", [
#                                "-1", "1", "-2", "2", "-1,1", "-2,2", "1,2", "-1,-2", "-1,-2,-3,1,2,3"])

extra_accdec_mode = FigureMode("extra_ego_accdec_policies", [
                               "", "-1,-2,1,2", "-1,-2,-3,1,2,3"])

method_mode = FigureMode("method", ["fixed", "tree", "mpdm", "eudm", "mcts"])
cfb_mode = FigureMode("use_cfb", ["false", "true"])

plot_metrics = ["cost", "cost.safety", "efficiency"]
evaluate_metrics = ["cost", "safety", "efficiency", "cost.efficiency",
                    "cost.safety", "cost.accel", "cost.steer", "seconds"]

# find_filters = [("method", "eudm"), ("use_cfb", "true"), ("samples_n", 2)]
# print(max(filter_extra(results, find_filters), key=lambda entry: entry["seconds"]))
# quit()

# print_all_parameter_values_used(results, [])

# print_all_parameter_values_used(
#     results, [("method", "eudm"), ("use_cfb", "true"), ("samples_n", 2), ("max.rng_seed", 2047)])
# quit()

# time cargo run --release rng_seed 0:2:4095 :: method eudm :: use_cfb true :: eudm.samples_n 2 4 8
# time cargo run --release rng_seed 1:2:4095 :: method eudm :: use_cfb true :: eudm.samples_n 2 4 8
# time cargo run --release rng_seed 0:2:4095 :: use_cfb true :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode klucb :: mcts.ucb_const 1.5 :: mcts.klucb_max_cost 4.7 :: mcts.repeat_const 2048
# time cargo run --release rng_seed 1:2:4095 :: use_cfb true :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode klucb :: mcts.ucb_const 1.5 :: mcts.klucb_max_cost 4.7 :: mcts.repeat_const 2048
if should_make_figure("cfb"):
    samples_n_vals = [1, 2, 4, 8, 16, 32]
    # samples_n_vals = [8, 16, 32, 64, 128, 256]
    samples_n_mode = FigureMode("samples_n", samples_n_vals)
    cfb_vals = ["false", "true"]
    cfb_mode = FigureMode("use_cfb", cfb_vals)

    # repeat_mode = FigureMode("repeat_const", [-1, 2048])
    # bound_mode = FigureMode("bound_mode", ["marginal", "marginal_prior"])

    if True:
        filters = [
            ("use_cfb", "true"),
            ("method", "eudm"),
            # ("samples_n", 32),
            # ("method", "mcts"),
            # ("bound_mode", "marginal_prior"),
            # ("selection_mode", "klucb"),
            # ("klucb_max_cost", 4.7),
            # ("ucb_const", 1.5),
            # ("min.samples_n", 256),
            ("max.rng_seed", 4095),
        ]

        fig = FigureBuilder(results, None, "cost", translations=t10s)
        fig.plot(samples_n_mode, filters)
        fig.ticks(samples_n_vals)
        # fig.legend()
        fig.show(file_suffix="_for_ucb")

# time cargo run --release rng_seed 0:2:511 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode ucb :: mcts.ucb_const -1e5 -2.2e5 -4.7e5 -1e6 -2.2e6 -4.7e6 -1e7 :: mcts.repeat_const 2048
# time cargo run --release rng_seed 1:2:511 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode ucb :: mcts.ucb_const -1e5 -2.2e5 -4.7e5 -1e6 -2.2e6 -4.7e6 -1e7 :: mcts.repeat_const 2048
if should_make_figure("ucb"):
    samples_n_vals = [8, 16, 32, 64, 128, 256]
    samples_n_mode = FigureMode("samples_n", samples_n_vals)
    ucb_const_vals = [-1e5, -2.2e5, -4.7e5, -1e6, -2.2e6, -4.7e6, -1e7]
    ucb_const_mode = FigureMode("ucb_const", ucb_const_vals)

    # repeat_mode = FigureMode("repeat_const", [-1, 2048])
    # bound_mode = FigureMode("bound_mode", ["marginal", "marginal_prior"])

    if True:
        filters = [
            ("method", "mcts"),
            ("bound_mode", "marginal_prior"),
            ("selection_mode", "ucb"),
            ("min.samples_n", 256),
            ("max.rng_seed", 1023),
        ]

        fig = FigureBuilder(results, None, "cost", translations=t10s)
        fig.plot(ucb_const_mode, filters)
        fig.ticks(ucb_const_vals)
        fig.legend()
        fig.show(file_suffix="_for_ucb")

# time cargo run --release rng_seed 0:2:1023 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode klucb :: mcts.klucb_max_cost 2.2 3.3 4.7 10 22 47 :: mcts.ucb_const 1.5
# time cargo run --release rng_seed 1:2:1023 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode klucb :: mcts.klucb_max_cost 2.2 3.3 4.7 10 22 47 :: mcts.ucb_const 1.5
# time cargo run --release rng_seed 0:2:1023 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 0.68 1 1.5 2.2 3.3 4.7
# time cargo run --release rng_seed 1:2:1023 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 0.68 1 1.5 2.2 3.3 4.7
if should_make_figure("klucb"):
    samples_n_vals = [8, 16, 32, 64, 128, 256]
    samples_n_mode = FigureMode("samples_n", samples_n_vals)
    klucb_max_cost_vals = [2.2, 3.3, 4.7, 10, 22, 47]
    klucb_max_cost_mode = FigureMode("klucb_max_cost", klucb_max_cost_vals)
    ucb_const_vals = [0.68, 1, 1.5, 2.2, 3.3, 4.7]
    ucb_const_mode = FigureMode("ucb_const", ucb_const_vals)

    bound_mode = FigureMode("bound_mode", ["marginal", "marginal_prior"])

    if True:
        filters = [
            ("method", "mcts"),
            ("bound_mode", "marginal_prior"),
            ("min.samples_n", 256),
            ("klucb_max_cost", 4.7),
        ]

        fig = FigureBuilder(results, None, "cost", translations=t10s)
        fig.plot(ucb_const_mode, filters)
        fig.ticks(ucb_const_vals)
        fig.legend()
        fig.show(file_suffix="_for_klucb")

    if True:
        filters = [
            ("method", "mcts"),
            ("bound_mode", "marginal_prior"),
            ("min.samples_n", 256),
            ("ucb_const", 1.5),
        ]

        fig = FigureBuilder(results, None, "cost", translations=t10s)
        fig.plot(klucb_max_cost_mode, filters)
        fig.ticks(klucb_max_cost_vals)
        fig.legend()
        fig.show(file_suffix="_for_klucb")

    if False:
        filters = [
            ("method", "mcts"),
            ("bound_mode", "marginal_prior"),
            ("klucb_max_cost", 4.7),
            ("ucb_const", 1.5),
        ]

        fig = FigureBuilder(results, None, "cost", translations=t10s)
        fig.plot(samples_n_mode, filters)
        fig.ticks(samples_n_vals)
        fig.legend()
        fig.show(file_suffix="_for_klucb")

    if False:
        filters = [
            ("method", "mcts"),
            ("bound_mode", "marginal_prior"),
            ("ucb_const", 0.47)
        ]

        fig = FigureBuilder(results, None, "cost", translations=t10s)
        fig.plot(samples_n_mode, filters, klucb_max_cost_mode)
        fig.ticks(samples_n_vals)
        fig.legend()
        fig.show(file_suffix="_for_klucb")

# time cargo run --release rng_seed 0:2:16383 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 1.5 :: mcts.repeat_const 0 64 128 256 512 1024 2048 8192 32768
# time cargo run --release rng_seed 1:2:16383 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 1.5 :: mcts.repeat_const 0 64 128 256 512 1024 2048 8192 32768
# time cargo run --release rng_seed 0:2:2047 :: method mcts :: mcts.samples_n 8 16 32 64 128 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 0.47 :: mcts.repeat_const 0 64 128 256 512 1024 2048
# time cargo run --release rng_seed 1:2:2047 :: method mcts :: mcts.samples_n 8 16 32 64 128 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 0.47 :: mcts.repeat_const 0 64 128 256 512 1024 2048
# time ../selfdriving rng_seed 0-4095 :: method mcts :: mcts.samples_n 8 16 32 64 128 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 0.47 :: mcts.repeat_const 4096 8192
# time ../selfdriving rng_seed 4096-8191 :: method mcts :: mcts.samples_n 128 256 :: mcts.bound_mode marginal_prior :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 0.47 :: mcts.repeat_const 0 64 128 256 512 1024 2048 4096 8192
if should_make_figure("repeat"):
    samples_n_value = 8
    samples_n_vals = [samples_n_value] #[8, 16, 32, 64, 128, 256]
    samples_n_mode = FigureMode("samples_n", samples_n_vals)
    repeat_const_vals = [0, 64, 128, 256, 512, 1024, 2048, 8192, 32768]
    repeat_const_mode = FigureMode("repeat_const", repeat_const_vals)

    if True:
        filters = [
            ("method", "mcts"),
            ("bound_mode", "marginal_prior"),
            ("klucb_max_cost", 4.7),
            ("ucb_const", 1.5),
        ]

        fig = FigureBuilder(results, None, "cost", translations=t10s)
        fig.plot(repeat_const_mode, filters, samples_n_mode) #, normalize="first")
        fig.line_from(filters + [("samples_n", samples_n_value), ("repeat_const", -1), ("prioritize_worst_particles_z", -1000)], "old_repeat")
        fig.ticks(repeat_const_vals)
        fig.legend()
        fig.show(file_suffix=f"_{samples_n_value}")

# time cargo run --release rng_seed 0:2:8191 :: method mcts :: mcts.samples_n 8 16 32 64 :: mcts.single_trial_discount_factor 0.7 0.75 0.8 0.85 0.9 0.95 1
# time cargo run --release rng_seed 1:2:8191 :: method mcts :: mcts.samples_n 8 16 32 64 :: mcts.single_trial_discount_factor 0.7 0.75 0.8 0.85 0.9 0.95 1
if should_make_figure("single_trial"):
    samples_n_mode = FigureMode("samples_n", [8, 16, 32, 64])
    single_trial_discount_factor_vals = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    single_trial_discount_factor_mode = FigureMode("single_trial_discount_factor", single_trial_discount_factor_vals)

    fig = FigureBuilder(results, None, "cost", translations=t10s)

    filters = [
        ("method", "mcts"),
        ("bound_mode", "marginal"),
        ("max.rng_seed", 4095),
        ("mcts.bootstrap_confidence_z", 0),
    ]

    if False:
        fig.plot(single_trial_discount_factor_mode, filters, samples_n_mode)
        fig.legend()
        fig.ticks(single_trial_discount_factor_vals)
        fig.show(file_suffix="_separate")
    else:
        fig.plot(single_trial_discount_factor_mode, filters)
        fig.ticks(single_trial_discount_factor_vals)
        fig.show(file_suffix="_combined")

# cargo run --release rng_seed 0:2:32767 :: method mcts :: mcts.samples_n 8 16 32 64 128 :: mcts.bound_mode marginal
# cargo run --release rng_seed 1:2:32767 :: method mcts :: mcts.samples_n 8 16 32 64 128 :: mcts.bound_mode marginal
# cargo run --release rng_seed 0:2:16383 :: method mcts :: mcts.samples_n 128 :: mcts.bound_mode marginal_prior :: mcts.zero_mean_prior_std_dev 220 470 680 10000 :: mcts.unknown_prior_std_dev_scalar 0 0.4 0.6 1
# cargo run --release rng_seed 1:2:16383 :: method mcts :: mcts.samples_n 128 :: mcts.bound_mode marginal_prior :: mcts.zero_mean_prior_std_dev 220 470 680 10000 :: mcts.unknown_prior_std_dev_scalar 0 0.4 0.6 1
# cargo run --release rng_seed 0:2:16383 :: method mcts :: mcts.samples_n 8 64 :: mcts.bound_mode marginal_prior :: mcts.zero_mean_prior_std_dev 100 220 470 680 1000 10000 :: mcts.unknown_prior_std_dev_scalar 0 0.2 0.4 0.6 0.8 1 1.2
# cargo run --release rng_seed 1:2:16383 :: method mcts :: mcts.samples_n 8 64 :: mcts.bound_mode marginal_prior :: mcts.zero_mean_prior_std_dev 100 220 470 680 1000 10000 :: mcts.unknown_prior_std_dev_scalar 0 0.2 0.4 0.6 0.8 1 1.2
# cargo run --release rng_seed 0:2:2047 :: method mcts :: mcts.samples_n 8 16 32 64 :: mcts.bound_mode marginal_prior :: mcts.zero_mean_prior_std_dev 22 47 68 100 150 220 330 :: mcts.unknown_prior_std_dev_scalar 0.3 0.4 0.5 0.6
# cargo run --release rng_seed 1:2:2047 :: method mcts :: mcts.samples_n 8 16 32 64 :: mcts.bound_mode marginal_prior :: mcts.zero_mean_prior_std_dev 22 47 68 100 150 220 330 :: mcts.unknown_prior_std_dev_scalar 0.3 0.4 0.5 0.6
# cargo run --release rng_seed 0:2:32767 :: method mcts :: mcts.samples_n 8 16 32 64 128 :: mcts.bound_mode marginal_prior :: mcts.zero_mean_prior_std_dev 150 :: mcts.unknown_prior_std_dev_scalar 0.5
# cargo run --release rng_seed 1:2:32767 :: method mcts :: mcts.samples_n 8 16 32 64 128 :: mcts.bound_mode marginal_prior :: mcts.zero_mean_prior_std_dev 150 :: mcts.unknown_prior_std_dev_scalar 0.5
if should_make_figure("zero_prior"):
    samples_n_val = 128
    # samples_n_vals = [8, 16, 32, 64]
    # samples_n_mode = FigureMode("samples_n", samples_n_vals)
    zero_mean_prior_std_dev_vals = [100, 220, 470, 680, 1000, 10000]
    zero_mean_prior_std_dev_mode = FigureMode("zero_mean_prior_std_dev", zero_mean_prior_std_dev_vals)
    unknown_prior_std_dev_scalar_mode = FigureMode("unknown_prior_std_dev_scalar", [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])

    fig = FigureBuilder(results, None, "cost", translations=t10s)

    filters = [
        ("max.rng_seed", 16383),
        ("method", "mcts"),
        ("bound_mode", "marginal_prior"),
        ("repeat_const", 32768),
        ("samples_n", samples_n_val),
    ]

    marginal_filters = [
        ("max.rng_seed", 16383),
        ("method", "mcts"),
        ("bound_mode", "marginal"),
        ("repeat_const", 32768),
        ("samples_n", samples_n_val),
    ]

    fig.plot(zero_mean_prior_std_dev_mode, filters, unknown_prior_std_dev_scalar_mode)
    fig.line_from(marginal_filters, label="marginal")
    fig.ticks(zero_mean_prior_std_dev_vals)
    fig.legend()
    fig.show(file_suffix=f"_{samples_n_val}")

# time cargo run --release rng_seed 0:2:8191 :: method mcts :: mcts.samples_n 8 16 32 64 128 256 :: mcts.bound_mode marginal :: mcts.bootstrap_confidence_z 0 0.01 0.032 0.1 0.32
# time cargo run --release rng_seed 1:2:8191 :: method mcts :: mcts.samples_n 8 16 32 64 128 256 :: mcts.bound_mode marginal :: mcts.bootstrap_confidence_z 0 0.01 0.032 0.1 0.32
if should_make_figure("bootstrap"):
    samples_n_vals = [8, 16, 32, 64]
    samples_n_mode = FigureMode("samples_n", samples_n_vals)
    bootstrap_confidence_z_vals = [0, 0.01, 0.032, 0.1, 0.32]
    bootstrap_confidence_z_mode = FigureMode("bootstrap_confidence_z", bootstrap_confidence_z_vals)

    fig = FigureBuilder(results, None, "cost", translations=t10s)

    filters = [
        ("method", "mcts"),
        ("bound_mode", "marginal"),
        ("single_trial_discount_factor", 1),
        ("max.rng_seed", 8191),
        ("ucb_const", -1),
    ]

    if True:
        fig.plot(bootstrap_confidence_z_mode, filters)
        fig.ticks(bootstrap_confidence_z_vals)
        fig.show()
        # fig.legend()
        # fig.show(file_suffix="_combined")
    else:
        fig.plot(samples_n_mode, filters + [("unknown_prior_std_dev_scalar", 0.45), ("zero_mean_prior_std_dev", 47)])
        fig.plot(samples_n_mode, marginal_filters, label="marginal")
        fig.ticks(samples_n_vals)
        fig.legend()
        fig.show(file_suffix="_separate")

# cargo run --release rng_seed 0-1023 :: method eudm :: use_cfb false true :: eudm.samples_n 2 4 8 16 32
# cargo run --release rng_seed 1024-2047 :: method eudm :: use_cfb false true :: eudm.samples_n 2 4 8 16 32
samples_n_mode = FigureMode("samples_n", [2, 4, 8, 16, 32])
if False:
    common_filters = [("method", "eudm"),
                      ("allow_different_root_policy", "true"),
                      ("max.rng_seed", 2047)]
    eudm_filters = common_filters + []
    for metric in plot_metrics:
        samples_n_kind.plot(
            results, metric, mode=cfb_mode, filters=eudm_filters)


# cargo run --release rng_seed 0-1023 :: method mcts :: use_cfb false :: mcts.bound_mode lower_bound marginal :: mcts.samples_n 4 8 16 32 64 :: mcts.prioritize_worst_particles_z -1000 1000
# cargo run --release rng_seed 1024-2047 :: method mcts :: use_cfb false :: mcts.bound_mode lower_bound marginal :: mcts.samples_n 4 8 16 32 64 :: mcts.prioritize_worst_particles_z -1000 1000
samples_n_mode = FigureMode("samples_n", [4, 8, 16, 32, 64, 128])
prioritize_worst_particles_z_mode = FigureMode("prioritize_worst_particles_z", ["-1000", "1000"])
if False:
    common_filters = [("use_cfb", "false"),
                      ("max.rng_seed", 2047)]
    for bound_mode in ["marginal"]:
        mcts_filters = [("method", "mcts"),
                        ("mcts.bound_mode", bound_mode)] + common_filters
        fixed_filters = [("method", "fixed")] + common_filters
        mpdm_filters = [("method", "mpdm")] + common_filters
        eudm_cfb_true_filters = [("method", "eudm"),
                                 ("allow_different_root_policy", "true"), ("use_cfb", "true"), ("max.rng_seed", 2047)]
        eudm_cfb_false_filters = [("method", "eudm"),
                                  ("allow_different_root_policy", "true"), ("use_cfb", "false"), ("max.rng_seed", 2047)]
        extra_lines = [
            # ("Fixed", fixed_filters),
            ("MPDM", mpdm_filters)]
        extra_modes = [("EUDM-CFB", eudm_cfb_true_filters), ("EUDM-NoCFB", eudm_cfb_false_filters)]
        for metric in plot_metrics:
            samples_n_kind.plot(
                results, metric, mode=prioritize_worst_particles_z_mode, filters=mcts_filters, extra_lines=extra_lines, extra_modes=extra_modes)

# print_all_parameter_values_used(
#     results, [("discount_factor", 0.8), ("replan_dt", 0.25), ("method", "mcts"), ("bound_mode", "marginal"), ("search_depth", 4), ("use_cfb", "false"), ("max.rng_seed", 8191), ("prioritize_worst_particles_z", -1000)])
# print_all_parameter_values_used(
#     results, [("discount_factor", 0.8), ("replan_dt", 0.05), ("method", "mpdm"), ("use_cfb", "false"), ("max.rng_seed", 8191)])
# quit()

# find_filters = [("method", "eudm"), ("use_cfb", "true"), ("samples_n", 16)]
# print(max(filter_extra(results, find_filters), key=lambda entry: entry["cost.safety"]))
# quit()

# cargo run --release rng_seed 0:2:16383 :: method mpdm :: use_cfb false :: mpdm.samples_n 2 4 8 16 32 64
# cargo run --release rng_seed 0:2:16383 :: method eudm :: use_cfb false true :: eudm.samples_n 1 2 4 8 16 32
# cargo run --release rng_seed 0:2:16383 :: method mcts :: use_cfb false :: mcts.bound_mode normal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.repeat_const 0
# cargo run --release rng_seed 0:2:16383 :: method mcts :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.repeat_const 0 32768
# cargo run --release rng_seed 1:2:16383 :: method mpdm :: use_cfb false :: mpdm.samples_n 2 4 8 16 32 64
# cargo run --release rng_seed 1:2:16383 :: method eudm :: use_cfb false true :: eudm.samples_n 1 2 4 8 16 32
# cargo run --release rng_seed 1:2:16383 :: method mcts :: use_cfb false :: mcts.bound_mode normal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.repeat_const 0
# cargo run --release rng_seed 1:2:16383 :: method mcts :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.repeat_const 0 32768
if should_make_figure("final"):
    for do_ablation in [False, True]:
        for metric in ["cost.efficiency", "cost.safety", "cost", "safety", "efficiency"]:
            seconds_fig = FigureBuilder(results, "95_ts", metric, translations=t10s)
            common_filters = [("max.rng_seed", 16383), ("discount_factor", 0.8), ("safety", 600)]

            if not do_ablation:
                mpdm_filters = [("method", "mpdm"), ("use_cfb", "false")] + common_filters
                seconds_fig.plot(FigureMode("samples_n", [2, 4, 8, 16, 32, 64]),
                                 mpdm_filters, label="MPDM")

                eudm_filters = [("method", "eudm"),
                                ("allow_different_root_policy", "true")] + common_filters
                seconds_fig.plot(FigureMode("samples_n", [1, 2, 4, 8, 16, 32]), eudm_filters, cfb_mode, label="EUDM, ")

            if do_ablation:
                mcts_filters = [("method", "mcts"),
                                ("use_cfb", "false"),
                                ("repeat_const", 0),
                                ("selection_mode", "klucb"),
                                ("klucb_max_cost", 4.7),
                                ("ucb_const", 1.5),
                                ("bound_mode", "normal")] + common_filters
                seconds_fig.plot(FigureMode(
                    "samples_n", [8, 16, 32, 64, 128, 256]), mcts_filters, label="MCPTDM (-repeat, -MAC)")

                mcts_filters = [("method", "mcts"),
                                ("use_cfb", "false"),
                                ("repeat_const", 0),
                                ("selection_mode", "klucb"),
                                ("klucb_max_cost", 4.7),
                                ("ucb_const", 1.5),
                                ("bound_mode", "marginal")] + common_filters
                seconds_fig.plot(FigureMode(
                    "samples_n", [8, 16, 32, 64, 128, 256]), mcts_filters, label="MCPTDM (-repeat)")

            mcts_filters = [("method", "mcts"),
                            ("use_cfb", "false"),
                            ("repeat_const", 32768),
                            ("selection_mode", "klucb"),
                            ("klucb_max_cost", 4.7),
                            ("ucb_const", 1.5),
                            ("bound_mode", "marginal")] + common_filters
            seconds_fig.plot(FigureMode(
                "samples_n", [8, 16, 32, 64, 128, 256]), mcts_filters, label="MCPTDM (proposed)")

            seconds_fig.legend()
            metric_name = seconds_fig.translate(metric).lower()
            title = f"MCPTDM ablation: {metric_name} by 95% computation time (s)" if do_ablation else f"Final comparison: {metric_name} by 95% computation time (s)"
            seconds_fig.show(title=title, file_suffix="_ablation" if do_ablation else "_final")

# cargo run --release rng_seed 0-2047 :: replan_dt 1 0.5 0.25 0.2 0.1 0.05 :: method mcts :: mcts.samples_n 64 :: mcts.bound_mode marginal :: mcts.prioritize_worst_particles_z -1000 :: use_cfb false
# cargo run --release rng_seed 0-2047 :: replan_dt 1 0.5 0.25 0.2 0.1 0.05 :: method mpdm :: mpdm.samples_n 16 :: use_cfb false
# cargo run --release rng_seed 0-2047 :: replan_dt 1 0.5 0.25 0.2 0.1 0.05 :: method eudm :: eudm.samples_n 8 :: use_cfb false
# cargo run --release rng_seed 2048-4095 :: replan_dt 1 0.5 0.25 0.2 0.1 0.05 :: method mcts :: mcts.samples_n 64 :: mcts.bound_mode marginal :: mcts.prioritize_worst_particles_z -1000 :: use_cfb false
# cargo run --release rng_seed 2048-4095 :: replan_dt 1 0.5 0.25 0.2 0.1 0.05 :: method mpdm :: mpdm.samples_n 16 :: use_cfb false
# cargo run --release rng_seed 2048-4095 :: replan_dt 1 0.5 0.25 0.2 0.1 0.05 :: method eudm :: eudm.samples_n 8 :: use_cfb false
# cargo run --release rng_seed 4096-8191 :: replan_dt 1 :: method mcts :: mcts.samples_n 64 :: mcts.bound_mode marginal :: mcts.prioritize_worst_particles_z -1000 :: use_cfb false
# cargo run --release rng_seed 4096-8191 :: replan_dt 1 :: method mpdm :: mpdm.samples_n 16 :: use_cfb false
# cargo run --release rng_seed 4096-8191 :: replan_dt 1 :: method eudm :: eudm.samples_n 8 :: use_cfb false
if False:
    common_filters = [("discount_factor", 0.8), ("use_cfb", "false")]
    mcts_filters = common_filters + [("method", "mcts"), ("samples_n", 64)]
    mpdm_filters = common_filters + [("method", "mpdm"), ("samples_n", 16)]
    eudm_filters = common_filters + [("method", "eudm"), ("samples_n", 8)]

    for metric in plot_metrics:
        replan_fig = FigureBuilder(results, "replan_dt", metric, translations=t10s)
        replan_fig.plot(FigureMode("replan_dt", [1, 0.5, 0.25,
                                                 0.2, 0.1, 0.05]), mcts_filters, label="PTDM")
        replan_fig.plot(FigureMode("replan_dt", [1, 0.5, 0.25,
                                                 0.2, 0.1, 0.05]), mpdm_filters, label="MPDM")
        replan_fig.plot(FigureMode("replan_dt", [1, 0.5, 0.25,
                                                 0.2, 0.1, 0.05]), eudm_filters, label="EUDM")
        replan_fig.legend()
        replan_fig.show()


# print_all_parameter_values_used(
#     results, [("discount_factor", 0.8), ("replan_dt", 0.25), ("method", "mcts"), ("total_forward_t", 8), ("samples_n", 64), ("mcts.bound_mode", "marginal"), ("use_cfb", "false"), ("max.rng_seed", 4095), ("mcts.prioritize_worst_particles_z", "-1000")])
# quit()

# cargo run --release rng_seed 0-2047 :: method mcts :: use_cfb false :: replan_dt 0.25 :: mcts.total_forward_t 8 :: mcts.search_depth 3 4 5 :: mcts.bound_mode marginal :: mcts.samples_n 64 :: mcts.prioritize_worst_particles_z -1000
# cargo run --release rng_seed 2048-2303 :: method mcts :: use_cfb false :: replan_dt 0.25 :: mcts.total_forward_t 8 :: mcts.search_depth 3 4 5 :: mcts.bound_mode marginal :: mcts.samples_n 64 :: mcts.prioritize_worst_particles_z -1000
if False:
    for metric in plot_metrics:
        seconds_fig = FigureBuilder(results, "95_ts", metric, translations=t10s)

        common_filters = [("max.rng_seed", 4095), ("discount_factor", 0.8), ("replan_dt", 0.25)]

        mcts_filters = [("method", "mcts"),
                        ("total_forward_t", 8),
                        ("samples_n", 64),
                        ("mcts.prioritize_worst_particles_z", "-1000"),
                        ("mcts.bound_mode", "marginal")] + common_filters
        seconds_fig.plot(FigureMode(
            "search_depth", [2, 3, 4, 5, 6]), mcts_filters, label="PTDM, ")

        # eudm_filters = [("method", "eudm"),
        #                 ("allow_different_root_policy", "true")] + common_filters
        # seconds_fig.plot(FigureMode("samples_n", [2, 4, 8, 16, 32]),
        #                  eudm_filters, label="EUDM, ")

        # mpdm_filters = [("method", "mpdm")] + common_filters
        # seconds_fig.plot(FigureMode("samples_n", [2, 4, 8, 16, 32, 64]),
        #                  mpdm_filters, label="MPDM, ")

        seconds_fig.legend()
        seconds_fig.show(file_suffix="_search_depth")

# cargo run --release rng_seed 2048-5119 :: method eudm :: discount_factor 0.6 0.7 0.8 0.9 1 :: use_cfb false :: eudm.samples_n 16
# cargo run --release rng_seed 2048-5119 :: method mpdm :: discount_factor 0.6 0.7 0.8 0.9 1 :: use_cfb false :: mpdm.samples_n 16
# cargo run --release rng_seed 2048-5119 :: method mcts :: discount_factor 0.6 0.7 0.8 0.9 1 :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 64 :: mcts.prioritize_worst_particles_z -1000
# cargo run --release rng_seed 5120-8191 :: method eudm :: discount_factor 0.6 0.7 0.8 0.9 1 :: use_cfb false :: eudm.samples_n 16
# cargo run --release rng_seed 5120-8191 :: method mpdm :: discount_factor 0.6 0.7 0.8 0.9 1 :: use_cfb false :: mpdm.samples_n 16
# cargo run --release rng_seed 5120-8191 :: method mcts :: discount_factor 0.6 0.7 0.8 0.9 1 :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 64 :: mcts.prioritize_worst_particles_z -1000
if False:
    common_filters = [("use_cfb", "false")]
    mcts_filters = common_filters + [("samples_n", 64), ("method", "mcts"),
                                     ("mcts.bound_mode", "marginal"), ("mcts.prioritize_worst_particles_z", "-1000")]
    mpdm_filters = common_filters + [("samples_n", 16), ("method", "mpdm")]
    eudm_filters = common_filters + [("samples_n", 16), ("method", "eudm")]

    fig = FigureBuilder(results, "discount_factor", "cost", translations=t10s)
    fig.plot(FigureMode("discount_factor", [
        0.6, 0.7, 0.8, 0.9, 1]), mcts_filters, label="PTDM")
    fig.plot(FigureMode("discount_factor", [
        0.6, 0.7, 0.8, 0.9, 1]), mpdm_filters, label="MPDM")
    fig.plot(FigureMode("discount_factor", [
        0.6, 0.7, 0.8, 0.9, 1]), eudm_filters, label="EUDM")
    fig.legend()
    fig.show()

# mcts.search_depth 4-7 :: mcts.samples_n 8 16 32 64 128 256 512
# cargo run --release rng_seed 0-15 :: method tree :: tree.samples_n 1 2 4 8 :: use_cfb false true

# cargo run --release rng_seed 0-15 :: method fixed mpdm mcts eudm :: use_cfb false true :: smoothness 0 0.1 0.3 1 3 10 30 100
# cargo run --release rng_seed 16-31 :: method fixed mpdm mcts eudm :: use_cfb false true :: smoothness 0 0.1 0.3 1 3 10 30 100
if False:
    smoothness_kind = FigureKind("smoothness", [0, 0.1, 0.3, 1, 3, 10, 30, 100])
    for metric in plot_metrics:
        smoothness_kind.plot(results, metric, mode=method_mode)

# cargo run --release rng_seed 0-255 :: method mcts :: use_cfb false :: safety 10 15 22 33 47 68 100 150 220 330 470 680 1000 :: mcts.selection_mode klucb :: mcts.klucb_max_cost 100 150 220 330 470 680 1000 :: mcts.bound_mode marginal :: mcts.samples_n 64
# cargo run --release rng_seed 256-511 :: method mcts :: use_cfb false :: safety 10 15 22 33 47 68 100 150 220 330 470 680 1000 :: mcts.selection_mode klucb :: mcts.klucb_max_cost 100 150 220 330 470 680 1000 :: mcts.bound_mode marginal :: mcts.samples_n 64
# cargo run --release rng_seed 2048-3071 :: method mcts :: use_cfb false :: safety 150 :: mcts.selection_mode klucb :: mcts.klucb_max_cost 100 150 220 330 470 680 1000 :: mcts.bound_mode marginal :: mcts.samples_n 64
# cargo run --release rng_seed 3072-4095 :: method mcts :: use_cfb false :: safety 150 :: mcts.selection_mode klucb :: mcts.klucb_max_cost 100 150 220 330 470 680 1000 :: mcts.bound_mode marginal :: mcts.samples_n 64
if False:
    safety_kind = FigureKind("safety", [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 1000])
    klucb_max_cost_kind = FigureKind("klucb_max_cost", [100, 150, 220, 330, 470, 680, 1000])
    safety_mode = FigureMode("safety", [150, 220, 330, 470])
    for metric in plot_metrics:
        # safety_kind.plot(results, metric, filters=[
        #     "_method_mcts_", "_selection_mode_klucb_", "_bound_mode_marginal_", "_use_cfb_false_"])
        klucb_max_cost_kind.plot(results, metric, filters=[
            "_method_mcts_", "_safety_150_", "_selection_mode_klucb_", "_bound_mode_marginal_", "_use_cfb_false_"])
        # klucb_max_cost_kind.plot(results, metric, filters=[
        #     "_method_mcts_", "_selection_mode_klucb_", "_bound_mode_marginal_", "_use_cfb_false_"])
        # klucb_max_cost_kind.plot(results, metric, mode=safety_mode, filters=[
        #     "_method_mcts_", "_selection_mode_klucb_", "_bound_mode_marginal_", "_use_cfb_false_"])

# cargo run --release rng_seed 0-1023 :: method mcts :: use_cfb false :: mcts.bound_mode normal bubble_best lower_bound marginal :: mcts.prioritize_worst_particles_z -1000 1000
# cargo run --release rng_seed 1024-2047 :: method mcts :: use_cfb false :: mcts.bound_mode normal bubble_best lower_bound marginal :: mcts.prioritize_worst_particles_z -1000 1000
if False:
    prioritize_worst_particles_z_kind = FigureKind(
        "prioritize_worst_particles_z", [-1000, 1000], translations=t10s)
    filters = [("method", "mcts"), ("selection_mode", "klucb"),
               ("search_depth", 4), ("samples_n", 64), ("use_cfb", "false")]
    for metric in plot_metrics:
        prioritize_worst_particles_z_kind.plot(results, metric, mode=bound_mode, filters=filters)

    for bound_mode in ["lower_bound", "marginal"]:
        for z in [-1000, 1000]:
            evaluate_conditions(results, plot_metrics, filters + [
                                ("bound_mode", bound_mode), ("prioritize_worst_particles_z", z)])

    # method=mcts,selection_mode=klucb,search_depth=4,samples_n=64,use_cfb=false,bound_mode=lower_bound,prioritize_worst_particles_z=-1000:
    #   efficiency has mean:  6.206 and mean std dev: 0.04445
    #   cost has mean:  427.5 and mean std dev:  9.176
    #   safety has mean: 0.003439 and mean std dev: 0.0009229

    # method=mcts,selection_mode=klucb,search_depth=4,samples_n=64,use_cfb=false,bound_mode=lower_bound,prioritize_worst_particles_z=1000:
    #   efficiency has mean:  6.178 and mean std dev: 0.04449
    #   cost has mean:  451.7 and mean std dev:  12.16
    #   safety has mean: 0.006107 and mean std dev: 0.001267

    # method=mcts,selection_mode=klucb,search_depth=4,samples_n=64,use_cfb=false,bound_mode=marginal,prioritize_worst_particles_z=-1000:
    #   efficiency has mean:  5.834 and mean std dev: 0.04181
    #   cost has mean:  448.7 and mean std dev:  10.45
    #   safety has mean: 0.003931 and mean std dev: 0.00112

    # method=mcts,selection_mode=klucb,search_depth=4,samples_n=64,use_cfb=false,bound_mode=marginal,prioritize_worst_particles_z=1000:
    #   efficiency has mean:  5.847 and mean std dev: 0.04242
    #   cost has mean:  454.1 and mean std dev:  11.49
    #   safety has mean: 0.00471 and mean std dev: 0.001229

if should_make_figure("latex"):
    table_metrics = ["cost", "safety", "efficiency"]
    latex_table = ""

    res = evaluate_conditions(results, table_metrics, filters=[("method", "fixed")])
    latex_table += f"    Fixed & {res[0]:.0f} & {res[1]:.4f} & {res[2]:.1f} \\\\\n"

    common_filters = [("discount_factor", 0.8), ("replan_dt", 0.25), ("max.rng_seed", 8191)]

    mpdm_filters = [("method", "mpdm"), ("use_cfb", "false"),
                    ("forward_t", 8), ("samples_n", 16)] + common_filters
    res = evaluate_conditions(results, table_metrics, filters=mpdm_filters)
    latex_table += f"    MPDM & {res[0]:.0f} & {res[1]:.4f} & {res[2]:.1f} \\\\\n"

    eudm_filters = [("method", "eudm"), ("use_cfb", "true"), ("search_depth", 4),
                    ("samples_n", 8)] + common_filters
    res = evaluate_conditions(results, table_metrics, filters=eudm_filters)
    latex_table += f"    EUDM w/ CFB & {res[0]:.0f} & {res[1]:.4f} & {res[2]:.1f} \\\\\n"

    eudm_filters = [("method", "eudm"), ("use_cfb", "false"), ("search_depth", 4),
                    ("samples_n", 8)] + common_filters
    res = evaluate_conditions(results, table_metrics, filters=eudm_filters)
    latex_table += f"    EUDM w/o CFB & {res[0]:.0f} & {res[1]:.4f} & {res[2]:.1f} \\\\\n"

    mcts_filters = [("method", "mcts"), ("use_cfb", "false"), ("bound_mode", "marginal"), ("selection_mode", "klucb"),
                    ("search_depth", 4), ("samples_n", 64), ("prioritize_worst_particles_z", -1000)] + common_filters
    res = evaluate_conditions(results, table_metrics, filters=mcts_filters)
    latex_table += f"    PTDM (proposed) & {res[0]:.0f} & {res[1]:.4f} & {res[2]:.1f} \\\\\n"
    print(latex_table)


if len(sys.argv) == 1 or "help" in sys.argv:
    print("Valid figure options:")
    for option in figure_cmd_line_options:
        print(option)
