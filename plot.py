#!/usr/bin/python3
from common_plot import parse_parameters, FigureBuilder, FigureKind, FigureMode, print_all_parameter_values_used, evaluate_conditions, filter_extra
import time

t10s = dict()
t10s["discount_factor"] = "Discount Factor"
t10s["safety"] = "Proportion unsafe"
t10s["cost.safety"] = "Safety cost"
t10s["cost"] = "Cost"
t10s["efficiency"] = "Efficiency"
t10s["ud"] = "Uncomfortable decelerations"
t10s["cc"] = "Curvature change"
t10s["tree"] = "Tree"
t10s["mpdm"] = "MPDM"
t10s["eudm"] = "EUDM"
t10s["mcts"] = "PTDM"
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

start_time = time.time()
results = []
with open("results.cache", "r") as f:
    for line in f:
        parts = line.split()
        if len(parts) > 9:
            entry = dict()
            entry["params"] = parse_parameters(parts[0])
            entry["efficiency"] = float(parts[5])
            entry["safety"] = float(parts[6])
            entry["ud"] = float(parts[7])
            entry["cc"] = float(parts[8])
            entry["mean_ts"] = float(parts[9])
            entry["95_ts"] = float(parts[10])
            entry["997_ts"] = float(parts[11])
            entry["max_ts"] = float(parts[12])
            entry["stddev_ts"] = float(parts[13])

            entry["cost.efficiency"] = float(parts[1])
            entry["cost.safety"] = float(parts[2])
            entry["cost.accel"] = float(parts[3])
            entry["cost.steer"] = float(parts[4])
            entry["cost"] = entry["cost.efficiency"] + entry["cost.safety"] + \
                entry["cost.accel"] + entry["cost.steer"]

            results.append(entry)
        else:
            continue
print(f"took {time.time() - start_time:.2f} seconds to load data")

method_kind = FigureKind("method", ["fixed", "tree", "mpdm", "eudm", "mcts"], translations=t10s)
discount_kind = FigureKind("discount_factor", [0.6, 0.7, 0.8, 0.9, 1], translations=t10s)
cfb_kind = FigureKind("use_cfb", ["false", "true"], translations=t10s)

# extra_accdec_kind = FigureKind("extra_ego_accdec_policies", [
#                                "-1", "1", "-2", "2", "-1,1", "-2,2", "1,2", "-1,-2", "-1,-2,-3,1,2,3"])

extra_accdec_kind = FigureKind("extra_ego_accdec_policies", [
                               "", "-1,-2,1,2", "-1,-2,-3,1,2,3"], translations=t10s)

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

# cargo run --release rng_seed 0-1023 :: method eudm :: use_cfb false true :: eudm.samples_n 2 4 8 16 32 :: thread_limit 24
# cargo run --release rng_seed 1024-2047 :: method eudm :: use_cfb false true :: eudm.samples_n 2 4 8 16 32 :: thread_limit 24
samples_n_kind = FigureKind("samples_n", [2, 4, 8, 16, 32], translations=t10s)
if False:
    common_filters = [("method", "eudm"),
                      ("allow_different_root_policy", "true"),
                      ("max.rng_seed", 2047)]
    eudm_filters = common_filters + []
    for metric in plot_metrics:
        samples_n_kind.plot(
            results, metric, mode=cfb_mode, filters=eudm_filters)


# cargo run --release rng_seed 0-1023 :: method mcts :: use_cfb false :: mcts.bound_mode lower_bound marginal :: mcts.samples_n 4 8 16 32 64 :: mcts.prioritize_worst_particles_z -1000 1000 :: thread_limit 24
# cargo run --release rng_seed 1024-2047 :: method mcts :: use_cfb false :: mcts.bound_mode lower_bound marginal :: mcts.samples_n 4 8 16 32 64 :: mcts.prioritize_worst_particles_z -1000 1000 :: thread_limit 24
# samples_n_kind = FigureKind("samples_n", [4, 8, 16, 32, 64, 128, 256, 512], translations=t10s)
samples_n_kind = FigureKind("samples_n", [4, 8, 16, 32, 64, 128], translations=t10s)
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

# cargo run --release rng_seed 0000-2047 :: method mpdm :: replan_dt 0.25 0.1 0.05 :: use_cfb false :: mpdm.samples_n 2 4 8 16 32 64 :: thread_limit 24
# cargo run --release rng_seed 0000-2047 :: method eudm :: replan_dt 0.25 0.1 0.05 :: use_cfb false true :: eudm.samples_n 2 4 8 16 32 :: thread_limit 24
# cargo run --release rng_seed 0000-2047 :: method mcts :: replan_dt 0.25 0.1 0.05 :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.prioritize_worst_particles_z -1000 :: thread_limit 24
# cargo run --release rng_seed 2048-4095 :: method mpdm :: replan_dt 0.25 0.1 0.05 :: use_cfb false :: mpdm.samples_n 2 4 8 16 32 64 :: thread_limit 24
# cargo run --release rng_seed 2048-4095 :: method eudm :: replan_dt 0.25 0.1 0.05 :: use_cfb false true :: eudm.samples_n 2 4 8 16 32 :: thread_limit 24
# cargo run --release rng_seed 2048-4095 :: method mcts :: replan_dt 0.25 0.1 0.05 :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.prioritize_worst_particles_z -1000 :: thread_limit 24
#
# cargo run --release rng_seed 4096-6143 :: method mpdm :: replan_dt 0.25 0.1 0.05 :: use_cfb false :: mpdm.samples_n 2 4 8 16 32 64 :: thread_limit 24
# cargo run --release rng_seed 4096-6143 :: method eudm :: replan_dt 0.25 0.1 0.05 :: use_cfb false true :: eudm.samples_n 2 4 8 16 32 :: thread_limit 24
# cargo run --release rng_seed 4096-6143 :: method mcts :: replan_dt 0.25 0.1 0.05 :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.prioritize_worst_particles_z -1000 :: thread_limit 24
# cargo run --release rng_seed 6144-8191 :: method mpdm :: replan_dt 0.25 0.1 0.05 :: use_cfb false :: mpdm.samples_n 2 4 8 16 32 64 :: thread_limit 24
# cargo run --release rng_seed 6144-8191 :: method eudm :: replan_dt 0.25 0.1 0.05 :: use_cfb false true :: eudm.samples_n 2 4 8 16 32 :: thread_limit 24
# cargo run --release rng_seed 6144-8191 :: method mcts :: replan_dt 0.25 0.1 0.05 :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.prioritize_worst_particles_z -1000 :: thread_limit 24
#
# cargo run --release rng_seed 8192-12287 :: method mpdm :: replan_dt 0.25 0.1 0.05 :: use_cfb false :: mpdm.samples_n 2 4 8 16 32 64 :: thread_limit 24
# cargo run --release rng_seed 8192-12287 :: method eudm :: replan_dt 0.25 0.1 0.05 :: use_cfb false true :: eudm.samples_n 2 4 8 16 32 :: thread_limit 24
# cargo run --release rng_seed 8192-12287 :: method mcts :: replan_dt 0.25 0.1 0.05 :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.prioritize_worst_particles_z -1000 :: thread_limit 24
# cargo run --release rng_seed 12288-16383 :: method mpdm :: replan_dt 0.25 0.1 0.05 :: use_cfb false :: mpdm.samples_n 2 4 8 16 32 64 :: thread_limit 24
# cargo run --release rng_seed 12288-16383 :: method eudm :: replan_dt 0.25 0.1 0.05 :: use_cfb false true :: eudm.samples_n 2 4 8 16 32 :: thread_limit 24
# cargo run --release rng_seed 12288-16383 :: method mcts :: replan_dt 0.25 0.1 0.05 :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.prioritize_worst_particles_z -1000 :: thread_limit 24
#
# cargo run --release rng_seed 0-8191 :: method fixed :: use_cfb false :: thread_limit 7
if True:
    for metric in plot_metrics + ["safety"]:
        seconds_fig = FigureBuilder(results, "95_ts", metric, translations=t10s)

        common_filters = [("max.rng_seed", 8191), ("discount_factor", 0.8), ("replan_dt", 0.25)]

        mcts_filters = [("method", "mcts"),
                        ("search_depth", 4),
                        ("use_cfb", "false"),
                        ("prioritize_worst_particles_z", -1000),
                        ("selection_mode", "klucb"),
                        ("bound_mode", "marginal")] + common_filters
        seconds_fig.plot(FigureMode(
            "samples_n", [8, 16, 32, 64, 128, 256]), mcts_filters, label="PTDM")

        eudm_filters = [("method", "eudm"),
                        ("allow_different_root_policy", "true")] + common_filters
        seconds_fig.plot(FigureMode("samples_n", [2, 4, 8, 16, 32]),
                         eudm_filters, cfb_mode, label="EUDM, ")

        mpdm_filters = [("method", "mpdm"), ("use_cfb", "false")] + common_filters
        seconds_fig.plot(FigureMode("samples_n", [2, 4, 8, 16, 32, 64]),
                         mpdm_filters, label="MPDM")

        seconds_fig.legend()
        seconds_fig.show()

# cargo run --release rng_seed 0-2047 :: replan_dt 1 0.5 0.25 0.2 0.1 0.05 :: method mcts :: mcts.samples_n 64 :: mcts.bound_mode marginal :: mcts.prioritize_worst_particles_z -1000 :: use_cfb false :: thread_limit 24
# cargo run --release rng_seed 0-2047 :: replan_dt 1 0.5 0.25 0.2 0.1 0.05 :: method mpdm :: mpdm.samples_n 16 :: use_cfb false :: thread_limit 24
# cargo run --release rng_seed 0-2047 :: replan_dt 1 0.5 0.25 0.2 0.1 0.05 :: method eudm :: eudm.samples_n 8 :: use_cfb false :: thread_limit 24
# cargo run --release rng_seed 2048-4095 :: replan_dt 1 0.5 0.25 0.2 0.1 0.05 :: method mcts :: mcts.samples_n 64 :: mcts.bound_mode marginal :: mcts.prioritize_worst_particles_z -1000 :: use_cfb false :: thread_limit 24
# cargo run --release rng_seed 2048-4095 :: replan_dt 1 0.5 0.25 0.2 0.1 0.05 :: method mpdm :: mpdm.samples_n 16 :: use_cfb false :: thread_limit 24
# cargo run --release rng_seed 2048-4095 :: replan_dt 1 0.5 0.25 0.2 0.1 0.05 :: method eudm :: eudm.samples_n 8 :: use_cfb false :: thread_limit 24
# cargo run --release rng_seed 4096-8191 :: replan_dt 1 :: method mcts :: mcts.samples_n 64 :: mcts.bound_mode marginal :: mcts.prioritize_worst_particles_z -1000 :: use_cfb false :: thread_limit 24
# cargo run --release rng_seed 4096-8191 :: replan_dt 1 :: method mpdm :: mpdm.samples_n 16 :: use_cfb false :: thread_limit 24
# cargo run --release rng_seed 4096-8191 :: replan_dt 1 :: method eudm :: eudm.samples_n 8 :: use_cfb false :: thread_limit 24
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

# cargo run --release rng_seed 0-2047 :: method mcts :: use_cfb false :: replan_dt 0.25 :: mcts.total_forward_t 8 :: mcts.search_depth 3 4 5 :: mcts.bound_mode marginal :: mcts.samples_n 64 :: mcts.prioritize_worst_particles_z -1000 :: thread_limit 24
# cargo run --release rng_seed 2048-2303 :: method mcts :: use_cfb false :: replan_dt 0.25 :: mcts.total_forward_t 8 :: mcts.search_depth 3 4 5 :: mcts.bound_mode marginal :: mcts.samples_n 64 :: mcts.prioritize_worst_particles_z -1000 :: thread_limit 24
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

# cargo run --release rng_seed 2048-5119 :: method eudm :: discount_factor 0.6 0.7 0.8 0.9 1 :: use_cfb false :: eudm.samples_n 16 :: thread_limit 24
# cargo run --release rng_seed 2048-5119 :: method mpdm :: discount_factor 0.6 0.7 0.8 0.9 1 :: use_cfb false :: mpdm.samples_n 16 :: thread_limit 24
# cargo run --release rng_seed 2048-5119 :: method mcts :: discount_factor 0.6 0.7 0.8 0.9 1 :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 64 :: mcts.prioritize_worst_particles_z -1000 :: thread_limit 24
# cargo run --release rng_seed 5120-8191 :: method eudm :: discount_factor 0.6 0.7 0.8 0.9 1 :: use_cfb false :: eudm.samples_n 16 :: thread_limit 24
# cargo run --release rng_seed 5120-8191 :: method mpdm :: discount_factor 0.6 0.7 0.8 0.9 1 :: use_cfb false :: mpdm.samples_n 16 :: thread_limit 24
# cargo run --release rng_seed 5120-8191 :: method mcts :: discount_factor 0.6 0.7 0.8 0.9 1 :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 64 :: mcts.prioritize_worst_particles_z -1000 :: thread_limit 24
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

# cargo run --release rng_seed 0-31 :: use_cfb false true :: method mcts :: mcts.samples_n 32 :: mcts.bound_mode normal lower_bound marginal :: mcts.selection_mode ucb klucb :: mcts.klucb_max_cost 10 30 100 300 1000 3000 :: thread_limit 24
# cargo run --release rng_seed 32-63 :: use_cfb false true :: method mcts :: mcts.samples_n 32 :: mcts.bound_mode normal lower_bound marginal :: mcts.selection_mode ucb klucb :: mcts.klucb_max_cost 10 30 100 300 1000 3000 :: thread_limit 24
klucb_max_cost_kind = FigureKind(
    "klucb_max_cost", [10, 30, 100, 300, 1000, 3000], translations=t10s)
selection_mode = FigureMode("selection_mode", ["ucb", "klucb"])
bound_mode = FigureMode("bound_mode", ["normal", "bubble_best", "lower_bound", "marginal"])
bound_mode_kind = FigureKind(
    "bound_mode", ["normal", "bubble_best", "lower_bound", "marginal"], translations=t10s)
if False:
    # for metric in plot_metrics:
    #     for use_cfb in ["false", "true"]:
    #         filters = ["_method_mcts_", "_samples_n_32_", f"_use_cfb_{use_cfb}_"]
    #         klucb_max_cost_kind.plot(results, metric, mode=bound_mode,
    #                                  filters=filters + ["_selection_mode_klucb_"])
    #         bound_mode_kind.plot(results, metric, mode=selection_mode, filters=filters)
    for use_cfb in ["false", "true"]:
        evaluate_conditions(results, evaluate_metrics, [
            ("method", "mcts"),
            ("samples_n", 32),
            ("use_cfb", use_cfb),
            ("bound_mode", "marginal"),
            ("selection_mode", "ucb")])

        evaluate_conditions(results, evaluate_metrics, [
            ("method", "mcts"),
            ("samples_n", 32),
            ("use_cfb", use_cfb),
            ("bound_mode", "marginal"),
            ("selection_mode", "klucb"),
            ("klucb_max_cost", 30)])

# mcts.search_depth 4-7 :: mcts.samples_n 8 16 32 64 128 256 512
# cargo run --release rng_seed 0-15 :: method tree :: tree.samples_n 1 2 4 8 :: use_cfb false true :: thread_limit 24

# cargo run --release rng_seed 0-15 :: method fixed mpdm mcts eudm :: use_cfb false true :: smoothness 0 0.1 0.3 1 3 10 30 100 :: thread_limit 24
# cargo run --release rng_seed 16-31 :: method fixed mpdm mcts eudm :: use_cfb false true :: smoothness 0 0.1 0.3 1 3 10 30 100 :: thread_limit 24
if False:
    smoothness_kind = FigureKind("smoothness", [0, 0.1, 0.3, 1, 3, 10, 30, 100])
    for metric in plot_metrics:
        smoothness_kind.plot(results, metric, mode=method_mode)

# cargo run --release rng_seed 0-255 :: method mcts :: use_cfb false :: safety 10 15 22 33 47 68 100 150 220 330 470 680 1000 :: mcts.selection_mode klucb :: mcts.klucb_max_cost 100 150 220 330 470 680 1000 :: mcts.bound_mode marginal :: mcts.samples_n 64 :: thread_limit 24
# cargo run --release rng_seed 256-511 :: method mcts :: use_cfb false :: safety 10 15 22 33 47 68 100 150 220 330 470 680 1000 :: mcts.selection_mode klucb :: mcts.klucb_max_cost 100 150 220 330 470 680 1000 :: mcts.bound_mode marginal :: mcts.samples_n 64 :: thread_limit 24
# cargo run --release rng_seed 2048-3071 :: method mcts :: use_cfb false :: safety 150 :: mcts.selection_mode klucb :: mcts.klucb_max_cost 100 150 220 330 470 680 1000 :: mcts.bound_mode marginal :: mcts.samples_n 64 :: thread_limit 24
# cargo run --release rng_seed 3072-4095 :: method mcts :: use_cfb false :: safety 150 :: mcts.selection_mode klucb :: mcts.klucb_max_cost 100 150 220 330 470 680 1000 :: mcts.bound_mode marginal :: mcts.samples_n 64 :: thread_limit 24
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

# cargo run --release rng_seed 0-1023 :: method mcts :: use_cfb false :: mcts.bound_mode normal bubble_best lower_bound marginal :: mcts.prioritize_worst_particles_z -1000 1000 :: thread_limit 24
# cargo run --release rng_seed 1024-2047 :: method mcts :: use_cfb false :: mcts.bound_mode normal bubble_best lower_bound marginal :: mcts.prioritize_worst_particles_z -1000 1000 :: thread_limit 24
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

table_metrics = ["cost", "safety", "efficiency"]
latex_table = ""
if False:
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
