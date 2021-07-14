#!/usr/bin/python3
from common_plot import FigureKind, FigureMode, print_all_parameter_values_used

t10s = dict()
t10s["discount_factor"] = "Discount Factor"
t10s["safety"] = "Safety"
t10s["cost"] = "Cost"
t10s["efficiency"] = "Efficiency"
t10s["ud"] = "Uncomfortable decelerations"
t10s["cc"] = "Curvature change"
t10s["tree"] = "Tree"
t10s["mpdm"] = "MPDM"
t10s["eudm"] = "EUDM"
t10s["mcts"] = "MCTS"
t10s["method"] = "Method"
t10s["false"] = "Normal"
t10s["true"] = "CFB"
t10s["seconds"] = "Computation time (s)"
t10s["search_depth"] = "Search depth"
t10s["samples_n"] = "# Samples"
t10s[None] = "Average"

# sed -i 's/_samples_n/,samples_n/g' results.cache
# sed -i 's/_search_depth/,search_depth/g' results.cache
# sed -i 's/_layer_t/,layer_t/g' results.cache
# sed -i 's/_forward_t/,forward_t/g' results.cache
# sed -i 's/_selection_mode/,selection_mode/g' results.cache
# sed -i 's/_bound_mode/,bound_mode/g' results.cache
# sed -i 's/_klucb_max_cost/,klucb_max_cost/g' results.cache
# sed -i 's/_prioritize_worst_particles_z/,prioritize_worst_particles_z/g' results.cache
# sed -i 's/_method/,method/g' results.cache
# sed -i 's/_use_cfb/,use_cfb/g' results.cache
# sed -i 's/_extra_ego_accdec_policies/,extra_ego_accdec_policies/g' results.cache
# sed -i 's/_max_steps/,max_steps/g' results.cache
# sed -i 's/_n_cars/,n_cars/g' results.cache
# sed -i 's/_safety_margin_low/,safety_margin_low/g' results.cache
# sed -i 's/_safety_margin_high/,safety_margin_high/g' results.cache
# sed -i 's/_safety/,safety/g' results.cache
# sed -i 's/_accel/,accel/g' results.cache
# sed -i 's/_steer/,steer/g' results.cache
# sed -i 's/_discount_factor/,discount_factor/g' results.cache
# sed -i 's/_rng_seed/,rng_seed/g' results.cache
# sed -i 's/^method/,method/g' results.cache
# sed -i -E 's/(rng_seed=[0-9]{1,9})/\1,/g' results.cache
#!!! sed -i 's/_/=/g' results.cache


results = []
with open("results.cache", "r") as f:
    for line in f:
        parts = line.split()
        if len(parts) > 9:
            entry = dict()
            entry["name"] = parts[0]
            entry["efficiency"] = float(parts[5])
            entry["safety"] = float(parts[6])
            entry["ud"] = float(parts[7])
            entry["cc"] = float(parts[8])
            entry["seconds"] = float(parts[9])

            entry["cost.efficiency"] = float(parts[1])
            entry["cost.safety"] = float(parts[2])
            entry["cost.accel"] = float(parts[3])
            entry["cost.steer"] = float(parts[4])
            entry["cost"] = entry["cost.efficiency"] + entry["cost.safety"] + \
                entry["cost.accel"] + entry["cost.steer"]

            results.append(entry)
        else:
            continue

print_all_parameter_values_used(
    results, [("method", "mcts"), ("bound_mode", "marginal"), ("prioritize_worst_particles_z", "1000"), ("max.rng_seed", 255)])

method_kind = FigureKind("method", ["fixed", "tree", "mpdm", "eudm", "mcts"], translations=t10s)
discount_kind = FigureKind("discount_factor", [0.6, 0.7, 0.8, 0.9, 1], translations=t10s)
cfb_kind = FigureKind("use_cfb", ["false", "true"], translations=t10s)
seconds_kind = FigureKind("seconds", None, xlim=(0, 1.0), translations=t10s)

# extra_accdec_kind = FigureKind("extra_ego_accdec_policies", [
#                                "-1", "1", "-2", "2", "-1,1", "-2,2", "1,2", "-1,-2", "-1,-2,-3,1,2,3"])

extra_accdec_kind = FigureKind("extra_ego_accdec_policies", [
                               "", "-1,-2,1,2", "-1,-2,-3,1,2,3"], translations=t10s)

method_mode = FigureMode("method", ["fixed", "tree", "mpdm", "eudm", "mcts"])
cfb_mode = FigureMode("use_cfb", ["false", "true"])

plot_metrics = ["efficiency", "cost", "safety"]
evaluate_metrics = ["efficiency", "cost", "safety", "cost.efficiency",
                    "cost.safety", "cost.accel", "cost.steer", "seconds"]

find_unsafest_filters = ["_method_mcts_", "_use_cfb_false_",
                         "_selection_mode_klucb_", "_bound_mode_marginal_", "_klucb_max_cost_30_"]
# print(max([r for r in results if all(f in r["name"]
#  for f in find_unsafest_filters)], key=lambda entry: entry["safety"]))

if False:
    for use_cfb in ["true"]:
        for samples_n in [8, 16, 32, 64, 128, 256, 512]:
            evaluate_conditions(results, plot_metrics, [
                                ("method", "mcts"), ("search_depth", 4), ("layer_t", "2"),
                                ("smoothness", 0), ("safety", 100), ("ud", 5),
                                ("samples_n", samples_n), ("use_cfb", use_cfb)])
    # print("layer_t 2")
    # for use_cfb in ["false", "true"]:
    #     for search_depth in [4, 5, 6, 7]:
    #         evaluate_conditions(results, plot_metrics, [
    #                             ("method", "mcts"), ("search_depth", search_depth), ("layer_t", "2"), ("samples_n", 128), ("use_cfb", use_cfb)])

    print("eudm")
    for use_cfb in ["true"]:
        for search_depth in [4]:
            evaluate_conditions(results, plot_metrics, [
                                ("method", "eudm"),
                                ("smoothness", 0), ("safety", 100), ("ud", 5),
                                ("search_depth", search_depth), ("use_cfb", use_cfb)])


#
# cargo run --release rng_seed 0-15 :: method tree :: tree.samples_n 1 2 4 8 :: use_cfb false true :: thread_limit 24

# cargo run --release rng_seed 0-127 :: method fixed mpdm eudm mcts :: use_cfb false true :: eudm.search_depth 3-7 :: mcts.search_depth 3-7 :: mcts.samples_n 8 16 32 64 128 256 512 :: thread_limit 24
# cargo run --release rng_seed 127-255 :: method fixed mpdm eudm mcts :: use_cfb false true :: eudm.search_depth 3-7 :: mcts.search_depth 3-7 :: mcts.samples_n 8 16 32 64 128 256 512 :: thread_limit 24
samples_n_kind = FigureKind("samples_n", [8, 16, 32, 64, 128, 256, 512], translations=t10s)
search_depth_kind = FigureKind("search_depth", [3, 4, 5, 6, 7], translations=t10s)
method_mode = FigureMode("method", ["fixed", "mpdm", "eudm", "mcts"])
if True:
    for metric in plot_metrics:
        samples_n_kind.plot(results, metric, mode=cfb_mode, filters=[
                            ("method", "mcts"), ("bound_mode", "marginal"), ("prioritize_worst_particles_z", "1000"), ("max.rng_seed", 255)])
        for use_cfb in ["false", "true"]:
            search_depth_kind.plot(results, metric, mode=FigureMode("method", ["eudm", "mcts"]), filters=[
                                   ("use_cfb", use_cfb),
                                   ("mcts.bound_mode", "marginal"),
                                   ("mcts.prioritize_worst_particles_z", "1000"),
                                   ("max.rng_seed", 255)])
        search_depth_kind.plot(results, metric, mode=cfb_mode, filters=[("mcts.bound_mode", "marginal"),
                                                                        ("mcts.prioritize_worst_particles_z", "1000"),
                                                                        ("max.rng_seed", 255)])

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

if False:
    for metric in plot_metrics:
        seconds_kind.plot(results, metric, mode=method_mode)
        seconds_kind.plot(results, metric, mode=cfb_mode)

# cargo run --release rng_seed 0-1023 :: method mcts :: use_cfb false :: mcts.bound_mode normal bubble_best lower_bound marginal :: mcts.prioritize_worst_particles_z -3 -2 -1 0 1 2 3 1000 :: thread_limit 24
# cargo run --release rng_seed 1024-2047 :: method mcts :: use_cfb false :: mcts.bound_mode normal bubble_best lower_bound marginal :: mcts.prioritize_worst_particles_z -3 -2 -1 0 1 2 3 1000 :: thread_limit 24
if False:
    prioritize_worst_particles_z_kind = FigureKind(
        "prioritize_worst_particles_z", [-3, -2, -1, 0, 1, 2, 3, 1000])
    for metric in plot_metrics:
        prioritize_worst_particles_z_kind.plot(results, metric, mode=bound_mode, filters=[
            "_method_mcts_", "_selection_mode_klucb_", "_use_cfb_false_"])
