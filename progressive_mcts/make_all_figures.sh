#!/bin/bash
cargo run --release rng_seed 0-8191 :: samples_n 128 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: final_choice_mode same :: ucb_const -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000 -15000 -22000 -33000 -47000 -68000 :: thread_limit 24
./plot.py 1

cargo run --release rng_seed 0-8191 :: samples_n 128 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal :: final_choice_mode marginal :: ucb_const -1000 -1500 -2200 -3300 -4700 -6800 -10000 -15000 -22000 -33000 -47000 -68000 :: thread_limit 24
cargo run --release rng_seed 0-8191 :: samples_n 128 :: selection_mode uniform :: final_choice_mode marginal :: thread_limit 24
./plot.py 2
