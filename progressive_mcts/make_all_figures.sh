#!/bin/bash
./make_expected_cost_figure.py

time cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal marginal_prior :: final_choice_mode same :: ucb_const 0 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 :: thread_limit 24
./plot.py 1

time cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode uniform :: final_choice_mode marginal_prior :: thread_limit 24
time cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal marginal_prior :: final_choice_mode marginal_prior :: ucb_const 0 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 :: thread_limit 24
./plot.py 2

time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode uniform :: final_choice_mode marginal_prior :: thread_limit 24
time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal marginal_prior :: final_choice_mode marginal_prior :: normal.ucb_const -1500 :: bubble_best.ucb_const -1500 :: lower_bound.ucb_const -1500 :: marginal.ucb_const -1000 :: marginal_prior.ucb_const -150 :: thread_limit 24
./plot.py 3

time cargo run --release rng_seed 0-16383 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: repeat_const 0 65536 131072 262144 524288 1048576 2097152 :: thread_limit 24
./plot.py repeat_const

./plot.py final
