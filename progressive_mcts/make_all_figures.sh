#!/bin/bash
./make_expected_cost_figure.py

time cargo run --release rng_seed 0-1023 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: ucb_const -150 :: bound_mode marginal_prior :: final_choice_mode same :: unknown_prior_std_dev_scalar 1.2 1.4 1.6 1.8 2 :: zero_mean_prior_std_dev 33 47 68 100 150 220 330 470 680 1000000000
./plot.py zero_prior

time cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal marginal_prior :: final_choice_mode same :: ucb_const 0 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800
./plot.py 1

time cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode uniform :: final_choice_mode marginal_prior
time cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal marginal_prior :: final_choice_mode marginal_prior :: ucb_const 0 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000
./plot.py 2

time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode uniform :: final_choice_mode marginal_prior
time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode normal bubble_best lower_bound marginal marginal_prior :: final_choice_mode marginal_prior :: normal.ucb_const -3300 :: bubble_best.ucb_const -4700 :: lower_bound.ucb_const -6800 :: marginal.ucb_const -6800 :: marginal_prior.ucb_const -100
./plot.py 3

time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal_prior :: selection_mode ucb ucbv ucbd klucb klucb+ uniform :: ucb.ucb_const -100 :: ucbv.ucb_const -22 :: ucbv.ucbv_const 0 :: ucbd.ucb_const -33 :: ucbd.ucbd_const 0.0001 :: klucb.ucb_const -0.0047 :: klucb.klucb_max_cost 4700 :: klucb+.ucb_const -0.01 :: klucb+.klucb_max_cost 680
./plot.py 4

time cargo run --release rng_seed 0-16383 :: samples_n 8 16 32 64 128 256 512 1024 :: repeat_const 0 1024 2048 4096 8192 16384 32768 65536
./plot.py repeat_const

time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode normal :: selection_mode ucb :: ucb_const -470
time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal :: selection_mode ucb :: ucb_const -470
time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096
time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: repeat_const 16384
./plot.py final
