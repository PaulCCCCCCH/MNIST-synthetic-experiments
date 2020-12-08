@echo off

for %%m in (
pure_half_1,
pure_half_2,
pure_half_3,
pure_half_12,
pure_half_13,
pure_half_23,
pure_half_123 ) do (
    python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode partial --test_mode %%m
)
