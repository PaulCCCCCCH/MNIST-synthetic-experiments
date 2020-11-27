@echo off

for %%t in (
    noise,
    mixture,
    random_pure,
    strips,
    pure,
) do (
    python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode partial --test_mode %%t
)
