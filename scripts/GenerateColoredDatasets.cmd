@echo off

:: These generated data sets will all have the same biased training and dev sets,
:: so it does not matter which one you choose for training.
:: The only difference will be in their test sets.
for %%t in (
    noise,
    mixture,
    random_pure,
    strips,
    pure,
) do (
    python generate_adversarial.py temp --attack_name colored --bias_mode partial --test_mode %%t --unbiased_data_mode random_pure
)


