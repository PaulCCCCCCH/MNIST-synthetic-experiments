:: `--unbiased_data_mode pure` means that
:: the background of unbiased data is pure colors PICKED FROM A POOL OF TEN COLORS (i.e. the initial setup)
:: If you want to choose pure colors by using random RGB values (i.e. the better setup),
:: then change `--unbiased_data_mode` to `random_pure`.
:: You can also change it to `mixture` if you want. See `README.md` for more details.
echo "Generating the biased data sets for training and evaluation"
for %%t in (
    noise,
    mixture,
    random_pure,
    strips,
    pure,
) do (
    python generate_adversarial.py model_name_placeholder --attack_name colored --bias_mode partial --test_mode %%t --unbiased_data_mode random_pure
)

:: Train a model named 'colored_biased' on a data set generated previously.
:: Note that it does not matter which one of the five data sets you choose,
:: since they all have the same training and dev sets, and the only difference is in their test sets.
echo "Training biased base model. Call it 'colored_biased'. "
python train.py colored_biased --data_path adversarial\\colored\\colored_partial_test_pure.pkl --epoch 100 --is_rgb_data


:: Generate different augmentation data sets and fine tune the base model with them.
:: See `README.md` for caveats.
for %%s in (noise, strips, mixture, basic, random_pure) do (

    echo Generating %%s augmented data set
    python generate_adversarial.py model_name_placeholder --attack_name colored --bias_mode partial --augment_mode %%s

    :: If you wish not to use L2 regularization during paired training,
    :: simply remove the --use_reg_model parameter below.
    echo Training base model on %%s augmented data set
    python train_paired.py colored_biased ^
    --data_path adversarial\\colored\\colored_partial_test_pure.pkl ^
    --is_rgb_data ^
    --paired_data_path adversarial\\colored\\colored_partial_aug_%%s.pkl ^
    --new_model_name partial_aug_%%s_onto_partial_biased ^
    --augment_data_mode pick_random ^
    --epoch 100 ^
    --use_reg_model ^
    --patience 5
)

:: Evaluating all models and produce confusion matrices
for %%m in (
colored_biased,
partial_aug_basic_onto_partial_biased,
partial_aug_mixture_onto_partial_biased,
partial_aug_noise_onto_partial_biased,
partial_aug_strips_onto_partial_biased,
partial_aug_random_pure_onto_partial_biased,
) do (

    for %%t in (
    colored_partial_test_noise.pkl
    colored_partial_test_mixture.pkl
    colored_partial_test_random_pure.pkl
    colored_partial_test_strips.pkl
    colored_partial_test_pure.pkl
    ) do (
        python eval.py %%m --data_path adversarial\\colored\\%%t --is_rgb_data --use_reg_model
    )

)
