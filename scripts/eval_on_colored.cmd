:: python eval.py partial_aug_onto_partial_biased --data_path adversarial\\colored\\colored_all.pkl --is_rgb_data --use_reg_model
:: python eval.py partial_aug_onto_partial_biased --data_path adversarial\\colored\\colored_none.pkl --is_rgb_data --use_reg_model
:: python eval.py partial_aug_onto_partial_biased --data_path adversarial\\colored\\colored_partial.pkl --is_rgb_data --use_reg_model
for %%m in (
partial_aug_noise_onto_partial_biased,
partial_aug_strips_onto_partial_biased,
partial_aug_random_pure_onto_partial_biased,
partial_aug_mixture_onto_partial_biased,
) do (

    for %%t in (
    colored_partial_test_noise.pkl
    colored_partial_test_mixture.pkl
    colored_partial_test_random_pure.pkl
    colored_partial_test_strips.pkl
    colored_partial.pkl
    ) do (
        python eval.py %%m --data_path adversarial\\colored\\%%t --is_rgb_data
    )

)
