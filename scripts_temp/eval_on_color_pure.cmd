:: python eval.py partial_aug_noise_onto_partial_biased --data_path adversarial\\colored\\colored_partial_test_noise.pkl --is_rgb_data
:: python eval.py partial_aug_onto_partial_biased --data_path adversarial\\colored\\colored_all.pkl --is_rgb_data --use_reg_model
:: python eval.py partial_aug_onto_partial_biased --data_path adversarial\\colored\\colored_none.pkl --is_rgb_data --use_reg_model
:: python eval.py partial_aug_onto_partial_biased --data_path adversarial\\colored\\colored_partial.pkl --is_rgb_data --use_reg_model

:: python eval.py partial_aug_noise_onto_partial_biased --data_path adversarial\\colored\\colored_partial_test_pure_black.pkl --is_rgb_data
for %%m in (
pure_half_1,
pure_half_2,
pure_half_3,
pure_half_12,
pure_half_13,
pure_half_23,
pure_half_123
) do (

python eval.py colored_biased --data_path adversarial\\colored\\colored_partial_test_%%m.pkl --is_rgb_data

)
