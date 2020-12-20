:: python eval.py partial_aug_noise_onto_partial_biased --data_path adversarial\\colored\\colored_partial_test_noise.pkl --is_rgb_data
:: python eval.py partial_aug_onto_partial_biased --data_path adversarial\\colored\\colored_all.pkl --is_rgb_data --use_reg_model

:: python eval.py partial_aug_noise_onto_partial_biased --data_path adversarial\\colored\\colored_partial_test_pure_black.pkl --is_rgb_data
:: python eval.py partial_aug_noise_onto_partial_biased --data_path adversarial\\colored\\colored_partial_test_pure_special.pkl --is_rgb_data

for %%m in (
mixture,
noise,
pure,
random_pure,
strips
) do (

python eval.py partial_aug_noise_onto_partial_biased --data_path adversarial\\colored\\colored_partial_test_%%m.pkl --is_rgb_data

)