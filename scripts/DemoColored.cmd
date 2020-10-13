@echo off

echo "Generating data with randomly colored background for all images"
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode none

echo "Generating data with color bias for digits 5-9 (designated background for each of the digits 5-9)"
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode partial

echo "Generating data with color bias for all digits (designated background for all digits)"
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode all

echo "Generating data with color bias for digits 5-9, clipping the rest.
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode partial --clipped

echo "Generating augmented dataset. The unbiased samples (which cannot be augmented) are simply copied."
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode partial --augment_mode basic

echo "Generating augmented dataset with only augmented data. Data not augmented is not included."
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode partial --augment_mode clipped

echo "Training and evaluating on biased dataset"
python train.py colored_biased --data_path adversarial\\colored\\colored_partial.pkl --epoch 100 --is_rgb_data
python eval.py colored_biased --data_path adversarial\\colored\\colored_partial.pkl --is_rgb_data

:: echo "Training and evaluating on unbiased dataset for comparison"
:: python train.py colored_unbiased --data_path adversarial\\colored\\colored_none.pkl --epoch 100 --is_rgb_data
:: python eval.py colored_unbiased --data_path adversarial\\colored\\colored_partial.pkl --is_rgb_data

echo "Use paired training with basic augmented dataset to de-bias the biased model"
python train_paired.py colored_biased ^
--data_path adversarial\\colored\\colored_partial.pkl ^
--is_rgb_data ^
--paired_data_path adversarial\\colored\\colored_partial_aug_basic.pkl ^
--new_model_name partial_aug_onto_partial_biased ^
--augment_data_mode pick_random ^
--epoch 100

echo "Evaluate the de-biased model"
python eval.py partial_aug_onto_partial_biased --data_path adversarial\\colored\\colored_partial.pkl --is_rgb_data


echo "Use paired training with only labels that need to be de-biased"
python train_paired.py colored_biased ^
--data_path adversarial\\colored\\colored_partial_clipped.pkl ^
--is_rgb_data ^
--paired_data_path adversarial\\colored\\colored_partial_aug_clipped.pkl ^
--new_model_name clipped_aug_onto_partial_biased ^
--augment_data_mode pick_random ^
--epoch 100

echo "Evaluate the de-biased model"
python eval.py clipped_aug_onto_partial_biased --data_path adversarial\\colored\\colored_partial_clipped.pkl --is_rgb_data