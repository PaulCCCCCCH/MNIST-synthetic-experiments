python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode partial --augment_mode basic

python train_paired.py colored_biased ^
--data_path adversarial\\colored\\colored_partial.pkl ^
--is_rgb_data ^
--paired_data_path adversarial\\colored\\colored_partial_aug_noise.pkl ^
--new_model_name partial_aug_noise_onto_partial_biased ^
--augment_data_mode pick_random ^
--epoch 100

echo "Evaluate the de-biased model"
python eval.py partial_aug_noise_onto_partial_biased --data_path adversarial\\colored\\colored_partial.pkl --is_rgb_data