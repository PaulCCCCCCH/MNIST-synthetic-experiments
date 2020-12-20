:: Train paired model
python train_paired.py partial_aug_noise_onto_partial_biased ^
--data_path adversarial\\colored\\colored_partial_test_pure.pkl ^
--is_rgb_data ^
--paired_data_path adversarial\\colored\\colored_partial_aug_noise.pkl ^
--new_model_name partial_aug_noise_onto_partial_biased ^
--augment_data_mode pick_random ^
--use_reg_model ^
--epoch 100 ^
--patience 7