:: Train paired model
python train_paired.py colored_fully_biased ^
--data_path adversarial\\colored\\colored_partial.pkl ^
--is_rgb_data ^
--paired_data_path adversarial\\colored\\colored_partial_aug.pkl ^
--new_model_name partial_aug_onto_fully_biased ^
--augment_data_mode pick_random ^
--epoch 100