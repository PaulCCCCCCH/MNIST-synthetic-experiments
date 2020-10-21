@echo off

:: for %%s in (noise, noise_weak, noise_minor, strips, other_colors) do (
:: for %%s in (noise_weak, noise_minor, strips) do (
for %%s in (strips) do (
    echo Generating %%s dataset
    python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode partial --augment_mode %%s

    python train_paired.py colored_biased ^
    --data_path adversarial\\colored\\colored_partial.pkl ^
    --is_rgb_data ^
    --paired_data_path adversarial\\colored\\colored_partial_aug_%%s.pkl ^
    --new_model_name partial_aug_%%s_onto_partial_biased ^
    --augment_data_mode pick_random ^
    --epoch 100

    echo "Evaluate the de-biased model"
    python eval.py partial_aug_%%s_onto_partial_biased --data_path adversarial\\colored\\colored_partial.pkl --is_rgb_data
)