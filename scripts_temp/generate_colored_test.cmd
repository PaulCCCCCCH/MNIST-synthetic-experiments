@echo off

echo "Generating augmented dataset. The unbiased samples (which cannot be augmented) are copied."
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode partial --test_mode pure_special
