@echo off

echo "Generating data with background randomly colored"
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode none

echo "Generating data with color bias for digits 5-9"
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode partial

echo "Generating data with color bias for all digits"
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode all

echo "Generating augmented dataset. The unbiased samples (which cannot be augmented) are copied."
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode partial --augment_mode basic
