@echo off

echo "Training LeNet on standard MNIST"
python train.py lenet_standard --data_path data\\mnist.pkl --epoch 120

echo Generating attack data using CW
python generate_adversarial.py lenet_standard --data_path data\\mnist.pkl --attack_name cw

echo "Generating attack data using FGSM"
python generate_adversarial.py lenet_standard --data_path data\\mnist.pkl --attack_name fgsm

echo "Generating data with background randomly colored"
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode none

echo "Generating data with color bias for digits 5-9"
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode partial

echo "Generating data with color bias for all digits"
python generate_adversarial.py temp --data_path data\\mnist.pkl --attack_name colored --bias_mode all
