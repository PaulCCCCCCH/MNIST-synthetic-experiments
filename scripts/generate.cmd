@echo off

echo "Training LeNet on standard MNIST"
python train.py lenet_standard --data_path data\\mnist.pkl --epoch 120

echo Generating attack data using CW
python generate_adversarial.py lenet_standard --data_path data\\mnist.pkl --attack_name cw

echo "Generating attack data using FGSM"
python generate_adversarial.py lenet_standard --data_path data\\mnist.pkl --attack_name fgsm

