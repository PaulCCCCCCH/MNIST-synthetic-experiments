@echo off

:: echo "Training LeNet on standard MNIST"
:: python train.py lenet_standard --data_path data\\mnist.pkl --epoch 120

:: echo Generating attack data using CW
:: python generate_adversarial.py lenet_standard --data_path data\\mnist.pkl --attack_name cw

for %%i in (0.1, 0.5, 1.0, 5.0) do (

    echo "Training previous LeNet on CW (c = %%i) adversarial data
    python train.py lenet_standard --data_path adversarial\\cw\\cw_c_%%i.pkl --new_model_name cw_%%i_onto_std

    echo "Evaluating new LeNet on CW (c = %%i) on reserved data of standard MNIST"
    python eval.py cw_%%i_onto_std --test_only_data_path data\\reserved.pkl

)

:: echo "Generating attack data using FGSM"
:: python generate_adversarial.py lenet_standard --data_path data\\mnist.pkl --attack_name fgsm

for %%i in (0.1, 0.2, 0.3, 0.4, 0.5) do (
    echo "Training LeNet on FGSM adv data with e = %%i"
    python train.py lenet_standard --data_path adversarial\\fgsm\\fgsm_epsilon_%%i.pkl --new_model_name fgsm_%%i_onto_std

    echo "Evaluating it on reserved data of standard MNIST"
    python eval.py fgsm_%%i_onto_std --test_only_data_path data\\reserved.pkl
)

