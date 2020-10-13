:: python train.py colored --data_path adversarial\\colored\\colored_partial.pkl --epoch 100 --is_rgb_data
python train.py colored_fully_biased --data_path adversarial\\colored\\colored_all.pkl --epoch 100 --is_rgb_data
:: python train.py test --data_path adversarial\\fgsm\\fgsm_epsilon_0.1.pkl --epoch 2
