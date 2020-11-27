
python train.py colored_biased --data_path adversarial\\colored\\colored_partial.pkl --epoch 100 --is_rgb_data

python eval.py colored_biased --data_path adversarial\\colored\\colored_partial_test_random_pure.pkl --is_rgb_data

python eval.py colored_biased --data_path adversarial\\colored\\colored_partial_test_pure.pkl --is_rgb_data

python eval.py colored_biased --data_path adversarial\\colored\\colored_partial_test_mixture.pkl --is_rgb_data

python eval.py colored_biased --data_path adversarial\\colored\\colored_partial_test_noise.pkl --is_rgb_data
