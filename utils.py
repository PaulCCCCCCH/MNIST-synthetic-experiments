import logging
import os

def set_logger(args):
    model_name = args.model_name if not args.new_model_name else args.new_model_name
    save_dir = os.path.join(args.save_dir, model_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s: %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                  logging.StreamHandler()])
