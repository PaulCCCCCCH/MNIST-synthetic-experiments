import logging
import os


def set_logger(args):
    if args.isGeneration:
        save_dir = args.adversarial_dir
    elif args.saveAsNew:
        save_dir = args.new_save_dir
    else:
        save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s: %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                  logging.FileHandler("all_output.txt"),
                                  logging.StreamHandler()])
