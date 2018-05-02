import argparse
import warnings

import colorama as c

from texter.utils.config_utils import keras_data_config as kdc
from texter.utils.config_utils import keras_model_config as kmc
from texter.utils.io_utils import load_config, save_config, save_model, save_text_model
from texter.utils import eval_utils as evu

warnings.simplefilter("ignore")


def get_args():
    parser = argparse.ArgumentParser(
        description='classification training api')
    parser.add_argument('-dc', '--data_config', type=str,
                        help='path to data config file', required=True, nargs='+')
    parser.add_argument('-mc', '--model_config', type=str,
                        help='path to data config file', required=True, nargs='+')
    parser.add_argument('-bs', '--batch_size', type=int,
                        help='batch size', required=True, nargs='+')
    parser.add_argument('-e', '--epoch', type=int,
                        help='number of passes', required=True, nargs='+')
    parser.add_argument('-v', '--verbose', type=int,
                        help='output state display mode', required=True, nargs='+')
    parser.add_argument('-mn', '--model_name', type=str,
                        help='classifier model name', required=True, nargs='+')

    args = parser.parse_args()
    data_config = args.data_config[0]
    model_config = args.model_config[0]
    bs = args.batch_size[0]
    e = args.epoch[0]
    v = args.verbose[0]
    mn = args.model_name[0]
    return data_config, model_config, bs, e, v, mn


if __name__ == '__main__':
    data_config, model_config, bs, e, v, mn = get_args()
    print(
        f"\nRunning model with\nBatch Size: {bs}\nepochs: {e}\nverbose: {v}\n")
    data_config = load_config(data_config)
    model_config = load_config(model_config)
    x_train, x_test, y_train, y_test, tokenizer, word_index, num_class = kdc(
        **data_config)
    model = kmc(**model_config, num_class=num_class)
    print(f"\n{c.Fore.RED}Classification Model summary:\n")
    print(f"\n\n{c.Fore.CYAN}{model.summary()}")
    model.fit(x_train, y_train, batch_size=bs, epochs=e, verbose=v,
              validation_data=(x_test, y_test), callbacks=evu.add_callbacks())
    print("\n")
    save_text_model(f"{mn}", tokenizer)
    save_model(f"{mn}", model)
    print(f"\n\n{c.Fore.GREEN}Text preprocessor(tokenizer) saved successfully.")
    print(f"\n\n{c.Fore.GREEN}Model weights saved successfully.")
    print(f"\n\n{c.Fore.BLUE}Model architecture saved successfully.")
