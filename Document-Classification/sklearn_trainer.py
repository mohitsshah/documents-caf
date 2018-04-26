import argparse
import warnings

import colorama as c

from texter.utils.config_utils import sklearn_data_config as sdc
from texter.utils.config_utils import sklearn_model_config as smc
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
    parser.add_argument('-mn', '--model_name', type=str,
                        help='classifier model name', required=True, nargs='+')

    args = parser.parse_args()
    data_config = args.data_config[0]
    model_config = args.model_config[0]
    mn = args.model_name[0]
    return data_config, model_config, mn


if __name__ == '__main__':
    data_config, model_config, mn = get_args()
    data_config = load_config(data_config)
    model_config = load_config(model_config)
    x_train, x_test, y_train, y_test, tokenizer = sdc(
        **data_config)
    model = smc(**model_config)
    print(f"\n{c.Fore.RED}Classification Model definition parameters:\n")
    print(f"{c.Fore.CYAN}{model.get_params()}\n")
    model.fit(x_train, y_train)
    y_pred = model.fit(x_train, y_train).predict(x_test)
    clf_report = evu.classifier_report(model, x_test, y_test, y_pred)
    print(f"\n\n{c.Fore.YELLOW}Classifier performance report:\n{clf_report}")
    save_text_model(f"{mn}", tokenizer)
    save_model(f"{mn}", model)
    print(f"\n\n{c.Fore.GREEN}Text preprocessor(tokenizer) saved successfully.")
    print(f"\n\n{c.Fore.BLUE}Model saved successfully.")
