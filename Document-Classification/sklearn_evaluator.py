import argparse
import warnings

import colorama as c

from texter.utils.io_utils import load_model, load_text_model
from texter.utils.text_utils import sklearn_text_vectorizer as stv

warnings.simplefilter("ignore")


def get_args():
    parser = argparse.ArgumentParser(
        description='classification prediction api')
    parser.add_argument('-d', '--data_path', type=str,
                        help='path to new document', required=True, nargs='+')
    parser.add_argument('-m', '--model_path', type=str,
                        help='path to saved model file', required=True, nargs='+')
    parser.add_argument('-t', '--text_model_path', type=str,
                        help='path to saved text model file', required=True, nargs='+')
    parser.add_argument('-e', '--preprocessing_engine', type=str,
                        help='text processing method, either word2vec or tfidf', required=True, nargs='+')
    parser.add_argument('-p', '--word2vec_path', type=str,
                        help='path to pretrained word2vec bin file, use if `e` is word2vec', required=False, nargs='+')

    args = parser.parse_args()
    dp = args.data_path[0]
    mp = args.model_path[0]
    tmp = args.text_model_path[0]
    engine = args.preprocessing_engine[0]
    try:
        w2v = args.word2vec_path[0]
    except Exception:
        w2v = "Not required"
    return dp, mp, tmp, engine, w2v


if __name__ == '__main__':
    dp, mp, tmp, engine, w2v = get_args()
    print(f"{c.Fore.RED}Successfully loaded the classifier and the text vectorization models.\n")
    model = load_model(mp)
    text_model = load_text_model(tmp)
    print(f"\n\n{c.Fore.CYAN}Successfully loaded the text document.\n")
    with open(f'{dp}', 'r') as f:
        t = f.readlines()
        text = ''.join([x for x in t])
    vector_text = stv(text, text_model, model=engine, w2v_path=w2v)
    try:
        prediction = model.predict_proba(vector_text)
        print(
            f"{c.Fore.GREEN}prediction: {prediction[0]}\nnote:(representing the class probabilities.)")
    except Exception:
        prediction = model.predict(vector_text)
        print(
            f"{c.Fore.GREEN}prediction: {prediction[0]}\nnote:(representing the class, this algorithm does not provide a probabilistic score.)")
