import json
import numpy as np
import tensorflow as tf
from QANet.model import Model
from QANet.prepro import convert_to_features, word_tokenize


class Eval(object):
    def __init__(self, config_path):
        with open(config_path, "r") as fi:
            configs = json.load(fi)
        with open(configs["word_emb_file"], "r") as fh:
            word_mat = np.array(json.load(fh), dtype=np.float32)
        with open(configs["char_emb_file"], "r") as fh:
            char_mat = np.array(json.load(fh), dtype=np.float32)
        # with open(configs["test_meta"], "r") as fh:
        #     meta = json.load(fh)

        model = Model(configs, None, word_mat, char_mat, trainable=False, demo=True)
        with open(configs["word_dictionary"], "r") as fh:
            word_dictionary = json.load(fh)
        with open(configs["char_dictionary"], "r") as fh:
            char_dictionary = json.load(fh)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with model.graph.as_default():
            sess = tf.Session(config=sess_config)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(configs["save_dir"]))
            if configs["decay"] < 1.0:
                sess.run(model.assign_vars)
        self.sess = sess

        self.word_dictionary = word_dictionary
        self.char_dictionary = char_dictionary
        self.configs = configs
        self.model = model

    def format_passage(self, passage):
        new_passage = ""
        for idx, chr in enumerate(passage):
            if idx > 0:
                if chr == "\n":
                    if passage[idx - 1] != "\n" and not passage[idx - 1].endswith("."):
                        new_passage += "."
            new_passage += chr
        return new_passage

    def extract(self, question, passage):
        sess = self.sess
        configs = self.configs
        model = self.model
        word_dictionary = self.word_dictionary
        char_dictionary = self.char_dictionary
        passage = passage.replace("''", '" ').replace("``", '" ')
        passage = self.format_passage(passage)
        query = (passage, question)
        if len(question) > 0 and len(passage) > 0:
            context = word_tokenize(query[0].replace("''", '" ').replace("``", '" '))
            c, ch, q, qh = convert_to_features(configs, query, word_dictionary, char_dictionary)
            fd = {'context:0': [c],
                  'question:0': [q],
                  'context_char:0': [ch],
                  'question_char:0': [qh]}
            yp1, yp2, lg1, lg2 = sess.run([model.yp1, model.yp2, model.lg1, model.lg2], feed_dict=fd)
            start_idx = yp1[0]
            stop_idx = yp2[0]
            yp2[0] += 1
            response = " ".join(context[yp1[0]:yp2[0]])
            return response
        else:
            return None


if __name__ == "__main__":
    config_path = "config.json"
    passage = open("collateral.txt", "r").read()
    eval = Eval(config_path)
    while True:
        question = input("Question >> ")
        if not question:
            break
        response = eval.extract(question, passage)
        print (response)
