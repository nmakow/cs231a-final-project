from __future__ import division
import tensorflow as tf

# Show Attend and Tell model notation:
# N : batch size
# L : spatial size of feature vector (196)
# D : dimension of feature vector (512)
# T : number of time steps, equal to caption's length - 1 (16)
# V : vocabulary size (about 10k)
# M : dimension of word vector (i.e. embedding size) (default=512)
# H : dimension of hidden state (default=1024)

class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024,
                 n_time_step=16, prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx["<START>"]
        self._null = word_to_idx["<NULL>"]

        # variable initializers
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # placeholders for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])

    def _get_initial_lstm(self, features, reuse=False):
        with tf.variable_scope("initial_lstm"):
            w_h_init = tf.get_variable("w_h_init", [self.D, self.H], initializer=self.weight_initializer)
            b_h_init = tf.get_variable("b_h_init", [self.H], initializer=self.const_initializer)
            w_c_init = tf.get_variable("w_c_init", [self.D, self.H], initializer=self.weight_initializer)
            b_c_init = tf.get_variable("b_c_init", [self.H], initializer=self.const_initializer)

            h = tf.nn.tanh(tf.matmul(features, w_h_init) + b_h_init)
            c = tf.nn.tanh(tf.matmul(features, w_c_init) + b_c_init)
            return c, h


    def _project_features(self, features, reuse=False):
        with tf.variable_scope("feature_projection", reuse=reuse):
            w_proj = tf.get_variable("w_proj", [self.D, self.D], initializer=self.weight_initializer)
            # flat_features = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features, w_proj)
            # features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
        return features_proj


    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope("word_embedding", reuse=reuse):
            embed = tf.get_variable("embed", [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(embed, inputs, name="word_vector")
            return x

    def _get_logits(self, x, h, dropout=False, reuse=False):
        with tf.variable_scope("logits", reuse=reuse):
            w_h = tf.get_variable("w_h", [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable("b_h", [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable("w_out", [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable("b_out", [self.V], initializer=self.const_initializer)

            # if dropout:
            #     # TODO: implement
            #     pass
            # if self.ctx2out:
            #     # TODO:
            #     pass
            # if self.prev2out:
            #     #TODO
            #     pass

            h_logits = tf.matmul(h, w_h) + b_h
            h_logits = tf.nn.tanh(h_logits)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def build_model(self):
        features = self.features
        captions = self.captions
        batch_size = tf.shape(features[0])

        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        # batch norm TODO
        # features = self._batch_norm(features, mode="train", name="conv_features")

        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=captions_in)
        features_proj = self._project_features(features=features)

        loss = 0.0
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            with tf.variable_scope("lstm", reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=x[:,t,:], state=[c, h])
            logits = self._get_logits(x[:,t,:], h, dropout=self.dropout, reuse=(t!=0))
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=captions_out[:, t], logits=logits)*mask[:, t])

        return loss / tf.to_float(batch_size)


    def build_sampler(self, max_len=20):
        features = self.features
        N = tf.shape(features)[0]

        # batch norm
        # TODO

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word = tf.fill([N], self._start)
        sampled_words = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(max_len):
            x = self._word_embedding(inputs=sampled_word, reuse=(t!=0))
            with tf.variable_scope("lstm", reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=x, state=[c,h])
            logits = self._get_logits(x, h, reuse=(t!=0))
            sampled_word = tf.argmax(logits, 1)
            sampled_words.append(sampled_word)

        # TODO: ALPHA AND BETAS

        sampled_captions = tf.transpose(tf.stack(sampled_words), (1, 0)) # (N, max_len)

        return None, None, sampled_captions
