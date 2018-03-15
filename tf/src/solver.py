import tensorflow as tf
import matplotlib.pyplot as plt
# import skimage.transform
import numpy as np
import time
import os
import cPickle as pickle
# from scipy import ndimage
from utils import *
# from bleu import evaluate

class CaptioningSolver(object):
    def __init__(self, model, data, **kwargs):
        """
        Required arguments:
         - model: image captioning model
         - data: dictionary of training and validation data from load_coco_data
        """
        self.model = model
        self.data = data
        # hyperparameters
        self.n_epochs = kwargs.pop("n_epochs", 10)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.update_rule = kwargs.pop("update_rule", "adam")
        self.learning_rate = kwargs.pop("learning_rate", 0.01)
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.print_bleu = kwargs.pop("print_bleu", False)
        self.print_every = kwargs.pop("print_every", 100)
        self.save_every = kwargs.pop("save_every", 1)
        self.log_path = kwargs.pop("log_path", "./log/")
        self.model_path = kwargs.pop("model_path", "./model/")
        self.pretrained_model = kwargs.pop("pretrained_model", None)
        self.test_model = kwargs.pop("test_model", "./model/lstm/model-1")
        # select an optimizer based on update_rule
        if self.update_rule == "adam":
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == "momentum":
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == "rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer
        # make sure model, log directories exists
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def train(self):
        n_train, n_val = self.data["train_features"].shape[0], \
                         self.data["val_features"].shape[0]
        n_batches_train = int(np.ceil(float(n_train) / self.batch_size))
        n_batches_val = int(np.ceil(float(n_val) / self.batch_size))
        features = self.data['train_features']
        captions = self.data['train_captions']
        image_idxs = self.data['train_image_idxs']
        val_features = self.data['val_features']

        # Loss op (from self.model)
        with tf.variable_scope(tf.get_variable_scope()):
            loss = self.model.build_model()
            tf.get_variable_scope().reuse_variables()
            _, _, generated_captions = self.model.build_sampler(max_len=20)
            print generated_captions
        # Training Op
        with tf.variable_scope(tf.get_variable_scope()):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
        # Summary op
        # tf.summary.scalar("batch_loss", loss)
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name, var)
        # for grad, var in grads_and_vars:
        #     print grad, var
        #     tf.summary.histogram(var.op.name+"/gradient", grad)
        # summary_op = tf.summary.merge_all()

        print "Num epochs: %d" % self.n_epochs
        print "Data size (train): %d" % n_train
        print "Batch size: %d" % self.batch_size
        print "Iterations per epoch: %d" % n_batches_train

        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print "Start training with pretrained model..."
                saver.restore(sess, self.pretrained_model)

            prev_loss, curr_loss = -1, 0
            start_time = time.time()

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_train)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]

                for i in range(n_batches_train):
                    captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = features[image_idxs_batch]
                    feed_dict = {
                        self.model.features: features_batch,
                        self.model.captions: captions_batch
                    }
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l

                    # Summary w/ tensorboard: TODO
                    # if i % 10 == 0:
                    #     summary = sess.run(summary_op, feed_dict)
                    #     summary_writer.add_summary(summary, e*n_batches_train + i)

                    if (i+1) % self.print_every == 0:
                        print "\nTrain loss at epoch %d iteration %d: %0.5f" % (e+1, i+1, l)
                        ground_truths = captions[image_idxs == image_idxs_batch[0]]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print "Ground truth %d: %s" % (j+1, gt)
                        gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print "Generated caption: %s\n" % (decoded[0])

                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_time
                prev_loss = curr_loss
                curr_loss = 0

                # TODO: BLEU evaluation
                if self.print_bleu:
                    all_gen_cap = np.ndarray((val_features.shape[0], 20))
                    for i in range(n_batches_val):
                        features_batch = val_features[i*self.batch_size:(i+1)*self.batch_size]
                        feed_dict = { self.model.features: features_batch }
                        all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = \
                            sess.run(generated_captions, feed_dict=feed_dict)
                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                    scores = evaluate(data_path="./data", split="val", get_scores=True)
                    write_bleu(scores=scores, path=self.model_path, epoch=e)

                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, "model"), global_step=e+1)
                    print "model-%s saved" % (e+1)


    def test(self, data, split="train", attention_visualization=False, save_sampled_captions=True):
        features = data[split + '_features']

        _, _, sampled_captions = self.model.build_sampler(max_len=20) # (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options_allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            _, features_batch, image_urls = sample_coco_minibatch(data, self.batch_size)
            feed_dict = { self.model.features: features_batch }
            out_captions = sess.run([sampled_captions], feed_dict) # (N, max_len)
            decoded = decode_captions(out_captions, self.model.idx_to_word)

            if attention_visualization:
                # TODO: if implement attention, add this
                pass

            if save_sampled_captions:
                all_out_captions = np.ndarray((features.shape[0], 20))
                num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
                for i in xrange(num_iter):
                    features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
                    feed_dict = { self.model.features: featuers_batch }
                    all_out_captions[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)
                all_decoded = decode_captions(all_out_captions, self.model.idx_to_word)
                save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" % (split, split))
