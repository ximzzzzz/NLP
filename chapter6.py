#!/usr/bin/env python
# coding: utf-8

# In[7]:


# p 247
# 6-1


import sys
sys.path.append('../models')
sys.path.append('../')
from bert.tokenization import FullTokenizer
from preprocess import get_tokenizer, post_processing


class Tuner(object):
    
    def __init__(self, train_corpus_fname = None,
                tokenized_train_corpus_fname = None,
                test_corpus_fname = None, tokenized_test_corpus_fname= None,
                model_name='bert', model_save_path = None, vocab_fname=None,
                eval_every=1000,
                batch_size=32, num_epochs=10, dropout_keep_prob_rate=0.9,
                model_ckpt_path=None):
        
        self.model_name = model_name
        self.eval_every = eval_every
        self.model_ckpt_path = model_ckpt_path
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_keep_prob_rate = dropout_keep_prob_rate
        self.best_valid_score = 0.0
        
        #tokenizer defining
        if self.model_name =='bert':
            self.tokenizer = FullTokenizer(vocab_file = vocab_fname, do_lower_case = False)
        else:
            self.tokenizer = get_tokenizer('mecab')
            
        #load or tokenize corpus
        
        self.train_data, self.train_data_size = self.load_or_tokenize_corpus(train_corpus_fname, tokenized_train_corpus_fname)
        self.test_data, self.test_data_size = self.load_or_tokenize_corpus(test_corpus_fname, tokenized_test_corpus_fname)
    
    
    def load_or_tokenize_corpus(self, corpus_fname, tokenized_corpus_fname):
        data_set = []
        if os.path.exists(tokenized_corpus_fname):
            tf.logging.info('load tokenized corpus : ' + tokenized_corpus_fname)
            with open(tokenized_corpus_fname, 'r') as f1:
                for line in f1:
                    tokens, label = line.strip().split('\u241E')
                    if len(tokens) > 0:
                        data_set.append([tokens.split(" "), int(label)])
                        
        else :
            with open(corpus_fname, 'r') as f2:
                next(f2) #skip head line
                for line in f2:
                    sentence, label = line.strip().split('\u241E')
                    if self.model_name == 'bert':
                        tokens = self.tokenizer.tokenize(sentence)
                    else:
                        tokens = self.tokenizer.morphs(sentence)
                        tokens = post_processing(tokens)
                    
                    #labelling
                    if int(label) >=1:
                        int_label = 1
                    else:
                        int_label = 0
                    data_set.append([tokens, int_label])
            with open(tokenized_corpus_fname, 'w') as f3:
                for tokens, label in data_set:
                    f3.writelines(' '.join(tokens) + '\u241E' + str(label) + '\n')
        return data_set, len(data_set)
    
    
    def get_batch(self, data, num_epochs, is_training = True):

        data_size = self.train_data_size
        num_batches_per_epoch = int((data_size - 1) / self.batch_size) + 1
        
        if is_training:
            for epoch in range(num_epocs):
                idx = random.sample(range(data_size), data_size)
                data = np.array(data)[idx]
                for batch_num in range(num_batches_per_epoch):
                    batch_sentences = []
                    batch_labels = []
                    start_index = batch_num * self.batch_size
                    end_index = min((batch_num+1)* self.batch_size, data_size)
                    features = data[start_index : end_index]
                    for features in features:
                        sentence, label = feature
                        batch_sentences.append(sentence)
                        batch_labels.append(int(label))
                    yield self.make_input(batch_sentences, batch_labels, is_training)
                    
                    
                    
    def train(self, sess, saver, global_step, output_feed):
        train_batches = self.get_batch(self.train_data, self.num_epochs, is_training=True)
        checkpoint_loss = 0.0
        for current_input_feed in train_batches:
            _,_,_, current_loss = sess.run(output_feed, current_input_feed)
            checkpoint_loss += current_loss
            if global_step.eval(sess) % self.eval_every == 0 :
                tf.logging.info("global step %d train loss %.4f" % (global_step.eval(sess), checkpoint_loss / self.eval_every))
                checkpoint_loss = 0.0
                self.validation(sess, saver, global_step)
                
                
    def validation(self, sess, saver, global_step):
        valid_loss, valid_pred, valid_num_data = 0,0,0
        output_feed = [self.logits, self.loss]
        test_batches = self.get_batch(self.test_data, num_epochs = 1, is_training= False)
        
        for current_input_feed, current_labels in test_batches:
            current_logits, current_loss = sess.run(output_feed, current_input_feed)
            current_preds = np.argmax(current_logits, axis= -1)
            valid_loss += current_loss
            valid_num_data += len(current_labels)
            for pred, label in zip(current_preds, current_labels):
                if pred == label :
                    valid_pred +=1
        valid_score = valid_pred / valid_num_data
        tf.logging.info('valid loss %.4f valid score %.4f'%(valid_loss, valid_score))
        
        if valid_score > self.best_valid_score:
            self.best_valid_score = valid_score
            path = self.model_save_path + '/' +str(valid_score)
            saver.save(sess, path, global_step=global_step)
            
            
        def make_input(self, sentences, labels, is_training):
            raise NotImplementedError
            
        def tune(self):
            raise NotImplementedError
            
            


# In[4]:


' '.join(['hey','you','there?'])


# In[ ]:


# 6.7

def make_word_embedding_graph(num_labels, vocab_size, embedding_size,  tune=False):
    ids_placeholder = tf.placeholder(tf.int32, [None, None], name='input_ids')
    input_lengths = tf.placeholder(tf.int32, [None], name='input_lengths')
    labels_placeholder = tf.placeholder(tf.int32, [None], name='label_ids')
    if tune:
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    else:
        dropout_keep_prob = tf.constant(1.0, dtype=tf.float32)
    We = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=True)
    
    embedding_placeholder = tf.placeholder(tf.float32, shape=[vocab_size, embedding_size])
    
    embed_init = tf.assign(embedding_placeholder)
    #shape : [batch_size, unroll_steps, dimension]
    embedded_words = tf.nn.embedding_lookup(We, ids_placeholder)
    features = tf.nn.dropout(embedded_words, dropout_keep_prob)
    
    # bi lstm layers
    lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units = embedding_size, cell_clip=5, proj_clip=5)
    lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units = embedding_size, cell_clip=5, proj_clip=5)
    
    lstm_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_cell_fw,
                                                     cell_bw = lstm_cell_bw,
                                                    inputs = features, 
                                                     sequence_length = input_lengths, dtype=tf.float32)
    
    
    #attention layer
    output_fw, output_bw = lstm_output
    H = tf.contrib.layers.fully_connected(inputs = output_fw + output_bw, num_outputs =256, activation_fn = tf.nn.tanh)
    attention_score = tf.softmax(tf.contrib.layers.fully_connected(inputs=H, num_outputs=1, activation_fn = None))

    attention_output = tf.squeeze(tf.matmul(tf.transpose(H, perm=[0, 2, 1]), attention_score, axis=-1))
    layer_output = tf.nn.dropout(attention_output,dropout_keep_prob)
    
    
    #feed foward layer
    
    
                                         

