import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', './toxiccc_lstmtf', 'Summaries directory')
if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
tf.gfile.MakeDirs(FLAGS.summaries_dir)


# Function to generate batches from the test/train datasets
def generate_batches(data, batch_size, num_epochs, shuffle=True, istrain=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    l = 0
    if istrain:
        for epoch in range(num_epochs):
            l += 1
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
    else:
        for block_num in range(num_batches_per_epoch):
            start_index = block_num * batch_size
            end_index = min((block_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]


# For Test data. Can use generate_batch function.
def test_split(data, block_size):
    data = np.array(data)
    data_size = len(data)
    nums = int((data_size-1)/block_size) + 1
    for block_num in range(nums):
        start_index = block_num * block_size
        end_index = min((block_num + 1) * block_size, data_size)
        yield data[start_index:end_index]


def get_coefs(word1, *arr):
    return word1, np.asarray(arr, dtype='float32')


embed_size = 300        # size of each word vector
max_features = 100000   # number of rows in the embedding vector (unique words)
maxlen = 170            # max words in a comment; pad if not this length
batch_size = 128        # batch size for processing train dataset
epochs = 20             # number of epochs to run
n_classes = 6           # number of output classes
lr = 0.00001            # learning rate
lr_decay = 0.008        # learning rate decay rate
lr_decay_steps = 10000  # learning rate decay steps
n_hidden = 256          # number of hidden lstm units

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
Y_tr = train[list_classes].values
tokenizer = Tokenizer(num_words=max_features)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_sentences_test = test["comment_text"].fillna("_na_").values

tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_tr = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

# load embedding file
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('glove.840B.300d.txt'))
print ("Loaded GloVe Embedding, embed size = 300")

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
print "Mean and SD of embedding %s, %s" % (emb_mean, emb_std)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

tf.reset_default_graph()

# Input placeholders
with tf.name_scope('input'):
    input_x = tf.placeholder(tf.int32, [None, maxlen], name="input_x")
    input_y = tf.placeholder(tf.int32, [None, n_classes], name="input_y")

# define global step
with tf.variable_scope("global_step"):
    global_step_tensor = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)

# placeholders for embedding layer
with tf.name_scope('embedding'):
    embedding = tf.get_variable("embedding", shape=[max_features, embed_size])
    input_lookup = tf.nn.embedding_lookup(embedding_matrix, input_x)
    input_lookup = tf.cast(input_lookup, tf.float32)

with tf.name_scope('dropout'):
    dropout_keep_prob = tf.placeholder(tf.float32)

# Forward Cell for LSTM
with tf.name_scope('lstm_forward'):
    lstm_cell_fw = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_fw, output_keep_prob=dropout_keep_prob)

# Backward Cell for LSTM
with tf.name_scope('lstm_backward'):
    lstm_cell_bw = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_bw,  output_keep_prob=dropout_keep_prob)

# BiDirectional RNN
outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, input_lookup, dtype=tf.float32)
output_rnn = tf.reduce_mean(tf.concat(outputs, axis=2), axis=1)

# Weights and biases
w = tf.get_variable("w", [n_hidden*2, n_classes], initializer=tf.random_normal_initializer(stddev=0.1))
b = tf.get_variable("b", [n_classes])

# RNN prediction
with tf.name_scope("output"):
    scores = tf.matmul(output_rnn, w) + b

# Loss
with tf.variable_scope("loss"):
    losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=scores))
    # l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * 1e-4
    l2_losses = (tf.nn.l2_loss(w) + tf.nn.l2_loss(b)) * 0.05
    loss = losses + l2_losses

# learning rate decay
lr = tf.train.exponential_decay(learning_rate=lr, global_step=global_step_tensor, decay_steps=lr_decay_steps,
                                decay_rate=lr_decay, staircase=False)

with tf.variable_scope("train"):
    optimizer = tf.train.AdamOptimizer(lr)
    optimizer = optimizer.apply_gradients(optimizer.compute_gradients(loss), global_step=global_step_tensor)

with tf.variable_scope("accuracy"):
    correct_scores = tf.equal(tf.argmax(scores, axis=1), tf.argmax(input_y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_scores, tf.float32))

sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
train_iters = len(X_tr) - batch_size
batches = generate_batches(list(zip(X_tr, Y_tr)), batch_size, epochs)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    globalcounter = 1
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        _, tr_loss, acc = sess.run([optimizer, loss, accuracy], feed_dict={input_x: x_batch, input_y: y_batch,
                                                                           dropout_keep_prob: 0.8})
        current_step = tf.train.global_step(sess, global_step_tensor)
        if current_step % 100 == 0:
            print('step:' + str(current_step) + ' Train Loss: ' + str(tr_loss) + ', Train Accuracy: ' + str(acc))

    print("Training complete")
    test_pred = pd.DataFrame()
    test_blocks = test_split(list(np.array(X_te)), 1000)
    for block in test_blocks:
        block = pd.DataFrame(block)
        pred = sess.run(scores, feed_dict={input_x: block, dropout_keep_prob: 1})
        test_pred = test_pred.append(pd.DataFrame(pred))

sess.close()
print("Predictions complete")
submission = pd.read_csv('sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = np.array(test_pred)
submission.to_csv('submission_rnn.csv', index=False)

