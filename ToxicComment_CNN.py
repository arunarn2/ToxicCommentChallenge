# Convolutional Neural Networks for Sentence Classification
# https://arxiv.org/abs/1408.5882
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', './cnn', 'Summaries directory')


# For Test data. Can use generate_batch function.
def test_split(data, block_size):
    data = np.array(data)
    data_size = len(data)
    nums = int((data_size-1)/block_size) + 1
    for block_num in range(nums):
        start_index = block_num * block_size
        end_index = min((block_num + 1) * block_size, data_size)
        yield data[start_index:end_index]


def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def get_coefs(word1, *arr):
    return word1, np.asarray(arr, dtype='float32')


# prevents multiple graphs warning in tensorboard
if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
tf.gfile.MakeDirs(FLAGS.summaries_dir)

sess = tf.InteractiveSession()
embed_size = 300        # size of each word vector
max_features = 100000   # number of rows in the embedding vector (unique words)
maxlen = 170            # max words in a comment; pad if not this length
batch_size = 256        # batch size for processing
epochs = 25             # number of epochs
num_filters = 32        # filters in the
num_classes = 6         # number of output classes
filter_sizes = [1, 2, 3, 4, 5]

num_filters_total = num_filters * len(filter_sizes)

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
with tf.name_scope('input'):
    input_x = tf.placeholder(tf.int32, [None, maxlen], name="input_x")
    input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

with tf.name_scope('dropout'):
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

with tf.name_scope('embedding'):
    embedding = tf.get_variable("embedding", shape=[max_features, embed_size])
    embed_lookup = tf.nn.embedding_lookup(embedding, input_x)
    embedded_chars_expanded = tf.expand_dims(embed_lookup, -1)

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    filter_shape = [filter_size, embed_size, 1, num_filters]

    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="w")
    b = tf.Variable(tf.truncated_normal([num_filters], stddev=0.05), name="b")

    conv = tf.nn.conv2d(embedded_chars_expanded, w, strides=[1, 1, 1, 1], padding="VALID", name="conv")
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    pooled = tf.nn.max_pool(h, ksize=[1, maxlen - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1], padding="VALID", name="pool")
    pooled_outputs.append(pooled)

with tf.name_scope('pooled_output'):
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

# In the first dense layer, reduce the node to half.
with tf.name_scope('FC1'):
    w_fc1 = tf.Variable(tf.truncated_normal([num_filters_total, int(num_filters_total / 2)], stddev=0.05), name="w_fc1")
    b_fc1 = tf.Variable(tf.truncated_normal([int(num_filters_total / 2)], stddev=0.05), name="b_fc1")
    layer1 = tf.nn.xw_plus_b(h_drop, w_fc1, b_fc1, name='fc1')
    layer1 = tf.nn.relu(layer1)

# Second dense layer, reduce the outputs to 6.
with tf.name_scope('FC2'):
    w_fc2 = tf.Variable(tf.truncated_normal([int(num_filters_total / 2), 6], stddev=0.05), name='w_fc2')
    b_fc2 = tf.Variable(tf.truncated_normal([6], stddev=0.05), name="b_fc2")
    layer2 = tf.nn.xw_plus_b(layer1, w_fc2, b_fc2, name='fc2')

with tf.name_scope('output'):
    prediction = tf.nn.sigmoid(layer2)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=layer2, labels=input_y))

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0007).minimize(loss)

# Calculate Accuracy
with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(prediction), input_y), tf.float32))

init = tf.global_variables_initializer()
train_iters = len(X_tr) - batch_size

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

with tf.Session() as sess:
    sess.run(init)
    globalcounter = 1
    for epoch in range(epochs):
        step = 0
        avg_acc = 0
        avg_loss = 0
        while step * batch_size < train_iters:
            x_batch = X_tr[step * batch_size:(step + 1) * batch_size]
            y_batch = Y_tr[step * batch_size:(step + 1) * batch_size]

            _, tr_loss, acc = sess.run([optimizer, loss, accuracy],
                                       feed_dict={input_x: x_batch, input_y: y_batch, dropout_keep_prob: 0.9})
            avg_loss += tr_loss
            avg_acc += acc
            step += 1
            # train_writer.add_summary(summary, globalcounter)
            globalcounter += 1
        avg_loss = avg_loss / (int(train_iters / batch_size) + 1)
        avg_acc = avg_acc / (int(train_iters / batch_size) + 1)
        print('Epoch:' + str(epoch) + ' Train Loss: ' + str(avg_loss) + ', Train Accuracy: ' + str(avg_acc))

    print("Training complete")
    test_pred = pd.DataFrame()
    test_blocks = test_split(list(np.array(X_te)), 1000)
    for block in test_blocks:
        block = pd.DataFrame(block)
        pred = sess.run(prediction, feed_dict={input_x: block, dropout_keep_prob: 1})
        test_pred = test_pred.append(pd.DataFrame(pred))
sess.close()
print("Predictions complete")
submission = pd.read_csv('sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = np.array(test_pred)
submission.to_csv('submission_cnn.csv', index=False)
