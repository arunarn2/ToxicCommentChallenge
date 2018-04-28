import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from keras.preprocessing import sequence


# Define a function to read the Pre-trained Word Embedding in to a dictionary.
def get_coefs(w1, *arr):
    return w1, np.asarray(arr, dtype='float32')


# Define batch generation function.
def generate_batch(data, batchsize, nepochs, shuffle=True, train=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batchsize) + 1
    if train:
        index = 0
        for epoch in range(nepochs):
            index += 1
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batchsize
                end_index = min((batch_num + 1) * batchsize, data_size)
                yield shuffled_data[start_index:end_index]
    else:
        for block_num in range(num_batches_per_epoch):
            if block_num == 0:
                print("prediction start!")
            start_index = block_num * batchsize
            end_index = min((block_num + 1) * batchsize, data_size)
            yield data[start_index:end_index]


train = pd.read_csv('./train.csv').fillna('fillna')
test = pd.read_csv('./test.csv').fillna('fillna')
x_train = train['comment_text']
x_test = test['comment_text']
y_train = train[['toxic', 'severe_toxic', "obscene", "threat", "insult", "identity_hate"]]

filter_sizes = [1, 2, 3, 4, 5]
num_filters = 32
batch_size = 256
num_filters_total = num_filters * len(filter_sizes)
embedding_size = 300
num_epochs = 10
dropout_keep_prob = 0.9
maxlen = 170

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('glove.840B.300d.txt'))
del embeddings_index['2000000']  # The first row of the file is useless, so delete it.

lst = []
for line in x_train:
    lst += line.split()

counter = Counter(lst)
for k in list(counter.keys()):
    if k not in embeddings_index:
        del counter[k]
counter = dict(sorted(counter.items(), key=lambda x: -x[1]))
counter = {key: value for (key, value) in counter.items() if value >= 2}
counter = dict(zip(list(counter.keys()), range(1, len(counter) + 1)))

embedding_matrix = {}
for key in counter:
    embedding_matrix[key] = embeddings_index[key]

# Create the word embedding matrix where the first element is all zeros
# which is for word that is not shown and the padding elements.
W = np.zeros((1, embedding_size))
W = np.append(W, np.array(list(embedding_matrix.values())), axis=0).astype(np.float32, copy=False)

lst = []
for line in x_test:
    lst += line.split()

counter_test = Counter(lst)
for k in list(counter_test.keys()):
    if k not in embedding_matrix:
        del counter_test[k]
    else:
        counter_test[k] = counter[k]

# Make the train dataset to be a sequence of ids of words.
for i in range(len(x_train)):
    temp = x_train[i].split()
    for word in temp[:]:
        if word not in counter:
            temp.remove(word)
    for j in range(len(temp)):
        temp[j] = counter[temp[j]]
    x_train[i] = temp

for i in range(len(x_test)):
    temp = x_test[i].split()
    for word in temp[:]:
        if word not in counter_test:
            temp.remove(word)
    for j in range(len(temp)):
        temp[j] = counter_test[temp[j]]
    x_test[i] = temp

train_x = sequence.pad_sequences(list(x_train), maxlen=maxlen)
test_x = sequence.pad_sequences(list(x_test), maxlen=maxlen)
del embeddings_index, lst, x_train, x_test

# Define convolutional layers
with tf.name_scope('input'):
    input_x = tf.placeholder(tf.int32, [None, maxlen], name="input_x")
    input_y = tf.placeholder(tf.float32, [None, 6], name="input_y")

with tf.name_scope('embedding'):
    embedded_chars = tf.nn.embedding_lookup(W, input_x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    filter_shape = [filter_size, embedding_size, 1, num_filters]

    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="w")
    b = tf.Variable(tf.truncated_normal([num_filters], stddev=0.05), name="b")

    conv = tf.nn.conv2d(embedded_chars_expanded, w, strides=[1, 1, 1, 1], padding="VALID", name="conv")
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    pooled = tf.nn.max_pool(h, ksize=[1, maxlen - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1], padding="VALID", name="pool")

    pooled_outputs.append(pooled)

with tf.name_scope('pooling'):
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

with tf.name_scope('dropout'):
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

# In the first fulluy connected layer, reduce the node to half the input size
with tf.name_scope('FC1'):
    w_fc1 = tf.Variable(tf.truncated_normal([num_filters_total, int(num_filters_total / 2)], stddev=0.05), name="w_fc1")
    bd1 = tf.Variable(tf.truncated_normal([int(num_filters_total / 2)], stddev=0.05), name="bd1")
    layer1 = tf.nn.xw_plus_b(h_drop, w_fc1, bd1, name='FC1')
    layer1 = tf.nn.relu(layer1)

# Second fully connected layer, reduce the outputs to 6.
with tf.name_scope('FC2'):
    w_fc2 = tf.Variable(tf.truncated_normal([int(num_filters_total / 2), 6], stddev=0.05), name='w_fc2')
    b_fc2 = tf.Variable(tf.truncated_normal([6], stddev=0.05), name="b_fc2")
    layer2 = tf.nn.xw_plus_b(layer1, w_fc2, b_fc2, name='FC2')

with tf.name_scope('output'):
    prediction = tf.nn.sigmoid(layer2)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=layer2, labels=input_y))

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(prediction), input_y), tf.float32))


batch1 = generate_batch(list(
    zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'],
        y_train['insult'], y_train['identity_hate'])), batch_size, 1)

batch2 = generate_batch(list(
    zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'],
        y_train['insult'], y_train['identity_hate'])), batch_size, 1)

batch3 = generate_batch(list(
    zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'],
        y_train['insult'], y_train['identity_hate'])), batch_size, 1)

batch4 = generate_batch(list(
    zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'],
        y_train['insult'], y_train['identity_hate'])), batch_size, 1)

batch5 = generate_batch(list(
    zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'],
        y_train['insult'], y_train['identity_hate'])), batch_size, 1)

batch6 = generate_batch(list(
    zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'],
        y_train['insult'], y_train['identity_hate'])), batch_size, 1)

batch7 = generate_batch(list(
    zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'],
        y_train['insult'], y_train['identity_hate'])), batch_size, 1)

batch8 = generate_batch(list(
    zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'],
        y_train['insult'], y_train['identity_hate'])), batch_size, 1)

batch9 = generate_batch(list(
    zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'],
        y_train['insult'], y_train['identity_hate'])), batch_size, 1)

batch10 = generate_batch(list(
    zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'],
        y_train['insult'], y_train['identity_hate'])), batch_size, 1)

listofbatches = [batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8, batch9, batch10]
test_blocks = generate_batch(list(np.array(test_x)), 1000, 0, shuffle=False, train=False)
total = int((train_x.shape[0] - 1) / batch_size) + 1
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    for batches in listofbatches:
        i += 1
        avg_acc = 0
        avg_loss = 0
        for batch in batches:
            batch = pd.DataFrame(batch, columns=['a', 'b', 'c', 'd', 'e', 'g', 'f'])
            x_batch = pd.DataFrame(list(batch['a']))
            y_batch = batch.loc[:, batch.columns != 'a']
            _, l, acc = sess.run([optimizer, loss, accuracy], feed_dict={input_x: x_batch, input_y: y_batch})
            avg_loss += l
            avg_acc += acc
        avg_loss = avg_loss / total
        avg_acc = avg_acc / total
        print('Epoch: ' + str(i) + ' - Loss:' + str(avg_loss) + '; Train Accuracy: ' + str(avg_acc))
    print('Training completed')

    df = pd.DataFrame()
    for block in test_blocks:
        block = pd.DataFrame(block)
        pred = sess.run(prediction, feed_dict={input_x: block})
        df = df.append(pd.DataFrame(pred))

    print('Prediction completed')
    submission = pd.read_csv('sample_submission.csv')
    submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = np.array(df)
    submission.to_csv('submission_CNN.csv', index=False)
