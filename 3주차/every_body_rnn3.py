import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
tf.set_random_seed(777)

data=("김예술은 성격파탄자다. "
            "그래도 먹을것을 잘준다"
            "돼지같은 김윤정은 뭉치를 날 안준다."
      "김윤정은 먹을 것을 주면화가 풀린다. 허접한 돼지인거 같다.")

char_set=list(set(data))
char2ix={c:i for i,c in  enumerate(char_set)}
ix2char={i: c for i,c in enumerate(char_set )}
vocab_size=len(char_set)
seq_length=10
learning_rate=0.1
hidden_size=len(char_set)
data_x=[]
data_y=[]

#sampling
for i in range(0,len(data)-seq_length):
    x_str=data[i:i+seq_length]
    y_str=data[i+1:i+seq_length+1]

    x=[char2ix[i] for i in x_str]
    y=[char2ix[i] for i in y_str]

    data_x.append(x)
    data_y.append(y)

batch_size=len(data_x)

print(batch_size)
x=tf.placeholder(tf.int32,shape=[None,seq_length])
y=tf.placeholder(tf.int32,shape=[None,seq_length])

x_one_hot=tf.one_hot(x,vocab_size)

def lstm_cell():
    cell=rnn.BasicLSTMCell(hidden_size,state_is_tuple=True)
    return cell

multi_cells=rnn.MultiRNNCell([lstm_cell() for _ in range(2)],state_is_tuple=True)

outputs,state=tf.nn.dynamic_rnn(multi_cells,x_one_hot,dtype=tf.float32)

# fc layer

x_for_fc=tf.reshape(outputs,[-1,hidden_size])

outputs=tf.contrib.layers.fully_connected(x_for_fc,vocab_size,activation_fn=None)


outputs=tf.reshape(outputs,[batch_size,seq_length,vocab_size])

weights=tf.ones([batch_size,seq_length])

seq_loss=tf.contrib.seq2seq.sequence_loss(logits=outputs,targets=y,weights=weights)
loss=tf.reduce_mean(seq_loss)
train=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)





sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, l, results,fc = sess.run(
        [train,loss, outputs,x_for_fc], feed_dict={x: data_x, y: data_y})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        # print(index)
        print(i, j, ''.join([char_set[t] for t in index]), l)
    #
results = sess.run(outputs, feed_dict={x: data_x})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')
