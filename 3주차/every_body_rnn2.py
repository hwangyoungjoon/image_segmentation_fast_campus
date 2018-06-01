import tensorflow as tf
import numpy as np

sample="if you want me"
idx2char=list(set(sample))
char2idx={c:i for i,c in enumerate(idx2char)}
sample_idx=[char2idx[c] for c in sample]
x_data=[sample_idx[:-1]]
y_data=[sample_idx[1:]]
hidden_size=len(char2idx)

#러닝을 위한 하이퍼 파라미터 설정
num_classes=len(char2idx)  #vocab_size
seq_length=len(sample)-1
batch_size=1
rnn_hidden_size= len(char2idx)



x=tf.placeholder(tf.int32,[None,seq_length])
y=tf.placeholder(tf.int32,[None,seq_length])

x_one_hot=tf.one_hot(x,num_classes)

cell=tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size,state_is_tuple=True)
initial_state=cell.zero_state(batch_size,tf.float32)
outputs,_state=tf.nn.dynamic_rnn(cell,x_one_hot,initial_state=initial_state,dtype=tf.float32)

x_for_fc=tf.reshape(outputs,[-1,hidden_size])
outputs=tf.contrib.layers.fully_connected(x_for_fc,num_classes,activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])

weights=tf.ones([batch_size,seq_length])
seq_loss=tf.contrib.seq2seq.sequence_loss(logits=outputs,targets=y,weights=weights)
loss=tf.reduce_mean(seq_loss)
train=tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(loss)

prediction=tf.argmax(outputs,axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3000):
        l,_=sess.run([loss,train],feed_dict={x:x_data,y:y_data})
        result=sess.run(prediction,feed_dict={x:x_data})
        result_str=[idx2char[c] for c in np.squeeze(result)]
        print(i,"loss:",l,"prediction: ", "".join(result_str))