import tensorflow as tf
import numpy as np
import pprint


idx2char = ['h', 'i', 'e', 'l', 'o']
# Teach hello: hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3
y_data = [[1, 0, 2, 3, 3, 4]]    # ihello

num_classes=5 #vocab_size
hidden_size=5
hidden_layer=2
seq_length=6
batch_size=1
learning_rate=0.1
input_dim=5

x=tf.placeholder(tf.float32,[None,seq_length,input_dim])
y=tf.placeholder(tf.int32,[None,seq_length])

cell=tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True) #셀정의
initial_state=cell.zero_state(batch_size,dtype=tf.float32) # 초기 스테이트
outputs,_state=tf.nn.dynamic_rnn(cell,x,initial_state=initial_state,dtype=tf.float32) #아웃풋과 다음 히든으로 보내줄 스테이트

#fully_connected layer
x_for_fc=tf.reshape(outputs,[-1,hidden_size])
output=tf.contrib.layers.fully_connected(inputs=x_for_fc,num_outputs=num_classes,activation_fn=None)

#reshape
output=tf.reshape(outputs,[batch_size,seq_length,num_classes])
weights=tf.ones([batch_size,seq_length])
seq_loss=tf.contrib.seq2seq.sequence_loss(logits=output,targets=y,weights=weights)


loss=tf.reduce_mean(seq_loss)
train=tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)

prediction=tf.argmax(output,axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l,_=sess.run([loss,train],feed_dict={x:x_one_hot,y:y_data})
        result=sess.run(prediction,feed_dict={x:x_one_hot})
        print(i,"loss:",l,"prediction:",result,"true y:",y_data)

        result_str=[idx2char[c] for c in np.squeeze(result)]
        print("\tprediction str:",''.join(result_str))
