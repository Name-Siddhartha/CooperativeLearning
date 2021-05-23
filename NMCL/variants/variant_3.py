import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

import h5py

label_size = 188
visual_size = 2048
audio_size = 200
textual_size = 100

feature_size = visual_size + audio_size + textual_size

is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(dtype=tf.float32)
lr = tf.placeholder(tf.float32)


weight1_size = 128


threshold_V_host = tf.nn.sigmoid(tf.Variable(0.0))
threshold_A_host = tf.nn.sigmoid(tf.Variable(0.0))
threshold_T_host = tf.nn.sigmoid(tf.Variable(0.0))

threshold_V_guest = tf.nn.sigmoid(tf.Variable(0.0))
threshold_A_guest = tf.nn.sigmoid(tf.Variable(0.0))
threshold_T_guest = tf.nn.sigmoid(tf.Variable(0.0))

weight_v = tf.Variable(1.0)
weight_a = tf.Variable(1.0)
weight_t = tf.Variable(1.0)
weight_c = tf.Variable(1.0)

input_V = tf.placeholder(shape=[None, visual_size], dtype=tf.float32)
input_A = tf.placeholder(shape=[None, audio_size], dtype=tf.float32)
input_T = tf.placeholder(shape=[None, textual_size], dtype=tf.float32)

input_label = tf.placeholder(shape=[None, label_size], dtype=tf.float32)

batch_size = tf.shape(input_label)[0]


def concat(input_1, input_2):
    input_concat = tf.concat([input_1, input_2], 1)
    return input_concat


def init_xavier(size1, size2):
    xavier_val = 4 * np.sqrt(6.0 / (size1 + size2))
    return tf.random_uniform(shape=[size1, size2], minval=-xavier_val, maxval=xavier_val)


def normalize(input):
    max_value = tf.reduce_max(input, axis=-1)
    min_value = tf.reduce_min(input, axis=-1)
    temp_value = tf.transpose(input, [1, 0])-tf.transpose(min_value)
    output = tf.div(temp_value, tf.transpose(max_value-min_value)+0.00001)
    return tf.transpose(output, [1, 0])


def do_attend(input_tensor, l_size, f_size):
    matrix = tf.Variable(tf.random_normal(
        shape=[l_size, f_size]), dtype=tf.float32)
    att_matrix = tf.nn.softmax(matrix, dim=1)

    exp_input = tf.expand_dims(input_tensor, dim=1)
    ones = tf.ones(shape=[batch_size, label_size, f_size])
    values = ones * exp_input
    values = tf.reshape(values, shape=[batch_size, -1])

    att_matrix_exp = tf.reshape(att_matrix, shape=[1, -1])

    # attention shape=[batch_size, label_size*visual_size]
    attention_values = values * att_matrix_exp
    attention_outputs = tf.reshape(attention_values, shape=[
                                   batch_size, l_size, f_size])
    attention_outputs = tf.reshape(attention_outputs, shape=[-1, f_size])
    return attention_outputs


def gen_relevent(host, guest, host_size, guest_size):
    concate = concat(host, guest)
    weight = tf.Variable(init_xavier(
        size1=host_size+guest_size, size2=host_size), dtype=tf.float32)
    bias = tf.Variable(tf.zeros(shape=[host_size]))
    relevent = tf.matmul(concate, weight)+bias
    relevent = normalize(relevent)
    return relevent


def gen_distribution(input_vector, input_size):
    attention_vector = do_attend(input_vector, label_size, input_size)
    weight1 = tf.Variable(init_xavier(
        size1=input_size, size2=weight1_size), dtype=tf.float32)
    bias1 = tf.Variable(tf.zeros(shape=[weight1_size]))
    value1 = tf.matmul(attention_vector, weight1) + bias1
    value1 = tf.nn.relu(value1)

    weight2 = tf.Variable(init_xavier(
        size1=weight1_size, size2=1), dtype=tf.float32)
    bias2 = tf.Variable(tf.zeros(shape=[1]))
    drop_value1 = tf.nn.dropout(value1, keep_prob=keep_prob)
    value2 = tf.matmul(drop_value1, weight2) + bias2
    value2 = tf.reshape(value2, shape=[batch_size, -1])
    #preds = tf.nn.softmax(value2)
    return value2


def enhance_con(common_value, input_size, out_size):
    weight = tf.Variable(tf.random_normal(
        shape=[input_size, out_size]), dtype=tf.float32)
    bias = tf.Variable(tf.zeros(shape=[out_size]))
    enhance_con = tf.matmul(common_value, weight) + bias
    enhance_con = tf.nn.relu(enhance_con)
    return enhance_con


#######################################common_specific###########################################
guest_V = concat(input_A, input_T)
guest_A = concat(input_V, input_T)
guest_T = concat(input_V, input_A)

relevent_V = gen_relevent(
    input_V, guest_V, visual_size, audio_size+textual_size)
relevent_A = gen_relevent(input_A, guest_A, audio_size,
                          visual_size+textual_size)
relevent_T = gen_relevent(
    input_T, guest_T, textual_size, visual_size+audio_size)


common_V = tf.nn.sigmoid(50*(relevent_V-threshold_V_host))
common_A = tf.nn.sigmoid(50*(relevent_A-threshold_A_host))
common_T = tf.nn.sigmoid(50*(relevent_T-threshold_T_host))

specific_V = 1-common_V
specific_A = 1-common_A
specific_T = 1-common_T

count_v = tf.reduce_sum(tf.round(common_V))
count_a = tf.reduce_sum(tf.round(common_A))
count_t = tf.reduce_sum(tf.round(common_T))

common_value = concat(common_V * input_V, common_A * input_A)
common_value = concat(common_value, common_T * input_T)

complementary_V = specific_V * input_V
complementary_A = specific_A * input_A
complementary_T = specific_T * input_T

##########################################KL#####################################################
con_V = common_V * input_V
con_A = common_A * input_A
con_T = common_T * input_T

pred_V_KL = tf.nn.softmax(gen_distribution(con_V, visual_size))
pred_A_KL = tf.nn.softmax(gen_distribution(con_A, audio_size))
pred_T_KL = tf.nn.softmax(gen_distribution(con_T, textual_size))

loss_KL_V = tf.abs(tf.reduce_mean(
    tf.reduce_sum(-pred_V_KL*tf.log(pred_V_KL/pred_A_KL))))
loss_KL_A = tf.abs(tf.reduce_mean(
    tf.reduce_sum(-pred_A_KL*tf.log(pred_A_KL/pred_T_KL))))
loss_KL_T = tf.abs(tf.reduce_mean(
    tf.reduce_sum(-pred_T_KL*tf.log(pred_T_KL/pred_V_KL))))
loss_KL = loss_KL_V + loss_KL_A + loss_KL_T
#################################################################################################
distribution_c = gen_distribution(common_value, feature_size)
distribution_v = gen_distribution(complementary_V, visual_size)
distribution_a = gen_distribution(complementary_A, audio_size)
distribution_t = gen_distribution(complementary_T, textual_size)

#################################################################################################
distribution_X = weight_v*distribution_v + weight_a * \
    distribution_a + weight_t*distribution_t + weight_c*distribution_c
preds_v = tf.nn.softmax(distribution_v)
preds_a = tf.nn.softmax(distribution_a)
preds_t = tf.nn.softmax(distribution_t)
preds_c = tf.nn.softmax(distribution_c)

preds = tf.nn.softmax(distribution_X)

loss_pred_c = tf.nn.softmax_cross_entropy_with_logits(
    labels=input_label, logits=distribution_c)
loss_pred_v = tf.nn.softmax_cross_entropy_with_logits(
    labels=input_label, logits=distribution_v)
loss_pred_a = tf.nn.softmax_cross_entropy_with_logits(
    labels=input_label, logits=distribution_a)
loss_pred_t = tf.nn.softmax_cross_entropy_with_logits(
    labels=input_label, logits=distribution_t)
loss_pred = tf.nn.softmax_cross_entropy_with_logits(
    labels=input_label, logits=distribution_X)

loss_c = tf.reduce_mean(loss_pred_c)
loss_v = tf.reduce_mean(loss_pred_v)
loss_a = tf.reduce_mean(loss_pred_a)
loss_t = tf.reduce_mean(loss_pred_t)
loss = loss_KL_V+loss_KL_A + loss_KL_T

train_c = tf.train.AdamOptimizer(lr).minimize(loss_c)
train_v = tf.train.AdamOptimizer(lr).minimize(loss_v)
train_a = tf.train.AdamOptimizer(lr).minimize(loss_a)
train_t = tf.train.AdamOptimizer(lr).minimize(loss_t)
train = tf.train.AdamOptimizer(lr).minimize(loss)

saver = tf.train.Saver()
init = tf.global_variables_initializer()


def train_model():
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            train_data_genor = extractData(128,
                                           l_file='../h5files/label_train.h5',
                                           v_file='../h5files/visual_train.h5',
                                           a_file='../h5files/acoustic_train.h5',
                                           t_file='../h5files/textual_train.h5')

            label_list, multi_list = extractMultiData(f_label='../h5files/label_train.h5',
                                                      f_visual='../h5files/visual_train.h5',
                                                      f_audio='../h5files/acoustic_train.h5',
                                                      f_textual='../h5files/textual_train.h5')

            sess.run(init)
            learning_rate = 0.01
            acc_prevous = 0.0
            for i in range(20000):
                features, labels = train_data_genor.next_multiple()
                features = np.array(features)
                v_input = features[:, :visual_size]
                a_input = features[:, visual_size:visual_size + audio_size]
                t_input = features[:, visual_size + audio_size:]

                _, _, _, _, _, l, l_v, l_a, l_t, pv, pa, pt, p, th_v, th_a, th_t = sess.run([train_c, train_v, train_a, train_t, train,
                                                                                            loss,
                                                                                            loss_v, loss_a, loss_t, preds_v, preds_a,
                                                                                             preds_t, preds, threshold_V_host,
                                                                                            threshold_A_host, threshold_T_host], feed_dict={
                    input_V: v_input,
                    input_A: a_input,
                    input_T: t_input,
                    input_label: labels,
                    keep_prob: 0.5,
                    is_training: True,
                    lr: learning_rate})

                f1_p = np.argmax(p, axis=1)
                f1_r = np.argmax(labels, axis=1)
                f1_p_v = np.argmax(pv, axis=1)
                f1_p_a = np.argmax(pa, axis=1)
                f1_p_t = np.argmax(pt, axis=1)
                microF1_v = f1_score(f1_r, f1_p_v, average='micro')
                macroF1_v = f1_score(f1_r, f1_p_v, average='macro')
                microF1_a = f1_score(f1_r, f1_p_a, average='micro')
                macroF1_a = f1_score(f1_r, f1_p_a, average='macro')
                microF1_t = f1_score(f1_r, f1_p_t, average='micro')
                macroF1_t = f1_score(f1_r, f1_p_t, average='macro')
                microF1 = f1_score(f1_r, f1_p, average='micro')
                macroF1 = f1_score(f1_r, f1_p, average='macro')

                multi_list = np.array(multi_list)
                if i % 100 == 0:
                    #_ = sess.run(train, feed_dict={input_V:v_input, input_A:a_input,input_T:t_input,input_label:labels,keep_prob:0.5,is_training:True,lr:learning_rate})
                    print('%d-th macroF1 is %f, microF1 is %f' %
                          (i, macroF1, microF1))
                    print('%d-th v_train loss is %f, v_macroF1 is %f, v_microF1 is %f' %
                          (i, l_v, macroF1_v, microF1_v))
                    print('%d-th a_train loss is %f, a_macroF1 is %f, a_microF1 is %f' %
                          (i, l_a, macroF1_a, microF1_a))
                    print('%d-th t_train loss is %f, t_macroF1 is %f, t_microF1 is %f' %
                          (i, l_t, macroF1_t, microF1_t))
                    print('%d-th train loss is %f' % (i, l))
                    print(
                        'v-threshold is %f, a-threshold is %f, t-threshold is %f' % (th_v, th_a, th_t))
                    p_evl_list = []
                    v_evl_list = []
                    a_evl_list = []
                    t_evl_list = []
                    loss_v_evl = loss_a_evl = loss_t_evl = 0
                    countv = countvi = counta = countai = countt = countti = 0
                    for k in range(500):
                        start = k * 162
                        end = (k + 1) * 162
                        v_test = multi_list[start:end, :visual_size]
                        a_test = multi_list[start:end,
                                            visual_size:visual_size + audio_size]
                        t_test = multi_list[start:end,
                                            visual_size + audio_size:]
                        v_evl, a_evl, t_evl, p_evl_v, p_evl_a, p_evl_t, p_evl, cv, ca, ct = sess.run([loss_v, loss_a,
                                                                                                     loss_t, preds_v, preds_a, preds_t, preds, count_v, count_a, count_t], feed_dict={input_V: v_test,
                                                                                                                                                                                      input_A: a_test,
                                                                                                                                                                                      input_T: t_test,
                                                                                                                                                                                      input_label: label_list[start:end],
                                                                                                                                                                                      keep_prob: 1.0,
                                                                                                                                                                                      is_training: False})
                        p_evl_list.extend(p_evl)
                        v_evl_list.extend(p_evl_v)
                        a_evl_list.extend(p_evl_a)
                        t_evl_list.extend(p_evl_t)
                        loss_v_evl += v_evl
                        loss_a_evl += a_evl
                        loss_t_evl += t_evl
                        countv += cv
                        #countvi += cvi
                        counta += ca
                        #countai += cai
                        countt += ct
                        #countti += cti

                    f1_p_evl = np.argmax(p_evl_list, axis=1)
                    f1_v_evl = np.argmax(v_evl_list, axis=1)
                    f1_a_evl = np.argmax(a_evl_list, axis=1)
                    f1_t_evl = np.argmax(t_evl_list, axis=1)
                    f1_r_evl = np.argmax(label_list[:81000], axis=1)
                    loss_v_evl = loss_v_evl
                    loss_a_evl = loss_a_evl
                    loss_t_evl = loss_t_evl

                    microF1_evl = f1_score(f1_r_evl, f1_p_evl, average='micro')
                    macroF1_evl = f1_score(f1_r_evl, f1_p_evl, average='macro')
                    microF1_v = f1_score(f1_r_evl, f1_v_evl, average='micro')
                    macroF1_v = f1_score(f1_r_evl, f1_v_evl, average='macro')
                    microF1_a = f1_score(f1_r_evl, f1_a_evl, average='micro')
                    macroF1_a = f1_score(f1_r_evl, f1_a_evl, average='macro')
                    microF1_t = f1_score(f1_r_evl, f1_t_evl, average='micro')
                    macroF1_t = f1_score(f1_r_evl, f1_t_evl, average='macro')
                    print('%d-th evl-macroF1 is %f, evl-microF1 is %f' %
                          (i,  macroF1_evl, microF1_evl))
                    print('%d-th v loss is %f, v-macroF1 is %f, v-microF1 is %f' %
                          (i, loss_v_evl, macroF1_v, microF1_v))
                    print('%d-th a loss is %f, a-macroF1 is %f, a-microF1 is %f' %
                          (i, loss_a_evl, macroF1_a, microF1_a))
                    print('%d-th t loss is %f, t-macroF1 is %f, t-microF1 is %f' %
                          (i, loss_t_evl, macroF1_t, microF1_t))
                    print('v-host consistent is %f, a-host consistent is %f, t-host consistent is %f' % (countv/81000,
                                                                                                         counta/81000, countt/81000))


if __name__ == '__main__':
    train_model()
