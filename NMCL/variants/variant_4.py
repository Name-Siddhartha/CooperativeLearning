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

weight_con = tf.Variable(1.0)
weight_com = tf.Variable(1.0)

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

complementary_value = concat(complementary_V, complementary_A)
complementary_value = concat(complementary_value, complementary_T)

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
distribution_con = gen_distribution(common_value, feature_size)
distribution_com = gen_distribution(complementary_value, feature_size)

#################################################################################################
distribution_X = weight_com*distribution_com + weight_con*distribution_con
preds_con = tf.nn.softmax(distribution_con)
preds_com = tf.nn.softmax(distribution_com)
preds = tf.nn.softmax(distribution_X)

loss_pred_con = tf.nn.softmax_cross_entropy_with_logits(
    labels=input_label, logits=distribution_con)
loss_pred_com = tf.nn.softmax_cross_entropy_with_logits(
    labels=input_label, logits=distribution_com)
loss_pred = tf.nn.softmax_cross_entropy_with_logits(
    labels=input_label, logits=distribution_X)

loss_con = tf.reduce_mean(loss_pred_con)
loss_com = tf.reduce_mean(loss_pred_com)
loss = loss_KL_V+loss_KL_A + loss_KL_T

train_com = tf.train.AdamOptimizer(lr).minimize(loss_com)
train_con = tf.train.AdamOptimizer(lr).minimize(loss_con)
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

                _, _, _, l, l_con, l_com, p_com, p_con, p, th_v, th_a, th_t = sess.run([train_con, train_com, train,
                                                                                        loss,
                                                                                       loss_com, loss_con, preds_com,
                                                                                        preds_con, preds, threshold_V_host,
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
                f1_p_con = np.argmax(p_con, axis=1)
                f1_p_com = np.argmax(p_com, axis=1)
                microF1_com = f1_score(f1_r, f1_p_com, average='micro')
                macroF1_com = f1_score(f1_r, f1_p_com, average='macro')
                microF1_con = f1_score(f1_r, f1_p_con, average='micro')
                macroF1_con = f1_score(f1_r, f1_p_con, average='macro')
                microF1 = f1_score(f1_r, f1_p, average='micro')
                macroF1 = f1_score(f1_r, f1_p, average='macro')

                multi_list = np.array(multi_list)
                if i % 100 == 0:
                    #_ = sess.run(train, feed_dict={input_V:v_input, input_A:a_input,input_T:t_input,input_label:labels,keep_prob:0.5,is_training:True,lr:learning_rate})
                    print('%d-th macroF1 is %f, microF1 is %f' %
                          (i, macroF1, microF1))
                    print('%d-th com_train loss is %f, com_macroF1 is %f, com_microF1 is %f' % (i, l_com, microF1_com,
                                                                                                macroF1_com))
                    print('%d-th con_train loss is %f, con_macroF1 is %f, con_microF1 is %f' % (i, l_con, microF1_con,
                                                                                                macroF1_con))
                    print('%d-th train loss is %f' % (i, l))
                    print(
                        'v-threshold is %f, a-threshold is %f, t-threshold is %f' % (th_v, th_a, th_t))
                    p_evl_list = []
                    com_evl_list = []
                    con_evl_list = []
                    loss_com_evl = loss_con_evl = 0
                    countv = countvi = counta = countai = countt = countti = 0
                    for k in range(500):
                        start = k * 162
                        end = (k + 1) * 162
                        v_test = multi_list[start:end, :visual_size]
                        a_test = multi_list[start:end,
                                            visual_size:visual_size + audio_size]
                        t_test = multi_list[start:end,
                                            visual_size + audio_size:]
                        com_evl, con_evl, p_evl_com, p_evl_con, p_evl, cv, ca, ct = sess.run([loss_com, loss_con, preds_com,
                                                                                             preds_con, preds, count_v, count_a, count_t], feed_dict={input_V: v_test,
                                                                                                                                                      input_A: a_test,
                                                                                                                                                      input_T: t_test,
                                                                                                                                                      input_label: label_list[start:end],
                                                                                                                                                      keep_prob: 1.0,
                                                                                                                                                      is_training: False})
                        p_evl_list.extend(p_evl)
                        com_evl_list.extend(p_evl_com)
                        con_evl_list.extend(p_evl_con)

                        loss_com_evl += com_evl
                        loss_con_evl += con_evl

                        countv += cv
                        counta += ca
                        countt += ct

                    f1_p_evl = np.argmax(p_evl_list, axis=1)
                    f1_com_evl = np.argmax(com_evl_list, axis=1)
                    f1_con_evl = np.argmax(con_evl_list, axis=1)
                    f1_r_evl = np.argmax(label_list[:81000], axis=1)
                    loss_com_evl = loss_com_evl
                    loss_con_evl = loss_con_evl

                    microF1_evl = f1_score(f1_r_evl, f1_p_evl, average='micro')
                    macroF1_evl = f1_score(f1_r_evl, f1_p_evl, average='macro')
                    microF1_com = f1_score(
                        f1_r_evl, f1_com_evl, average='micro')
                    macroF1_com = f1_score(
                        f1_r_evl, f1_com_evl, average='macro')
                    microF1_con = f1_score(
                        f1_r_evl, f1_con_evl, average='micro')
                    macroF1_con = f1_score(
                        f1_r_evl, f1_con_evl, average='macro')
                    print('%d-th evl-macroF1 is %f, evl-microF1 is %f' %
                          (i,  macroF1_evl, microF1_evl))
                    print('%d-th com loss is %f, com-macroF1 is %f, com-microF1 is %f' % (i, loss_com_evl,
                                                                                          macroF1_com, microF1_com))
                    print('%d-th con loss is %f, con-macroF1 is %f, con-microF1 is %f' % (i, loss_con_evl,
                                                                                          macroF1_con, microF1_con))
                    print('v-host consistent is %f, a-host consistent is %f, t-host consistent is %f' % (countv/81000,
                                                                                                         counta/81000, countt/81000))


if __name__ == '__main__':
    train_model()
