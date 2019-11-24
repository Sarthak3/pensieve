"""
This code is for illustration purpose only.
Use multi_agent.py for better performance and speed.
"""

import os
import numpy as np
import tensorflow as tf
import env
import a3c
import load_trace

S_INFO = 8  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end

S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
GRADIENT_BATCH_SIZE = 16
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = None


def decision_tree(loc_last_stall,delay,num_stall,tst):
    rel_loc_stall = loc_last_stall
    delay = delay / M_IN_K
    rel_tst = tst / M_IN_K
    if(rel_loc_stall > 55):
        if(delay < 5):
            if(rel_tst < 2):
                return 4.9
            else:
                return 4.45
        else:
            return 4.05
    else:
        if(num_stall < 4):
            if(rel_tst < 4):
                if(num_stall < 2):
                    return 3.65
                else:
                    return 3.3
            else:
                if(rel_tst < 7):
                    return 3.5
                else:
                    return 2.5
        else:
            return 1.4
def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    with tf.Session() as sess, open(LOG_FILE, 'wb') as log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = 0
        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        actor_gradient_batch = []
        critic_gradient_batch = []

        num_stall = 0
        tst = 0
        loc_last_stall = 0
        STALL_THRES = 4.05
        CONV_TIME = 1200 / (120*M_IN_K*M_IN_K)
        
        ALPHA = 0.8
        thrp = 0
        dtime = 0

        avg_bitrate = 0
        avg_rebuffer = 0
        avg_smooth = 0
        stalls = []
        avg_qoe = 0
        tot_data = 0

        epochs = set()
        while epoch<100:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smooth penalty
            
            avg_bitrate += (VIDEO_BIT_RATE[bit_rate] / M_IN_K)
            avg_rebuffer += (REBUF_PENALTY * rebuf)
            avg_smooth += (SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K) 

            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            avg_qoe += reward
            tot_data = tot_data + 1

            chunk_time = video_chunk_size * CONV_TIME

            if buffer_size < STALL_THRES:
                num_stall = num_stall + 1
                tst += chunk_time
                loc_last_stall = 0
                stalls.append(1)
            else:
                loc_last_stall += chunk_time
                stalls.append(0)

            if(num_stall > 10):
                num_stall = 0
            if(tst > 14000):
                tst = 0

            mos = decision_tree(loc_last_stall,delay,num_stall,tst)
            reward += mos

            r_batch.append(reward)
           

            
            last_bit_rate = bit_rate

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            cthrp = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            thrp = (1.0 - ALPHA) * thrp + ALPHA * cthrp

            cdtime  = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            dtime = (1.0 - ALPHA) * dtime + ALPHA * cdtime 

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = thrp
            state[3, -1] = dtime
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            state[6, -1] = loc_last_stall
            state[7,-1] = tst

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:  # do training once

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(s_batch=np.stack(s_batch[1:], axis=0),  # ignore the first chuck
                                          a_batch=np.vstack(a_batch[1:]),  # since we don't have the
                                          r_batch=np.vstack(r_batch[1:]),  # control over it
                                          terminal=end_of_video, actor=actor, critic=critic)
                td_loss = np.mean(td_batch)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                if(epoch not in epochs):
                    print "Epoch", epoch, " Bitrate ",VIDEO_BIT_RATE[bit_rate], " Avg_reward", np.mean(r_batch)
                    epochs.add(epoch)

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: td_loss,
                    summary_vars[1]: np.mean(r_batch),
                    summary_vars[2]: np.mean(entropy_record)
                })

                writer.add_summary(summary_str, epoch)
                writer.flush()

                entropy_record = []

                if len(actor_gradient_batch) >= GRADIENT_BATCH_SIZE:

                    assert len(actor_gradient_batch) == len(critic_gradient_batch)
                  
                    for i in xrange(len(actor_gradient_batch)):
                        actor.apply_gradients(actor_gradient_batch[i])
                        critic.apply_gradients(critic_gradient_batch[i])

                    actor_gradient_batch = []
                    critic_gradient_batch = []

                    epoch += 1
                    if epoch % MODEL_SAVE_INTERVAL == 0:
                        # Save the neural net parameters to disk.
                        save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                               str(epoch) + ".ckpt")
                        print("Model saved in file: %s" % save_path)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)

            else:
                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)

        avg_bitrate /= tot_data
        avg_rebuffer /= tot_data
        avg_smooth /= tot_data
        avg_stall = np.mean(stalls)
        avg_qoe /= tot_data

        print "Avg bitrate: ",avg_bitrate
        print "avg_rebuffer: ",avg_rebuffer
        print "avg_smooth: ",avg_smooth
        print "avg_stall: ",avg_stall
        print "avg_qoe: ",avg_qoe


if __name__ == '__main__':
    main()
