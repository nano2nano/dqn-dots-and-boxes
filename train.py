import datetime
import os

import tensorflow as tf

from agent import Agent
from dots_and_boxes import DotsAndBoxes
from model import create_model, load_my_model
from per_rank_base_memory import PERRankBaseMemory
from send_result2line import send_result2line
from vs_random import vs_random

if __name__ == "__main__":
    ############################### for my environment ###############################
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    ###################################################################################

    LOAD_PARAMS = True
    EPISODE_NUM = 1000000
    MEMORY_SIZE = 10000

    PER_ALPHA = 0.6
    MEMORY_FILE_PATH = './params/memory_dmp'
    MODEL_FILE_PATH = './params/model.h5'
    WEIGHTS_FILE_PATH = './params/latest_weights.h5'
    HIGHT_SCORE_DIRECTORY_PATH = './hight_score_params/'

    GAMMA = 0.99
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 32
    EPSILON = 0.01

    env = DotsAndBoxes()

    if LOAD_PARAMS:
        main = load_my_model(MODEL_FILE_PATH, WEIGHTS_FILE_PATH)
        target = load_my_model(MODEL_FILE_PATH, WEIGHTS_FILE_PATH)
        memory = PERRankBaseMemory(
            capacity=MEMORY_SIZE, alpha=PER_ALPHA, memory_path=MEMORY_FILE_PATH)
    else:
        main = create_model(state_size=env.state_size,
                            action_size=env.action_size, learning_rate=LEARNING_RATE)
        target = create_model(state_size=env.state_size,
                              action_size=env.action_size, learning_rate=LEARNING_RATE)
        memory = PERRankBaseMemory(
            capacity=MEMORY_SIZE, alpha=PER_ALPHA)

    agent = Agent(main=main, target=target, memory=memory,
                  batch_size=BATCH_SIZE, gamma=GAMMA, epsilon=EPSILON)

    for episode in range(EPISODE_NUM):
        state, _, done, _ = env.reset()
        step = 0
        episode_td_error = list()
        next_valid_actions = env.available_actions.copy()
        next_invalid_actions = env.invalid_actions.copy()
        while not done:
            step += 1
            turn = env.turn
            valid_actions = next_valid_actions
            invalid_actions = next_invalid_actions
            action = agent.get_action(state, turn, valid_actions)
            next_state, reward, done, _ = env.step(action)
            next_turn = env.turn
            next_valid_actions = env.available_actions.copy()
            next_invalid_actions = env.invalid_actions.copy()
            agent.memory.add((state, turn, action, reward, next_state, next_turn,
                              valid_actions, invalid_actions, next_valid_actions, next_invalid_actions))

            state = next_state

            batch_td_error = agent.replay()
            if batch_td_error != 0:
                episode_td_error.append(batch_td_error)
        # finish episode
        agent.update_target_weights()
        if len(episode_td_error) != 0:
            print(
                "--------------------- %d Episode finished---------------------" % (episode))
            print("Finish turn：", step)
            print('average td_error: ', (sum(episode_td_error) /
                                         len(episode_td_error))/BATCH_SIZE)
            print("----------------------------------------------------------")
        if (episode+1) % 50 == 0:
            win, lose, draw, _, _, _ = vs_random(
                agent, battle_num=100)
            winning_percentage = win/(win+lose+draw) * 100
            print("win : {}, lose : {}, draw : {}, WP : {}".format(
                win, lose, draw, winning_percentage))
            send_result2line(win, lose, draw, winning_percentage)
            agent.save_params(
                MODEL_FILE_PATH, WEIGHTS_FILE_PATH, MEMORY_FILE_PATH)
            if winning_percentage > 95:
                date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
                base_name = date+str(int(winning_percentage))
                base_name = os.path.join(HIGHT_SCORE_DIRECTORY_PATH, base_name)
                agent.save_params(model_file_path=base_name+'_model.h5', weights_file_path=base_name +
                                  '_weights.h5', memory_file_path=base_name+'_memory_dmp')
