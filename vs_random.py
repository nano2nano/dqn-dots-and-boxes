import random

from dots_and_boxes import DotsAndBoxes


def vs_random(agent, battle_num=1000):
    win = 0
    lose = 0
    drow = 0
    step_cnt = 0
    max_step = 0
    min_step = 59
    env = DotsAndBoxes()
    for _ in range(battle_num):
        step = 0
        next_state, _, done, _ = env.reset()
        agent_turn = random.choice([-1, 1])

        while not done:
            step += 1
            state = next_state
            # choice action
            if env.turn == agent_turn:
                # by agent
                action = agent.get_best_action(
                    state, env.turn, sorted(list(env.available_actions)))
            else:
                # by random
                action = agent.get_random_action(
                    sorted(list(env.available_actions)))
            # proceed to next step by the action
            next_state, _, done, _ = env.step(action)
        # finish episode
        step_cnt += step
        if step > max_step:
            max_step = step
        if step < min_step:
            min_step = step
        if env.winner == agent_turn:
            win += 1
        elif env.winner == -1*agent_turn:
            lose += 1
        elif env.winner == 0:
            drow += 1
    avg_step = step_cnt/battle_num
    return win, lose, drow, max_step, min_step, avg_step


if __name__ == "__main__":
    from agent import Agent
    from model import load_my_model

    MODEL_FILE_PATH = './params/model.h5'
    WEIGHTS_FILE_PATH = './params/latest_weights.h5'

    env = DotsAndBoxes()
    main = load_my_model(MODEL_FILE_PATH, WEIGHTS_FILE_PATH)
    agent = Agent(main=main, target=None, memory=None,
                  batch_size=1, gamma=1, epsilon=1)
    win, lose, drow, max_turn, min_turn, avg_turn = vs_random(
        agent, battle_num=100)
    print("戦績: ", win, "勝", lose, "敗")
    print("最大ターン数: {}, 最小ターン数: {}, 平均ターン数: {}".format(
        max_turn, min_turn, avg_turn))
