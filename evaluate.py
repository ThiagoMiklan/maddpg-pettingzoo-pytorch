import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from MADDPG import MADDPG
from main import get_env
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env',
                        choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2'])
    parser.add_argument('folder', type=str, help='name of the folder where model is saved')
    parser.add_argument('--episode-num', type=int, default=10, help='total episode num during evaluation')
    parser.add_argument('--episode-length', type=int, default=50, help='steps per episode')

    args = parser.parse_args()


    #model_dir = os.path.join('./results', args.env_name, args.folder)
    #model_dir criado com abspath para facilitar o uso com debugger do Pycharm.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'results', args.env_name, args.folder)
    assert os.path.exists(model_dir)
    gif_dir = os.path.join(model_dir, 'gif')
    print('GIF DIR', gif_dir)
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif

    env, dim_info = get_env(args.env_name, args.episode_length)
    maddpg = MADDPG.load(dim_info, os.path.join(model_dir, 'model.pt'))

    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent: np.zeros(args.episode_num) for agent in env.agents}
    for episode in range(args.episode_num):
        states = env.reset()
        agent_reward = {agent: 0 for agent in env.agents}  # agent reward of the current episode
        frame_list = []  # used to save gif
        while env.agents:  # interact with the env for an episode
            actions = maddpg.select_action(states)
            #next_states, rewards, dones, infos = env.step(actions)
            next_states, rewards, dones, truncations, infos = env.step(actions)


            render_result = env.render()
            if render_result is not None:  # Verifique se o render não é None
                frame_list.append(Image.fromarray(render_result))

            states = next_states

            for agent_id, reward in rewards.items():  # update reward
                agent_reward[agent_id] += reward

        # Salvar o gif no final do episódio
        message = f'episode {episode + 1}, '
        for agent_id, reward in agent_reward.items():  # episode finishes, record reward
            episode_rewards[agent_id][episode] = reward
            message += f'{agent_id}: {reward:>4f}; '
        print(message)
        # save gif
        if frame_list:
            frame_list[0].save(os.path.join(gif_dir, f'out{gif_num + episode + 1}.gif'),
                               save_all=True, append_images=frame_list[1:], duration=1, loop=0)

    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, rewards in episode_rewards.items():
        ax.plot(x, rewards, label=agent_id)
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    total_files = len([file for file in os.listdir(model_dir)])
    title = f'evaluate result of maddpg solve {args.env_name} {total_files - 3}'
    ax.set_title(title)
    plt.savefig(os.path.join(model_dir, title))

    env.close()  # Agora chamamos o close no final, após o loop de episódios
