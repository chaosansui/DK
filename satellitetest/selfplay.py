import numpy as np
import os
import argparse
from tqdm import tqdm
import copy
import nashpy
from satellitetest.satellite_env import SatelliteEnv
from agents import Agent as DDPGAgent
import logging

logging.basicConfig(
    filename='app.log',
    filemode='w',  # 'w'表示覆盖写，'a'表示追加写
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def set_seed(seed):
    if seed is not None:
        import subprocess
        import sys

        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
        if 'torch' in installed_packages:
            import torch
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)


def get_device():
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    return device


def empty_list_generator(num_dimensions):
    result = []
    for _ in range(num_dimensions - 1):
        result = [result]
    return result


class PSRO:
    def __init__(self):
        self.Agent_1 = None
        self.Agent_0 = None
        self.state_dim = 21
        self.action_dim = 3
        self.eval_env = None
        self.env = None
        self._meta_strategy = None
        self._meta_payoff = None
        self._policies = None
        self._new_policies = None
        self._iterations = None
        self._num_players = 2  # 玩家数目
        self.device = get_device()
        set_seed(args.seed)

    def estimate_policies(self, estimated_policies):
        # (1) 加载两个智能体对应的模型
        agent_blue, agent_redacc = None, None
        if args.algorithm == 'dqn':
            # TODO 加载模型
            pass
        elif args.algorithm == 'ppo':
            # TODO 加载模型
            pass
        elif args.algorithm == 'ddpg':
            agent_blue = DDPGAgent.from_checkpoint(checkpoint=estimated_policies[0])
            agent_redacc = DDPGAgent.from_checkpoint(checkpoint=estimated_policies[1])

        # (2) 设置双方并进行评估
        # TODO 设置对手
        # TODO 评估
        result = None

        return result

    def train_oracle(self, meta_strategy):
        # Agent_0
        for episode in tqdm(range(args.num_episodes)):
            # (1) 加载对手
            agent_redacc = None
            sample_prob = np.abs(np.array(list(meta_strategy[1]))) / np.sum(
                np.abs(np.array(list(meta_strategy[1]))))  # 采样概率
            sample_policy_idx = np.random.choice(np.arange(len(meta_strategy[1])), p=sample_prob)
            if args.algorithm == 'dqn':
                # TODO 加载模型
                pass
            elif args.algorithm == 'ppo':
                # TODO 加载模型
                pass
            elif args.algorithm == 'ddpg':
                agent_redacc = DDPGAgent.from_checkpoint(checkpoint=self._policies[1][sample_policy_idx])

            # (2) 设置对手并进行训练（一个episode）
            episode_reward_blue = 0
            episode_reward_redacc = 0
            epsilon = max(0.1, 1.0 - episode / args.num_episodes)

            state = self.env.reset()
            for step in range(args.max_steps):
                action_blue = self.Agent_0.get_action(state, noise_scale=epsilon)
                action_redacc = agent_redacc.get_action(state, noise_scale=epsilon)
                actions = np.concatenate([action_blue, action_redacc])

                # 与环境交互
                next_state, rewards, done, _ = self.env.step(actions)
                reward_blue, reward_redacc = rewards
                done = np.any(done)

                # 将经验添加到重放缓冲区中
                self.Agent_0.replay_buffer.add(state, action_blue, reward_blue, next_state, done)

                # 训练智能体
                self.Agent_0.train(args.batch_size)

                state = next_state
                episode_reward_blue += reward_blue
                episode_reward_redacc += reward_redacc

                if done:
                    break

            logging.info(
                f"Agent Red Train Episode {episode}, Red Reward: {episode_reward_blue}, Blueacc Reward: {episode_reward_redacc}")

            # # 记录每个episode的总奖励
            # rewards_blue.append(episode_reward_blue)
            # rewards_redacc.append(episode_reward_redacc)

        # Agent_1
        for episode in tqdm(range(args.num_episodes)):
            # (1) 加载对手
            agent_blue = None
            sample_prob = np.abs(np.array(list(meta_strategy[0]))) / np.sum(
                np.abs(np.array(list(meta_strategy[0]))))  # 采样概率
            sample_policy_idx = np.random.choice(np.arange(len(meta_strategy[0])), p=sample_prob)
            if args.algorithm == 'dqn':
                # TODO 加载模型
                pass
            elif args.algorithm == 'ppo':
                # TODO 加载模型
                pass
            elif args.algorithm == 'ddpg':
                agent_blue = DDPGAgent.from_checkpoint(checkpoint=self._policies[0][sample_policy_idx])

            # (2) 设置对手并进行训练（一个episode）
            episode_reward_blue = 0
            episode_reward_redacc = 0
            epsilon = max(0.1, 1.0 - episode / args.num_episodes)

            state = self.env.reset()
            for step in range(args.max_steps):
                action_blue = agent_blue.get_action(state, noise_scale=epsilon)
                action_redacc = self.Agent_1.get_action(state, noise_scale=epsilon)
                actions = np.concatenate([action_blue, action_redacc])

                # 与环境交互
                next_state, rewards, done, _ = self.env.step(actions)
                reward_blue, reward_redacc = rewards
                done = np.any(done)

                # 将经验添加到重放缓冲区中
                self.Agent_1.replay_buffer.add(state, action_blue, reward_blue, next_state, done)

                # 训练智能体
                self.Agent_1.train(args.batch_size)

                state = next_state
                episode_reward_blue += reward_blue
                episode_reward_redacc += reward_redacc

                if done:
                    break

            logging.info(
                f"Agent Blueacc Train Episode {episode}, Red Reward: {episode_reward_blue}, Blueacc Reward: {episode_reward_redacc}")

            # # 记录每个episode的总奖励
            # rewards_blue.append(episode_reward_blue)
            # rewards_redacc.append(episode_reward_redacc)

    def init(self):
        # (1) 创建训练环境 (env) 和评估环境 (eval_env)
        self.env = SatelliteEnv()
        self.eval_env = SatelliteEnv()

        # (2) 创建智能体
        if args.algorithm == 'dqn':
            # TODO 创建新智能体
            pass
        elif args.algorithm == 'ppo':
            # TODO 创建新智能体
            pass
        elif args.algorithm == 'ddpg':
            self.Agent_0 = DDPGAgent(self.state_dim, self.action_dim)  # Agent_blue
            self.Agent_1 = DDPGAgent(self.state_dim, self.action_dim)  # Agent_red

        # (3) 初始化其他参数
        self._iterations = 0  # 迭代次数
        self._initialize_policy()  # 初始化策略
        self._initialize_game_state()  # 初始化游戏状态
        self.update_meta_strategies()  # 获得meta-strategy

    def _initialize_policy(self):
        """
            初始化策略集合
        """
        self._policies = [[] for _ in range(self._num_players)]  # 原策略集合
        self._new_policies = [[copy.deepcopy(self.Agent_0.checkpoint_attributes())],
                              [copy.deepcopy(self.Agent_1.checkpoint_attributes())], ]  # 新增策略集合

    def _initialize_game_state(self):
        """
            初始化meta_payoff并合并策略集合
        """
        self._meta_payoff = [np.array(empty_list_generator(self._num_players)) for _ in range(self._num_players)]
        self.update_empirical_gamestate()

    def update_empirical_gamestate(self):
        """
            增加新智能体，并更新游戏矩阵
        """
        # (1) 策略合并
        updated_policies = [self._policies[i] + self._new_policies[i] for i in range(self._num_players)]
        total_number_policies = len(updated_policies[0])  # △假设：这里假设参与博弈的所有玩家的策略数目相同
        number_older_policies = len(self._policies[0])  # △假设：这里假设参与博弈的所有玩家的策略数目相同

        if args.sp_type == 'psro':
            # (2) 创建新的meta-payoff，并将原meta-payoff填充进去
            meta_payoff = []
            # 玩家0
            meta_payoff_0 = np.full(tuple([total_number_policies, total_number_policies]), np.nan)
            older_policies_slice_0 = tuple([slice(number_older_policies) for _ in range(self._num_players)])
            meta_payoff_0[older_policies_slice_0] = self._meta_payoff[0]
            meta_payoff.append(meta_payoff_0)
            # 玩家1
            meta_payoff_1 = np.full(tuple([total_number_policies, total_number_policies]), np.nan)
            older_policies_slice_1 = tuple([slice(number_older_policies) for _ in range(self._num_players)])
            meta_payoff_1[older_policies_slice_1] = self._meta_payoff[1]
            meta_payoff.append(meta_payoff_1)

            # (3) 填充其他元素
            for current_index in range(total_number_policies):
                index_tuple_0 = (number_older_policies, current_index)
                index_tuple_1 = (current_index, number_older_policies)

                if index_tuple_0[0] == index_tuple_0[1]:
                    # 位置0和1对应的评估效果
                    estimated_policies = [updated_policies[0][index_tuple_0[0]],
                                          updated_policies[1][index_tuple_0[1]]]
                    winning_rate_0 = self.estimate_policies(estimated_policies)
                    winning_rate_1 = -winning_rate_0
                    # 效果
                    meta_payoff[0][index_tuple_0] = winning_rate_0
                    meta_payoff[1][index_tuple_0] = winning_rate_1
                else:
                    # 位置0对应的评估效果
                    estimated_policies = [updated_policies[0][index_tuple_0[0]],
                                          updated_policies[1][index_tuple_0[1]]]
                    winning_rate_0_0 = self.estimate_policies(estimated_policies)
                    winning_rate_1_0 = -winning_rate_0_0
                    # 位置1对应的评估效果
                    estimated_policies = [updated_policies[0][index_tuple_1[0]],
                                          updated_policies[1][index_tuple_1[1]]]
                    winning_rate_0_1 = self.estimate_policies(estimated_policies)
                    winning_rate_1_1 = -winning_rate_0_1
                    # 效果
                    meta_payoff[0][index_tuple_0] = winning_rate_0_0
                    meta_payoff[0][index_tuple_1] = winning_rate_0_1
                    meta_payoff[1][index_tuple_0] = winning_rate_1_0
                    meta_payoff[1][index_tuple_1] = winning_rate_1_1

        # (4) 更新
        if args.sp_type == 'psro':
            self._meta_payoff = meta_payoff
        self._policies = updated_policies

        return self._meta_payoff

    def update_agents(self):
        """
            训练新智能体
        """
        meta_strategy = self._meta_strategy  # 元策略
        self.train_oracle(meta_strategy)  # 智能体训练
        self._new_policies = [[copy.deepcopy(self.Agent_0.checkpoint_attributes())],
                              [copy.deepcopy(self.Agent_1.checkpoint_attributes())]]

    def update_meta_strategies(self, ):
        """
            更新元博弈策略
        """
        if args.sp_type == 'psro':
            zero_sum_matrix = nashpy.Game(self._meta_payoff[0])
            p0_sol, p1_sol = zero_sum_matrix.lemke_howson(initial_dropped_label=0)
            self._meta_strategy = [list(p0_sol), list(p1_sol)]
        elif args.sp_type == 'self_play':
            meta_strategy_0 = [0.0] * (len(self._policies[0]) - 1) + [1.0]
            meta_strategy_1 = [0.0] * (len(self._policies[1]) - 1) + [1.0]
            self._meta_strategy = [meta_strategy_0, meta_strategy_1]
        elif args.sp_type == 'fictious_play':
            values = [1.0] * len(self._policies[0])
            total_sum = sum(values)
            meta_strategy_0 = [value / total_sum for value in values]
            meta_strategy_1 = [value / total_sum for value in values]
            self._meta_strategy = [meta_strategy_0, meta_strategy_1]
        else:
            raise ValueError("This type of Selfplay method does not exist")

    def iteration(self):
        """
            自博弈迭代
        """
        self._iterations += 1  # 迭代次数加一

        self.update_agents()  # (1) BR求解 强化学习算法

        self.update_empirical_gamestate()  # (2) 更新游戏矩阵
        print(self._meta_payoff)

        self.update_meta_strategies()  # (3) 下一个iteration的对手选择
        print(self._meta_strategy)

        self.Agent_0.save_checkpoint(path=args.log_dir, filename='Agent_0_' + str(self._iterations) + '.pt')
        self.Agent_1.save_checkpoint(path=args.log_dir, filename='Agent_1_' + str(self._iterations) + '.pt')

    def train_psro(self, psro_loop):
        for i in range(psro_loop):
            print('')
            print('----------------------------------------')
            print('  SELFPLAY_LOOP      |  ' + str(i))
            print('----------------------------------------')
            self.iteration()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Selfplay Example for DK")
    parser.add_argument(
        '--algorithm',
        type=str,
        default='ddpg',
        choices=[
            'dqn',
            'ppo',
            'ddpg'
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=200,
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=50,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/selfplay/',
    )
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default='',
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10000
    )
    parser.add_argument(
        '--sp_type',
        type=str,
        default='self_play',
        choices=[
            'self_play',
            'fictious_play',
            'psro'
        ],
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    psro = PSRO()
    psro.init()
    psro.train_psro(10)
