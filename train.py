import argparse
import importlib
import logging
import multiprocessing
import os
import sys
import time
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from utils import *

# Using TF 2.0, one solution is to switch off eager execution,
# which is what prevents multi_gpu_model from working.
# This can be done as follows
# tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_name', action="store", dest="model_name", default='DQN', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2010-2015', help="stock name")
parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int,
                    help="span (days) of observation")
parser.add_argument('--num_episode', action="store", dest="num_episode", default=10, type=int, help='episode number')
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int,
                    help='initial balance')
parser.add_argument('--volume', action="store", dest="volume", default=False, type=bool, help='is Volume Considered')
inputs = parser.parse_args()

model_name = inputs.model_name
stock_name = inputs.stock_name
window_size = inputs.window_size
num_episode = inputs.num_episode
initial_balance = inputs.initial_balance
volume_considered = inputs.volume

stock_prices = stock_close_prices(stock_name)
stock_volumes = get_stock_volumes(stock_name)
trading_period = len(stock_prices) - 1
returns_across_episodes = []
num_experience_replay = 0
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

# select learning model
model = importlib.import_module(f'agents.{model_name}')
# agent = model.Agent(state_dim=window_size + 3, balance=initial_balance, sess=session)
agent = model.Agent(state_dim=window_size + 3, balance=initial_balance)


def config_hard_GPU(GPU=True, GPU_mem_use=0.5):
    """
    Configuration of CPU and GPU, Hardware Parameters
    GPU = True  # Selection between CPU or GPU
    GPU_mem_use = 0.5  # In both cases the GPU mem is going to be used, choose fraction to use
    https://www.tensorflow.org/tutorials/distribute/parameter_server_training
    """
    cpu_cores = multiprocessing.cpu_count()

    if GPU:
        # tensorflow1.14以及之后的版本（tf2.x）中的分配与使用策略
        # https://www.codenong.com/cs106002237/
        gpus = tf.config.list_physical_devices('GPU')
        # tf.config.set_visible_devices(gpus, 'GPU')
        # tf.config.set_soft_device_placement(True)
        try:
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # 查看逻辑GPU的数量
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

            for gpu in gpus:
                # 限制内存是具体的多少
                # tf.config.set_logical_device_configuration(
                #     gpu,  # 指定的一块可见的GPU
                #     [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])

                # 自动增长
                tf.config.experimental.set_memory_growth(gpu, True)

                # 虚拟GPU技术
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
                )
        except RuntimeError as e:
            print(e)

        # tensorflow1.13 以及之前的版本对于GPU的常见的一些设置
        # gpu_options = tf.compat.v1.GPUOptions(
        #     per_process_gpu_memory_fraction=GPU_mem_use,
        #     allow_growth=True,
        # )
        # 每个gpu占用0.8的显存
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/ConfigProto
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=cpu_cores,
                                          inter_op_parallelism_threads=cpu_cores,
                                          allow_soft_placement=True,
                                          # gpu_options=gpu_options
                                          )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=cpu_cores,
                                          inter_op_parallelism_threads=cpu_cores,
                                          allow_soft_placement=True)

    # 如果电脑有多个GPU，tensorflow默认全部使用。如果想只使用部分GPU，可以设置CUDA_VISIBLE_DEVICES。
    # sess = tf.compat.v1.Session(config=config)
    # set_session(sess)
    # return sess


# session = config_hard(GPU_mem_use=0.1)
# config_hard_GPU(GPU_mem_use=0.1)


def config_hard(CPU=True, CPU_cores=1, GPU_mem_use=0.25):
    """
    Configuration of CPU and GPU, Hardware Parameters
    CPU = True  # Selection between CPU or GPU
    CPU_cores = 1  # If CPU, how many cores
    GPU_mem_use = 0.25  # In both cases the GPU mem is going to be used, choose fraction to use
    """
    config = tf.compat.v1.ConfigProto()

    if CPU:
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0},
                                          intra_op_parallelism_threads=CPU_cores,
                                          inter_op_parallelism_threads=CPU_cores)

    config.gpu_options.per_process_gpu_memory_fraction = GPU_mem_use

    # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


config_hard(CPU_cores=12)


def hold(actions):
    # encourage selling for profit and liquidity
    next_probable_action = np.argsort(actions)[1]
    if next_probable_action == 2 and len(agent.inventory) > 0:
        max_profit = stock_prices[t] - min(agent.inventory)
        if max_profit > 0:
            sell(t)
            actions[next_probable_action] = 1  # reset this action's value to the highest
            return 'Hold', actions


def buy(t):
    if agent.balance > stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        return 'Buy: ${:.2f}'.format(stock_prices[t])


def sell(t):
    if len(agent.inventory) > 0:
        agent.balance += stock_prices[t]
        bought_price = agent.inventory.pop(0)
        profit = stock_prices[t] - bought_price
        global reward
        reward = profit
        return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)


volume_config = 'Volume' if volume_considered else 'Default'
# configure logging
logging.basicConfig(filename=f'logs/{model_name}_training_{stock_name}_{volume_config}_{num_episode}episodes.log',
                    filemode='w',
                    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logging.info(f'Trading Object:           {stock_name}')
logging.info(f'Trading Period:           {trading_period} days')
logging.info(f'Window Size:              {window_size} days')
logging.info(f'Training Episode:         {num_episode}')
logging.info(f'Model Name:               {model_name}')
logging.info('Initial Portfolio Value: ${:,}'.format(initial_balance))

start_time = time.time()

for e in range(1, num_episode + 1):
    print(f'\nEpisode: {e}/{num_episode}')
    logging.info(f'\nEpisode: {e}/{num_episode}')

    agent.reset()  # reset to initial balance and hyperparameters
    # state_size = window_size加3个维度：stock_prices[end_index], balance, num_holding
    state = generate_combined_state(0, window_size, stock_prices, agent.balance,
                                    len(agent.inventory)) if not volume_considered else \
        generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory), stock_volumes)

    for t in range(1, trading_period + 1):
        if t % 100 == 0:
            logging.info(f'\n-------------------Period: {t}/{trading_period}-------------------')

        reward = 0
        next_state = generate_combined_state(t, window_size, stock_prices, agent.balance,
                                             len(agent.inventory)) if not volume_considered else \
            generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory),
                                    stock_volumes)

        previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

        if model_name == 'DDPG':
            actions = agent.act(state, t)
            action = np.argmax(actions)
        else:
            actions = agent.model.predict(state)[0]
            action = agent.act(state)

        # execute position
        logging.info(
            'Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}'.format(t, actions[0],
                                                                                           actions[1],
                                                                                           actions[2]))
        if action != np.argmax(actions):
            logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
        if action == 0:  # hold
            execution_result = hold(actions)
        if action == 1:  # buy
            execution_result = buy(t)
        if action == 2:  # sell
            execution_result = sell(t)

        # check execution result
        if execution_result is None:
            reward -= treasury_bond_daily_return_rate() * agent.balance  # missing opportunity
        else:
            if isinstance(execution_result, tuple):  # if execution_result is 'Hold'
                actions = execution_result[1]
                execution_result = execution_result[0]
            logging.info(execution_result)

        # calculate reward
        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        unrealized_profit = current_portfolio_value - agent.initial_portfolio_value
        reward += unrealized_profit

        agent.portfolio_values.append(current_portfolio_value)
        agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)

        done = True if t == trading_period else False
        agent.remember(state, actions, reward, next_state, done)

        # update state
        state = next_state

        # experience replay
        if len(agent.memory) > agent.buffer_size:
            num_experience_replay += 1
            loss = agent.experience_replay()
            logging.info(
                'Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}'.format(
                    e,
                    loss,
                    action_dict[
                        action],
                    reward,
                    agent.balance,
                    len(agent.inventory)))
            agent.tensorboard.on_batch_end(num_experience_replay,
                                           {'loss': loss, 'portfolio value': current_portfolio_value})

        if done:
            portfolio_return = evaluate_portfolio_performance(agent, logging)
            returns_across_episodes.append(portfolio_return)

    # save models periodically
    if e % 5 == 0:
        if model_name == 'DQN':
            agent.model.save('saved_models/DQN_ep' + str(e) + '.h5')
        elif model_name == 'DDPG':
            agent.actor.model.save_weights('saved_models/DDPG_ep{}_actor.h5'.format(str(e)))
            agent.critic.model.save_weights('saved_models/DDPG_ep{}_critic.h5'.format(str(e)))
        logging.info('model saved')

logging.info('total training time: {0:.2f} min'.format((time.time() - start_time) / 60))
plot_portfolio_returns_across_episodes(model_name, returns_across_episodes, stock_name, num_episode, volume_considered)
