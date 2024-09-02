from satellitetest.specific_time import specific_time
from test import create
from train import test,Args


def main():
    create()
    # # 创建参数对象
    # args = Args(max_episodes=1000, max_steps=200, batch_size=32, algorithm='ddpg')
    #
    # # 调用 test 函数
    # test(args)


if __name__ == "__main__":
    main()

