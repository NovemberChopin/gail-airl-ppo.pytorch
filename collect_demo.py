import os
import argparse
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.utils import collect_demo


def run(args):
    env = make_env(args.env_id)

    algo = ALGOS[args.algo](
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=args.weight
    )

    buffer = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed
    )
    buffer.save(os.path.join(
        'buffers',
        args.algo,
        args.env_id,
        f'size{args.buffer_size}_std{args.std}_prand{args.p_rand}.pth'
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weight', type=str, default='weights/ppo/Hopper-v3.pth')
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    p.add_argument('--algo', type=str, default='ppo_exp')
    p.add_argument('--buffer_size', type=int, default=int(5e5))
    p.add_argument('--std', type=float, default=0.01)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--cuda', action='store_true', default=False)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
