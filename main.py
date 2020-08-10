import click

# --------------------------- Algorithms ---------------------------

from algorithms.policy_gradient.cart_pole import train as train_cart_pole_pg
from algorithms.policy_gradient.cart_pole import play as play_cart_pole_pg
from algorithms.policy_gradient.cart_pole import example  \
    as example_cart_pole_pg

from algorithms.policy_gradient.moon_lander import play as play_moon_lander
from algorithms.policy_gradient.moon_lander import train as train_moon_lander

from algorithms.DQN.cart_pole import train as train_cart_pole_dqn
from algorithms.DQN.cart_pole import play as play_cart_pole_dqn
from algorithms.DQN.cart_pole import example as example_cart_pole_dqn

from algorithms.actor_critic.cart_pole import train as train_cart_pole_ac
from algorithms.actor_critic.cart_pole import play as play_cart_pole_ac
# from algorithms.actor_critic.cart_pole import example as example_cart_pole_ac

from algorithms.DDPG.moon_lander import train as train_moon_lander_ddpg
from algorithms.DDPG.moon_lander import play as play_moon_lander_ddpg
from algorithms.DDPG.moon_lander import example as example_moon_lander_ddpg
from algorithms.DDPG.moon_lander import test_Q

from algorithms.evo.moon_lander import train as train_moon_lander_evo
from algorithms.evo.moon_lander import play as play_moon_lander_evo

from algorithms.TD3.moon_lander import train as train_moon_lander_td3
from algorithms.TD3.moon_lander import play as play_moon_lander_td3
# from algorithms.TD3.moon_lander import example as example_moon_lander_td3


# --------------------------- Tests --------------------------------

from tests.critic_learn import test_critic
# from algorithms.policy_gradient.test import train_test, play_test

# --------------------------- Cli ----------------------------------


cli_map = {
    'td3': {
        'luner-lander': {
            'train': train_moon_lander_td3,
            'play': play_moon_lander_td3,
            'example': None
        }
    },
    'evo': {
        'luner-lander': {
            'train': train_moon_lander_evo,
            'play': play_moon_lander_evo,
            'example': None
        }
    },
    'pg': {
        'luner-lander': {
            'train': train_moon_lander,
            'play': play_moon_lander,
            'example': None
        },
        'cart-pole': {
            'train': train_cart_pole_pg,
            'play': play_cart_pole_pg,
            'example': example_cart_pole_pg
        }
    },
    'ddpg': {
        'luner-lander': {
            'train': train_moon_lander_ddpg,
            'play': play_moon_lander_ddpg,
            'example': example_moon_lander_ddpg,
            'test_Q': test_Q
        }
    },
    'dqn': {
        'cart-pole': {
            'train': train_cart_pole_dqn,
            'play': play_cart_pole_dqn,
            'example': example_cart_pole_dqn
        },
        'luner-lander': {
            'train': None,
            'play': None,
            'example': None
        }
    },
    'ac': {
        'cart-pole': {
            'train': train_cart_pole_ac,
            'play': play_cart_pole_ac,
            'example': None
        },
        'luner-lander': {
            'train': None,
            'play': None,
            'example': None
        }
    }
}


def invoke(algorithm, target, intent, num_episodes, num_steps):
    func = cli_map[algorithm][target][intent]
    if func:
        func(num_episodes, num_steps)
    else:
        print((algorithm, target, intent), 'option not available.')


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        pass


@cli.command()
@click.pass_context
@click.option('--target', '-t', default=None, help='training target')
@click.option('--num_episodes', '-e', default=2500, type=int,
              help='Number of epsiodes of training')
@click.option('--num_steps', '-s', default=None, type=int,
              help='Max number of steps per episode')
@click.option('--algorithm', '-a', default='pg',
              help='Algorithm to use')
def train(ctx, algorithm, target, num_episodes, num_steps):
    invoke(algorithm, target, 'train', num_episodes, num_steps)


@cli.command()
@click.pass_context
@click.option('--target', '-t', default='moon-lander', help='training target')
@click.option('--num_episodes', '-e', default=2500, type=int,
              help='Number of epsiodes of training')
@click.option('--num_steps', '-s', default=200, type=int,
              help='Max number of steps per episode')
@click.option('--algorithm', '-a', default='pg',
              help='Algorithm to use')
def play(ctx, algorithm, target, num_episodes, num_steps):
    invoke(algorithm, target, 'play', num_episodes, num_steps)


@cli.command()
@click.pass_context
@click.option('--target', '-t', default='moon-lander', help='training target')
@click.option('--num_episodes', '-e', default=2500, type=int,
              help='Number of epsiodes of training')
@click.option('--num_steps', '-s', default=200, type=int,
              help='Max number of steps per episode')
@click.option('--algorithm', '-a', default='pg',
              help='Algorithm to use')
def example(ctx, algorithm, target, num_episodes, num_steps):
    invoke(algorithm, target, 'example', num_episodes, num_steps)


@cli.command()
@click.pass_context
def test_q(ctx):
    invoke('ddpg', 'luner-lander', 'test_Q', 1, 200)


# test_map = {
#     'train': train_test,
#     'play': play_test,
#     'crtic': test_critic
# }


@cli.command()
@click.pass_context
@click.option('--target', default='all', help='training target')
def test(ctx, target):
    test_critic()


if __name__ == '__main__':
    cli()
