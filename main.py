import click

# --------------------------- Algorithms ---------------------------

from algorithms.policy_gradient.cart_pole import train_cart_pole, \
    play_cart_pole, play_trained_soln_cart_pole
from algorithms.policy_gradient.moon_lander import play_moon_lander, \
    train_moon_lander

from algorithms.DQN.cart_pole import train_cart_pole as train_cart_pole_dqn
from algorithms.DQN.cart_pole import play_cart_pole as play_cart_pole_dqn

# --------------------------- Tests --------------------------------

from tests.critic_learn import test_critic
from algorithms.policy_gradient.test import train_test, play_test

# --------------------------- Cli ----------------------------------


cli_map = {
    'po': {
        'luner-lander': {
            'train': train_moon_lander,
            'play': play_moon_lander,
            'example': None
        },
        'cart-pole': {
            'train': train_cart_pole,
            'play': play_cart_pole,
            'example': play_trained_soln_cart_pole
        }
    },
    'dqn': {
        'cart-pole': {
            'train': train_cart_pole_dqn,
            'play': play_cart_pole_dqn,
            'example': None
        }
    },
    'ac': {}
}

test_map = {
    'train': train_test,
    'play': play_test,
    'crtic': test_critic
}


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        pass


@cli.command()
@click.pass_context
@click.option('--target', '-t', default=None, help='training target')
@click.option('--num_episodes', '-e', default=2500,
              help='Number of epsiodes of training')
@click.option('--num_steps', '-s', default=None,
              help='Max number of steps per episode')
@click.option('--algorithm', '-a', default='po',
              help='Algorithm to use')
def train(ctx, algorithm, target, num_episodes, num_steps):
    cli_map[algorithm][target]['train'](num_episodes, num_steps)


@cli.command()
@click.pass_context
@click.option('--target', '-t', default='moon-lander', help='training target')
@click.option('--num_episodes', '-e', default=2500,
              help='Number of epsiodes of training')
@click.option('--num_steps', '-s', default=None,
              help='Max number of steps per episode')
@click.option('--algorithm', '-a', default='po',
              help='Algorithm to use')
def play(ctx, algorithm, target, num_episodes, num_steps):
    cli_map[algorithm][target]['play']()


@cli.command()
@click.pass_context
@click.option('--target', default='all', help='training target')
def test(ctx, target):
    test_critic()


if __name__ == '__main__':
    cli()
