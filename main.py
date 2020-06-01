import click
from algorithms.policy_gradient.cart_pole import train_cart_pole, \
    play_cart_pole, play_trained_soln_cart_pole
from algorithms.policy_gradient.moon_lander import play_moon_lander, \
    train_moon_lander, compute_path_stat_moon_lander
from algorithms.policy_gradient.test import train_test, play_test


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        pass


@cli.command()
@click.pass_context
@click.option('--target', default=None, help='training target')
@click.option('--num_episodes', default=2500,
              help='Number of epsiodes of training')
@click.option('--num_steps', default=None,
              help='Max number of steps per episode')
def train_model(ctx, target, num_episodes, num_steps):
    if target == 'moon-lander':
        train_moon_lander(num_episodes, num_steps)
    elif target == 'cart-pole':
        train_cart_pole(num_episodes, num_steps)
    elif target == 'test':
        play_test()
    else:
        print('param: --target | options: moon-lander, cart-pole, test')


@cli.command()
@click.pass_context
@click.option('--target', default='moon-lander', help='training target')
@click.option('--num_episodes', default=2500,
              help='Number of epsiodes of training')
@click.option('--num_steps', default=None,
              help='Max number of steps per episode')
@click.option('--example', '-e', is_flag=True,
              help='Play example solution')
def play_model(ctx, target, num_episodes, num_steps, example):
    if target == 'moon-lander':
        play_moon_lander()
    elif target == 'cart-pole':
        if example:
            play_trained_soln_cart_pole()
        else:
            play_cart_pole()
    elif target == 'test':
        train_test()
    else:
        print('param: --target | options: moon-lander, cart-pole, test')


@cli.command()
@click.pass_context
@click.option('--target', default=None, help='training target')
@click.option('--num_episodes', default=2500,
              help='Number of epsiodes of training')
@click.option('--num_steps', default=None,
              help='Max number of steps per episode')
def path_stats(ctx, target, num_episodes, num_steps):
    if target == 'moon-lander':
        compute_path_stat_moon_lander()
    else:
        print('param: --target, | options: moon-lander, -')


if __name__ == '__main__':
    cli()
