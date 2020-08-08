## Reinforcement Learning Solutions To Openai Gym Problems:

___


### Setup:

Create new virtual environment, activate and install requirements:

```sh
python3 -m venv venv
source venv/bin/activate
pip install --update pip
pip install -r requirements
```

### Running Examples:

This repo uses [click](https://click.palletsprojects.com/en/7.x/) as a command line interface to reinforcment algorithms written on top of [TensorFlow](https://www.tensorflow.org/).

To see pretrained examples use `python main.py example`. The options are `--target`, `--algorithm` and `--num_steps`. `target` refers to the openai gym environment. Choices are `cart-pole` and `luner-lander`. Algorithm is the algorithm used to train the solution, choices are `pg` for Policy Gradient, `dqn` for Deep Q Network and `ac` for Actor Critic. `--num_steps` is just the number of iterations of the trained example solution we run.

#### Example:

```sh
python main.py example --target='luner-lander' --algorithm='pg'
```


### TODO:
- [ ] add burn in for Q learning in ddpg.
- [ ] add checks for Q convergence in Memory
