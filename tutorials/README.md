# Welcome to the Tutorials! 
This folder includes a bunch of Python scripts for you to get yourself familiarized with:
- Maps in the simulation that are used for the challenges
- Simulation and it's dynamics
- An agent/expert that drives itself (PID controller)
- A noise profile that's used in Reinforcement Learning training (DDPG)
- Train DDPG on [OpenAI's Cartpole](https://gym.openai.com/envs/CartPole-v0/)!

---
<details>
<summary><b><i>maps/</i></b></summary>

By the time this code is published, these are all the available maps we have in Duckietown:
```shell script
cd challenge-aido_RL-IL
tree tutorials/maps
``` 
![show maps](images/maps.png)

Every map is stored in a `.yaml` file where each map is constructed tile-by-tile, as well as some objects such as; 
static & non-static Duckiebots, pedestrians, traffic lights, etc. Please see [this repository](https://github.com/duckietown/gym-duckietown#map-file-format) if you want to construct a
new map or want to get more information. In addition, check out this [great notebook](https://github.com/duckietown/duckietown-world/blob/daffy/notebooks/30-DuckietownWorld-maps.ipynb)
for a better and broader perspective on the maps available!


</details>

---
<details>
<summary><b><i>get_good_sample_pose.py</i></b></summary>

This script outputs a 'good' starting position and angle for an agent, given a map. That means, agent starts at a point
that is close to the center of a lane and starts at an angle that is close to zero, which means agent is aligned with
the lane. Example usage:

```shell script
cd challenge-aido_RL-IL/tutorials
python get_good_sample_pose.py -m "zigzag_dists"
```
which returns the starting position of the car in `(x,y,z)` coordinates and the alignment with the lane in `degrees`
for the map `zigzag_dists`. e.g. Angle `0` means the car is perfectly aligned with the right-lane's center line. Angle `45` means the car needs to turn 45 degrees to be
aligned with the  right-lane's center line.

Keep it in mind that a map name has to be given as an argument(`-m` which stands for 'map')


</details>

---
<details>
<summary><b><i>get_dynamics.py</i></b></summary>

Let's get our hands dirty and play with the simulator/environment!
In other words, let's have a look at the dynamics of the simulator and try to get an answer to the
following questions:
- What's the full-speed of the our car/duckie-bot? 
- How many time-steps or seconds does it need to reach to full-speed? 
- Can we go backwards and what the max. speed we can reach while diriving to backwards?
- How many time-steps or seconds does it need to stop?

In order to answer some of the questions above, if not all, I performed a test where I manually control the car and
collect some data. 

Here's the processed version of the data collected from my test: ![get_dynamics_processed](images/get_dynamics_processed.png)
> Please see the raw data and the graph [here](https://docs.google.com/spreadsheets/d/1Z7T850Boy9YJm9lRytTkmAFld-AV8DBCbTb3Lo4PRIM/edit?usp=sharing) for more information.

#### Take-aways:

- The full-speed is: `0.6 m/s` 
- It takes `≈0.858 seconds` or `≈26 time-steps` to reach to full-speed from `0 m/s`
- It takes `≈0.858 seconds` or `≈26 time-steps` to reach to `0 m/s` from full-speed(`0.6 m/s`)

These details will become important when building/training algorithms!

<details>
<summary><i><b>***</b> Click to see the <b>spoiler</b> where this will be important! <b>***</b></i></summary>

When training a reinforcement learning algorithm (the one we use is called DDPG) we let the agent apply the same action
for a fixed number of times, because it takes some time to achieve speed! That fixed number will be called `frame_skip`
which you will encounter when creating the simulator instance [as in here](https://github.com/rizavelioglu/challenge-aido_RL-IL/blob/82f84a31ce46585b97498ed56ee6d794e8bd0038/duckietown_rl/env.py#L5)! 
</details>

---

##### Important Note:

As soon as the script is executed, the data is getting aggregated. And once
"ENTER" is hit from the keyboard, then the data is saved to a file: `get_dynamics_raw.csv`  
Here the data collection is done manually, but it can also be done automatically, obviously:
- Initialize an environment in `straight_road` map 
- Generate a good sampling position to start in the map using `get_good_sample_pose.py`
- Create a dummy agent that applies the max. action to go full speed for a specified time-steps
- Collect all the data, store, process, and gather info!

</details>

---
<details>
<summary><b><i>PIDcontroller.py</i></b></summary>

This script implements a PID controller[, see Wikipedia for more info,](https://en.wikipedia.org/wiki/PID_controller) for the car to drive and
navigate itself within an environment. Go check out the code cause every line of code is explained!

#### Take-aways:

- The action space is in 2-d. That means an action is a 2-dimensional vector which corresponds to the left & right wheel
velocities.
- The structure of running an agent in a simulation shown in a pseudocode-ish way:
```shell script
Initialize the environment                                // Line [9]
Reset the environment and store observation               // Line [23]
Render the environment                                    // Line [25]
for 0 to EPISODES:                                        // Line [29]
    for 0 to STEPS:                                       // Line [35]
        Calculate the action according to your algorithm  // Line [69-71]
        Apply the action and store information            // Line [73]
        render the environment                            // Line [75]
    reset the environment                                 // Line [88]
```

##### Note:
The reason why we store only last 10 angle errors [[Line 31]](https://github.com/rizavelioglu/challenge-aido_RL-IL/blob/82f84a31ce46585b97498ed56ee6d794e8bd0038/tutorials/PIDcontroller.py#L31) is based on this [paper](https://www.robotshop.com/community/forum/t/pid-tutorials-for-line-following/13164)
</details>

---
<details>
<summary><b><i>get_tile_coordinates.py</i></b></summary>

This script runs the PID agent to get each of the tile coordinates in some of the maps available in `maps/` folder.
For simplicity, only some of the maps' tile data is collected,not all. In addition, some maps are preffered amongst
others due to simplifications in the chosen maps such as; there's no obstacle or no other car in the chosen maps, which
make things easier for us. 

After the following code is run, `tile_coordinates.csv` file is created inside `tutorials/`
folder: 

```shell script
python get_tile_coordinates.py
```

> #### Why do we need this script?
We will need the tile coordinates data for constructing the feature vector which will be given to Reinforcement Learning
algorithm (DDPG) as input. Therefore, it is essential to have the tile coordinates of the maps where we are building our
approach to self-driving car. You can see that these maps are inside the `duckietown_rl/maps` folder, for which we stored
the tile coordinates. And you can see that we copied the data from `tile_coordinates.csv` and pasted inside [this function](https://github.com/rizavelioglu/challenge-aido_RL-IL/blob/362feae4f058c6db897021c47c98759c79ea1ed2/duckietown_rl/gym_duckietown/simulator.py#L2036)
in `duckietown_rl/gym_duckietown/simulator.py`

> #### Tile coordinates of the map "zigzag_dists"

In the following figure, the tile coordinates of the map **"zigzag_dists"** are given:

![show maps](images/tile_coordinates_zigzag_dists.png)

- The coordinates assigned to each tile are the `tile coordinates` . So you can see that the origin is the tile that is
on left-most bottom part of the map. 

> Another important fact about tile coordinates is the relation it has with the **position of the car**. We can reach the
current position of the car through the environment: `env.cur_pos` gives the `(x,y,z)` coordinates of the car. That
returned position tells where the car is, see the following examples:

- On the right-hand-side of the figure, we calculate the `env.cur_pos`--> `tile_coordinates` is multiplied with the `road_tile_size`.

    - Say that the car is in the middle of the tile `(3,6)`. Then the car's position is calculated to be `[2.0475, 3.8025]`(see figure above).
    Therefore the `env.cur_pos` variable in that point would be `[2.0475, 0, 3.8025]`. 

    - Another example: The mid-point of the tile `(2,7)` is `[1.4625, 4.3875]`. Therefore the `env.cur_pos` variable in that
    point would be `[1.4625, 0, 4.3875]`. 

    - `y` coordinate is always 0, because that's how the environment is built (2-D map), see
    the coordinate axis in the middle of the image.


</details>

---
<details>
<summary><b><i>OU_action_noise.py</i></b></summary>

<i>"In Reinforcement learning for discrete action spaces, exploration is done via probabilistically selecting a random action
(such as epsilon-greedy or Boltzmann exploration). For continuous action spaces, exploration is done via adding noise to
the action itself. In the DDPG paper, the authors use Ornstein-Uhlenbeck Process to add noise to the action output
(Uhlenbeck & Ornstein, 1930)"</i> [[Source]](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b)

<i>"The Ornstein-Uhlenbeck Process generates noise that is correlated with the previous noise, as to prevent the noise from
canceling out or “freezing” the overall dynamics"</i> [[Source]](https://www.quora.com/Why-do-we-use-the-Ornstein-Uhlenbeck-Process-in-the-exploration-of-DDPG/answer/Edouard-Leurent?ch=10&share=4b79f94f&srid=udNQP)

We will see OU noise in detail when we train a Reinforcement Learning agent using DDPG algorithm in `duckietown_rl/`. The purpose of this
script is just to get the user familiarized with OU noise. What this script does is that it generates an OU noise profile
and visualizes it. 

**Example usage #1:** Scatter plot & save the plot: `--save-img = 1`
```shell script
python OU_action_noise.py --mu 0 --sigma 0.2 --timesteps 1000 --reset-after 500 --save-img 1
```
![OU_1](images/OrnsteinUhlenbeckActionNoise(mu=%5B0.%200.%5D,%20sigma=%5B0.2%200.2%5D)-1000%20steps%20-%20resets%20after%20500%20steps.png)
>-  As the action is a 2-D vector (left & right wheel velocities) 2 noise profiles are generated where the blue dots belong
    to the left wheel velocity's noise profile and the orange ones to the right wheel velocity's noise profile.
>- As the arguments:
>>   - `--mu` & `--sigma` are set to 0 and 0.2, respectively, both of the two noise profiles have mean 0
    and standard deviation 0.2.
>>   - `--timesteps` is set to 1000, both profiles are generated for 1000 timesteps (x-axis).
>>   - `--reset-after` is set to 500, both profiles are reset after 500 timesteps: they start from 0 again.
>>   - `--save-img` is set to 1, the plot is saved.

**Example usage #2:** Line chart: `--line = 1` & don't save the image(default):
```shell script
python OU_action_noise.py --mu 0 --sigma 0.5 --timesteps 500 --reset-after 0 --line 1
```
![OU_2](images/OrnsteinUhlenbeckActionNoise(mu=%5B0.%200.%5D,%20sigma=%5B0.5%200.5%5D)-500%20steps%20-%20resets%20after%200%20steps.png)
>-  This is another plot with minor differences explained below.
>- As the arguments:
>>   - `--mu` & `--sigma` are set to 0 and 0.5, respectively, both of the two noise profiles have mean 0
    and standard deviation 0.5.
>>   - `--timesteps` is set to 500, both profiles are generated for 500 timesteps (x-axis).
>>   - `--reset-after` is set to 0, both profiles are reset after 0 timesteps: so, they are never reset after time step 0.
>>   - `--line` is set to 1, the plot is a line plot, instead of a scatter plot.
>>   - `--save-img` is not given, the plot is not saved.

See the following links to get more info on OU Noise:
- [Wikipedia](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
- [Blogpost](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b)


#### Take-aways:

- OU noise generates a noise that is correlated with the previous noise, until you reset it.
- Every reset OU noise is unique!
- Since we only have 2 wheels, hence 2 wheel velocities, 2 noise profile is generated: one for left wheel, and another
one for right wheel.

</details>

---
<details>
<summary><b><i>cartpole/</i></b></summary>

[CartPole Problem](https://gym.openai.com/envs/CartPole-v0/) a.k.a. Inverted Pendulum, is the "Hello World" of Reinforcement
Learning. Here's the official explanation on CartPole by the creators: 
> <i>"A pole is attached by an un-actuated joint to a cart, which moves
along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright,
and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright.
The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center."</i>

While there are much simpler algorithms that solve this problem, we would like to use DDPG just to double-check that the
implementation of DDPG is correct and that it actually works on this simple problem. 

Here's what's inside the `cartpole` folder:

- **<i>models/</i>**: This is where we store our trained (final)model! The model in this folder is used by `train_ddpg.py` & `test_ddpg.py`
- `args.py`: where we set parameters related to training (e.g. batch size, ddpg parameters)
- `ddpg.py`: the implementation of DDPG
- `test_ddpg.py`: where we test the trained agent and visualize it
- `train_ddpg.py`: where the training loop lies
- `utils.py`: the implementation of Replay Buffer & the evaluation done during training
 
</details>

---