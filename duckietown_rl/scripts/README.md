### Train, Test & Evaluate a Reinforcement Learning (DDPG) agent 
This folder contains the scripts that train, test & compare the Reinforcement Learning (DDPG) approach with other approaches.


<b>Note:</b> All of the scripts has to be run as a module. In that way, we could import the scripts that are in `scripts` folder (e.g. `args.py`) to
our original script (e.g. `train_ddpg.py`) easily. 

--------------------------------------------------
<details>
<summary><b><i>train_ddpg.py</i></b></summary>

## Training
Three things are done when this script is executed:
1. Training of our RL agent with <b>DDPG</b> ([paper](https://arxiv.org/abs/1509.02971), [more info](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)) in [PyTorch](https://github.com/pytorch/pytorch).
2. Creating a folder `challenge-aido_RL-IL/duckietown_rl/pytorch_models`, where the trained models will be stored.
3. Creating a folder `challenge-aido_RL-IL/duckietown_rl/results`, where two .csv files (some training statistics/logs) will be stored for debugging
purposes:
    1. `df_eval.csv`: Logs of exploration
    2. `df_test.csv`: Log of exploitation
    
    Check out [Multi-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) if you're not sure what's meant by exploration vs. exploitation.

> Simply run the following code snippet:
```shell script
cd challenge-aido_RL-IL/duckietown_rl      # cd into the directory
python -m scripts.train_ddpg               # run the script as a module (-m parameter)
```

</details>

--------------------------------------------------
<details>
<summary><b><i>test_ddpg.py</i></b></summary>

## Testing

This script is for testing the trained agent: 
- It loads the trained model and runs it on a specified map
- Renders the environment
- Prints some logs: <b><i>reward, action taken, speed of the car</i></b>, and <b><i>the distance to the center of the road</i></b>

#### Important Note!
> The trained models have to be moved to `duckietown_rl/models/` and renamed as follows: `model_actor.pth` & `model_critic.pth`  

> Simply run the following code snippet:
```shell script
cd challenge-aido_RL-IL/duckietown_rl      # cd into the directory
python -m scripts.test_ddpg                # run the script as a module (-m parameter)
```

</details>

--------------------------------------------------
<details>
<summary><b><i>evalutation.py</i></b></summary>

## Evaluating
Three things are done when this script is executed:
- Creating a folder `challenge-aido_RL-IL/duckietown_rl/scripts/evaluation_results`, where the following figures will be saved
- Creating 3 figures:
    - distance traveled in meters per action & cumulative distance
    - reward achieved per action & cumulative reward
    - distance in meters vs. timesteps taken

![plot_dist](../../tutorials/images/Approach%231_plot_dist.png)

![plot reward](../../tutorials/images/Approach%231_plot_reward.png)

![plot distvstime](../../tutorials/images/Approach%231_plot_DistvsTime.png)

# TODO: 
Add images


This script can be used to evaluate a trained model/agent, in other words, it can be used to see how good/bad a model is
 doing. Additionaly, it can be used to compare different models/approaches.

> Simply run the following code snippet:
```shell script
cd challenge-aido_RL-IL/duckietown_rl      # cd into the directory
python -m scripts.evaluation               # run the script as a module (-m parameter)
```

</details>

--------------------------------------------------