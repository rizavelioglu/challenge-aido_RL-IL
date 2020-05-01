# Welcome to the Tutorials! 
This folder includes a bunch of Python scripts for you to get yourself familiarized with:
- Maps in the simulation that are used for challenges
- Simulation and it's dynamics
- An agent/expert that drives itself (PID controller)
- A noise profile that's used in Reinforcement Learning training (DDPG)

---
<details>
<summary><b><i>get_tile_coordinates.py</i></b></summary>

</details>

---
<details>
<summary><b><i>get_good_sample_pose.py</i></b></summary>

</details>

---
<details>
<summary><b><i>get_dynamics.py</i></b></summary>
  
  ## Heading
  1. A numbered
  2. list
     * With some
     * Sub bullets

##### yes, even hidden code blocks!
```python
print("hello world!")
```

Here's a processed version of the data collected: ![get_dynamics_processed](images/get_dynamics_processed.png)
> Please see the raw data and the graph [here](https://docs.google.com/spreadsheets/d/1Z7T850Boy9YJm9lRytTkmAFld-AV8DBCbTb3Lo4PRIM/edit?usp=sharing).

Here the data collection is done manually, but it can also be done automatically, obviously:
- Initialize an environment in `straight_road` map 
- Generate a good sampling position to start in the map using `get_good_sample_pose.py`
- Create a dummy agent that applies the max. action to go full speed for a limited time
- Collect all the data and store

</details>

---
<details>
<summary><b><i>OU_action_noise.py</i></b></summary>

</details>

---
<details>
<summary><b><i>PIDcontroller.py</i></b></summary>

The reason why we store only last 10 angle errors is based on this [paper](https://www.robotshop.com/community/forum/t/pid-tutorials-for-line-following/13164)
</details>

---