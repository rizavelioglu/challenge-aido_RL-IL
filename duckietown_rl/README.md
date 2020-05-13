### This folder contains all the scripts required for the Reinforcement Learning based approach! 

- gym_duckietown: My version of gym_duckietown package
- maps: the folder where the map info are stored 
- scipts: 
--------------------------------------------------
<details>
<summary><b><i>gym_duckietown</i></b></summary>

</details>

--------------------------------------------------
<details>
<summary><b><i>maps</i></b></summary>

</details>

--------------------------------------------------
<details>
<summary><b><i>scripts</i></b></summary>

#### This folder contains the scripts that trains, tests & compares the Reinforcement Learning (DDPG) approach with other approaches 

---
<details>
<summary><b><i>train_ddpg.py</i></b></summary>

```
cd duckietown_rl
python -m scripts.train_ddpg
```
</details>

---
<details>
<summary><b><i>test_ddpg.py</i></b></summary>

```
cd duckietown_rl
python -m scripts.test_ddpg
```
</details>

---
<details>
<summary><b><i>evalutation.py</i></b></summary>

```
cd duckietown_rl
python -m scripts.evaluation
```
</details>

</details>

--------------------------------------------------
<details>
<summary><b><i>OU_action_noise.py</i></b></summary>

OU explanations:
- [Link1](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
- [Link2](https://www.sciencedirect.com/topics/mathematics/ornstein-uhlenbeck-process)
- [Link3](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b)

<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">

[see this](https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b) for more info on latex


</details>

--------------------------------------------------
<details>
<summary><b><i>PIDcontroller.py</i></b></summary>

</details>

---