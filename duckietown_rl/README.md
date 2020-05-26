# Reinforcement Learning
This folder contains all the scripts required for the Reinforcement Learning based approach! 

- **<i>gym_duckietown/</i>**: My version of gym_duckietown package
- **<i>maps/</i>**: where the map info are stored 
- **<i>models/</i>**: This is where we store our trained (final)model! The model in this folder is used by `scripts/test_ddpg.py` & `scripts/evaluation.py`
- **<i>scripts/</i>**: where the training, testing scripts are. See [scripts/README](scripts/README.md) for more info
- `args.py`: where we set parameters related to training (e.g. batch size, ddpg parameters)
- `ddpg.py`: the implementation of DDPG
- `env.py`: includes the function that launches the environment
- `ornstein_uhlenbeck.py`: the implemenataion of OU Noise
- `utils.py`: the implementation of Replay Buffer & the evaluation done during training
- `wrappers.py`: image processing wrappers
  
--------------------------------------------------
<details>
<summary><b><i>gym_duckietown/</i></b></summary>

`gym_duckietown` is a Python package build on top of OpenAI's Gym. It is basically "the Simulator". You can find all the
scripts that creates the simulator; its physics, maps, objects, etc. This simulator was created as part of work done at [Mila](https://mila.quebec/).
The latest version of `gym_duckietown` can be found [on this link](https://github.com/duckietown/gym-duckietown/tree/daffy).
 
But in this folder I edited some scripts for my approach: `simulator.py` & `graphics.py`, to be specific. Here's my approach:
- **Sensor lines:** 

    The idea is to 'attach' distance sensors to the car where the sensor readings will correspond to the distance between
    the car and the center of the lane. Please refer to the following figure:
    
    ![](../tutorials/images/sensors.png)
    
    - The red dots seen on the right-lane shows the *center of the right-lane*, which can be assumed as the optimal trajectory. 
    - The red square in the middle of the image is the car (duckie-bot), in other words the *car's corners*.
    - The blue dot in the red square is *the center of the car*.
    - The vertical green line that passes through the center of the car is *the directory line* which is aligned with the direction of the car.
    - The other green lines, which we call ***sensor lines***, correspond to the sensor readings. 

#### Note
If you would like to render the above information such as; rendering the sensor lines, you need to set the corresponding
parameter of the simulator while constructing the environment. For example, the image above was taken from the simulator
which was constructed as follow:
```python
env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                randomize_maps_on_reset=True, draw_curve=False, draw_bbox=True, frame_skip=4, draw_DDPG_features=True)
```
> See [this script in tutorials](../tutorials/PIDcontroller.py) for explanations of the other parameters.


- **Reinforcement Learning**

    Once we have the above representation (in other words, state representation), we can use Reinforcement Learning algorithms
    to train an agent (a neural network) that drives itself! How? Here's the input (features) to the neural network:
    
    - **24 numbers**: `12` sensor lines, `2` values each
    
        Refer to the image above. We have `12` sensor lines, so `12` sensor readings which are the distances from the sensor
        to the center line on the lane. We store **2** values for each sensor line:
     
        1. A binary value (**0** or **1**): **1** if sensor line intersects with the center line, and **0** otherwise
        2. The distance.
        
            If the binary value is **0** which means the sensor line does not intersect with the center line, then
            its distance value is also set to **0**
            
            If the binary value is **1**, then the Euclidean distance is measured. If the sensor is to the right-hand side
            of the center line, the distance is multiplied by `-1`. So, the distance `-0.5` would mean that the sensor is on
            the right-hand side of the lane by `0.5`.
    
    - **Wheel velocities**: `2` numbers referring to the left & right wheel velocity
    - **Speed**: `1` number which is the speed of the car in `m/s`
    
    In total, we have `24+2+1=27` values/features. When an agent is trained with those features, it does not drive smoothly.
    In other words, the agent does oscillations/manoeuvres. To overcome this issues, we stack previous `7` states to the
    features/inputs of the neural network, which yields in a size of `27*7=189`.
        

</details>

--------------------------------------------------
<details>
<summary><b><i>maps/</i></b></summary>

This is where we store our maps in which our RL agent will learn driving!
```shell script
tree challenge-aido_RL-IL/duckietown_rl/maps
``` 
![show maps](../tutorials/images/duckietown_rl-maps.png)

We see that there are only 3 maps in this folder. That's because these are the most reasonable maps amongst the others, in my opinion.
Because these maps do not have any other car/duckiebot or any pedestrian. At the same time, they have all the features
required for learning how to drive; turns, straight roads, zigzags, etc. But these are not the only maps available within
the environment: see [tutorials/maps](../tutorials/maps) for more info.

</details>

--------------------------------------------------