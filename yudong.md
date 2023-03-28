# Building the New Environment

The goal is to build the environment for constraint inference. There will be two types on environment:

### Robot Environment
The type of environments will be based on the mujoco. Currently, I have build four environments/constraints including:

1. HalfCheetah / The x_position must be larger than -3
2. Ant / The x_position must be larger than -3
3. Walker / The x_position must be larger than -3
4. Pendulum / The x_position must be larger than -0.015.

The definition of these ***new*** environments can be found at [here](./constraint-RL-auto-driving/mujuco_environment/custom_envs/envs).

I need your help to build more environments. To validate whether the environment works, you should do two tests:
1. Create your branch in github. 
2. Set up the mujoco and environment by following README.md.
3. Run PPO and see if the agent will always break the constraints without knowing the constraint.  
```python train_ppo.py ../config/mujuco_HCWithPos-v0/train_ppo_HCWithPos-v0.yaml -n 5 ```
4. Run PPO-Lag and see if the agent will respect the constraints when knowing the real constraint.  
```python train_ppo.py ../config/mujuco_HCWithPos-v0/train_ppo_lag_HCWithPos-v0.yaml -n 5 ```
5. If both work, run ICRL and see how it works. Remember that you need to build your dataset with ```generate_data_for_constraint_inference.py```.  
```python train_ppo.py ../config/mujuco_HCWithPos-v0/train_ICRL_HCWithPos-v0_with-action.yaml -n 5 ```

The candidate environments can be select from traditional mujoco environments (e.g, swimmer, hopper and reacher).