import numpy as np
import gym

g = [   "SFFH",
        "FFFF",
        "HFHF",
        "HFFG"]
env = gym.make("FrozenLake-v1"
                            , is_slippery=True, desc=g)
env.action_space.seed(42)
observation, info = env.reset(seed=42)
Action = env.action_space.n
State = env.observation_space.n
Q_table = np.zeros([State, Action])
alpha = .01
discount_factor = .5
alpha = .01
discount_factor = .5
action = env.action_space.sample()
# print(env.observation_space.n)
def run(iteration):
    
    for itr in range(0, iteration):
        state,_ = env.reset()
        # print(itr)
        for i in range((State)):
            # print(i)
            if itr < 10000:
                action = env.action_space.sample()
                
            else:
                action = np.argmax(Q_table[state,:])
            
            state_new, reward, terminated, truncated, info = env.step(action)
            # print(reward)
            maxQ = np.max(Q_table[state_new,:])
            # print(type(state))
            # print(state[0],state[1])
            # print(action)

            # bellamn equation for weight updation
            Q_table[state, action]  = (1-alpha)*Q_table[state, action] + alpha*(reward + discount_factor* maxQ)

            state = state_new
            if  terminated or truncated:
                break
    env.close()
run(100000)
print(Q_table)

env = gym.make("FrozenLake-v1",render_mode="human",
                                is_slippery=True,desc=g)
env.action_space.seed(42)
observation, info = env.reset(seed=42)
env.reset()
max_steps = 99   
for episode in range(5):
    state = env.reset()[0]
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Q_table[state,:])
        
        new_state, reward, terminated, terminated, info = env.step(action)
    
        # if terminated or terminated:
        #     # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
        #     env.render()
        #     print(new_state)
    
        state = new_state

        if new_state == 15:
            print("We reached our Goal 🏆")
            break
    if new_state != 15:
        print("We fell into a hole ☠️")
    env.reset()
env.close()