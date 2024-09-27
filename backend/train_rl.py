# train_rl.py

import torch
import torch.nn.functional as F
import numpy as np
from shogi_env import ShogiEnv
from ppo_agent import PPOAgent

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = ShogiEnv()
    env.init_move_list()
    action_size = len(env.move_list)
    agent = PPOAgent(action_size, device)
    
    max_episodes = 10000
    print_interval = 20
    score = 0.0
    
    for episode in range(max_episodes):
        state = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            policy, value = agent.model(state_tensor)
            policy = F.softmax(policy, dim=1)
            dist = torch.distributions.Categorical(policy)
            action = dist.sample().item()
            move = env.decode_action(action)
            
            # 行動が合法手でない場合はランダムに選択
            if move not in env.legal_moves():
                move = np.random.choice(env.legal_moves())
                action = env.encode_action(move)
                if action is None:
                    continue
            
            prob = dist.log_prob(torch.tensor(action).to(device)).item()
            next_state, reward, done, info = env.step(move)
            agent.put_data((state, action, reward, next_state, prob, done))
            
            state = next_state
            score += reward
        
        if episode % print_interval == 0 and episode != 0:
            agent.train_net()
            print("# of episode :{}, avg score : {:.1f}".format(episode, score / print_interval))
            score = 0.0

if __name__ == '__main__':
    main()
