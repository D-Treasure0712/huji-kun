# evaluate.py

import torch
import torch.nn.functional as F
import numpy as np
from shogi_env import ShogiEnv
from model import ShogiModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = ShogiEnv()
    env.init_move_list()
    action_size = len(env.move_list)
    model = ShogiModel(action_size).to(device)
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()
    
    state = env.reset()
    done = False
    
    while not done:
        env.render()
        if env.board.turn == 0:  # 先手（AI）の手番
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            policy, value = model(state_tensor)
            policy = F.softmax(policy, dim=1)
            dist = torch.distributions.Categorical(policy)
            action = dist.sample().item()
            move = env.decode_action(action)
            
            # 行動が合法手でない場合はランダムに選択
            if move not in env.legal_moves():
                move = np.random.choice(env.legal_moves())
            
            env.step(move)
            state = env.get_observation()
        else:  # 後手（人間）の手番
            move_str = input('Your move (in CSA format): ')
            move = None
            try:
                move = env.board.move_from_csa(move_str)
                if move not in env.legal_moves():
                    print('Illegal move. Try again.')
                    continue
            except:
                print('Invalid input. Try again.')
                continue
            env.step(move)
            state = env.get_observation()
    
    env.render()
    if env.winner == 'BLACK':
        print('AI wins!')
    elif env.winner == 'WHITE':
        print('You win!')
    else:
        print('Draw!')

if __name__ == '__main__':
    main()
