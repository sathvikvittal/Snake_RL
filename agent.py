import torch
import random
import numpy as np
from snake import Snake, Direction, Point
from collections import deque
from model import QNet,QTrainer
from plotter import plot

MAX_MEM = 50_000
BATCH_SIZE = 250
LEARNING_RATE = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.eps = 0
        self.gamma = 0.9
        self.mem = deque(maxlen=MAX_MEM)
        self.model = QNet(11,256,3)
        self.trainer = QTrainer(self.model,lr=LEARNING_RATE,gam=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x-20,head.y)
        point_r = Point(head.x+20,head.y)
        point_u = Point(head.x,head.y-20)
        point_d = Point(head.x,head.y+20)

        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        """
        state = [
        Danger_straight,Danger_right,Danger_left,

        direction left,right,up,down

        food left,right,up,down
        ]
        """

        state = [
            (dir_right and game.is_coll(point_r)) or 
            (dir_left and game.is_coll(point_l)) or 
            (dir_up and game.is_coll(point_u)) or 
            (dir_down and game.is_coll(point_d)),

            (dir_right and game.is_coll(point_d)) or 
            (dir_left and game.is_coll(point_u)) or 
            (dir_up and game.is_coll(point_r)) or 
            (dir_down and game.is_coll(point_l)),

            (dir_right and game.is_coll(point_u)) or 
            (dir_left and game.is_coll(point_d)) or 
            (dir_up and game.is_coll(point_l)) or 
            (dir_down and game.is_coll(point_r)),

            dir_left,
            dir_right,
            dir_up,
            dir_down,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y

        ]

        return np.array(state,dtype=int)



    def remember(self,state,action,reward, next_state, end):
        self.mem.append((state,action,reward, next_state, end))
        

    def train_long(self):
        if len(self.mem) >= BATCH_SIZE:
            sample = random.sample(self.mem,BATCH_SIZE)
        else:
            sample = self.mem
        states, actions, rewards, next_states, ends = zip(*sample)
        self.trainer.train_step(states,actions,rewards,next_states,ends)

    def train_short(self,state,action,reward, next_state, end):
        self.trainer.train_step(state, action, reward, next_state, end)


    def get_action(self,action):
        self.epsilon = 75 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            action0 = torch.tensor(action,dtype=torch.float)
            pred = self.model(action0)
            move = torch.argmax(pred).item()
            final_move[move] = 1
        
        return final_move





def train():
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Snake()

    while True:
        old_state = agent.get_state(game)

        next_move = agent.get_action(old_state)

        reward ,end,score = game.play_step(next_move)
        new_state = agent.get_state(game)

        agent.train_short(old_state,next_move,reward, new_state, end)

        agent.remember(old_state,next_move,reward, new_state, end)

        if end:
            game.reset()
            agent.n_games += 1
            agent.train_long()

            if score > record:
                record = score
                agent.model.save_model()
            
            print("Game",agent.n_games, "Score",score,"Record",record)

            scores.append(score)
            total_score += score
            mean_scores.append(total_score/len(scores))
            plot(scores,mean_scores)


if __name__ == "__main__":
    train()