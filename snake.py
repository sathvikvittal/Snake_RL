import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font("STIXTwoText-VariableFont_wght.ttf",25)


class Direction(Enum):
    RIGHT = 1
    UP = 2
    DOWN = 3
    LEFT = 4


Point = namedtuple("Point", ["x", "y"])
PIXEL_BLOCK = 20
SPEED = 60



class Snake:
    def __init__(self,width=640,height=640):
        self.width = width
        self.height = height
        #intialize display
        self.display = pygame.display.set_mode((self.width,self.height))
        pygame.display.set_caption("SNAKE")
        self.clock = pygame.time.Clock()

        #initialize game state
        self.reset()
        


    def manDist(self):
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)


    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.width//2,self.height//2)
        self.snake = [self.head,
                      Point(self.head.x - PIXEL_BLOCK,self.head.y),
                      Point(self.head.x - (2*PIXEL_BLOCK),self.head.y)]
        
        self.score = 0
        self.food = None
        self._random_food()
        self.frame_iteration = 0




    
    def _random_food(self):
        x = random.randint(0, (self.width - PIXEL_BLOCK)//PIXEL_BLOCK) * PIXEL_BLOCK
        y = random.randint(0, (self.height - PIXEL_BLOCK)//PIXEL_BLOCK) * PIXEL_BLOCK
        self.food = Point(x,y)
        if self.food in self.snake:
            self._random_food()




    def _move(self,action):
        #directions -> [straight, right, left]
        clock_wise = [Direction.DOWN,Direction.LEFT,Direction.UP,Direction.RIGHT]
        ind = clock_wise.index(self.direction)

        if np.array_equal(action ,[1,0,0]):
            new_dir = clock_wise[ind]
        elif np.array_equal(action ,[0,1,0]):
            ind = (ind + 1) % 4
            new_dir = clock_wise[ind]
        else:
            ind = (ind - 1) % 4
            new_dir = clock_wise[ind]
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += PIXEL_BLOCK
        elif self.direction == Direction.LEFT:
            x -= PIXEL_BLOCK 
        elif self.direction == Direction.UP:
            y -= PIXEL_BLOCK 
        elif self.direction == Direction.DOWN:
            y += PIXEL_BLOCK 
        self.head = Point(x,y)




    def play_step(self,action):
        self.frame_iteration += 1
        #User input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Move
        before = self.manDist()
        self._move(action)
        self.snake.insert(0,self.head)
        after = self.manDist()
        
        #check if game over
        reward = 0
        game_over = False

        if self.is_coll() or self.frame_iteration > (100 * len(self.snake)):
            reward = -10
            game_over = True
            return reward,game_over,self.score
        
        #Place food random and/or move
        if self.head == self.food:
            reward = 10
            self.score += 1
            self._random_food()
        else:
            if after < before:
                reward += 1
            else:
                reward -= 1
            self.snake.pop()

        #Update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        #return score,game over
        return reward,game_over,self.score
    




    def is_coll(self,pt = None):
        if pt == None:
            pt = self.head
        if (pt.x > (self.width - PIXEL_BLOCK)) or (pt.x < 0) or (pt.y > (self.height - PIXEL_BLOCK)) or (pt.y < 0):
            return True
        
        if pt in self.snake[1:]:
            return True
        
        return False





    def _update_ui(self):
        self.display.fill((0,0,0)) #Black background
        for s in self.snake:
            #Draw snake
            pygame.draw.rect(self.display,(0,123,255),pygame.Rect(s.x,s.y, PIXEL_BLOCK,PIXEL_BLOCK))
            pygame.draw.rect(self.display,(0,200,255),pygame.Rect(s.x+5,s.y+5, 10,10))

        #Draw food
        pygame.draw.rect(self.display,(255,0,0),pygame.Rect(self.food.x,self.food.y,PIXEL_BLOCK,PIXEL_BLOCK))

        text = font.render("Score : "+ str(self.score), True,(255,255,255))
        self.display.blit(text,[0,0])
        pygame.display.flip()





