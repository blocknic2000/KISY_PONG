import pygame
pygame.init()
import random
from deepQ_agent import my_agent

WIDTH, HEIGHT = 700, 500
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

FPS = 160

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PADDLE_WIDTH, PADDLE_HEIGHT = 20, 100
BALL_RADIUS = 7

SCORE_FONT = pygame.font.SysFont("comicsans", 50)
WINNING_SCORE = 100000
LOAD_MODEL = True  
ai = my_agent(8,3,LOAD_MODEL)


class Paddle:
    COLOR = WHITE
    VEL = 11

    def __init__(self, x, y, width, height):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.width = width
        self.height = height

    def draw(self, win):
        pygame.draw.rect(
            win, self.COLOR, (self.x, self.y, self.width, self.height))

    def move(self, up=True):
        if up:
            self.y -= self.VEL
        else:
            self.y += self.VEL

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y


class Ball:
    MAX_VEL = 12
    COLOR = WHITE

    def __init__(self, x, y, radius):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.radius = radius
        self.x_vel = self.MAX_VEL
        self.y_vel = 0

    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.radius)

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
        self.y_vel = 0
        self.x_vel *= -1


def draw(win, paddles, ball, left_score, right_score):
    win.fill(BLACK)

    left_score_text = SCORE_FONT.render(f"{left_score}", 1, WHITE)
    right_score_text = SCORE_FONT.render(f"{right_score}", 1, WHITE)
    win.blit(left_score_text, (WIDTH//4 - left_score_text.get_width()//2, 20))
    win.blit(right_score_text, (WIDTH * (3/4) -
                                right_score_text.get_width()//2, 20))

    for paddle in paddles:
        paddle.draw(win)

    for i in range(10, HEIGHT, HEIGHT//20):
        if i % 2 == 1:
            continue
        pygame.draw.rect(win, WHITE, (WIDTH//2 - 5, i, 10, HEIGHT//20))

    ball.draw(win)
    pygame.display.update()


def handle_collision(ball, left_paddle, right_paddle):
    reward=0
    rewardleft=0
    if ball.y + ball.radius >= HEIGHT:
        ball.y_vel *= -1
    elif ball.y - ball.radius <= 0:
        ball.y_vel *= -1

    if ball.x_vel < 0:
        if ball.y >= left_paddle.y and ball.y <= left_paddle.y + left_paddle.height:
            if ball.x - ball.radius <= left_paddle.x + left_paddle.width:
                ball.x_vel *= -1

                middle_y = left_paddle.y + left_paddle.height / 2
                difference_in_y = middle_y - ball.y
                reduction_factor = (left_paddle.height / 2) / ball.MAX_VEL
                y_vel = difference_in_y / reduction_factor
                ball.y_vel = -1 * y_vel
                rewardleft=1

    else:
        if ball.y >= right_paddle.y and ball.y <= right_paddle.y + right_paddle.height:
            if ball.x + ball.radius >= right_paddle.x:
                ball.x_vel *= -1

                middle_y = right_paddle.y + right_paddle.height / 2
                difference_in_y = middle_y - ball.y
                reduction_factor = (right_paddle.height / 2) / ball.MAX_VEL
                y_vel = difference_in_y / reduction_factor
                ball.y_vel = -1 * y_vel
                reward=1
    return reward,rewardleft


def handle_paddle_movement(keys, actionrightpaddle,human,actionleftpaddle,left_paddle, right_paddle):
    if human:
        if keys[pygame.K_w] and left_paddle.y - left_paddle.VEL >= 0:
            left_paddle.move(up=True)
        if keys[pygame.K_s] and left_paddle.y + left_paddle.VEL + left_paddle.height <= HEIGHT:
            left_paddle.move(up=False)
    else:
        if actionleftpaddle==0 and left_paddle.y - left_paddle.VEL >= 0:
            left_paddle.move(up=True)
        if actionleftpaddle==1 and left_paddle.y + left_paddle.VEL + left_paddle.height <= HEIGHT:
            left_paddle.move(up=False)


    if actionrightpaddle==0 and right_paddle.y - right_paddle.VEL >= 0:
        right_paddle.move(up=True)
    if actionrightpaddle==1 and right_paddle.y + right_paddle.VEL + right_paddle.height <= HEIGHT:
        right_paddle.move(up=False)



class pong_environment:
    def __init__(self,**kwargs):
        self.render=False
        for key, value in kwargs.items():
            self.render=value
            if self.render:
                print("Rendering on")
            else:
                print("Rendering off")
        
        self.clock = pygame.time.Clock()
        self.left_paddle = Paddle(10, HEIGHT//2 - PADDLE_HEIGHT //
                         2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle = Paddle(WIDTH - 10 - PADDLE_WIDTH, HEIGHT //
                          2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball = Ball(WIDTH // 2, HEIGHT // 2, BALL_RADIUS)

        self.left_score = 0
        self.right_score = 0
        if self.render:
            draw(WIN, [self.left_paddle, self.right_paddle], self.ball, self.left_score, self.right_score)

    def one_step(self,actionrightpaddle,human=True,actionleftpaddle=2):
               
        if self.render:
            self.clock.tick(FPS)
            draw(WIN, [self.left_paddle, self.right_paddle], self.ball, self.left_score, self.right_score)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break

        if human:
            keys = pygame.key.get_pressed()
        else:
            keys=[]
        
        
        
        
            
        handle_paddle_movement(keys,actionrightpaddle,human,actionleftpaddle, self.left_paddle, self.right_paddle)

        # state before the environment step (after paddles moved)
        prev_state = self.get_state()

        # advance ball and compute immediate rewards
        self.ball.move()
        reward=0 # reward for right paddle (AI)
        rewardleft=0 # reward for left paddle (if training both)
        rewright,rewleft=handle_collision(self.ball, self.left_paddle, self.right_paddle)
        reward += rewright
        rewardleft += rewleft

        done = False
        if self.ball.x < 0:
            self.right_score += 1
            reward += 2   # AI scored -> +2
            rewardleft -= 1
            self.ball.reset()
            done = True
        elif self.ball.x > WIDTH:
            self.left_score += 1
            reward -= 1   # Player scored -> negative reward for AI
            rewardleft += 2
            self.ball.reset()
            done = True

        # wenn der Ball das paddle berührt gibt es einen reward von +1
        if self.ball.x + self.ball.radius >= self.right_paddle.x and self.ball.y >= self.right_paddle.y and self.ball.y <= self.right_paddle.y + self.right_paddle.height:
            reward += 1
            print("Ball berührt rechten Paddle Reward +1")
        elif  self.ball.x - self.ball.radius <= self.left_paddle.x + self.left_paddle.width and self.ball.y >= self.left_paddle.y and self.ball.y <= self.left_paddle.y + self.left_paddle.height:
            rewardleft += 1

        won = False
        if self.left_score >= WINNING_SCORE:
            won = True
            win_text = "Left Player Won!"
        elif self.right_score >= WINNING_SCORE:
            won = True
            win_text = "Right Player Won!"

        if won:
            if self.render:
                text = SCORE_FONT.render(win_text, 1, WHITE)
                WIN.blit(text, (WIDTH//2 - text.get_width() //
                            2, HEIGHT//2 - text.get_height()//2))
                pygame.display.update()
                pygame.time.delay(1000)
            self.ball.reset()
            self.left_paddle.reset()
            self.right_paddle.reset()
            self.left_score = 0
            self.right_score = 0
        
       
        balldata = [
        self.ball.x / WIDTH,
        self.ball.y / HEIGHT,
        self.ball.x_vel / self.ball.MAX_VEL,
        self.ball.y_vel / self.ball.MAX_VEL
        ]   

        paddledata = [
        self.left_paddle.x / WIDTH,
        self.left_paddle.y / HEIGHT,
        self.right_paddle.x / WIDTH,
        self.right_paddle.y / HEIGHT
        ]
        # next state after environment step (may be after reset)
        next_state = self.get_state()

        # store experience for right-paddle agent
        ai.memory.append((prev_state, actionrightpaddle, reward, next_state, done))

        #ai.train()
        return next_state, reward, rewardleft, done

    def get_state(self):
        balldata = [
            self.ball.x / WIDTH,
            self.ball.y / HEIGHT,
            self.ball.x_vel / self.ball.MAX_VEL,
            self.ball.y_vel / self.ball.MAX_VEL
        ]
        paddledata = [
            self.left_paddle.x / WIDTH,
            self.left_paddle.y / HEIGHT,
            self.right_paddle.x / WIDTH,
            self.right_paddle.y / HEIGHT
        ]
        return balldata + paddledata

def ai_train():
    env = pong_environment(render=False)  # render=False für schnelleres Training
    print("="*60)
    print("STARTING DEEP Q-LEARNING TRAINING FOR PONG")
    print("="*60)
    
    # Trainiere für eine Anzahl von Episodes/Spielen
    num_episodes = 10
    steps_per_episode = 100
    total_steps = 0
    
    for episode in range(num_episodes):
        state = env.get_state()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(steps_per_episode):
            # Agent wählt Aktion
            actionrightpaddle = ai.get_action(state)
            
            # Speichere den alten State
            prev_state = state
            
            # Spiel einen Schritt weiter - AI vs AI Training
            positiondata, reward, rewardleft, done = env.one_step(
                actionrightpaddle, 
                human=False, 
                actionleftpaddle=2  # linker Paddle macht nichts
            )
            
            # Speichere Experience in Memory
            ai.memory.append((prev_state, actionrightpaddle, reward, positiondata, done))
            
            # Trainiere das Modell
            ai.train()
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            state = positiondata
            
            # Decay Epsilon
            if ai.EPSILON > ai.EPSILON_MIN:
                ai.EPSILON *= ai.EPSILON_DECAY
            
            if done:
                break
        
        # Episode beendet - gib Fortschritt aus
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Steps: {total_steps} | Episode Reward: {episode_reward:.1f} | Epsilon: {ai.EPSILON:.4f}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    ai.save_model()  # Speichern am Ende
  
if __name__ == '__main__':
    if(not LOAD_MODEL):
        ai_train()
    print("\n\nStarting game - Human vs AI (W/S to move)...")
    env2 = pong_environment(render=True)
    # Enable online training while playing: small exploration
    ai.EPSILON = 0.05
    
    # Human controls left paddle with W/S, AI controls right paddle.
    while True:
        state = env2.get_state()
        actionrightpaddle = ai.get_action(state)
        # action=0: up, action=1: down, action=2: do nothing
        positiondata, reward, rewardleft, done = env2.one_step(actionrightpaddle, human=True)


  