import pygame
import random
import neat
import pickle

pygame.init()


WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
MAGENTA = (255, 0, 255)


WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Catch the ball!")

class Ball:
    def __init__(self, level=3):
        self.x = random.randint(0, WIDTH)
        self.y = 0
        self.speed = level
        self.radius = 10
        
    def move(self):
        self.y += self.speed

    def draw(self, surface):
        pygame.draw.circle(surface, RED, (self.x, self.y), self.radius)

class Player:
    def __init__(self):
        self.width = 60
        self.height = 10
        self.x = WIDTH // 2 - self.width // 2
        self.y = HEIGHT - 20
        self.speed = 3
        self.points = 0
        self.lives = 5
        self.level = 0

    def move(self, direction):
        self.x += direction * self.speed
        self.x = max(0, min(WIDTH - self.width, self.x))

    def draw(self, surface):
        pygame.draw.rect(surface, BLUE, (self.x, self.y, self.width, self.height))

    def check_collision(self, ball):
        if ball.y + ball.radius > self.y and ball.x > self.x and ball.x < self.x + self.width:
            self.points += 100
            if self.points % 1 == 0:
                self.level += 1
                if ball.speed < 30:
                    ball.speed += 1
                if self.speed < 40:
                    self.speed += 1
            return True
        return False



def draw_text(surface, text, size, x, y, color):
    font_name = pygame.font.match_font('arial')
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surface.blit(text_surface, text_rect)

def eval_genomes(genomes, config):
    best_genome = None
    best_fitness = -1
    initial_speed = 3
    points_threshold = 5
    max_level = 100 // points_threshold
    max_speed = initial_speed + max_level

    #Train all genomes without rendering
    for genome_id, genome in genomes:
        # Create the neural network for the genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        ball = Ball()
        player = Player()

        while player.lives > 0:
            # Get the current state
            distance_to_paddle = abs(ball.y - player.y)
            inputs = (
                ball.x / WIDTH,
                ball.y / HEIGHT,
                distance_to_paddle / HEIGHT,
                player.x / WIDTH, 
                ball.speed / max_speed  
            )
            output = net.activate(inputs)

            # The AI decides to move based on the output
            if output[0] > 0.5:
                player.move(-1)
            if output[1] > 0.5:
                player.move(1)

            ball.move()

            # Check collision with the player paddle
            if player.check_collision(ball):
                ball = Ball(ball.speed)

            # Ball goes out of the screen
            if ball.y > HEIGHT:
                player.lives -= 1
                player.points -= 100
                if player.lives > 0:
                    ball = Ball()
                else:
                    break

        # Assign fitness value to the genome based on the performance
        genome.fitness = player.points

        # Track the best genome
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome = genome

        
# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward.txt')

p = neat.Population(config)

# Add reporters for neat to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Run for a maximum of 200 gerations.
winner = p.run(eval_genomes, 200)

with open('best_genome.pkl', 'wb') as output:
    pickle.dump(winner, output, pickle.HIGHEST_PROTOCOL)

