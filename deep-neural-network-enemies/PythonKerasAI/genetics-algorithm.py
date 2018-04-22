from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import math
import numpy as np
import time
from random import *

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

class Enemy:
  # Player color
  color = None

  # Identification number in the pool
  number = -1

  # Selected as the fittest member in the pool:
  #  1 => yes
  #  0 => no
  selectable = 0

  # Output
  turn_left = 0
  turn_right = 0
  move_forward = 0
  shoot = 0
  change_vision_longer = 0
  change_vision_shorter = 0

  # Input
  see_enemy = 0
  see_bullet = 0
  enemy_fired = 0
  vision_width = 0

  # Neural network model
  model = None

  # Fitness score
  fitness = 0





class GeneticAlgorithm:

  # Setting up game
  def __init__(self):
    # create a new Firefox session
    self.driver = webdriver.Firefox()
    self.driver.implicitly_wait(30)
    self.driver.maximize_window()

    # Navigate to the application home page
    self.driver.get("file:///home/mantas/Desktop/Kursinis%20Darbas/Kursinis/deep-neural-network-enemies/index.html")

  # Core genetic algorithm, performing operations: crossover, mutation, fitness function etc.
  def runAlgorithm(self):
    self.players_pool = []

    x = 0
    enemyNum = 1
    while x < 5:
      # Creating two artificial enemies
      self.enemy_blue = Enemy()
      self.enemy_blue.color = 'blue'
      self.enemy_blue.number = enemyNum
      enemyNum += 1
      self.enemy_brown = Enemy()
      self.enemy_brown.color = 'brown'
      self.enemy_brown.number = enemyNum
      enemyNum += 1

      # Creating feed forward neural network for each player
      self.initiliazeNeuralNetwork(self.enemy_blue)
      self.initiliazeNeuralNetwork(self.enemy_brown)

      # Starting new game round
      self.game_round()

      print("================== Fitness Score ==================")

      fitnes_scores = self.driver.find_element_by_id("result").get_attribute('innerHTML').split(":")
      self.enemy_brown.fitness = int(fitnes_scores[0])
      self.enemy_blue.fitness = int(fitnes_scores[1])

      self.players_pool.append(self.enemy_brown)
      self.players_pool.append(self.enemy_blue)

      # Restarting game with new parameters
      self.driver.refresh()

      x += 1

    # Picking only the fittests and setting selectable to 1
    self.roulette_selection_fittest(self.players_pool)
    self.model_crossover(self.players_pool)

    for player in self.players_pool:
      print("============== Player ================")
      print("player color: ", player.color)
      print("player number: ", player.number)
      print("player fitness: ", player.fitness)
      print("player selectable: ", player.selectable)



  # Initiliazing enemy neural network, 1 generation is made from random weights
  def initiliazeNeuralNetwork(self, enemy):
    enemy.model = Sequential()
    enemy.model.add(Dense(output_dim=6, input_dim=4))
    enemy.model.add(Activation("sigmoid"))
    enemy.model.add(Dense(output_dim=6))
    enemy.model.add(Activation("sigmoid"))
    enemy.model.add(Dense(output_dim=6))
    enemy.model.add(Activation("sigmoid"))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    enemy.model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])



  # Game single round, each round consist of new population
  def game_round(self):
    x = 0
    while x < 70:
      self.upadateCurrentInput(self.enemy_blue, "first", "second")
      self.upadateCurrentInput(self.enemy_brown, "second", "first")

      # Blue predict
      self.predict_action(self.enemy_blue, self.enemy_brown)

      # Brown predict
      self.predict_action(self.enemy_brown, self.enemy_blue)

      x += 1



  # Updates neural network input parameters
  def upadateCurrentInput(self, enemy, first, second):

    # Updating enemy blue
    enemy_detected = self.driver.find_element_by_id(("enemy-detected-" + first)).get_attribute('innerHTML')
    bullet_detected = self.driver.find_element_by_id(("bullet-detected-" + first)).get_attribute('innerHTML')
    degrees = self.driver.find_element_by_id(("degress-" + first)).get_attribute('innerHTML')
    shooted_time = self.driver.find_element_by_id(("shooting-time-" + second)).get_attribute('innerHTML')

    enemy.see_enemy = 0 if enemy_detected == 'false' else 1
    enemy.see_bullet = 0 if bullet_detected == 'false' else 1
    enemy.vision_width = float(degrees) / 124
    enemy.enemy_fired = 1 if float(shooted_time) < 100  else 0

  # Predicts enemy next movement
  def predict_action(self, player, enemy):
    print(player.color, player.see_enemy, player.see_bullet, player.enemy_fired, enemy.vision_width)
    neural_input = np.asarray([player.see_enemy, player.see_bullet, player.enemy_fired, enemy.vision_width])
    neural_input = np.atleast_2d(neural_input)

    output_prob = player.model.predict(neural_input, 1)[0]


    self.controlEnemy(player.color, int(round(output_prob[0], 0)), int(round(output_prob[1], 0)), int(round(output_prob[2], 0)), int(round(output_prob[3], 0)), int(round(output_prob[4], 0)), int(round(output_prob[5], 0)))


  # Controls enemy movement
  def controlEnemy(self, color, turn_left, turn_right, move_forward, shoot, change_vision_longer, change_vision_shorter):
    if(color == 'blue'):
      self.driver.execute_script("circle_two_keys['a_key'] = " + self.toString(turn_right))
      self.driver.execute_script("circle_two_keys['d_key'] = " + self.toString(turn_left))
      self.driver.execute_script("circle_two_keys['w_key'] = " + self.toString(move_forward))
      self.driver.execute_script("circle_two_keys['o_key'] = " + self.toString(change_vision_shorter))
      self.driver.execute_script("circle_two_keys['p_key'] = " + self.toString(change_vision_longer))
      self.driver.execute_script("circle_two_keys['spacebar_key'] = " + self.toString(shoot))
    else:
      self.driver.execute_script("circle_one_keys['right_key'] = " + self.toString(turn_right))
      self.driver.execute_script("circle_one_keys['left_key'] = " + self.toString(turn_left))
      self.driver.execute_script("circle_one_keys['up_key'] = " + self.toString(move_forward))
      self.driver.execute_script("circle_one_keys['z_key'] = " + self.toString(change_vision_longer))
      self.driver.execute_script("circle_one_keys['x_key'] = " + self.toString(change_vision_shorter))
      self.driver.execute_script("circle_one_keys['one_key'] = " + self.toString(shoot))

  def toString(self, value):
    return 'true' if value else 'false'


  def roulette_selection_fittest(self, players):
    self.normalize_fitness_score(players)

    total_fit = float(sum(player.fitness for player in players))
    relative_fitness = [player.fitness / total_fit for player in players]
    probabilities = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]

    found = 0
    while found < 4:
        r = random()
        for (i, player) in enumerate(players):
            # only pick if it haven't been picked before
            if r <= probabilities[i] and players[i].selectable == 0:
                players[i].selectable = 1
                found += 1
                print(i)
                break


  def normalize_fitness_score(self, players):

    # Orders players ascending order
    players.sort(key=lambda x: x.fitness)

    # Find minimal fitness
    min_fitness = 0
    for player in players:
      if player.fitness < min_fitness:
        min_fitness = player.fitness

    # If it's equal or bellow zero, it makes sure it's always positive by adding lowest value + 1
    if min_fitness < 0:
      for player in players:
        player.fitness += abs(min_fitness) + 1
    elif min_fitness == 0:
      for player in players:
        player.fitness += 1


  def model_crossover(self, players):

    indexes = []
    for i, player in enumerate(players):
      if player.selectable == 1:
        indexes.append(i)




    for iteration in range(6):

      # Generating random numbers two chose random parents
      print("Generating random numbers: ")
      rand_num_one = randint(0, 3)
      rand_num_two = randint(0, 3)
      print(rand_num_one)
      print(rand_num_two)

      # Performing crossover, then parents produces offsprings
      weights1 = players[indexes[rand_num_one]].model.get_weights()
      weights2 = players[indexes[rand_num_two]].model.get_weights()
      weightsnew1 = weights1
      weightsnew2 = weights2
      weightsnew1[0] = weights2[0]
      weightsnew2[0] = weights1[0]



      # Performing mutation with probability, probability that offsprint will mutate
      #mutated_weights1 = self.model_mutate(weightsnew1)
      #mutated_weights2 = self.model_mutate(weightsnew2)

      # Adding new offsprings (new members) to populations of fittest


  # Mutates players genes
  def model_mutate(self, weights):
    for xi in range(len(weights)):
      for yi in range(len(weights[xi])):
        if np.random.uniform(0, 1) > 0.50:
          change = np.random.uniform(5, 6)
          weights[xi][yi] += change
    return weights




genetic = GeneticAlgorithm()
genetic.runAlgorithm()
