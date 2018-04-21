from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import math
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

class Enemy:
  # Identification number
  number = 0

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



class GeneticAlgorithm:

  def __init__(self):
    # Creating two artificial enemies
    self.enemy_blue = Enemy()
    self.enemy_blue.number = 1
    self.enemy_brown = Enemy()
    self.enemy_brown.number = 2

    # create a new Firefox session
    self.driver = webdriver.Firefox()
    self.driver.implicitly_wait(30)
    self.driver.maximize_window()

    # Navigate to the application home page
    self.driver.get("file:///home/mantas/Desktop/Kursinis%20Darbas/Kursinis/JavaScript-Shooters/deep-neural-network-enemies/index.html")

    #self.controlEnemy(1, 1, 0, 1, 0, 0, 0)
    #self.controlEnemy(2, 0, 1, 1, 0, 0, 0)

    self.initiliazeNeuralNetwork(self.enemy_blue)
    self.initiliazeNeuralNetwork(self.enemy_brown)

    self.runAlgorithm()


  def runAlgorithm(self):
    #self.controlEnemy(1, 1, 0, 1, 0, 0, 0)

    while True:
      self.upadateCurrentInput(self.enemy_blue, "first", "second")
      self.upadateCurrentInput(self.enemy_brown, "second", "first")

      #print(self.enemy_brown.see_enemy)
      #print(self.enemy_brown.see_bullet)
      #print(self.enemy_brown.vision_width)
      #print(self.enemy_brown.enemy_fired)
      print("---------------------------------")

      #if self.enemy_blue.see_enemy == 1:
        #self.controlEnemy(1, 1, 0, 1, 1, 0, 0)
      #else:
        #self.controlEnemy(1, 1, 0, 1, 0, 0, 0)

      self.predict_action(self.enemy_blue)
      self.predict_action(self.enemy_brown)

  def initiliazeNeuralNetwork(self, enemy):
    enemy.model = Sequential()
    enemy.model.add(Dense(output_dim=7, input_dim=4))
    enemy.model.add(Activation("sigmoid"))
    enemy.model.add(Dense(output_dim=5))
    enemy.model.add(Activation("sigmoid"))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    enemy.model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])

    print(enemy.model.get_weights())


  # Controls enemy movement
  def controlEnemy(self, enemy, turn_left, turn_right, move_forward, shoot, change_vision_longer, change_vision_shorter):
    if(enemy == 1):
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

  def predict_action(self, enemy):
    neural_input = np.asarray([enemy.see_enemy, enemy.see_bullet, enemy.enemy_fired, enemy.vision_width])
    neural_input = np.atleast_2d(neural_input)

    output_prob = enemy.model.predict(neural_input, 1)[0]
    print(output_prob)
    print(round(output_prob[0], 0))
    print(round(output_prob[1], 0))
    print(round(output_prob[2], 0))
    print(round(output_prob[3], 0))

    #round(output_prob[2], 0)



    self.controlEnemy(enemy.number, int(round(output_prob[0], 0)), int(round(output_prob[1], 0)), int(round(output_prob[2], 0)), int(round(output_prob[3], 0)), 0, 0)
    #print(output_prob)


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

genetic = GeneticAlgorithm()
