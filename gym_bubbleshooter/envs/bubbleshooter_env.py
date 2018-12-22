import gym
import numpy as np
import copy
import random
import math
from gym import spaces, error


class Bubble():
    def __init__(self, center_x=0, center_y=0, color=None):
        # If color is None, bubble is empty
        self.center_x = center_x
        self.center_y = center_y
        self.color = color


class BubbleShooterEnv(gym.Env):
    # TODO: Add seeding function and fps lock
    metadata = {'render.modes': ['human', 'console'],
                'video.frames_per_second':350}

    gray = (100, 100, 100)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)
    orange = (255, 128, 0)
    purple = (255, 0, 255)
    cyan = (0, 255, 255)
    black = (0, 0, 0)

    colors = [red, green, blue, yellow, orange, purple, cyan]

    def __init__(self):
        self.array_height = 14
        self.array_width = 16
        self.death_line = self.array_height - 2
        self.initial_lines = 5
        self.spacing = 5  # in pixel
        self.bubble_radius = 20  # in pixel
        self.window_height = (self.array_height * self.bubble_radius * 2
                              + self.spacing * (self.array_height + 2)
                              + 6 * self.bubble_radius)
        self.window_width = (self.array_width * self.bubble_radius * 2
                             + self.spacing * (self.array_width + 1)
                             + self.bubble_radius + 0.5 * self.spacing)
        self.start_x = self.window_width / 2.0
        self.start_y = self.window_height - self.spacing - self.bubble_radius
        self.speed = 1  # pixels, affects performance, be careful with too high values!
        self.color_dictionary = {}
        for i in range(len(self.colors)):
            self.color_dictionary[self.colors[i]] = i
        self.action_space = spaces.Discrete(179)
        self.observation_space = spaces.Dict({"next_bubble": spaces.Discrete(len(self.colors)), "board": spaces.MultiDiscrete([
                                             len(self.colors) for i in range(self.array_height * self.array_height)])})
        self.reset()

    def reset(self):
        """
        This function resets the environment and returns the game state.
        """
        self.color_list = copy.deepcopy(self.colors)
        random.shuffle(self.color_list)
        self.board = self._make_blank_board()
        self._set_bubble_positions()
        self._fill_board()
        self.next_bubble = Bubble(
            self.start_x,
            self.start_y,
            self.color_list[0])

        # for rendering
        self.screen = None
        self.last_board = copy.deepcopy(self.board)
        self.last_positions = []
        self.last_color = None

        return self._get_game_state()

    def render(self, mode='human', close=False):
        """
        This function renders the current game state in the given mode.
        """
        if mode == 'console':
            print(self._get_game_state)
        elif mode == "human":
            try:
                import pygame
                from pygame import gfxdraw
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    "{}. (HINT: install pygame using `pip install pygame`".format(e))
            if close:
                pygame.quit()
            else:
                if self.screen is None:
                    pygame.init()
                    self.screen = pygame.display.set_mode(
                        (round(self.window_width), round(self.window_height)))
                clock = pygame.time.Clock()
                
                # Draw old bubbles
                self.screen.fill((255, 255, 255))
                for row in range(self.array_height):
                    for column in range(self.array_width):
                        if self.last_board[row][column].color is not None:
                            bubble = self.last_board[row][column]
                            pygame.gfxdraw.filled_circle(
                                self.screen, round(
                                    bubble.center_x), round(
                                    bubble.center_y), self.bubble_radius, bubble.color)
                pygame.display.update()

                # Draw flying bubble
                last_x, last_y = None, None
                for position in self.last_positions:
                    if last_x is not None and last_y is not None:
                        pygame.gfxdraw.filled_circle(
                            self.screen, round(
                            last_x), round(
                            last_y), self.bubble_radius, (255,255,255))
                    last_x, last_y = position[0], position[1]
                    pygame.gfxdraw.filled_circle(
                            self.screen, round(
                            position[0]), round(
                            position[1]), self.bubble_radius, self.last_color)
                    pygame.display.update()
                    clock.tick(self.metadata["video.frames_per_second"])
                
                # Draw new bubbles
                self.screen.fill((255, 255, 255))
                for row in range(self.array_height):
                    for column in range(self.array_width):
                        if self.board[row][column].color is not None:
                            bubble = self.board[row][column]
                            pygame.gfxdraw.filled_circle(
                                self.screen, round(
                                    bubble.center_x), round(
                                    bubble.center_y), self.bubble_radius, bubble.color)
                pygame.display.update()
        else:
            raise error.UnsupportedMode("Unsupported render mode: " + mode)

    def step(self, action):
        """
        This method steps the game forward one step and
        shoots a bubble at the given angle.

        Parameters
        ----------
        action : int
            The action is an angle between 0 and 180 degrees, that
            decides the direction of the bubble.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing the
                state of the environment.
            reward (float) :
                amount of reward achieved by the previous action.
            episode_over (bool) :
                whether it's time to reset the environment again.
            info (dict) :
                diagnostic information useful for debugging.
        """
        # add one due to discrete action_space
        action += 1
        # test if action is valid
        if action <= 0 or action >= 180:
            raise Exception("Invalid action: {}".format(action))

        self.last_board = copy.deepcopy(self.board)
        self.last_color = self.next_bubble.color

        # shoot bubble until it collides and set it to its new position
        angle = copy.deepcopy(action)
        self.last_positions = []
        while True:
            angle = self._move_next_bubble(angle)
            self.last_positions.append((self.next_bubble.center_x, self.next_bubble.center_y))
            if self._is_collided():
                break
        row, column = self._set_next_bubble_position()
        self.last_positions.append((row, column))

        # calculate all neighbors and delete if two or more of the same color
        # were hit
        neighborhood = self._get_neighborhood(row, column)
        if len(neighborhood) >= 3:
            self._delete_bubbles(neighborhood)
            self._delete_floaters()
            self._update_color_list()

        # create new next_bubble
        random.shuffle(self.color_list)
        self.next_bubble = Bubble(
            self.start_x,
            self.start_y,
            self.color_list[0])

        result, done = self._is_over()
        state = self._get_game_state()
        reward = self._get_reward(len(neighborhood), result)
        return state, reward, done, {}

    def _get_reward(self, bubbles, result):
        """
        This function calculates the reward.
        """
        rewards = {"hit": 1,
                   "miss": -1,
                   "pop": 10,
                   "win": 200,
                   "lost": -200}

        # Return win or loose
        if len(result) > 0:
            return rewards[result]
        # Nothing hit
        elif bubbles == 1:
            return rewards["miss"]
        # Hit at least one bubble of the same color
        elif bubbles < 3:
            return rewards["hit"]
        # Hit enough bubbles to delete
        else:
            return bubbles * rewards["pop"]

    def _update_color_list(self):
        """
        This function updates the color list
        based on what colors are still in the game.
        """
        remaining_colors = set()

        for row in range(self.array_height):
            for column in range(self.array_width):
                if self.board[row][column].color is not None:
                    remaining_colors.add(self.board[row][column].color)

        return list(remaining_colors)

    def _make_blank_board(self):
        """
        This function creates an empty array with the global size and returns it.
        """
        array = []
        for _ in range(self.array_height):
            column = []
            for _ in range(self.array_width):
                column.append(Bubble())
            array.append(column)
        return array

    def _set_bubble_positions(self):
        """
        This function sets the positional arguments of the bubbles
        in the game board.
        """
        # set the x-values for every bubble
        for row in range(self.array_height):
            for column in range(self.array_width):
                self.board[row][column].center_x = (
                    self.bubble_radius * 2 + self.spacing) * column + self.bubble_radius + self.spacing

        # adjust the x-value in every second row
        for row in range(1, self.array_height, 2):
            for column in range(self.array_width):
                self.board[row][column].center_x += self.bubble_radius + \
                    0.5 * self.spacing

        # calculate the row distance on the y-axis based on the spacing
        y_distance = abs(math.sqrt((2 * self.bubble_radius + self.spacing)
                                   ** 2 - (self.bubble_radius + 0.5 * self.spacing)**2))

        # set the y-values for every bubble
        for row in range(self.array_height):
            for column in range(self.array_width):
                self.board[row][column].center_y = self.spacing + \
                    self.bubble_radius + row * y_distance

    def _fill_board(self):
        """
        This function fills the game board's initial
        lines with bubbles.
        """
        for row in range(self.initial_lines):
            for column in range(self.array_width):
                random.shuffle(self.color_list)
                self.board[row][column].color = self.color_list[0]

    def _move_next_bubble(self, angle):
        """
        Moves the next_bubble forward at the global speed.
        """
        # Calculate the movement in x- and y-direction
        if angle == 90:
            xmove = 0
            ymove = self.speed * -1
        elif angle < 90:
            xmove = math.cos(math.radians(angle)) * self.speed * -1
            ymove = math.sin(math.radians(angle)) * self.speed * -1
        else:
            xmove = math.cos(math.radians(180 - angle)) * self.speed
            ymove = math.sin(math.radians(180 - angle)) * self.speed * -1
        self.next_bubble.center_x += xmove
        self.next_bubble.center_y += ymove

        # collision with left wall
        if self.next_bubble.center_x - self.bubble_radius <= self.spacing:
            angle = 180 - angle
        # collision with right wall
        elif self.next_bubble.center_x + self.bubble_radius >= self.window_width - self.spacing:
            angle = 180 - angle
        return angle

    def _set_next_bubble_position(self):
        """
        Sets the next_bubble to its new position in the game board
        and returns the postion.
        """
        # calculate distances to all empty places
        empty_bubble_list = []
        for row in range(self.array_height):
            for column in range(self.array_width):
                if self.board[row][column].color is None:
                    current = self.board[row][column]
                    distance = self._bubble_center_distance(
                        self.next_bubble, current)
                    empty_bubble_list.append(
                        (distance, current.center_x, current.center_y, row, column))

        # select place with smallest distance to next_bubble
        minimum = min(empty_bubble_list, key=lambda t: t[0])

        # set the next_bubble to its new loaction
        self.next_bubble.center_x = minimum[1]
        self.next_bubble.center_y = minimum[2]
        self.board[minimum[3]][minimum[4]] = self.next_bubble

        return minimum[3], minimum[4]

    def _delete_bubbles(self, bubbles):
        """
        Deletes all given bubbles (tuples with x and y coordinate).
        """
        for bubble in bubbles:
            self.board[bubble[0]][bubble[1]].color = None

    def _delete_floaters(self):
        """
        Deletes all floating bubbles.
        """
        # All bubbles to keep
        connected_to_top = set()
        # All bubbles to examine
        pending = set()

        # Add all bubbles in the first row to pending
        for column in range(self.array_width):
            if self.board[0][column].color is not None:
                pending.add((0, column))

        # Calculate all bubbles that are connected to the top
        while len(pending) > 0:
            current = pending.pop()
            connected_to_top.add(current)
            for bubble in self._get_neighbors(
                    current[0], current[1], check_color=False):
                if bubble not in connected_to_top and self.board[bubble[0]
                                                                 ][bubble[1]].color is not None:
                    pending.add(bubble)

        # Get a set of all bubbles
        all_bubbles = set()
        for row in range(self.array_height):
            for column in range(self.array_width):
                if self.board[row][column].color is not None:
                    all_bubbles.add((row, column))

        # Delete bubbles
        to_be_deleted = all_bubbles.difference(connected_to_top)
        self._delete_bubbles(to_be_deleted)

    def _get_neighborhood(self, row, column):
        """
        Returns a list of all coherent bubbles of the same color.
        """
        # all visited bubbles in the neighborhood
        neighborhood = set()
        # all unvisited bubbles in the neighborhood
        pending = set()
        pending.add((row, column))

        while (len(pending) > 0):
            current = pending.pop()
            neighborhood.add(current)
            for bubble in self._get_neighbors(
                    current[0], current[1], self.board[current[0]][current[1]].color, check_color=True):
                if bubble not in neighborhood:
                    pending.add(bubble)

        return neighborhood

    def _get_neighbors(self, row, column, color=None, check_color=True):
        """
        Returns all direct neighbors of a bubble that are in the given color.
        """
        neighbors = []
        if row % 2 == 0:
            if column + 1 < self.array_width:
                if self.board[row][column +
                                   1].color == color or not check_color:
                    neighbors.append((row, column + 1))  # right
            if column - 1 >= 0:
                if self.board[row][column -
                                   1].color == color or not check_color:
                    neighbors.append((row, column - 1))  # left
            if row - 1 >= 0:
                if self.board[row -
                              1][column].color == color or not check_color:
                    neighbors.append((row - 1, column))  # top right
            if row - 1 >= 0 and column - 1 >= 0:
                if self.board[row - 1][column -
                                       1].color == color or not check_color:
                    neighbors.append((row - 1, column - 1))  # top left
            if row + 1 < self.array_height:
                if self.board[row +
                              1][column].color == color or not check_color:
                    neighbors.append((row + 1, column))  # bottom right
            if row + 1 < self.array_height and column - 1 >= 0:
                if self.board[row + 1][column -
                                       1].color == color or not check_color:
                    neighbors.append((row + 1, column - 1))  # bottom left
        else:
            if column + 1 < self.array_width:
                if self.board[row][column +
                                   1].color == color or not check_color:
                    neighbors.append((row, column + 1))  # right
            if column - 1 >= 0:
                if self.board[row][column -
                                   1].color == color or not check_color:
                    neighbors.append((row, column - 1))  # left
            if row - 1 >= 0:
                if self.board[row -
                              1][column].color == color or not check_color:
                    neighbors.append((row - 1, column))  # top left
            if row - 1 >= 0 and column + 1 < self.array_width:
                if self.board[row - 1][column +
                                       1].color == color or not check_color:
                    neighbors.append((row - 1, column + 1))  # top right
            if row + 1 < self.array_height:
                if self.board[row +
                              1][column].color == color or not check_color:
                    neighbors.append((row + 1, column))  # bottom left
            if row + 1 < self.array_height and column + 1 < self.array_width:
                if self.board[row + 1][column +
                                       1].color == color or not check_color:
                    neighbors.append((row + 1, column + 1))  # bottom right
        return neighbors

    def _is_collided(self):
        """
        This function returns true if next_bubble is collided
        with another bubble or the top of the screen, false otherwise.
        """
        # collsion with top
        if self.next_bubble.center_y - self.bubble_radius <= self.spacing:
            return True
        # collision with another bubble
        for row in range(self.array_height):
            for column in range(self.array_width):
                if self.board[row][column].color is None:
                    continue
                distance = self._bubble_center_distance(
                    self.next_bubble, self.board[row][column]) - self.bubble_radius * 2
                if distance < 0:
                    return True
        return False

    def _bubble_center_distance(self, bubble1, bubble2):
        """
        Calculates the distance between the centers of two given bubbles.
        """
        return math.sqrt((bubble1.center_x - bubble2.center_x)
                         ** 2 + (bubble1.center_y - bubble2.center_y)**2)

    def _get_game_state(self):
        """
        This function returns the current game state.
        len(self.colors) means None
        """
        state = {}
        state["next_bubble"] = self.color_dictionary[self.next_bubble.color]
        bubbles = []
        for row in range(self.array_height):
            for column in range(self.array_width):
                if self.board[row][column].color is not None:
                    bubbles.append(
                        self.color_dictionary[self.board[row][column].color])
                else:
                    bubbles.append(len(self.colors))
        state["board"] = bubbles
        return state

    def _is_over(self):
        """
        Returns a string and a bool in which way the game is
        over or not.
        """
        # check if deadline is reached
        for row in range(self.death_line, self.array_height):
            for column in range(self.array_width):
                if self.board[row][column].color is not None:
                    return "lost", True
        # check if board is blank
        for row in range(self.array_height):
            for column in range(self.array_width):
                if self.board[row][column].color is not None:
                    return "", False
        return "win", True
