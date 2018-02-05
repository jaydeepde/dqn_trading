"""The ALEExperiment class handles the logic for training a deep
Q-learning agent in the Arcade Learning Environment.

Author: Nathan Sprague

"""
import logging
import numpy as np
#import cv2
from preprocess_1d import timeseries_into_1d
# Number of rows to crop off the bottom of the (downsampled) screen.
# This is appropriate for breakout, but it may need to be modified
# for other games.
#CROP_OFFSET = 8

ts_granularity = 4 # num of time points per hour

class TradingExperiment(object):
    def __init__(self, agent, resized_width, resized_height,
                 resize_method, num_epochs, epoch_length, test_length,
                 frame_skip, death_ends_episode, max_start_nullops, rng):
        # self.ale = ale
        self.agent = agent
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.frame_skip = frame_skip
        # self.death_ends_episode = death_ends_episode
        self.min_action_set = np.arange(25)
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.resize_method = resize_method
        self.oneD_data, self.twoD_data = timeseries_into_1d("data/napcrack_73.csv",4)
        self.width = self.twoD_data[0][0].shape[0]
        self.height = 1
        self.buffer_length = 1
        self.buffer_count = 0
        self.frame_pointer = 0
        self.last_action = 12
        self.screen_buffer = np.empty((self.buffer_length,
                                        self.width),
                                      dtype=np.float32)

        # self.terminal_lol = False # Most recent episode ended on a loss of life
        # self.max_start_nullops = max_start_nullops
        self.rng = rng

    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        for epoch in range(1, self.num_epochs + 1):
            self.run_epoch(epoch, self.epoch_length)
            self.agent.finish_epoch(epoch)

            if self.test_length > 0:
                self.agent.start_testing()
                self.run_epoch(epoch, self.test_length, True)
                self.agent.finish_testing(epoch)

        self.agent.close_open_files()

    def run_epoch(self, epoch, num_steps, testing=False):
        """ Run one 'epoch' of training or testing, where an epoch is defined
        by the number of steps executed.  Prints a progress report after
        every trial

        Arguments:
        epoch - the current epoch number
        num_steps - steps per epoch
        testing - True if this Epoch is used for testing and not training

        """
        self.terminal_lol = False # Make sure each epoch starts with a reset.
        max_id = self.oneD_data.shape[0] - 12 * ts_granularity
        nofdays = 30 # days
        
        steps_left = num_steps
        while steps_left > 0:
            episode_length = np.random.randint(low=12,high=(12*ts_granularity)*nofdays,size=1)
            self.frame_pointer = np.random.randint(max_id - episode_length, size=1)
            prefix = "testing" if testing else "training"
            logging.info(prefix + " epoch: " + str(epoch) + " steps_left: " +
                         str(steps_left))
           # print('%s epoch: %d steps_left:%d ' %(prefix,epoch,steps_left))
            _, num_steps = self.run_episode(steps_left, testing, episode_length)

            steps_left -= num_steps


    def _init_episode(self):
        """ This method resets the game if needed, performs enough null
        actions to ensure that the screen buffer is ready and optionally
        performs a randomly determined number of null action to randomize
        the initial game state."""

#==============================================================================
#         if not self.terminal_lol or self.ale.game_over():
#             self.ale.reset_game()
#
#             if self.max_start_nullops > 0:
#                 random_actions = self.rng.randint(0, self.max_start_nullops+1)
#                 for _ in range(random_actions):
#                     self._act(0) # Null action
#==============================================================================

        # Make sure the screen buffer is filled at the beginning of
        # each episode...
        self._act(12)
        #self._act(0)


    def _act(self, action):
        """Perform the indicated action for a single frame, return the
        resulting reward and store the resulting screen image in the
        buffer

        """
        # max_id = self.oneD_data.shape[0]-144
        index = self.buffer_count % self.buffer_length
        self.getScreenGrayscale(self.frame_pointer,index)
        self.buffer_count += 1

        curr_rand = self.frame_pointer
        next_rand = curr_rand + 1
        reward = self.act(curr_rand,next_rand,action)
        return reward

    def _step(self, action):
        """ Repeat one action the appopriate number of times and return
        the summed reward. """
        reward = 0
        for _ in range(self.frame_skip):
            reward += self._act(action)

        return reward

    def run_episode(self, max_steps, testing, episode_length):
        """Run a single training episode.

        The boolean terminal value returned indicates whether the
        episode ended because the game ended or the agent died (True)
        or because the maximum number of steps was reached (False).
        Currently this value will be ignored.

        Return: (terminal, num_steps)

        """

        self._init_episode()

        # start_lives = self.ale.lives()

        action = self.agent.start_episode(self.get_observation())

        num_steps = 0
        action_trial = []
        terminal = False
        for _ in range(episode_length):
            
            reward = self._step(self.min_action_set[action])
            action_trial.append([self.frame_pointer, action, reward])
            if self.frame_pointer >= len(self.oneD_data) - 12*ts_granularity:
                break
            #print reward*100
#==============================================================================
#             self.terminal_lol = (self.death_ends_episode and not testing and
#                                  self.ale.lives() < start_lives)
#             terminal = self.ale.game_over() or self.terminal_lol
#==============================================================================
            num_steps += 1 + self.action_forward_duration(action)
            action = self.agent.step(reward, self.get_observation())
        
        terminal = True
        self.agent.end_episode(reward, terminal, action_trial)


        return terminal, num_steps

    def getScreenGrayscale(self,next_rand,index):
        #print str(next_rand) + "," + str(len(self.twoD_data[0]))
        self.screen_buffer[index,...] = self.twoD_data[0][int(next_rand)] # change it for multichanel

    def action_forward_duration(self, action):
        to_add = 0
        if action <= 11:
            to_add = ts_granularity * (action + 1)
        elif action >= 13 <= 24:
            to_add = ts_granularity * (action - 12)
        return to_add

    def act(self, curr_rand, next_rand, action):

        curr_rand = int(curr_rand)
        next_rand = int(next_rand)

        curr_time_slice = self.oneD_data[curr_rand:curr_rand+12*ts_granularity]

        to_add = self.action_forward_duration(action)

        next_rand += to_add
        next_time_slice = self.oneD_data[next_rand:next_rand+12*ts_granularity]
        
        if (next_time_slice.shape[0]!=0):
            reward = (next_time_slice[-1] - curr_time_slice[-1])*100
        else:
            reward = 0
            
        if action == 12:  # do nothing
            reward = 0
        if action <= 11:  # buy
            reward = reward
        if action >= 13 <= 24:  # sell
            reward = -reward

        if self.last_action != action:
            reward -= 0.2  # penalty for changing the action
        self.last_action = action
        self.frame_pointer += to_add + 1
        return reward

    def get_observation(self):

        return self.screen_buffer[0,...]
#==============================================================================
#     def get_observation(self):
#         """ Resize and merge the previous two screen images """
#
#         assert self.buffer_count >= 2
#         index = self.buffer_count % self.buffer_length - 1
#         max_image = np.maximum(self.screen_buffer[index, ...],
#                                self.screen_buffer[index - 1, ...])
#         return self.resize_image(max_image)
#
#     def resize_image(self, image):
#         """ Appropriately resize a single image """
#
#         if self.resize_method == 'crop':
#             # resize keeping aspect ratio
#             resize_height = int(round(
#                 float(self.height) * self.resized_width / self.width))
#
#             resized = cv2.resize(image,
#                                  (self.resized_width, resize_height),
#                                  interpolation=cv2.INTER_LINEAR)
#
#             # Crop the part we want
#             crop_y_cutoff = resize_height - CROP_OFFSET - self.resized_height
#             cropped = resized[crop_y_cutoff:
#                               crop_y_cutoff + self.resized_height, :]
#
#             return cropped
#         elif self.resize_method == 'scale':
#             return cv2.resize(image,
#                               (self.resized_width, self.resized_height),
#                               interpolation=cv2.INTER_LINEAR)
#         else:
#             raise ValueError('Unrecognized image resize method.')
#
#==============================================================================
