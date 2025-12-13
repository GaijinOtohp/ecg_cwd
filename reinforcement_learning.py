import copy
import numpy as np


class Agent (object):

    def __init__(self, environment):
        self._environment = environment
        self._state: list[int] = []

        for dim in range(len(environment._dimensions_list)):
            self._state.append(np.random.randint(0, environment._dimensions_list[dim]._size))
    

    def get_state(self):
        return copy.deepcopy(self._state)
    

    def set_state(self, new_state: list[int]):
        for dim_index in range(len(new_state)):
            if dim_index < len(self._state):
                self._state[dim_index] = new_state[dim_index]


    def convert_to_base3(self, n, size=None):
        """
        Converts a non-negative integer to its base 3 (ternary) representation as a list of int.
        """
        if n == 0:
            digits = [0]
        else:
            digits = []

        while n > 0:
            remainder = n % 3
            digits.append(remainder)
            n //= 3  # Integer division

        if size != None:
            digits.extend([0] * (size - len(digits)))  # Pad with zeros to the left if needed

        digits.reverse() # Reverse the list of digits

        return digits


    def convert_action_to_movement(self, choice: int):
        # Convert the action choice to the new movement
        # Each dimension should have three actions of movements (forward, backward, stay) and multiple mouvements are possible
        raw_movements = self.convert_to_base3(choice, len(self._state))
        movements = [-1 + raw_movement for raw_movement in raw_movements] # Convert 0,1,2 to -1,0,1
            
        return movements


    def observe(self, choice: int):
        # Return the the expected state of the chosen action
        movements = self.convert_action_to_movement(choice)

        return movements, self.move(movements)


    def action(self, choice: int):
        # Change the state of the agnet according to the chosen action
        movements = self.convert_action_to_movement(choice)

        self._state = self.move(movements)


    def move(self, movements: list[int]):
        new_state = self.get_state()
        for dim_index in range(len(movements)):
            movement = movements[dim_index]
            if dim_index >= len(self._state) or movement == 0:
                continue
                
            new_state[dim_index] += movement
            # The new state should be in the right borders
            if (new_state[dim_index] < 0):
                new_state[dim_index] = 0
            elif (new_state[dim_index] >= self._environment._dimensions_list[dim_index]._size):
                new_state[dim_index] = self._environment._dimensions_list[dim_index]._size - 1

        return new_state
    

    def deep_move(self, predicted_state: list[float]):
        new_state = []

        for dimension_index in range(len(predicted_state)):
            dim = self._environment._dimensions_list[dimension_index]
            new_state.append(round(predicted_state[dimension_index] * dim._size))

        for dim_index in range(len(new_state)):
            new_state[dim_index] = max(0, new_state[dim_index])
            new_state[dim_index] = min(new_state[dim_index], self._environment._dimensions_list[dim_index]._size - 1)

        return new_state
            

class Dimension (object):
    
    def __init__(self, name: str, size: int, min: int, max: int):
        self._Name = name
        self._size = size
        self._min = min
        self._max = max
        self._step = (max - min) / size


class Environment (object):

    def __init__(self, dimensions_list: list[Dimension], learning_rate = 0.1, discount = 0.95, compute_reward_delegate = None, check_if_done_delegate = None, _include_additional_state_delegate = None):
        self._q_table_dict = dict()

        self._learning_rate = learning_rate
        self._discount = discount
        self._compute_reward_delegate = compute_reward_delegate
        self._check_if_done_delegate = check_if_done_delegate
        self._include_additional_state_delegate = _include_additional_state_delegate
        self._randomness_epsi = 0.95
        self._epsi_decay = 0.2
        self._epsi_min = 0.001

        self._dimensions_list = dimensions_list

        self.agent_reset()


    def agent_reset(self):
        # Just recreate the agents, which will reset their states
        '''for (int i = 0; i < _AgentsList.Count; i++)
            _AgentsList[i] = new Agent(this)'''
        self._agent = Agent(self)
        # You can try setting the starting states of the agents randomly


    def q_table_reset(self):
        self._q_table_dict = dict()


    def get_q_table_dict(self):
        return self._q_table_dict


    '''Take a step without actually chaging the state of the agent'''
    def step(self, agent: Agent, action: int):
        # Try moving the agent with the corresponding action
        movements, new_state = agent.observe(action)
        reward, bad_state = self._compute_reward_delegate(new_state, movements, self)
        new_state_extended = self.extend_state(new_state)

        return new_state, new_state_extended, reward, bad_state


    '''Append the chosen action to the given state'''
    def set_action_to_state(self, state: list[int], action: int):
        state_with_action = copy.deepcopy(state)
        state_with_action.append(action)
        
        return state_with_action
    

    def extend_state(self, state: list[int]):
        if self._include_additional_state_delegate is not None:
            return self._include_additional_state_delegate(state)
        return state


    def deep_train(self, predicted_init_state: list[float]):
        rd = np.random

        # Reste the new episode's stuff
        self._randomness_epsi = 1
        episode_steps = []
        deep_move_state = self._agent.deep_move(predicted_init_state)
        self._agent.set_state(deep_move_state)
        current_state_extended_with_action = None
        done = False

        last_state = self._agent._state
        bad_state = True

        while (not done):
            new_action = 0
            new_state = None
            include_additional_state = None
            reward = float('-inf')
            # Check if the epsilon of chosing a random action is less than a random value
            if (rd.rand() > self._randomness_epsi):
                # Take the action with the highest reward value
                for action in range(3 ** len(self._dimensions_list)): # There are 3^dim of actions (3 choices for each dimension, and multiple movements are possible)
                    current_state_extended_with_action = self.set_action_to_state(self.extend_state(self._agent.get_state()), action)
                    if tuple(current_state_extended_with_action) in self._q_table_dict:
                        if self._q_table_dict[tuple(current_state_extended_with_action)]["reward"] > reward:
                                new_action = action
                                new_state, new_state_extended, reward, bad_state = self.step(self._agent, action)
                    elif -rd.rand() > reward:
                            new_action = action
                            new_state, new_state_extended, reward, bad_state = self.step(self._agent, action)
            else:
                # Take a random action
                new_action = rd.randint(2 ** len(self._dimensions_list))
                new_state, new_state_extended, reward, bad_state = self.step(self._agent, new_action)
                self._randomness_epsi -= self._epsi_decay
            
            # Set the new reward in qTable by computing the new Q value
            max_future_q = float('-inf')
            for action in range(3 ** len(self._dimensions_list)):
                new_state_extended_with_action = self.set_action_to_state(new_state_extended, action)
                if tuple(new_state_extended_with_action) in self._q_table_dict:
                    if self._q_table_dict[tuple(new_state_extended_with_action)]["reward"] > max_future_q:
                        max_future_q = self._q_table_dict[tuple(new_state_extended_with_action)]["reward"]
                elif -rd.rand() > max_future_q:
                    max_future_q = -rd.rand()

            current_q = -rd.rand()
            current_state_extended_with_action = self.set_action_to_state(self.extend_state(self._agent.get_state()), new_action)
            if tuple(current_state_extended_with_action) in self._q_table_dict:
                current_q = self._q_table_dict[tuple(current_state_extended_with_action)]["reward"]
            new_q = (1 - self._learning_rate) * current_q + self._learning_rate * (reward + self._discount * max_future_q);

            # Update the _QTableDict and episodeQTable with the new Q value
            self._q_table_dict[tuple(current_state_extended_with_action)] = { "reward": new_q, "bad_state": bad_state }

            episode_steps.append({ "state": tuple(current_state_extended_with_action), "reward": new_q, "bad_state": bad_state })

            # Move the agent with the new action
            self._agent.action(new_action)

            # Check if this episode is done
            done = self._check_if_done_delegate(new_state)
            last_state = new_state

            '''if (_randomnessEpsi > _EpsiMin)
            {
                _randomnessEpsi -= _EpsiDecay;
                _randomnessEpsi = Math.Max(_EpsiMin, _randomnessEpsi);
            }'''
        
        # Return the best state in episodeQTable
        bad_state = True
        acceptable_states = {tuple["reward"]:tuple["state"] for tuple in episode_steps if tuple["bad_state"] == False}.items()
        # Check if there was no best state
        if len(acceptable_states) == 0:
            # Then just take the whole Q table
            acceptable_states = {tuple["reward"]:tuple["state"] for tuple in episode_steps}.items()
        else:
            bad_state = False
        # Take the state with the highest reward
        #best_state = max(acceptable_states)[1]
        best_state = last_state
        return (self._q_table_dict, episode_steps, best_state, bad_state)


if __name__ == "__main__":

    import reinforcement_learning

    reinforcementL = reinforcement_learning.ReinforcementLearning()

    input("Press the <Enter> key to continue...\n")

    # Dir 3fays tsuprimé bihom lmemoire