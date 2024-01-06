from index_advisor_selector.index_selection.swirl_selection.swirl_utils.swirl_com import b_to_mb


class RewardCalculator(object):
    def __init__(self, SCALER=1):
        # (0919): newly added.
        self.SCALER = SCALER
        self.reset()

    def reset(self):
        self.accumulated_reward = 0

    def calculate_reward(self, environment_state):
        current_cost = environment_state["current_cost"]
        previous_cost = environment_state["previous_cost"]
        initial_cost = environment_state["initial_cost"]
        new_index_size = environment_state["new_index_size"]

        assert new_index_size is not None

        reward = self._calculate_reward(current_cost, previous_cost, initial_cost, new_index_size)

        self.accumulated_reward += reward

        return reward

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        raise NotImplementedError


class AbsoluteDifferenceRelativeToStorageReward(RewardCalculator):
    def __init__(self, SCALER=1):
        RewardCalculator.__init__(self, SCALER)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = (previous_cost - current_cost) / new_index_size

        return reward * self.SCALER


class AbsoluteDifferenceToPreviousReward(RewardCalculator):
    def __init__(self, SCALER=1):
        RewardCalculator.__init__(self, SCALER)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = previous_cost - current_cost

        return reward * self.SCALER


# SWIRL

class RelativeDifferenceToPreviousReward(RewardCalculator):
    def __init__(self, SCALER=1):
        RewardCalculator.__init__(self, SCALER)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = (previous_cost - current_cost) / initial_cost

        return reward * self.SCALER


class RelativeDifferenceRelativeToStorageReward(RewardCalculator):
    def __init__(self, SCALER=1):
        RewardCalculator.__init__(self, SCALER)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        assert new_index_size > 0

        # : newly added.
        if initial_cost == 0:
            reward = 0
        else:
            reward = ((previous_cost - current_cost) / initial_cost) / b_to_mb(new_index_size)

        return reward * self.SCALER


# DRLinda

class DRLindaReward(RewardCalculator):
    def __init__(self, SCALER=100):
        RewardCalculator.__init__(self, SCALER)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = (initial_cost - current_cost) / initial_cost

        return reward * self.SCALER


class DRLindaRewardToStorage(RewardCalculator):
    def __init__(self, SCALER=100):
        RewardCalculator.__init__(self, SCALER)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = ((initial_cost - current_cost) / initial_cost) / b_to_mb(new_index_size)

        return reward * self.SCALER


# DQN

class DQNReward(RewardCalculator):
    def __init__(self, SCALER=1):
        RewardCalculator.__init__(self, SCALER)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = (previous_cost - current_cost) / initial_cost

        return reward * self.SCALER


class DQNRewardToStorage(RewardCalculator):
    def __init__(self, SCALER=1):
        RewardCalculator.__init__(self, SCALER)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = ((previous_cost - current_cost) / initial_cost) / b_to_mb(new_index_size)

        return reward * self.SCALER
