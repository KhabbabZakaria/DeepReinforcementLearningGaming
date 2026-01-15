import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.bernoulli import Bernoulli

state_size = 5  # Define the size of the state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)

        # actor's layer - output probability for binary action
        self.action_head = nn.Linear(64, 1)

        # critic's layer
        self.value_head = nn.Linear(64, 1)


    def forward(self, state):
        """
        forward of both actor and critic
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Actor: probability of action=1
        action_logits = self.action_head(x)
        action_prob = torch.sigmoid(action_logits)

        # Critic: state value
        state_value = self.value_head(x)

        return action_prob, state_value


policy = Policy()
policy = policy.to(device)




###################################################################
################ Define Agent #####################################
###################################################################
class Agent():
    def __init__(self, policy, is_eval=False, model_name=""):
      self.model_name = model_name
      self.is_eval = is_eval
      self.policy = policy
      if is_eval:
        if model_name:
          self.policy.load_state_dict(torch.load(model_name, map_location=device))
        self.policy.eval()
        # Disable gradients for eval mode
        for param in self.policy.parameters():
          param.requires_grad = False


    def act(self, state):
      # No gradient computation in eval mode
      if self.is_eval:
        with torch.no_grad():
          action_prob, state_value = self.policy(state)
          action_prob_clamped = torch.clamp(action_prob, 1e-6, 1 - 1e-6)
          m = Bernoulli(action_prob_clamped)
          action = m.sample()
          log_prob = m.log_prob(action)
          return action, state_value, log_prob, action_prob
      else:
        action_prob, state_value = self.policy(state)
        action_prob_clamped = torch.clamp(action_prob, 1e-6, 1 - 1e-6)
        m = Bernoulli(action_prob_clamped)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action, state_value, log_prob, action_prob

