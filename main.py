import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import argparse
from gameEngineForRLTraining import gamePlay, saved_log_probs_p1, saved_state_values_p1, saved_log_probs_p2, saved_state_values_p2, saved_action_probs_p1
from actorcritic import policy, device, Agent, Policy
from helpers import plot_training_losses

from cardsNpoints import cardsGeneration

# Hyperparameters
episodes = 1000
learning_rate = 0.001
gamma = 0.99  # discount factor
entropy_coef = 0.01  # entropy regularization for exploration

# Create two agents: one for training, one frozen for opponent
policy_train = policy  # This is the training policy
policy_opponent = Policy().to(device)  # Separate policy for opponent
policy_opponent.load_state_dict(policy_train.state_dict())  # Copy weights initially

agent_p1 = Agent(policy_train, is_eval=False)  # Training agent
agent_p2 = Agent(policy_opponent, is_eval=True)  # Frozen opponent

# Optimizer
optimizer = torch.optim.Adam(policy_train.parameters(), lr=learning_rate)


def train(noLLM=True):
    loss_history = []
    actor_loss_history = []
    critic_loss_history = []
    
    for e in range(episodes):
        
        cards, cardsOriginal = cardsGeneration()

        # Reset episode storage for both players
        saved_log_probs_p1.clear()
        saved_state_values_p1.clear()
        saved_action_probs_p1.clear()
        saved_log_probs_p2.clear()
        saved_state_values_p2.clear()



        player1_points, player2_points, winner, intrinsic_reward_list = gamePlay(cards, cardsOriginal, agent_p1, agent_p2, noLLM=noLLM)
        
        # Assign terminal reward based on winner (zero-sum game)
        if winner == 1:
            terminal_reward = 1.0  # Player 1 wins
        elif winner == 2:
            terminal_reward = -1.0  # Player 1 loses
        else:
            terminal_reward = 0.0  # Tie

        # Train ONLY Player 1's experiences
        # Player 2 uses the same model but doesn't contribute to training
        if len(saved_log_probs_p1) == 0:
            continue
        
        # Calculate proper discounted returns combining intrinsic and terminal rewards
        if noLLM or len(intrinsic_reward_list) == 0:
            # Pure RL: all actions get same terminal reward
            returns_p1 = [torch.tensor([terminal_reward], dtype=torch.float32).to(device) 
                          for _ in saved_log_probs_p1]
        else:
            # With LLM: proper discounted returns (Bellman backup)
            # Start from terminal reward and work backwards through intrinsic rewards
            returns_p1 = []
            G = terminal_reward  # Initialize with terminal reward
            
            # Pad or trim intrinsic_reward_list to match number of actions
            num_actions = len(saved_log_probs_p1)
            if len(intrinsic_reward_list) < num_actions:
                # Pad with zeros if we have fewer intrinsic rewards than actions
                padded_intrinsic = intrinsic_reward_list + [0.0] * (num_actions - len(intrinsic_reward_list))
            else:
                # Use only the first num_actions intrinsic rewards
                padded_intrinsic = intrinsic_reward_list[:num_actions]
            
            # Work backwards through the episode
            for intrinsic_r in reversed(padded_intrinsic):
                G = intrinsic_r + gamma * G  # Bellman equation
                returns_p1.insert(0, torch.tensor([G], dtype=torch.float32).to(device))
        
        returns = torch.stack(returns_p1)
        saved_state_values_tensor = torch.stack(saved_state_values_p1).squeeze()
        saved_log_probs_tensor = torch.stack(saved_log_probs_p1)
        
        # Calculate advantage
        advantage = returns.squeeze() - saved_state_values_tensor.detach()
        
        # Normalize advantage for stability
        if len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        # Actor loss (policy gradient)
        actor_loss = -(saved_log_probs_tensor * advantage).mean()
        
        # Add entropy regularization for exploration
        if len(saved_action_probs_p1) > 0:
            action_probs_tensor = torch.stack(saved_action_probs_p1).squeeze()
            # Clamp to avoid log(0)
            action_probs_tensor = torch.clamp(action_probs_tensor, 1e-6, 1 - 1e-6)
            # Bernoulli entropy: -p*log(p) - (1-p)*log(1-p)
            entropy = -(action_probs_tensor * torch.log(action_probs_tensor) +
                       (1 - action_probs_tensor) * torch.log(1 - action_probs_tensor))
            actor_loss -= entropy_coef * entropy.mean()  # Encourage exploration
        
        # Critic loss (value function loss) - lower weight to prevent dominating
        critic_loss = F.mse_loss(saved_state_values_tensor, returns.squeeze())
        
        # Total loss with reduced critic weight
        loss = actor_loss + 0.1 * critic_loss
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(policy_train.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # Periodically update opponent to current policy
        if e % 100 == 0 and e > 0:
            policy_opponent.load_state_dict(policy_train.state_dict())
            print(f"Updated opponent policy at episode {e}")
        
        # Track loss
        loss_history.append(loss.item())
        actor_loss_history.append(actor_loss.item())
        critic_loss_history.append(critic_loss.item())
        
        if e % 10 == 0:
            avg_loss = sum(loss_history[-10:]) / len(loss_history[-10:])
            num_actions_p1 = len(saved_log_probs_p1)
            num_actions_p2 = len(saved_log_probs_p2)
            avg_return = returns.mean().item() if len(returns) > 0 else 0.0
            num_intrinsic = len(intrinsic_reward_list)
            print(f"Episode {e}, Loss: {loss.item():.4f}, Actor: {actor_loss.item():.4f}, Critic: {critic_loss.item():.4f} | Winner: {winner} | P1 actions: {num_actions_p1} | Terminal R: {terminal_reward:.1f}, Avg Return: {avg_return:.3f}, Intrinsic: {num_intrinsic}")
    
    # Save final model
    if noLLM:
        torch.save(policy_train.state_dict(), "model_final_pureDL.pth")
        print("Final pure DL model saved as model_final_pureDL.pth")
    else:
        torch.save(policy_train.state_dict(), "model_final_withLLM.pth")
        print("Final model saved as model_final_withLLM.pth")
    
    # Plot loss curves
    if noLLM:
        plot_training_losses(loss_history, actor_loss_history, critic_loss_history, filename="training_losses_pureDL.png")
    else:
        plot_training_losses(loss_history, actor_loss_history, critic_loss_history, filename="training_losses_withLLM.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the card game AI')
    parser.add_argument('--noLLM', action='store_true', help='Run without LLM (pure RL training)')
    args = parser.parse_args()
    
    train(noLLM=args.noLLM)


