import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from gameEngineForRLTraining import gamePlay, saved_log_probs_p1, saved_state_values_p1, saved_log_probs_p2, saved_state_values_p2, saved_action_probs_p1
from actorcritic import policy, device, Agent, Policy

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


def train():
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

        player1_points, player2_points, winner = gamePlay(cards, cardsOriginal, agent_p1, agent_p2)
        
        # Assign rewards based on winner (zero-sum game)
        # Using smaller rewards for better critic learning
        if winner == 1:
            reward1 = 1.0  # Player 1 wins
            reward2 = -1.0  # Player 2 loses
        elif winner == 2:
            reward1 = -1.0  # Player 1 loses
            reward2 = 1.0  # Player 2 wins
        else:
            reward1 = 0.0  # Tie
            reward2 = 0.0  # Tie

        # Convert to tensors
        reward1 = torch.tensor([reward1], dtype=torch.float32).to(device)
        reward2 = torch.tensor([reward2], dtype=torch.float32).to(device)

        # Train ONLY Player 1's experiences
        # Player 2 uses the same model but doesn't contribute to training
        if len(saved_log_probs_p1) == 0:
            continue
        
        # Only use Player 1's experiences
        returns_p1 = [reward1 for _ in saved_log_probs_p1]
        
        returns = torch.stack(returns_p1)
        saved_state_values_tensor = torch.stack(saved_state_values_p1).squeeze()
        saved_log_probs_tensor = torch.stack(saved_log_probs_p1)
        
        # Calculate advantage
        advantage = returns - saved_state_values_tensor.detach()
        
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
            print(f"Episode {e}, Loss: {loss.item():.4f}, Actor: {actor_loss.item():.4f}, Critic: {critic_loss.item():.4f} | Winner: {winner} | P1 actions: {num_actions_p1}, P2 actions: {num_actions_p2} | Rewards: P1={reward1.item():.1f}, P2={reward2.item():.1f}")
    
    # Save final model
    torch.save(policy_train.state_dict(), "model_final.pth")
    print("Final model saved as model_final.pth")
    
    # Plot loss curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total Loss
    ax1 = axes[0, 0]
    ax1.plot(loss_history, alpha=0.3, label='Total Loss', color='blue')
    window_size = 50
    if len(loss_history) >= window_size:
        moving_avg = [sum(loss_history[i:i+window_size])/window_size 
                      for i in range(len(loss_history)-window_size+1)]
        ax1.plot(range(window_size-1, len(loss_history)), moving_avg, 
                linewidth=2, label=f'Moving Avg ({window_size} episodes)', color='darkblue')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Actor Loss
    ax2 = axes[0, 1]
    ax2.plot(actor_loss_history, alpha=0.3, label='Actor Loss', color='red')
    if len(actor_loss_history) >= window_size:
        moving_avg_actor = [sum(actor_loss_history[i:i+window_size])/window_size 
                           for i in range(len(actor_loss_history)-window_size+1)]
        ax2.plot(range(window_size-1, len(actor_loss_history)), moving_avg_actor, 
                linewidth=2, label=f'Moving Avg ({window_size} episodes)', color='darkred')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Actor Loss Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Critic Loss
    ax3 = axes[1, 0]
    ax3.plot(critic_loss_history, alpha=0.3, label='Critic Loss', color='green')
    if len(critic_loss_history) >= window_size:
        moving_avg_critic = [sum(critic_loss_history[i:i+window_size])/window_size 
                            for i in range(len(critic_loss_history)-window_size+1)]
        ax3.plot(range(window_size-1, len(critic_loss_history)), moving_avg_critic, 
                linewidth=2, label=f'Moving Avg ({window_size} episodes)', color='darkgreen')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title('Critic Loss Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Combined comparison
    ax4 = axes[1, 1]
    if len(actor_loss_history) >= window_size:
        ax4.plot(range(window_size-1, len(actor_loss_history)), moving_avg_actor, 
                linewidth=2, label='Actor Loss', color='red')
        ax4.plot(range(window_size-1, len(critic_loss_history)), moving_avg_critic, 
                linewidth=2, label='Critic Loss', color='green')
        ax4.plot(range(window_size-1, len(loss_history)), moving_avg, 
                linewidth=2, label='Total Loss', color='blue')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Loss')
    ax4.set_title('All Losses Comparison (Moving Avg)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_loss.png')
    print("Loss plots saved as training_loss.png")
    plt.savefig('training_loss.png')
    print("Loss plot saved as training_loss.png")
    plt.show()


if __name__ == "__main__":
    train()


