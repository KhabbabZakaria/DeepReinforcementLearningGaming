import torch
import argparse
from gameEngineForRLTraining import gamePlay
from actorcritic import Policy, device, Agent
from cardsNpoints import cardsGeneration


def fight(model1_path, model2_path, num_games=100):
    """
    Pit two trained models against each other
    
    Args:
        model1_path: Path to first model .pth file
        model2_path: Path to second model .pth file
        num_games: Number of games to play
    """
    
    # Load the two models
    print(f"Loading Model 1 from: {model1_path}")
    policy1 = Policy().to(device)
    policy1.load_state_dict(torch.load(model1_path, map_location=device))
    policy1.eval()  # Set to evaluation mode
    
    print(f"Loading Model 2 from: {model2_path}")
    policy2 = Policy().to(device)
    policy2.load_state_dict(torch.load(model2_path, map_location=device))
    policy2.eval()  # Set to evaluation mode
    
    # Create agents (both in eval mode)
    agent1 = Agent(policy1, is_eval=True)
    agent2 = Agent(policy2, is_eval=True)
    
    # Track results
    model1_wins = 0
    model2_wins = 0
    ties = 0
    
    print(f"\n{'='*60}")
    print(f"Starting {num_games} games battle!")
    print(f"{'='*60}\n")
    
    # Play the games
    for game in range(num_games):
        cards, cardsOriginal = cardsGeneration()
        
        # Play one game (noLLM=True for pure RL, no intrinsic rewards during eval)
        player1_points, player2_points, winner, _ = gamePlay(
            cards, cardsOriginal, agent1, agent2, noLLM=True
        )
        
        # Track winner
        if winner == 1:
            model1_wins += 1
        elif winner == 2:
            model2_wins += 1
        else:
            ties += 1
        
        # Print progress every 10 games
        if (game + 1) % 10 == 0:
            print(f"Game {game + 1}/{num_games} - Model1: {model1_wins}, Model2: {model2_wins}, Ties: {ties}")
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS after {num_games} games:")
    print(f"{'='*60}")
    print(f"Model 1 Wins: {model1_wins} ({model1_wins/num_games*100:.1f}%)")
    print(f"Model 2 Wins: {model2_wins} ({model2_wins/num_games*100:.1f}%)")
    print(f"Ties:         {ties} ({ties/num_games*100:.1f}%)")
    print(f"{'='*60}")
    
    if model1_wins > model2_wins:
        winner_text = f"🏆 Model 1 WINS the tournament! ({model1_path})"
        print(f"\n{winner_text}")
    elif model2_wins > model1_wins:
        winner_text = f"🏆 Model 2 WINS the tournament! ({model2_path})"
        print(f"\n{winner_text}")
    else:
        winner_text = "🤝 It's a TIE! Both models are equally matched!"
        print(f"\n{winner_text}")
    
    # Save results to file
    with open('fight_results.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Fight Results - {num_games} games\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model 1: {model1_path}\n")
        f.write(f"Model 2: {model2_path}\n")
        f.write(f"\n")
        f.write(f"Model 1 Wins: {model1_wins} ({model1_wins/num_games*100:.1f}%)\n")
        f.write(f"Model 2 Wins: {model2_wins} ({model2_wins/num_games*100:.1f}%)\n")
        f.write(f"Ties:         {ties} ({ties/num_games*100:.1f}%)\n")
        f.write(f"\n")
        f.write(f"{winner_text}\n")
        f.write(f"{'='*60}\n")
    
    print(f"\nResults saved to fight_results.txt")
    
    return model1_wins, model2_wins, ties


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pit two trained models against each other')
    parser.add_argument('--model1', type=str, default='model_final_pureDL.pth', 
                        help='Path to first model .pth file')
    parser.add_argument('--model2', type=str, default='model_final_withLLM.pth', 
                        help='Path to second model .pth file')
    parser.add_argument('--games', type=int, default=100, 
                        help='Number of games to play')
    
    args = parser.parse_args()
    
    fight(args.model1, args.model2, args.games)
