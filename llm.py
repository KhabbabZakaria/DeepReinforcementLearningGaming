from openai import OpenAI, RateLimitError
import os
from cardsNpoints import card_names

# Load environment variables from project directory
try:
    from dotenv import load_dotenv
    from pathlib import Path
    # Look for .env in the same directory as this file
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)
    # Also try loading from current directory
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, loading from environment variables")
    pass


# Initialize the OpenAI client with API key from environment (lazy initialization)
client = None

def get_client():
    """Lazy initialization of OpenAI client"""
    global client
    if client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            # Debug: show what directory we're looking in
            env_path = Path(__file__).resolve().parent / ".env"
            print(f"Looking for .env at: {env_path}")
            print(f"File exists: {env_path.exists()}")
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it to use LLM features.")
        client = OpenAI(api_key=api_key)
    return client


def model(prompt):
    completions = get_client().chat.completions.create(
        model="gpt-4o-mini",  # Fixed model name (gpt-4.1-mini doesn't exist)
        messages=[{"role": "user", "content": prompt}]
    )
    # Convert to dict format to maintain compatibility with existing code
    return {
        'choices': [
            {
                'message': {
                    'content': completions.choices[0].message.content
                }
            }
        ],
        'usage': {
            'total_tokens': completions.usage.total_tokens
        }
    }


def generate_response(prompt):
    try:
        completions = model(prompt)
        message = completions['choices'][0]['message']['content']
        return message, completions['usage']['total_tokens']
    except RateLimitError:
        return "Sorry, but due to overload of usage, we are currently out of service. Please try again later.", 0



def reward_model_LLM_label(state, cardToThrowList, player_action_str, cardsOriginal):
    # Convert tensor state to list of integers
    if hasattr(state, 'tolist'):
        state = state.tolist()
    else:
        state = list(state)
    
    # Convert to integers
    state = [int(x) for x in state]
    
    # Read the game rules from the markdown file
    rules_path = os.path.join(os.path.dirname(__file__), 'GAME_RULES.txt')
    
    with open(rules_path, 'r', encoding='utf-8') as f:
            game_rules = f.read()

    
    prompt = "The rule of the game is: " + game_rules

    if state[0] == -1:
        playercard1 = ' first card is unknown to the player.'
    else:
        playercard1 = cardsOriginal[state[0]]

    if state[1] == -1:
        playercard2 = ' second card is unknown to the player.'
    else:
        playercard2 = cardsOriginal[state[1]]

    playersCards = 'Player has cards: ' + playercard1 + ' and ' + playercard2 + '.'

    if state[2] == -1:
        opponentCard1 = ' first card is unknown to the player.'
    else:
        opponentCard1 = cardsOriginal[state[2]]

    if state[3] == -1:
        opponentCard2 = ' second card is unknown to the player.'
    else:
        opponentCard2 = cardsOriginal[state[3]]

    OpponentsCards = 'Opponent has cards: ' + opponentCard1 + ' and ' + opponentCard2 + '.'

    prompt += ' ' + playersCards + ' ' + OpponentsCards

    
    if len(cardToThrowList) > 0:
        cardToThrowStr = 'Cards thrown so far in the game are: ' + ', '.join([cardsOriginal[card] for card in cardToThrowList]) + '.'
        prompt += ' ' + cardToThrowStr

    action_by_player = state[-1]
    if action_by_player == 62 or action_by_player == 63:
        prompt += ' Player was asked Do you want to show? Y/n'
    elif action_by_player == 65:
        prompt += ' Player was asked Which card do you wanna swap? Choose between 0 or 1'
    elif action_by_player == 66:
        prompt += ' Player was asked Want to use the special power? Y/n '
    elif action_by_player == 67:
        prompt += ' Player was asked Choose which card of yours to see. Choose between 0 or 1. The last thrown card was special with power of 10 '
    elif action_by_player == 68:
        prompt += ' Player was asked Choose which card of opponent to see. Choose between 0 or 1. The last thrown card was special with power of J '

     
    prompt += ' The player chose to: ' + player_action_str + '.'
    prompt += ' Based on the rules of the game, was this action good or bad for the player? Answer with "very good", "good" or "bad" or "very bad" only.'

    response, tokens_used = generate_response(prompt)
    if response == 'very good':
        return 1.0
    elif response == 'good':
        return 0.5
    elif response == 'bad':
        return -0.5
    elif response == 'very bad':
        return -1.0
    else:
        return 0.0  # Neutral if the response is unexpected


