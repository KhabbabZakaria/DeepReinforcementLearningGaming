from cardsNpoints import *
from helpers import *
import time
import torch
from llm import reward_model_LLM_label

# Global storage for training - separate for each player
saved_log_probs_p1 = []
saved_state_values_p1 = []
saved_action_probs_p1 = []  # For entropy calculation
saved_log_probs_p2 = []
saved_state_values_p2 = []
saved_action_probs_p2 = []  # For entropy calculation

#let us initially play 2x2 1vs1: 2C 2P


def gamePlay(cards, cardsOriginal, agent_p1, agent_p2, noLLM=True):
    ########################################
    ######## Starting game, dealing ########
    ########################################
    #print('let us deal cards. 2 players playing. Each player gets 2 cards')
    # Shuffle and deal the cards
    player1_cards, player2_cards, cards = shuffle_and_deal(cards)
    cardToThrowList = []
    intrinsic_reward_list = []


    ############################################################
    ######## Which indices cards to check from own deck ########
    ############################################################
    # player1ChoiceIdx = int(input('Player 1, which card do you want to see? Choose between 0 or 1: ')) #60
    # player2ChoiceIdx = int(input('Player 2, which card do you want to see? Choose between 0 or 1: ')) #61

    state1 = torch.tensor([-1,-1, -1, -1, 60], dtype=torch.float32)
    state2 = torch.tensor([-1,-1, -1, -1, 61], dtype=torch.float32)

    action1, state_value1, log_prob1, action_prob1 = agent_p1.act(state1)
    player1ChoiceIdx = int(action1.item())
    saved_log_probs_p1.append(log_prob1)
    saved_state_values_p1.append(state_value1)
    saved_action_probs_p1.append(action_prob1)
    
    action2, state_value2, log_prob2, action_prob2 = agent_p2.act(state2)
    player2ChoiceIdx = int(action2.item())
    saved_log_probs_p2.append(log_prob2)
    saved_state_values_p2.append(state_value2)





    ########################################
    ########## Checking own cards ##########
    ########################################
    #print('Player 1, now see your chosen card. Player 2 shut your eyes.')
    state1List = state1.tolist()
    state1List[player1ChoiceIdx] = player1_cards[player1ChoiceIdx]
    state1 = torch.tensor(state1List, dtype=torch.float32)


   # print('Player 2, now see your chosen card. Player 1 shut your eyes.')
    state2List = state2.tolist()
    state2List[player2ChoiceIdx] = player2_cards[player2ChoiceIdx]
    state2 = torch.tensor(state2List, dtype=torch.float32)
   # print(show_card(player2_cards, player2ChoiceIdx, cardsOriginal))
    #print('Now, let us start the game!')




    ########################################
    ########## Actual Game Starts ##########
    ########################################
    while len(cards) > 0:
        #print('Player 1 plays')
        turn = Turn(player1_cards, player2_cards, cards, cardsOriginal, state1, agent_p1, cardToThrowList, reward_model_LLM_label, noLLM)
        turn.singleTurnANDNexts(specialCards)
        player1_cards, player2_cards, cards, state1 = turn.player1_cards, turn.player2_cards, turn.cards, turn.state

        if not noLLM:
            intrinsic_reward_list.extend(turn.intrinsic_reward_list)
            cardToThrowList.extend(turn.cardToThrowList)

        #showChoice = input("Do you want to show? Y/n: ") #62
        state1List = state1.tolist()
        state1List[-1] = 62
        state1 = torch.tensor(state1List, dtype=torch.float32)
        action1, state_value1, log_prob1, action_prob1 = agent_p1.act(state1)
        player1ChoiceIdx = int(action1.item())
        saved_log_probs_p1.append(log_prob1)
        saved_state_values_p1.append(state_value1)
        saved_action_probs_p1.append(action_prob1)

        if player1ChoiceIdx == 1:
            showFirstPlayer = True
            reward_model_LLM_label(state1, cardToThrowList, 'Yes', cardsOriginal)
        else:
            showFirstPlayer = False
            reward_model_LLM_label(state1, cardToThrowList, 'No', cardsOriginal)

        

        #print('Player 2 plays')
        # Player 2 always runs with noLLM=True (no intrinsic rewards)
        turn = Turn(player2_cards, player1_cards, cards, cardsOriginal, state2, agent_p2, cardToThrowList, reward_model_LLM_label, noLLM=True)
        turn.singleTurnANDNexts(specialCards)
        player2_cards, player1_cards, cards, state2 = turn.player1_cards, turn.player2_cards, turn.cards, turn.state
        # Don't extend intrinsic_reward_list for player 2
        if showFirstPlayer is True:
            break

        
        #showChoice = input("Do you want to show? Y/n: ") #63
        state2List = state2.tolist()
        state2List[-1] = 63
        state2 = torch.tensor(state2List, dtype=torch.float32)
        action2, state_value2, log_prob2, action_prob2 = agent_p2.act(state2)
        player2ChoiceIdx = int(action2.item())
        saved_log_probs_p2.append(log_prob2)
        saved_state_values_p2.append(state_value2)
        if player2ChoiceIdx == 1:
            showSecondPlayer = True
            reward_model_LLM_label(state2, cardToThrowList, 'Yes', cardsOriginal)
        else:
            showSecondPlayer = False
            reward_model_LLM_label(state2, cardToThrowList, 'No', cardsOriginal)

        if showSecondPlayer is True:
            #print('Player 1 plays')
            turn = Turn(player1_cards, player2_cards, cards, cardsOriginal, state1, agent_p1, cardToThrowList, reward_model_LLM_label, noLLM)
            turn.singleTurnANDNexts(specialCards)
            player1_cards, player2_cards, cards, state1 = turn.player1_cards, turn.player2_cards, turn.cards, turn.state
            if not noLLM:
                intrinsic_reward_list.extend(turn.intrinsic_reward_list)
                cardToThrowList.extend(turn.cardToThrowList)
            break









    ########################################
    ########## Winner Calculation ##########
    ########################################
    if len(cards) == 0:
        print('...Game Tied...')
        winner = 0
        player1_points, player2_points = 500, 500
    else:
        player1_points, player2_points, winner = winnerCalculation(player1_cards, player2_cards, cardsOriginal, points)
        
    return player1_points, player2_points, winner, intrinsic_reward_list