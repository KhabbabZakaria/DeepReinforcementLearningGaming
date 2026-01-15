import random
import time
import torch

# Function to shuffle and deal cards to two players
def shuffle_and_deal(cards):
    deck_keys = list(cards.keys())  # Get a list of all card keys
    random.shuffle(deck_keys)       # Shuffle the keys

    # Pop the first 2 cards for player 1 and 2 cards for player 2
    player1_keys = [deck_keys.pop(0), deck_keys.pop(0)]  # Remove and assign 2 cards to player 1
    player2_keys = [deck_keys.pop(0), deck_keys.pop(0)]  # Remove and assign 2 cards to player 2

    # Remove the dealt cards from the cards dictionary
    for key in player1_keys + player2_keys:
        cards.pop(key)
    
    return player1_keys, player2_keys, cards  # Return the dealt card keys and the updated cards deck




# Function to show a specific card from a player's hand
def show_card(player_keys, index, cardsOriginal):
    if index < 0 or index > 1:
        return "Invalid index! Please choose 0 or 1."
    else:
        card_key = player_keys[index]  # Get the card key from the player's hand
        return cardsOriginal[card_key] # Reveal the actual card from the cardsOriginal dictionary



#let us check some rules
#these rules apply when the player throws the card from hand to grave
#This can be a just picked card, or one exchanged card from own self

class Rules:
    def __init__(self, cardToThrow, player1Cards, player2Cards, cardsOriginal, state, agent, show_card=show_card):
        self.cardToThrow = cardToThrow
        self.player1Cards = player1Cards
        self.player2Cards = player2Cards
        self.cardsOriginal = cardsOriginal
        self.state = state
        self.agent = agent
        self.show_card = show_card

    def _rules10(self, index):
        #checks one of player's own card
        return  self.player1Cards[index]
    
    def _rulesJack(self, index):
        #checks one of opponent's own card
        return self.player2Cards[index]

    def _rulesQueen(self, index1, index2):
        #interchange 1x1 cards without checking them
        self.player1Cards[index1], self.player2Cards[index2] = self.player2Cards[index2], self.player1Cards[index1]
        return self.player1Cards, self.player2Cards
    
    def _rulesKing(self, index1, index2):
        #interchange 1x1 cards after checking them
        card1, card2 = self.player1Cards[index1], self.player2Cards[index2]
        card1, card2 = self.show_card(self.player1Cards, index1, self.cardsOriginal), self.show_card(self.player2Cards, index2, self.cardsOriginal)
        # print('Player 1 card is ', card1)
        # print('Player 2 card is ', card2)
        # swap = input('Do you wanna swap? Y/n: ')
        if swap.upper() == 'Y':
            self.player1Cards[index1], self.player2Cards[index2] = self.player2Cards[index2], self.player1Cards[index1]
        return self.player1Cards, self.player2Cards
    
    def act(self):
        if '10' in self.cardToThrow:
            #print('Showing one of your cards. Other player please shut your eyes!')
            #choiceIndex = int(input('Choose which card of yours to see. Choose between 0 or 1: ')) #67
            stateList = self.state.tolist()
            stateList[-1] = 67
            self.state = torch.tensor(stateList, dtype=torch.float32)
            choiceIndex = int(self.agent.act(self.state)[0].item())
            cardToSee = self._rules10(choiceIndex)
            #print('The card you wanted to see is: ', cardToSee)
            stateList = self.state.tolist()
            stateList[choiceIndex] = cardToSee
            self.state = torch.tensor(stateList, dtype=torch.float32)


        if 'J' in self.cardToThrow:
            #print('Showing one of opponents cards. Other player please shut your eyes!')
            #choiceIndex = int(input('Choose which card of opponets to see. Choose between 0 or 1: ')) #68
            stateList = self.state.tolist()
            stateList[-1] = 68
            self.state = torch.tensor(stateList, dtype=torch.float32)
            choiceIndex = int(self.agent.act(self.state)[0].item())
            cardToSee = self._rulesJack(choiceIndex)
            #print('The card you wanted to see is: ', cardToSee)
            stateList = self.state.tolist()
            stateList[choiceIndex+2] = cardToSee
            self.state = torch.tensor(stateList, dtype=torch.float32)

        if 'Q' in self.cardToThrow:
            pass
            # #useQueenPower = input('Use power of Queen? Y/n: ') #69
            # stateList = self.state.tolist()
            # stateList[-1] = 69
            # self.state = torch.tensor(stateList, dtype=torch.float32)
            # choiceIdx = int(agent.act(self.state)[0].item())
            # if choiceIdx == 1:
            #     #swapCardIdxYours = int(input('Choose your card that you wanna swap. Choose between 0 or 1: ')) #70
            #     stateList = self.state.tolist()
            #     stateList[-1] = 70
            #     self.state = torch.tensor(stateList, dtype=torch.float32)
            #     swapCardIdxYours = int(agent.act(self.state)[0].item())
            #     #swapCardIdxOpponents = int(input('Choose opponents card that you wanna swap. Choose between 0 or 1: ')) #71
            #     stateList = self.state.tolist()
            #     stateList[-1] = 71
            #     self.state = torch.tensor(stateList, dtype=torch.float32)
            #     swapCardIdxOpponents = int(agent.act(self.state)[0].item())
            #     self.player1Cards, self.player2Cards = self._rulesQueen(swapCardIdxYours, swapCardIdxOpponents)

        if 'K' in self.cardToThrow:
            pass
            # useQueenPower = input('Use power of King? Y/n: ')
            # if useQueenPower.upper() == 'Y':
            #     swapCardIdxYours = int(input('Choose your card that you wanna see & swap. Choose between 0 or 1: '))
            #     swapCardIdxOpponents = int(input('Choose opponents card that you wanna see & swap. Choose between 0 or 1: '))
            #     self.player1Cards, self.player2Cards = self._rulesKing(swapCardIdxYours, swapCardIdxOpponents)
        
        return self.player1Cards, self.player2Cards

    


class Turn:
    def __init__(self, player1_cards, player2_cards, cards, cardsOriginal, state, agent):
        self.player1_cards = player1_cards
        self.player2_cards = player2_cards
        self.cards = cards
        self.cardsOriginal = cardsOriginal
        self.state = state
        self.agent = agent

    # function to take one card from the deck during the game
    def _draw_card(self):
        deck_keys = list(self.cards.keys())  
        card_key = deck_keys.pop(0) 
        card_value = self.cards.pop(card_key)  
        return card_key, card_value

    def _singleTurn(self):
        # Check if deck is empty before drawing
        if len(self.cards) == 0:
            return None
            
        card_key, card_value = self._draw_card()
        #print('Your new card is: ', card_value)
        #choice = input('Do you want to keep the card or release? Y/n: ') #64
        stateList = self.state.tolist()
        stateList[-1] = 64
        self.state = torch.tensor(stateList, dtype=torch.float32)
        choiceIdx = int(self.agent.act(self.state)[0].item())
        if choiceIdx == 1:
            choice = 'Y'
        else:
            choice = 'N'

        if choice.upper() == 'Y':
            #choiceIndex = int(input('Which card do you wanna swap? Choose between 0 or 1: ')) #65
            stateList = self.state.tolist()
            stateList[-1] = 65
            self.state = torch.tensor(stateList, dtype=torch.float32)
            choiceIndex = int(self.agent.act(self.state)[0].item())

            cardToThrow = show_card(self.player1_cards, choiceIndex, self.cardsOriginal)
            self.player1_cards[choiceIndex] = card_key
        else:
            #print('You did not swap, Going to next')
            cardToThrow = card_value
        return cardToThrow

    def singleTurnANDNexts(self, specialCards):
        cardToThrow = self._singleTurn()
        
        # If no card was drawn (deck empty), return early
        if cardToThrow is None:
            return
            
        #print('You threw away: ', cardToThrow)
        stateList = self.state.tolist()

        if cardToThrow in specialCards:
            #useSpecialPower = input('Want to use the special power? Y/n: ') #66
            stateList[-1] = 66
            self.state = torch.tensor(stateList, dtype=torch.float32)
            choiceIdx = int(self.agent.act(self.state)[0].item())
            if choiceIdx == 1:
                rules = Rules(cardToThrow, self.player1_cards, self.player2_cards, self.cardsOriginal, self.state, self.agent)
                self.player1_cards, self.layer2_cards = rules.act()






def winnerCalculation(player1_cards, player2_cards, cardsOriginal, points):
    player1_points, player2_points = 0, 0
    player1_cardsActual, player2_cardsActual = [], []
    for key in player1_cards:
        value = cardsOriginal[key]
        player1_cardsActual.append(value)
        player1_points += points[value]

    for key in player2_cards:
        value = cardsOriginal[key]
        player2_cardsActual.append(value)
        player2_points += points[value]

    #time.sleep(3)
    # print('Player 1 has these cards: ')
    # print(player1_cardsActual)
    # print('And Player 1 got: ' + str(player1_points) + ' points.')

    #time.sleep(3)

    # print('Player 2 has these cards: ')
    # print(player2_cardsActual)
    # print('And Player 2 got: ' + str(player2_points) + ' points.')

   # time.sleep(3)
    winner =  -1
    if player1_points < player2_points:
        winner = 1
    elif player1_points > player2_points:
        winner = 2
    else:
        winner = 0

    return player1_points, player2_points, winner
    