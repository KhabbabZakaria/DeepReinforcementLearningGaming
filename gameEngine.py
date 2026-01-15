from cardsNpoints import *
from helpers import *
import time

#let us initially play 2x2 1vs1: 2C 2P


########################################
######## Starting game, dealing ########
########################################
print('let us deal cards. 2 players playing. Each player gets 2 cards')
time.sleep(3)
# Shuffle and deal the cards
player1_cards, player2_cards, cards = shuffle_and_deal(cards)









############################################################
######## Whcih indices cards to check from own deck ########
############################################################
player1ChoiceIdx = int(input('Player 1, which card do you want to see? Choose between 0 or 1: '))
player2ChoiceIdx = int(input('Player 2, which card do you want to see? Choose between 0 or 1: '))









########################################
########## Checking own cards ##########
########################################
print('Player 1, now see your chosen card. Player 2 shut your eyes.')
time.sleep(3)
print(show_card(player1_cards, player1ChoiceIdx, cardsOriginal))

time.sleep(2)

print('Player 2, now see your chosen card. Player 1 shut your eyes.')
time.sleep(3)
print(show_card(player2_cards, player2ChoiceIdx, cardsOriginal))

time.sleep(2)

print('Now, let us start the game!')










########################################
########## Actual Game Starts ##########
########################################
while len(cards) > 0:
    print('Player 1 plays')
    turn = Turn(player1_cards, player2_cards, cards, cardsOriginal)
    turn.singleTurnANDNexts(specialCards)
    player1_cards, player2_cards, cards = turn.player1_cards, turn.player2_cards, turn.cards

    showChoice = input("Do you want to show? Y/n: ")
    if showChoice.upper() == 'Y':
        showFirstPlayer = True
    else:
        showFirstPlayer = False
    
    time.sleep(5)

    print('Player 2 plays')
    turn = Turn(player2_cards, player1_cards, cards, cardsOriginal)
    turn.singleTurnANDNexts(specialCards)
    player2_cards, player1_cards, cards = turn.player1_cards, turn.player2_cards, turn.cards

    if showFirstPlayer is True:
        break

    showChoice = input("Do you want to show? Y/n: ")
    if showChoice.upper() == 'Y':
        showSecondPlayer = True
    else:
        showSecondPlayer = False

    if showSecondPlayer is True:
        print('Player 1 plays')
        turn = Turn(player1_cards, player2_cards, cards, cardsOriginal)
        turn.singleTurnANDNexts(specialCards)
        player1_cards, player2_cards, cards = turn.player1_cards, turn.player2_cards, turn.cards
        break









########################################
########## Winner Calculation ##########
########################################
if len(cards) == 0:
    print('...Game Tied...')
else:
    print(winnerCalculation(player1_cards, player2_cards, cardsOriginal, points))