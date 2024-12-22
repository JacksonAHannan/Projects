import random
# The below function picks a number between 1 and x, where x is input by the user as the maximum value that can be selected
def guess(x):
    random_number = random.randint(1,x)
    guess = 0
    
    while guess != random_number:
        guess = int(input("Guess a number between 1 and 10: "))
        guess = int(guess)
        
        if guess < random_number:
                        print("Sorry, guess is too low!")
        
        elif guess > random_number:
                        print("Sorry, guess is too high!")
            
    print("Good job! Guess is correct!")

guess(10)