import random
from collections import Counter
import tkinter as tk
from tkinter import ttk
import math

def simulate_draws(num_simulations, num_suits):
    all_suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades', 'Swirls', 'Suns', 'Moons', 'Stars', 'Swords', 'Shields']
    suits = all_suits[:num_suits]
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 
             'Jack', 'Queen', 'King', 'Ace']
    
    pair_probability = (math.comb(13,1) * math.comb(num_suits,2) * math.comb(12,3) * math.comb(num_suits,1) * math.comb(num_suits,1) * math.comb(num_suits,1) / (math.comb((13 * num_suits),5)))
    pair_expected = pair_probability * num_simulations
    
    two_pair_probability = (math.comb(13,2) * math.comb(num_suits,2) * math.comb(num_suits,2) * math.comb(11,1) * math.comb(num_suits,1)) / (math.comb((13 * num_suits),5))
    two_pair_expected = two_pair_probability * num_simulations
    
    triple_probability = (math.comb(13,1) * math.comb(num_suits,3) * math.comb(12,2) * math.comb(num_suits,1) * math.comb(num_suits,1) / (math.comb((13 * num_suits),5)))
    triple_expected = triple_probability * num_simulations
    
    full_house_probability = (math.comb(13,1) * math.comb(num_suits,3) * math.comb(12,1) * math.comb(num_suits,2) / (math.comb((13 * num_suits),5)))
    full_house_expected = full_house_probability * num_simulations
    
    quadruple_probability = (math.comb(13,1) * math.comb(num_suits,4) * math.comb(12,1) * math.comb(num_suits,1) / (math.comb((13 * num_suits),5)))
    quadruple_expected = quadruple_probability * num_simulations
    
    quintuple_probability = (math.comb(13,1) * math.comb(num_suits,5) / (math.comb((13 * num_suits),5)))
    quintuple_expected = quintuple_probability * num_simulations
    
    flush_probability = (math.comb(num_suits,1) * math.comb(13,5) / (math.comb((13 * num_suits),5)))
    flush_expected = flush_probability * num_simulations

    
    pair_count = 0
    two_pair_count = 0
    three_of_a_kind_count = 0
    straight_count = 0
    flush_count = 0
    full_house_count = 0
    four_of_a_kind_count = 0
    five_of_a_kind_count = 0

    deck = [f"{rank} of {suit}" for suit in suits for rank in ranks]

    for _ in range(num_simulations):
        drawn_cards = random.sample(deck, 5)
        ranks_drawn = [card.split(' of ')[0] for card in drawn_cards]
        suits_drawn = [card.split(' of ')[1] for card in drawn_cards]
        rank_frequencies = Counter(ranks_drawn)
        freq_values = list(rank_frequencies.values())

        if 5 in freq_values:
            five_of_a_kind_count += 1
        elif 4 in freq_values:
            four_of_a_kind_count += 1
        elif sorted(freq_values) == [2, 3]:
            full_house_count += 1
        elif 3 in freq_values:
            three_of_a_kind_count += 1
        elif freq_values.count(2) == 2:
            two_pair_count += 1
        elif 2 in freq_values:
            pair_count += 1

        if len(set(suits_drawn)) == 1:
            flush_count += 1

    def format_result(name, count, expected):
        probability = count / num_simulations
        return f"{name}: {count} ({probability:.2%}), Expected: {expected:.2f}"

    results = (
        f"Out of {num_simulations} simulations with {num_suits} suits:\n"
        + format_result("Pairs", pair_count, pair_expected) + "\n"
        + format_result("Two Pairs", two_pair_count, two_pair_expected) + "\n"
        + format_result("Three of a Kind", three_of_a_kind_count, triple_expected) + "\n"
        + format_result("Full House", full_house_count, full_house_expected) + "\n"
        + format_result("Four of a Kind", four_of_a_kind_count, quadruple_expected) + "\n"
        + format_result("Five of a Kind", five_of_a_kind_count, quintuple_expected) + "\n"
        + format_result("Flushes", flush_count, flush_expected)
    )

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, results)

def run_simulation_from_gui():
    num_suits = int(suit_slider.get())
    num_simulations = int(simulation_slider.get())
    simulate_draws(num_simulations=num_simulations, num_suits=num_suits)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Five Card Draw Simulator")

    tk.Label(root, text="Select number of suits:").pack()
    suit_slider = tk.Scale(root, from_=2, to=10, orient='horizontal', length=200, tickinterval=1)
    suit_slider.set(6)
    suit_slider.pack()

    tk.Label(root, text="Select number of simulations:").pack()
    simulation_slider = tk.Scale(root, from_=1000, to=100000, resolution=100, orient='horizontal', length=200)
    simulation_slider.set(1000)
    simulation_slider.pack()

    simulate_button = ttk.Button(root, text="Run Simulation", command=run_simulation_from_gui)
    simulate_button.pack(pady=10)

    result_text = tk.Text(root, height=15, width=60)
    result_text.pack(pady=10)

    root.mainloop()
