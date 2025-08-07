import random
from collections import Counter
import tkinter as tk
from tkinter import ttk
import math
import threading




def simulate_draws(num_simulations, num_suits):
    """
    Simulate five-card draws and count poker hands.
    Args:
        num_simulations (int): Number of hands to simulate.
        num_suits (int): Number of suits in the deck (2-10).
    """
    all_suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades', 'Swirls', 'Suns', 'Moons', 'Stars', 'Swords', 'Shields']
    suits = all_suits[:num_suits]
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 
             'Jack', 'Queen', 'King', 'Ace']
    
    # Hand counters (initialize to zero)
    pair_count = 0
    two_pair_count = 0
    three_of_a_kind_count = 0
    flush_count = 0
    full_house_count = 0
    four_of_a_kind_count = 0
    five_of_a_kind_count = 0

    # Probability calculations
    total_hands = math.comb(13 * num_suits, 5)
    pair_probability = (math.comb(13,1) * math.comb(num_suits,2) * math.comb(12,3) * math.comb(num_suits,1) * math.comb(num_suits,1) * math.comb(num_suits,1)) / total_hands
    pair_expected = pair_probability * num_simulations
    pair_sd = math.sqrt(num_simulations * pair_probability * (1 - pair_probability))

    two_pair_probability = (math.comb(13,2) * math.comb(num_suits,2) * math.comb(num_suits,2) * math.comb(11,1) * math.comb(num_suits,1)) / total_hands
    two_pair_expected = two_pair_probability * num_simulations
    two_pair_sd = math.sqrt(num_simulations * two_pair_probability * (1 - two_pair_probability))

    three_of_a_kind_probability = (math.comb(13,1) * math.comb(num_suits,3) * math.comb(12,2) * math.comb(num_suits,1) * math.comb(num_suits,1)) / total_hands
    three_of_a_kind_expected = three_of_a_kind_probability * num_simulations
    three_of_a_kind_sd = math.sqrt(num_simulations * three_of_a_kind_probability * (1 - three_of_a_kind_probability))

    full_house_probability = (math.comb(13,1) * math.comb(num_suits,3) * math.comb(12,1) * math.comb(num_suits,2)) / total_hands
    full_house_expected = full_house_probability * num_simulations
    full_house_sd = math.sqrt(num_simulations * full_house_probability * (1 - full_house_probability))

    four_of_a_kind_probability = (math.comb(13,1) * math.comb(num_suits,4) * math.comb(12,1) * math.comb(num_suits,1)) / total_hands
    four_of_a_kind_expected = four_of_a_kind_probability * num_simulations
    four_of_a_kind_sd = math.sqrt(num_simulations * four_of_a_kind_probability * (1 - four_of_a_kind_probability))

    five_of_a_kind_probability = (math.comb(13,1) * math.comb(num_suits,5)) / total_hands
    five_of_a_kind_expected = five_of_a_kind_probability * num_simulations
    five_of_a_kind_sd = math.sqrt(num_simulations * five_of_a_kind_probability * (1 - five_of_a_kind_probability))

    flush_probability = (math.comb(num_suits,1) * math.comb(13,5)) / total_hands
    flush_expected = flush_probability * num_simulations
    flush_sd = math.sqrt(num_simulations * flush_probability * (1 - flush_probability))


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

    # Calculate z-scores and percentiles (after all counts are done)
    pair_z = (pair_count - pair_expected) / pair_sd
    pair_percentile = 100 * (0.5 * (1 + math.erf(pair_z / math.sqrt(2))))

    two_pair_z = (two_pair_count - two_pair_expected) / two_pair_sd
    two_pair_percentile = 100 * (0.5 * (1 + math.erf(two_pair_z / math.sqrt(2))))

    three_of_a_kind_z = (three_of_a_kind_count - three_of_a_kind_expected) / three_of_a_kind_sd
    three_of_a_kind_percentile = 100 * (0.5 * (1 + math.erf(three_of_a_kind_z / math.sqrt(2))))

    full_house_z = (full_house_count - full_house_expected) / full_house_sd
    full_house_percentile = 100 * (0.5 * (1 + math.erf(full_house_z / math.sqrt(2))))  

    four_of_a_kind_z = (four_of_a_kind_count - four_of_a_kind_expected) / four_of_a_kind_sd
    four_of_a_kind_percentile = 100 * (0.5 * (1 + math.erf(four_of_a_kind_z / math.sqrt(2))))

    five_of_a_kind_z = (five_of_a_kind_count - five_of_a_kind_expected) / five_of_a_kind_sd
    five_of_a_kind_percentile = 100 * (0.5 * (1 + math.erf(five_of_a_kind_z / math.sqrt(2))))

    flush_z = (flush_count - flush_expected) / flush_sd
    flush_percentile = 100 * (0.5 * (1 + math.erf(flush_z / math.sqrt(2))))

    # Format results for display
    def format_result(name, count, expected, percentile):
        sample_percentage = count / num_simulations
        if expected is not None:
            return f"{name}: {count} ({sample_percentage:.2%}), Expected: {expected:.2f}, More than {percentile:.2f}% of simulations"
        else:
            return f"{name}: {count} ({sample_percentage:.2%})"

    results = (
        f"Out of {num_simulations} simulations with {num_suits} suits:\n"
        + format_result("Pairs", pair_count, pair_expected, pair_percentile) + "\n"
        + format_result("Two Pairs", two_pair_count, two_pair_expected, two_pair_percentile) + "\n"
        + format_result("Three of a Kind", three_of_a_kind_count, three_of_a_kind_expected, three_of_a_kind_percentile) + "\n"
        + format_result("Flushes", flush_count, flush_expected, flush_percentile) + "\n"
        + format_result("Full House", full_house_count, full_house_expected, full_house_percentile) + "\n"
        + format_result("Four of a Kind", four_of_a_kind_count, four_of_a_kind_expected, four_of_a_kind_percentile) + "\n"
        + format_result("Five of a Kind", five_of_a_kind_count, five_of_a_kind_expected, five_of_a_kind_percentile)
    )

    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, results)
    result_text.config(state=tk.DISABLED)

def run_simulation_from_gui():
    try:
        num_suits = int(suit_slider.get())
        num_simulations = int(simulation_slider.get())
        if not (2 <= num_suits <= 10):
            raise ValueError("Number of suits must be between 2 and 10.")
        if not (1000 <= num_simulations <= 100000):
            raise ValueError("Number of simulations must be between 1,000 and 100,000.")
    except Exception as e:
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Input error: {e}")
        result_text.config(state=tk.DISABLED)
        return

    # Run simulation in a separate thread for GUI responsiveness
    threading.Thread(target=simulate_draws, args=(num_simulations, num_suits), daemon=True).start()

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

    result_text = tk.Text(root, height=50, width=120, state=tk.DISABLED)
    result_text.pack(pady=10)

    root.mainloop()