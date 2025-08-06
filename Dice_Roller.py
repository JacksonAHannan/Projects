import tkinter as tk
from tkinter import ttk
import random

def roll_dice():
    results = []
    grand_total = 0

    # Flip multiple coins if selected
    coin_count = coin_count_var.get()
    if coin_count > 0:
        flips = [random.choice(["Heads", "Tails"]) for _ in range(coin_count)]
        heads = flips.count("Heads")
        tails = flips.count("Tails")
        results.append(f"Flipped {coin_count} coins: {flips}")
        results.append(f"Heads: {heads}, Tails: {tails}")

    for die in sorted(dice_vars.keys()):
        count = dice_vars[die].get()
        if count > 0:
            rolls = [random.randint(1, die) for _ in range(count)]
            total = sum(rolls)
            grand_total += total
            results.append(f"{count}d{die}: {rolls} (Subtotal: {total})")

    result_box.config(state='normal')
    result_box.delete("1.0", tk.END)

    if results:
        if grand_total > 0:
            results.append(f"\nGrand Total: {grand_total}")
        result_box.insert(tk.END, "\n".join(results))
    else:
        result_box.insert(tk.END, "No dice selected.")
    result_box.config(state='disabled')

# GUI setup
root = tk.Tk()
root.title("Dice Roller")

main_frame = ttk.Frame(root, padding=10)
main_frame.grid(row=0, column=0)

# Coin flip count slider
ttk.Label(main_frame, text="Number of Coins to Flip:").grid(row=0, column=0, sticky="w")
coin_count_var = tk.IntVar(value=0)
coin_spinbox = ttk.Spinbox(main_frame, from_=0, to=10, textvariable=coin_count_var, width=5)
coin_spinbox.grid(row=0, column=1, padx=5)

# Dice sliders (using Spinbox for integer-only selection)
dice_types = [4, 6, 8, 10, 12, 20, 100]
dice_vars = {}

for idx, die in enumerate(dice_types, start=1):
    ttk.Label(main_frame, text=f"d{die}:").grid(row=idx, column=0, sticky="w")
    var = tk.IntVar(value=0)
    dice_vars[die] = var
    spinbox = ttk.Spinbox(main_frame, from_=0, to=10, textvariable=var, width=5)
    spinbox.grid(row=idx, column=1, padx=5)

# Roll button
roll_button = ttk.Button(main_frame, text="Roll!", command=roll_dice)
roll_button.grid(row=len(dice_types) + 1, column=0, columnspan=3, pady=10)

# Result box
result_box = tk.Text(main_frame, width=50, height=12, state='disabled', wrap='word')
result_box.grid(row=len(dice_types) + 2, column=0, columnspan=3)

root.mainloop()
