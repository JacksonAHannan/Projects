import csv
import datetime
from collections import defaultdict

# Define the CSV file to save and read data
FILENAME = "monthly_expenses.csv"

# Function to add a new expense
def add_expense():
    date = input("Enter the date (YYYY-MM-DD): ")
    category = input("Enter the category (e.g., Food, Rent, Transport): ")
    description = input("Enter a description: ")
    try:
        amount = float(input("Enter the amount: $"))
    except ValueError:
        print("Invalid amount. Try again.")
        return

    with open(FILENAME, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([date, category, description, amount])
    print("Expense added successfully!\n")

# Function to generate a breakdown
def generate_breakdown():
    expenses_by_category = defaultdict(float)

    try:
        with open(FILENAME, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) != 4:
                    continue  # skip bad rows
                date_str, category, description, amount_str = row
                try:
                    amount = float(amount_str)
                    expenses_by_category[category] += amount
                except ValueError:
                    continue
    except FileNotFoundError:
        print("No expense file found.")
        return

    print("\nMonthly Expense Breakdown:")
    print("----------------------------")
    for category, total in expenses_by_category.items():
        print(f"{category}: ${total:.2f}")

    # Optionally write the breakdown to another CSV
    with open("expense_breakdown.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Category", "Total Amount"])
        for category, total in expenses_by_category.items():
            writer.writerow([category, f"{total:.2f}"])

    print("\nBreakdown also saved to 'expense_breakdown.csv'.")

# Main loop
def main():
    print("Welcome to the Monthly Expense Tracker!")
    while True:
        print("\nOptions:")
        print("1. Add a new expense")
        print("2. Generate monthly breakdown")
        print("3. Exit")

        choice = input("Choose an option: ")
        if choice == '1':
            add_expense()
        elif choice == '2':
            generate_breakdown()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()
