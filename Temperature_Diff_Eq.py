import math

def coffee_cooling_time(T_initial, T_ambient, T_final, k):
    """
    Calculate the time it takes for coffee to cool down to a specific temperature.

    Parameters:
        T_initial (float): Initial temperature of the coffee (in degrees Celsius).
        T_ambient (float): Ambient (room) temperature (in degrees Celsius).
        T_final (float): Desired final temperature of the coffee (in degrees Celsius).
        k (float): Cooling constant (in 1/min).

    Returns:
        float: Time required (in minutes) for the coffee to cool down to T_final.
    """
    if T_final >= T_initial:
        raise ValueError("Final temperature must be less than initial temperature.")
    if T_final < T_ambient:
        raise ValueError("Final temperature cannot be less than ambient temperature.")

    # Newton's Law of Cooling formula rearranged to solve for time:
    # t = (1 / k) * ln((T_initial - T_ambient) / (T_final - T_ambient))
    time = (1 / k) * math.log((T_initial - T_ambient) / (T_final - T_ambient))
    return time

# Example Usage
if __name__ == "__main__":
    # Define parameters
    T_initial = input("Enter an initial coffee temperature: ")  # Initial coffee temperature (°C)
    T_initial = float(T_initial)
    T_ambient = input("Enter an ambient room temperature: ")  # Room temperature (°C)
    T_ambient = float(T_ambient)
    T_final = input("Enter desired final drinking temperature: ")    # Desired drinking temperature (°C)
    T_final = float(T_final)
    k = input("Enter k: ")        # Cooling constant (1/min), adjust based on environment
    k = float(k)

    try:
        time_to_cool = coffee_cooling_time(T_initial, T_ambient, T_final, k)
        print(f"Time required for the coffee to cool to {T_final}°C: {time_to_cool:.2f} minutes")
    except ValueError as e:
        print(f"Error: {e}")
