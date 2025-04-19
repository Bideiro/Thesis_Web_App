import random

def pick_random_lowest_value(d, exclude_value=None):
    # Step 1: Find the minimum value in the dictionary
    min_value = min(d.values())

    # Step 2: Find all keys with the minimum value, excluding those with a value >= exclude_value
    min_keys = [key for key, value in d.items() if value == min_value and (exclude_value is None or value < exclude_value)]

    # Step 3: If no valid keys remain (after excluding), handle the case
    if not min_keys:
        raise ValueError("No valid keys left after excluding the specified value.")
    
    # Step 4: If there are multiple keys with the minimum value, pick one randomly
    return random.choice(min_keys)

# Example usage:
my_dict = {
    'a': 0,
    'b': 0,
    'c': 0,
    'd': 5
}

# Run the function with the exclusion parameter
picked_key = pick_random_lowest_value(my_dict, exclude_value=5)
print(picked_key)  # Will pick a key randomly from the minimum values, excluding 5 and values >= 5
