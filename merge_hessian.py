import torch
import os

# Configuration list where each item is a dictionary containing base_dir, save_dir, and groups
configurations = [
    {
        "base_dir": "./Hessians-Qwen2-57B-A14B-Instruct-6144-8k-seed-",
        "save_dir": "./Hessians-Qwen2-57B-A14B-Instruct-6144-8k",
        "groups": [4, 5]
    },
    # You can add more configuration combinations here
]

def merge_and_save_hessian(base_dir, groups, save_dir, entry):
    """
    Merges Hessian components across multiple groups and saves the merged result.
    """
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize variables
    total_flatH = None
    total_mu = None
    total_ct = 0

    # Load and merge data from all groups
    for group in groups:
        full_path = os.path.join(f'{base_dir}{group}', entry)
        data = torch.load(full_path)

        if total_flatH is None:
            total_flatH = torch.zeros_like(data['flatH'])
            total_mu = torch.zeros_like(data['mu'])

        total_flatH += data['flatH']
        total_mu += data['mu'] * data['ct']
        total_ct += data['ct']

    # Compute the average mu
    average_mu = total_mu / total_ct if total_ct > 0 else total_mu

    # Save the merged data
    merged_data = {
        'flatH': total_flatH / len(groups),
        'mu': average_mu,
        'n': data['n'],  # Assuming 'n' is the same across all groups
        'ct': total_ct
    }
    
    save_path = os.path.join(save_dir, entry)
    torch.save(merged_data, save_path)
    print(f"Merged data saved to {save_path}")

# Iterate over configurations and process each one
for config in configurations:
    for entry in os.listdir(f'{config["base_dir"]}4'):
        merge_and_save_hessian(config["base_dir"], config["groups"], config["save_dir"], entry)
        print('----')
