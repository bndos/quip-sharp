import torch
import os

# Example usage
base_dir = "./Hessians-Llama-31-70B-Instruct-6144-8k-seed-"
groups = [0, 1, 2, 3]
save_dir = "./Hessians-Llama-31-70B-Instruct-6144-8k" 

def merge_and_save_hessian(data, groups, save_dir):
    """
    Merges Hessian components across multiple groups and saves the merged result.

    :param base_dir: Base directory pattern where the Hessian data for each group is stored.
    :param num_groups: Number of groups that contain the Hessians.
    :param save_dir: Directory where the merged Hessians should be saved.
    """
    # Create the save directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Iterate through the files in the first group's directory to get the list of files to process
    total_flatH = None
    total_mu = None
    total_ct = 0
    num_groups = len(groups)
    # Load and sum the data from all groups
    # for group in range(num_groups):
    for group in groups:
        full_path = os.path.join(f'{base_dir}{group}', entry)
        data = torch.load(full_path)

        # Initialize the sum tensors if they're None
        if total_flatH is None:
            total_flatH = torch.zeros_like(data['flatH'])
            total_mu = torch.zeros_like(data['mu'])
        
        # Aggregate the flatH and weighted mu
        total_flatH += data['flatH']
        total_mu += data['mu'] * data['ct']
        total_ct += data['ct']

    # Compute the average mu
    average_mu = total_mu / total_ct if total_ct > 0 else total_mu

    # Save the merged data
    merged_data = {
        'flatH': total_flatH / num_groups,
        'mu': average_mu,
        'n': data['n'],  # Assuming 'n' is the same across all groups
        'ct': total_ct
    }
    
    save_path = os.path.join(save_dir, entry)
    torch.save(merged_data, save_path)
    print(f"Merged data saved to {save_path}")
    

for entry in os.listdir(f'{base_dir}0'):
    data = {}
    for group in groups:
    # for group in range(0, num_groups):
        full_path = os.path.join(f'{base_dir}{group}', entry) 
        print(full_path)
        data[group] = torch.load(full_path)
    merge_and_save_hessian(data, groups, save_dir)
    # exit()
    print('----')
