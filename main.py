import os

# Define the path for the new directory
dir_path = os.path.join("datasets", "agtab")

# Create the directory (and parent directories if they don't exist)
os.makedirs(dir_path, exist_ok=True)

print(f"Directory '{dir_path}' created or already exists.")