import os
import pandas as pd

# ✅ Your real dataset path (use forward slash to avoid unicode error)
DATASET_PATH = "C:/Users/Asus/OneDrive/Desktop/face/bollywood_celeb_faces_0"

folders = os.listdir(DATASET_PATH)

metadata_list = []

for idx, folder in enumerate(sorted(folders), start=1):

    folder_path = os.path.join(DATASET_PATH, folder)

    # Skip if not a folder
    if not os.path.isdir(folder_path):
        continue

    metadata_list.append({
        "person_id": idx,
        "folder_name": folder,
        "name": folder.replace("_", " "),
        "profession": "Actor/Actress",
        "description": f"Indian Bollywood celebrity {folder.replace('_', ' ')}"
    })

df = pd.DataFrame(metadata_list)

# Save metadata.csv in your main face folder
df.to_csv("metadata.csv", index=False)

print("✅ metadata.csv created successfully!")