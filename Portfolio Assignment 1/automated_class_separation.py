
import os
import shutil

# Source folders
source_folder = r"C:\Users\fycce\Documents\GitHub\computah-vision\GarbageDataset\GarbageDataset"
images_folder = os.path.join(source_folder, "images")
labels_folder = os.path.join(source_folder, "labels")

# Destination folder
target_folder = r"C:\Users\fycce\Documents\GitHub\computah-vision\Portfolio Assignment 1"
# Categories
categories = [
    "biodegradable",
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic"
]

# Create destination folders
for category in categories:
    os.makedirs(os.path.join(target_folder, category, "images"), exist_ok=True)
    os.makedirs(os.path.join(target_folder, category, "labels"), exist_ok=True)

# Process images
for filename in os.listdir(images_folder):

    image_path = os.path.join(images_folder, filename)

    if not os.path.isfile(image_path):
        continue

    # Identify category
    for category in categories:

        if filename.lower().startswith(category.lower()):

            # Get image filename without extension
            image_name = os.path.splitext(filename)[0]

            # Destination image folder
            destination_images = os.path.join(
                target_folder, category, "images"
            )

            # Copied image
            shutil.copy(
                image_path,
                os.path.join(destination_images, filename)
            )

            # Find corresponding label
            label_filename = image_name + ".txt"
            label_path = os.path.join(labels_folder, label_filename)

            # Destination label folder
            destination_labels = os.path.join(
                target_folder, category, "labels"
            )

            # Copy corresponding label if it exists
            if os.path.isfile(label_path):

                shutil.copy(
                    label_path,
                    os.path.join(destination_labels, label_filename)
                )

                # print(f"Copied image and label: {filename}")

            else:
                print(f"WARNING: No label found for {filename}")

            break

print("Image and label organization completed!")
