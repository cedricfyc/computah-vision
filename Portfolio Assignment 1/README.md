# Raw Dataset Distribution

The distribution is as follows in descending order:

| Class         | Count  | Share  |
|---------------|-------:|-------:|
| Biodegradable | 2,287  | 22.6%  |
| Glass         | 2,220  | 22.0%  |
| Paper         | 1,746  | 17.3%  |
| Cardboard     | 1,439  | 14.2%  |
| Metal         | 1,248  | 12.4%  |
| Plastic       | 1,162  | 11.5%   |
| **Total**     | **10,102** | **100.0%** |

# Balanced Dataset Distribution

With the re-distribution of the dataset to augment minority classes, the classes should be re-balanced so that the following is considered:
1. Plastic, metal and cardboard are underrepresented.
2. Biodegradable has high intra-class variance. It ranges from fruits, vegetables to bread and fries.
3. The images are already partly flipped and augmented, but can be augmented further for better generalisation.

The following table was aimed for the balanced augmented dataset.

| Class         | Count  | Share |
|---------------|-------:|------:|
| Biodegradable | 5,000  | 25%   |
| Glass         | 3,000  | 15%   |
| Paper         | 3,000  | 15%   |
| Cardboard     | 3,000  | 15%   |
| Metal         | 3,000  | 15%   |
| Plastic       | 3,000  | 15%   |
| **Total**     | **20,000** | **100.0%** |


# Choice of Augmentations

The following augmentations were applied:

1. Version Preprocessing
- Auto-Orient
- Stretch to 512*512

The 512\*512 is expected to display more of the image compared to the 416\*416 although it would make the fine-tuning process more computationally expensive.

2. Version Augmentations
- Outputs per training example: 3
- Rotation: Between -5° and +5°
- Saturation: Between -10% and +10%
- Brightness: Between -10% and +10%
- Exposure: Between -5% and +5%
- Blur: Up to 1px
- Noise: Up to 0.1% of pixels

**Note**: 
- No augmentation was applied other than the preprocessing on the glass dataset. The dataset was large enough and glass is expected to be easily distinguishable.
- No saturation modifier was applied on the biodegradable class due to the already high intra-class variance. The items within the class are expected to be reliably identified based on the colours.

# Final Augmented Dataset Distribution

In the end, the following dataset was achieved after augmentation:

| Class         | Count  | Share  |
|---------------|-------:|-------:|
| Biodegradable | 5,328  | 26.4%  |
| Cardboard     | 3,453  | 17.1%  |
| Metal         | 2,996  | 14.8%  |
| Paper         | 2,968  | 14.7%  |
| Plastic       | 2,788  | 13.8%  |
| Glass         | 2,649  | 13.1%  |
| **Total**     | **20,182** | **100.0%** |

**Note**: The biodegradable class was deliberately oversampled due to the high intra-class variance.

The final split for the datasets before augmentation is 70%, 20%, 10% for training, validation and testing respectively.
The distribution of the augmented dataset is now: 84.4%, 10.4%, 5.2%.
**Note**: Only the images in the training dataset are augmented. Validation and testing datasets remain the same in terms of size.
- train: 17,042 images
- valid: 2,093 images
- test: 1,047 images

# Possible Improvements
1. Implement subclasses within the biodegradable class. Problem: images are difficult to sort and re-classify along subclasses
2. Higher resolution to images
3. Images with several classes can be oversampled to increase detection of several classes in one image.