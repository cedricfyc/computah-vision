# Raw Dataset Distribution

The distribution is as follows in descending order:

| Class         | Count  | Share  |
|---------------|-------:|-------:|
| Glass         | 3,141  | 26.0%  |
| Biodegradable | 2,287  | 18.9%  |
| Paper         | 2,341  | 19.4%  |
| Cardboard     | 1,883  | 15.6%  |
| Metal         | 1,257  | 10.4%  |
| Plastic       | 1,166  | 9.7%   |
| **Total**     | **12,075** | **100%** |

# Balanced Dataset Distribution

With the re-distribution of the dataset to augment minority classes, the classes should be re-balanced so that the following is considered:
1. Plastic, metal and cardboard are underrepresented.
2. Biodegradable has high intra-class variance. It ranges from fruits, vegetables to bread and fries.
3. The images are already partly flipped and augmented, but can be augmented further for better generalisation.

| Class         | Count  | Share |
|---------------|-------:|------:|
| Glass         | 3,000  | 15%   |
| Biodegradable | 5,000  | 25%   |
| Paper         | 3,000  | 15%   |
| Cardboard     | 3,000  | 15%   |
| Metal         | 3,000  | 15%   |
| Plastic       | 3,000  | 15%   |
| **Total**     | **20,000** | **100%** |

# Possible Improvements
1. Implement subclasses within the biodegradable class. Problem: images are difficult to sort and re-classify along subclasses
2. Higher resolution to images