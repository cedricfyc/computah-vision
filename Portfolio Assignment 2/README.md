# Assignment 2, Task 2: Using and Analyzing the Learned Embedding Space

## Setup

The embedding model is trained with triplet loss. It turns each waste image (crop) into a vector of numbers (for example 128), called an embedding. Triplet loss uses an anchor sample (a), a positive sample (p) of the same class, and a negative sample (n) of a different class:

```
L = max( 0,  d(a, p) - d(a, n) + m )

d(x, y) = || f(x) - f(y) ||        (Euclidean distance between embeddings)
m       = margin
```

The loss makes the distance to the positive smaller than the distance to the negative by at least the margin m. As a result, images of the same class end up close together and images of different classes end up far apart.

After training, a labeled **gallery** (training images) serves as the reference set. Test images are used as **queries**.

## Summary of Choices

| Question | Chosen method |
|---|---|
| 1. Find similar objects | k-NN search with Euclidean distance |
| 2. Assign a new sample to a known class | Distance-weighted k-NN voting (k = 10) |
| 3. Visualize the embedding space | UMAP |
| 4. Evaluate the embedding | Precision@K (K = 10) |

Questions 1 and 2 use the same neighbor search on the same gallery. Question 1 returns the neighbors, and Question 2 lets the same neighbors vote for a class.

---

## 1. Question 1: Finding Similar Waste Objects

### 1.1 Options for the search method

| Option | How it works | Advantage | Disadvantage |
|---|---|---|---|
| **k-NN (exact search)** | Compare the query with every gallery vector, return the K closest | Simple, gives the exact neighbors | Search time grows with the gallery size |
| **ANN (approximate nearest neighbor, e.g. FAISS, HNSW)** | Same goal as k-NN, but uses an index to find the neighbors faster | Much faster for very large galleries | Neighbors are only approximately correct, extra tool |
| **Class prototype retrieval** | Compare the query only with one average vector per class | Very fast | Returns classes, not individual similar images |

### 1.2 Options for the distance measure

| Measure | Formula | What it compares | Notes |
|---|---|---|---|
| **Euclidean distance** | `d(x, y) = sqrt( sum( (x_i - y_i)^2 ) )` | The difference in every feature (every number of the vector) | The distance used by triplet loss during training |
| **Cosine similarity** | `cos(x, y) = (x . y) / (‖x‖ ‖y‖)` | Only the angle between the vectors | Ignores vector length, which is not what the model was trained on |
| **Dot product** | `dot(x, y) = x . y` | Angle and lengths together | Long vectors get high scores just for being long |


### 1.3 Chosen method: k-NN search with Euclidean distance

**Steps**
1. Compute the embeddings of all gallery images once and store them.
2. Compute the embedding of the query image.
3. Measure the distance from the query to every gallery vector:

```
d(q, g) = sqrt( (q1 - g1)^2 + (q2 - g2)^2 + ... + (q128 - g128)^2 )
```

4. Return the K = 10 closest images with their class labels.

**Why Euclidean distance**
Triplet loss trained the model by pushing distances up and down, and the distance it used is the Euclidean distance. Using the same distance for retrieval measures similarity in the same way the model learned it. Euclidean distance also looks at every feature, not only at the angle between vectors, so no information from the embedding is thrown away.

**Why k-NN and not ANN**
ANN is the same method as k-NN, only faster and slightly less exact. The gallery has about 10,000 images (about 20,000 after balancing). At this size exact search takes only a fraction of a second, so ANN would add speed that is not needed and a small loss of exactness. If the gallery grew to millions of images, ANN (for example HNSW) would be the way to scale up.

**Why neighbors and not one average vector per class**
The classes are visually varied. Biodegradable images range from fruit and vegetables to bread, and the other classes also contain different-looking items. One average vector for a class would fall between these groups, while neighbor search looks at individual nearby images and works when a class has several groups.

**Interpretation**
If most of the K results have the same class as the query, retrieval works well. If a glass query returns plastic images, the two classes overlap in the embedding space.

---

## 2. Question 2: Assigning a New Sample to a Known Class

### 2.1 Options

| Option | How it works | Advantage | Disadvantage |
|---|---|---|---|
| **k-NN** | The k closest gallery images vote, the most common class wins | No training needed, works when a class has several groups | Large classes can dominate the vote |
| **ANN** | The same vote, with a faster approximate search | Fast for huge galleries | No benefit for a gallery of this size |
| **Class prototype (nearest class mean)** | Average all embeddings of a class into one vector (the prototype); assign the new sample to the class with the closest prototype | Very fast, a new class only needs a new average | Fails if a class has several different-looking groups, because the average lies between them |
| **Multiple prototypes per class** | Use several averages per class (for example found by k-means) | Handles groups inside a class | The number of prototypes must be chosen |
| **Linear classifier (e.g. logistic regression) on the embeddings** | Train a small classifier on the fixed vectors | Often accurate | Needs retraining when a class is added |

### 2.2 Chosen method: distance-weighted k-NN (k = 10)

**Steps**
1. Compute the embedding of the new image.
2. Find its k nearest gallery neighbors.
3. Each neighbor votes for its own class, and closer neighbors count more:

```
weight(neighbor i) = 1 / d_i
score(class c)     = sum of the weights of the neighbors that belong to class c
predicted class    = class with the largest score
```

4. To reduce the effect of class size, divide each class's score by the number of gallery images of that class.

**Why this method**
- Question 1 already finds the neighbors, and every gallery image has a known label. The class of the new image is simply the class most of its neighbors belong to. No extra training is needed.
- The classes are visually varied, so the neighbors of a new image can come from different groups of the same class (for example a bread image next to other bread images). One average vector per class would not capture this.
- The classes have different sizes: 2,287 biodegradable images against 1,162 plastic images in the original set (5,000 against 3,000 in the balanced set). Dividing the scores by class size prevents the bigger classes from winning just because they have more images.
- With 1,162 images in the smallest class, k = 10 is small enough that the neighbors of a plastic image are not forced into other classes.


**Interpretation**
The predicted class is checked against the true class on the test images. Per-class accuracy shows which classes are confused.

---

## 3. Question 3: Visualizing the Embedding Space

### 3.1 Why a projection is needed
Each image is a point in many dimensions (for example 128), which cannot be drawn. We reduce it to 2 dimensions. Every method loses some information, and the methods differ in what they keep.

### 3.2 Options

| Method | What it keeps | Advantage | Disadvantage |
|---|---|---|---|
| **PCA** | The directions with the most spread (linear projection) | Fast, always the same result | Separate classes can end up on top of each other |
| **LDA** | The directions that separate the classes best (uses labels) | Shows class separation | Uses labels, so it can make the separation look better than it is |
| **MDS / Isomap** | Distances between points | Intuitive idea | Slow, crowded plots on larger data |
| **t-SNE** | Nearby points stay nearby | Very clear groups | Distances between groups and group sizes are not meaningful, slow |
| **UMAP** | Nearby points stay nearby, plus more of the overall layout | Clear groups, better overall layout, faster than t-SNE | Distances are still distorted, results depend on settings |


### 3.3 Chosen method: UMAP on the test embeddings, colored by true class

**Why UMAP**
1. Biodegradable images include fruit, vegetables and bread, so the class is expected to split into several groups. Other varied classes may split too. PCA tends to blur such groups, while UMAP shows them.
2. UMAP keeps the overall layout better than t-SNE, so we can see which classes sit next to each other and whether any classes overlap.
3. The overlap seen in the plot can be compared with the confusion matrix from Assignment 1.
4. UMAP handles several thousand points quickly, which suits a test set of this size.

t-SNE would also work. UMAP is preferred for its better overall layout and faster computation.

**Steps**
1. Compute the embeddings of the test images (original, non-augmented).
2. Run UMAP to get 2 numbers per image (fixed random seed; repeat with 2 or 3 seeds to check that the picture is stable).
3. Plot each image as a point colored by class, and mark images that were misclassified in Question 2.
4. Limit the number of points per class (for example 500) so that the larger classes do not hide the smaller ones.

**Interpretation**

| What we see | What it means |
|---|---|
| Each class forms its own region | Good embedding |
| Biodegradable forms several regions | Expected, because the class contains very different items |
| Two classes share a region | Confusable classes, compare with Assignment 1 |
| A point inside another class's region | Hard sample or possible label error |
| Train and test points in the same regions | The embedding generalizes |

Distances and sizes in the 2D plot are distorted, so the plot is only used to judge overlap and separation.

---

## 4. Question 4: Evaluating the Embedding

### 4.1 Options

| Metric | Definition | What it measures | Problem for this dataset |
|---|---|---|---|
| **Precision@K** | Correct images in the top K divided by K | How clean the whole list of K results is | Does not show the order inside the top K |
| **Recall@K** | Share of test images with at least one correct-class image in the top K | Does the search find something correct? | Easy to reach with six classes, and wrong results in the list are ignored |
| **MRR** | Average of 1 / (position of the first correct result) | How early the first correct result appears | Looks only at the first correct result |
| **F1@K** | Harmonic mean of Precision@K and the search-engine Recall@K | One combined number | Dominated by the very small recall value |
| **NDCG@K** | Rewards correct results that appear higher in the list | Ranking quality with graded relevance | No graded relevance in this task |
| **MAP@R** | Average precision over the top R results (R = number of correct items) | Quality of the whole neighborhood | R would be in the thousands, and it is strict for classes with several groups |
| **NMI / ARI** | How well automatic clusters match the true classes | Overall cluster quality | Assume one compact cluster per class |
| **Silhouette score** | Compactness of clusters versus separation between them | Cluster shape | Assumes one compact cluster per class |

### 6.2 Chosen metric: Precision@K with K = 10

**Approach**
Take one test image and look at its 10 closest images in the gallery. Count how many of them have the same class as the test image, and divide by 10. Do this for every test image and take the average.

```
Precision@10 = (1 / N) * sum( correct_i / 10 )        N = number of test images
                                                      correct_i = number of the 10 closest images with the correct class
```
- On original, non-augmented test images only. Many images in the dataset are already flipped or augmented, so copies of one photo must stay on the same side of the split. Otherwise the closest gallery image of a test image would be its own copy and the result would look better than it is.
- The query itself is excluded from the gallery.
- Per class first, then averaged over the six classes (macro average). This gives plastic (11.5% of the images) the same weight as biodegradable (22.6%).


**Why this metric**
1. The method uses many neighbors. Question 1 returns the 10 closest images, and Question 2 lets 10 neighbors vote for the class. Precision@10 judges all of these neighbors.
2. The dataset has thousands of images per class (the smallest class has 1,162), so there are always at least 10 correct images that could appear in the list.
3. It counts the wrong results too. With only six classes, a metric that needs just one correct image in the list (Recall@K) is easy to reach even by guessing, and wrong results would not matter.
4. It has a clear reference for guessing: a random list has a precision equal to the share of the class (22.6% for biodegradable, 11.5% for plastic), whatever K is.
5. The classes are visually varied (for example biodegradable with fruit, bread, and fries). Looking at only 10 neighbors, not the whole class, means a bread image is judged by other bread-like images. Metrics that need the whole class to be close (MAP@R, NMI, silhouette).

**Interpretation**
A high Precision@10 means most of the retrieved images, and most of the neighbors that vote in Question 2, have the correct class. A low value for one class points to confusable class pairs or too few varied training images for that class.
