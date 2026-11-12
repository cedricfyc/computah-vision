# Waste detection: training and evaluation

## Task 2: Model training and evaluation

### Training

We trained **YOLO26m** to detect six waste categories: biodegradable, cardboard, glass, metal, paper, and plastic. The selected [best model](5th_30%20epochs_2gpu_chosen/best.pt) started from pretrained weights and used 512 × 512 images, batch size 8, and two T4 GPUs. The training set contained 17,042 images; 2,093 images were used for validation.

The final run was configured for up to 30 epochs, with early stopping after 10 epochs without improvement. It stopped at epoch 24. The best monitored validation result was at **epoch 14**. Cedric's dataset already contained augmented training images, so additional YOLO color, geometric, flip, and mixing augmentations were configured off for the run.

### Code

The  [Python script](waste_detection.py) reconstructs the 5th run's training and evaluation settings from the saved configuration and evaluation notebook. It is using our dataset.


### Test-set results

We evaluated the selected model on 1,047 test images containing 7,626 annotated objects, at image size 512. The table shows precision, recall, and average precision (AP) for each class. Higher threshold is better :)

| Category | Precision | Recall | AP50 | AP50–95 |
| --- | ---: | ---: | ---: | ---: |
| Biodegradable | 63.3% | 56.6% | 58.9 | 30.9 |
| Cardboard | 68.9% | 41.7% | 51.9 | 34.5 |
| Glass | 85.1% | 36.6% | 52.6 | 36.1 |
| Metal | 61.6% | 56.0% | 53.7 | 35.3 |
| Paper | 62.3% | 50.7% | 52.7 | 37.3 |
| Plastic | 62.9% | 48.9% | 53.2 | 31.5 |
| **Overall** | **67.4%** | **48.4%** | **53.8** | **34.3** |
 

**Findings:** Glass has the highest precision 85.1% but lowest recall 36.6%. The predictions are often correct, but the detector misses many labeled glass objects. Biodegradable and metal have the highest recall 56.6% and 56.0%. Paper has the highest AP50–95 (37.3), biodegradable has the lowest (30.9), so its score drops more across stricter box-overlap requirements. Overall test AP50–95 is 34.3, compared with 37.2 when the same selected checkpoint was re-evaluated on validation.

### Comparison of five experiments

All rows below are validation results, not test results. AP values come from the listed training epochs

| Experiment | Setup | Result epoch | Precision | Recall | AP50 | AP50–95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1st | Batch 4, one T4, extra YOLO augmentation | 3 | 59.6% | 49.0% | 52.5 | 34.6 |
| 2nd | Batch 4,one T4, resumed on two | 12 | 65.3% | 50.7% | 55.1 | 35.1 |
| 3rd | Batch 8, one GPU, no extra YOLO augmentation | 3 | 65.8% | 50.8% | 56.7 | 36.8 |
| 4th | Batch 8, one T4, no extra YOLO augmentation | 11 | 65.8% | 53.0% | 57.0 | 35.2 |
| **5th** | **Batch 8, two T4s, no extra YOLO augmentation** | **14** | **68.9%** | **52.8%** | **57.4** | **37.1** |

The fifth model had the highest recorded validation AP50–95 and was selected for the test. The runs differed in training length, augmentation, training platform, so the table does **not** prove that batch size or GPU count caused the differences. The dataset version was identical when reproducing the comparison.

## Task 3: Detection examples and error analysis

In the images, **green boxes are dataset labels** and **red boxes are model predictions**. The label on a red box includes the predicted class and confidence. The examples were selected from the test set using confidence 0.25 and class-aware matching at IoU 0.50. A single image can contain both missed and incorrect detections.

### Correct detections

| Glass bottle | Plastic item |
| --- | --- |
| <img src="5th_30%20epochs_2gpu_chosen/analysis_examples/correct_1.png" alt="Glass bottle correctly detected" width="280"> | <img src="5th_30%20epochs_2gpu_chosen/analysis_examples/correct_2.png" alt="Plastic item correctly detected" width="280"> |
| The bottle is predicted as **glass** and the red and green boxes closely overlap. | The blue item is predicted as **plastic** and the predicted box closely covers the labeled object. |

### Missed detections

| Cardboard and metal | Glass container and metal region |
| --- | --- |
| <img src="5th_30%20epochs_2gpu_chosen/analysis_examples/missed_2.png" alt="Missed metal labels in a mixed scene" width="280"> | <img src="5th_30%20epochs_2gpu_chosen/analysis_examples/missed_3.png" alt="Glass detected but labeled metal region missed" width="280"> |
| The labels include **metal**, but the visible predictions are cardboard; the metal labels have no matching metal prediction at the chosen threshold. Overlapping objects and unclear boundaries may contribute. | The glass container is detected, but the region labeled **metal** has no matching metal prediction; an overlapping region is predicted as plastic. Adjacent materials may make the category difficult. |

### Incorrect detections

| Cardboard predicted as paper | Books labeled as paper |
| --- | --- |
| <img src="5th_30%20epochs_2gpu_chosen/analysis_examples/incorrect_1.png" alt="Cardboard predicted as paper" width="280"> | <img src="5th_30%20epochs_2gpu_chosen/analysis_examples/incorrect_2.png" alt="Paper labels predicted as cardboard" width="280"> |
| The object is labeled **cardboard**, but the model predicts **paper**. Similar colors and textures are a possible reason. | Some objects labeled **paper** are predicted as **cardboard**. A book cover and cardboard can look similar, and class definitions for mixed materials may be ambiguous. |

The examples are compared with the supplied labels, which are not guaranteed to be correct. In the crowded mixed scene, labels overlap and should rewied by all of us.

### Strengths, limitations, and improvements

- **Strength:** High test precision for glass, with well-aligned detections on some isolated glass and plastic objects.
- **Limitations:** Many glass objects are missed; visually similar classes can be confused; overlapping and crowded objects make localization and matching difficult. The test score is lower than validation (lol) and may not represent performance on an actual sorting conveyor.
- **Possible improvements:** Review missing or ambiguous boxes and class definitions; add varied, correctly labeled examples of difficult materials and crowded scenes; and compare augmentation or model settings on validation data. The run warned about images containing many objects, so the model's detection limit when assessing crowded scenes should be verified.

