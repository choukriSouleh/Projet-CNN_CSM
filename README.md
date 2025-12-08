# Projet Deepfake Detection with XceptionNet

# Table of Contents

Introduction </br> 
What is FaceForensics++? </br>
Our Model </br>
Results </br>
Key Code Sections </br>
Installation </br>
References </br>


# Introduction
=> What is the Problem? </br>
Deepfakes are fake videos or images created by AI. They can: </br>

- Spread false information </br>
- Damage people's reputation </br>
- Make us lose trust in media </br>

=> Our Solution </br>
We built a system that can detect fake faces automatically. Our model can tell if an image is real or fake with 97% accuracy.

# What is FaceForensics++?
FaceForensics++ (Rössler et al., 2019) is a famous research paper. It created a big dataset to train AI models.
 Main Contributions
1. Big Dataset

- 18 million images from 4,000 fake videos </br>
- 1,000 real videos from YouTube </br> 
- Much bigger than other datasets </br>

2. Four Manipulation Methods
 Expression Change: </br>

- Face2Face: Changes facial expressions in real-time </br>
- NeuralTextures: Uses AI to change mouth movements </br>

 Identity Change (Face Swap):

- FaceSwap: Replaces one person's face with another </br>
- DeepFakes: Uses deep learning to swap faces </br>

3. Three Quality Levels

- Raw: No compression (best quality) </br>
- HQ: Light compression (like YouTube) </br>
- LQ: Strong compression (like Facebook) </br>

# Performance Results
```
Method	      Human Accuracy	XceptionNet Accuracy
Raw videos	    68.69%	             99.26%
HQ videos	    66.57%	             95.73%
LQ videos	    58.73%	             81.00%

```
# Detection Difficulty by Method
```
Manipulation	 Accuracy (LQ)	          Why?
DeepFakes	      96.36%	         Easy - same artifacts every time
FaceSwap	      90.29%	         Medium - changes the whole face
Face2Face	      86.86%	         Hard - only small changes
NeuralTextures	  80.67%	         Hardest - different artifacts each time
```
# Main Challenges

1. Compression destroys evidence

- Social media compresses videos
- This removes fake traces
- Hard to detect after compression


2. Many different fake methods

- New AI methods appear every day
- Models trained on old fakes fail on new ones


3. Need big datasets

- Small datasets = bad performance
- Need 700+ videos minimum


4. Face detection can fail

- If we can't find the face, detection fails


# Current Research (2024-2025)
=> Why Detection is Still Hard Today
I studied recent research: "Deepfake Detection that Generalizes Across Benchmarks" (Yan et al., 2025)
# New Problems:
1. Generalization Problem

- Models work well on training data
- Models fail on new fake methods
- Fake generators improve faster than detectors

2. Adversarial Attacks

- Bad people can trick detectors
- They optimize fakes to avoid detection

3. Shortcut Learning

- Models learn wrong patterns
- Example: learning image size instead of fake artifacts

4. Social Impact

- People don't trust any media anymore
- Even real content is questioned


# New Detection Techniques
1. HiFE Network (2024) - Frequency Analysis
Simple Explanation:

JPEG compression removes high-frequency details
Deepfakes leave traces in high frequencies
HiFE recovers these hidden traces
Result: +15-20% accuracy on compressed videos

2. LNCLIP-DF (2025) - Minimal Training
Simple Explanation:

Uses pre-trained CLIP model (knows many images)
Only trains 0.03% of parameters
Works on 13 different datasets
Result: Best generalization across datasets

3. Multi-Modal Detection
Simple Explanation:

Checks audio AND video together
Detects lip-sync problems
Finds voice mismatches

4. FakeCatcher (Intel) - Biological Signals
Simple Explanation:

Measures blood flow under skin
Real people have blood flow patterns
Deepfakes don't have correct blood flow
Result: Detection in milliseconds


# Our Model 
=> Why XceptionNet? </br>
XceptionNet is recommended by FaceForensics++ because:

1. Efficient Architecture

- Uses "separable convolutions"
- Fewer parameters = faster training
- Still very powerful


2. Pre-trained on ImageNet

- Already knows 1.2 million images
- Learns faster on our task


3. Deep Network

- 36 convolutional layers
- Can learn complex patterns


4. Residual Connections

- Helps training very deep networks
- Better gradient flow
# Model Structure </br>

```
Input Image (299×299 pixels)
     ^
     |
Early Layers (find basic patterns)
     ^
     |
Middle Layers (find complex patterns)
     ^
     |
Final Layers (make decision)
     ^
     |
Output: [Real or Fake]

```

# Test set: 1,500 images
```
750 real images
750 fake images
```
# Data Preprocessing
```
For Training:
- Resize to 299×299 pixels
- Random horizontal flip (50% chance)
- Random rotation (±15 degrees)
- Change brightness/contrast randomly
- Normalize (ImageNet standards)
=> Why? </br> Because this helps the model learn better and not memorize.
```
# Training Settings

```
Batch Size: 32 images
Learning Rate: 0.0001
Optimizer: Adam
Epochs: 15+
Scheduler: ReduceLROnPlateau

**Scheduler explanation**: If validation doesn't improve for 2 epochs, reduce learning rate by 50%.

---

## Results

### Overall Performance
Test Accuracy: 97.07%
Test Loss:     0.0920
Test F1-Score: 97.00%

### Confusion Matrix
 Predicted
              Real   Fake
Actual Real   745     5     ← Only 5 mistakes!
       Fake    39   711     ← 39 fakes missed

### Detailed Metrics
Class	Precision	Recall	F1-Score
Real	95.03%	    99.33%	 97.13%
Fake	99.30%	    94.80%	 97.00%

=> What this means:
Precision (Fake): When we say "fake", we're right 99.30% of the time
Recall (Real): We find 99.33% of all real images
Balanced: Good at detecting both real and fake

### Comparison with FaceForensics++
Method	Our Model	FaceForensics++ (LQ)
Accuracy	97.07%	81.00%
Note: Our data seems higher quality than their "LQ" test set.


```
 # Key Code Sections
1. Loading the Model
```
# Load XceptionNet with ImageNet weights
model = timm.create_model('xception', pretrained=True, num_classes=2)
model = model.to(device)

# Train all layers
for name, param in model.named_parameters():
    param.requires_grad = True

Note :
Here we use XceptionNet pre-trained on ImageNet.
This means it already knows basic image patterns.
We change the last layer to output 2 classes: real or fake.
We train all layers to adapt to deepfakes.
```
2. Handling Class Imbalance
```
# Count samples per class
counter = Counter(train_labels)
class_counts = np.array([counter[0], counter[1]])

# Calculate weights (minority class gets higher weight)
class_weights = class_counts.sum() / (2.0 * class_counts)
class_weights = torch.tensor(class_weights).to(device)

# Use weighted loss
criterion = nn.CrossEntropyLoss(weight=class_weights)

Note :
If we have more real images than fake images, the model might always predict 'real'.
Class weights fix this.
The minority class gets higher importance.
This balances the training.
```
3. Training Loop
```
best_val_f1 = 0.0
for epoch in range(EPOCHS):
    # Train for one epoch
    train_loss, train_acc = train_one_epoch(epoch)
    
    # Validate
    val_loss, val_f1 = evaluate(val_loader)
    
    # Reduce learning rate if stuck
    scheduler.step(val_loss)
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pth')

Note :
We train for multiple epochs.
After each epoch, we check validation performance.
If the model stops improving, we reduce the learning rate.
We save the best model based on F1-score, not just accuracy.
```
4. Evaluation Function
```
def evaluate(loader):
    model.eval()  # Set to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Don't compute gradients
        for images, labels in loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    return acc, f1, cm
Note :
This function tests the model.
We turn off training mode.
We don't compute gradients to save memory.
We collect all predictions and calculate accuracy, F1-score, and confusion matrix.
```
5. Data Augmentation
```
transform_train = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),           # Flip left-right
    transforms.RandomRotation(15),               # Rotate ±15°
    transforms.ColorJitter(brightness=0.2),      # Change brightness
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])   # ImageNet std
])
Note :
Data augmentation creates variations of training images.
Horizontal flip simulates different face orientations.
Rotation adds small angle changes.
Color jitter simulates different lighting.
This helps the model generalize better.
```
# Conclusion
=> What We Achieved </br>

- 97% accuracy in detecting deepfakes
- Better than human performance (68%)
- Fast and efficient detection

# Limitations 

- Only tested on one dataset
- May not work on new fake methods
- Needs face detection to work

# Future Improvements 

1. Add frequency analysis (HiFE)
2. Test on more datasets
3. Make robust against compression
4. Combine audio and video analysis

# Final Thoughts
Deepfakes are improving every day. Detection is a race between generators and detectors. We need better models that work on many different types of fakes.

# Author: Choukri SOULEIMAN
# Date: December 2025
