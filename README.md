# Deepfake Detection with XceptionNet
# Table of Contents

Introduction </br> 
What is FaceForensics++? </br>
Our Model </br>
Results </br>
Key Code Sections </br>
Installation </br>
References </br>


# Introduction
What is the Problem? </br>
Deepfakes are fake videos or images created by AI. They can: </br>

Spread false information </br>
Damage people's reputation </br>
Make us lose trust in media </br>

Our Solution </br>
We built a system that can detect fake faces automatically. Our model can tell if an image is real or fake with 97% accuracy.

# What is FaceForensics++?
FaceForensics++ (Rössler et al., 2019) is a famous research paper. It created a big dataset to train AI models.
Main Contributions
1. Big Dataset

18 million images from 4,000 fake videos
1,000 real videos from YouTube
Much bigger than other datasets

2. Four Manipulation Methods
Expression Change:

Face2Face: Changes facial expressions in real-time
NeuralTextures: Uses AI to change mouth movements

Identity Change (Face Swap):

FaceSwap: Replaces one person's face with another
DeepFakes: Uses deep learning to swap faces

3. Three Quality Levels

Raw: No compression (best quality)
HQ: Light compression (like YouTube)
LQ: Strong compression (like Facebook)

Performance Results
MethodHuman AccuracyXceptionNet AccuracyRaw videos68.69%99.26%HQ videos66.57%95.73%LQ videos58.73%81.00%
Important: AI is much better than humans at detecting fakes!

Detection Difficulty by Method
ManipulationAccuracy (LQ)Why?DeepFakes96.36%Easy - same artifacts every timeFaceSwap90.29%Medium - changes the whole faceFace2Face86.86%Hard - only small changesNeuralTextures80.67%Hardest - different artifacts each time

Main Challenges

Compression destroys evidence

Social media compresses videos
This removes fake traces
Hard to detect after compression


Many different fake methods

New AI methods appear every day
Models trained on old fakes fail on new ones


Need big datasets

Small datasets = bad performance
Need 700+ videos minimum


Face detection can fail

If we can't find the face, detection fails




🔬 Current Research (2024-2025)
Why Detection is Still Hard Today
I studied recent research: "Deepfake Detection that Generalizes Across Benchmarks" (Yan et al., 2025)
New Problems:
1. Generalization Problem

Models work well on training data
Models fail on new fake methods
Fake generators improve faster than detectors

2. Adversarial Attacks

Bad people can trick detectors
They optimize fakes to avoid detection

3. Shortcut Learning

Models learn wrong patterns
Example: learning image size instead of fake artifacts

4. Social Impact

People don't trust any media anymore
Even real content is questioned


New Detection Techniques
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


🏗️ Our Model
Why XceptionNet?
XceptionNet is recommended by FaceForensics++ because:

Efficient Architecture

Uses "separable convolutions"
Fewer parameters = faster training
Still very powerful


Pre-trained on ImageNet

Already knows 1.2 million images
Learns faster on our task


Deep Network

36 convolutional layers
Can learn complex patterns


Residual Connections

Helps training very deep networks
Better gradient flow
