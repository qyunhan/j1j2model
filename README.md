# j1j2model Supervised CNN/MLP

## Project

### Models:
Contains the trainable ML models used in the application.
1. cnn_regressor: 1D Convolutional Neural network to predict energy of quantum spin configurations
- Input: (batch, 3, n_spins) representing (x, y, z) spin components
- Layers: Conv1d (3->16) + ReLU, Conv1d (16->32) + ReLU, AdaptiveAvgPool1d (output=1) + Flatten, Linear + ReLU, Linear + Output
          Conv1d layers extract spatial features from spin configurations. ReLU makes it non-linear.
          Second convolution leans higher level features, multi-spin correlations to capture more abstract spatial relationships
          AdaptiveAvgPool1d is used so that CNN is size invariant and stable. 
          Flatten prepares for fully connected layers, where (batch, 32, 1) is flattened to a vector.
          Fully connected layers learn complex mappings from extracted features to energy prediction learning global interactions.
- Output: Predicted scalar energy

2. mlp_regressor: Multilayer Perceptron model using angles (theta, phi) as inputs
- Input: 2*n_spins
- Layers: Fully connected layers with ReLu activations
- Output: Scalar energy

### Physics:
Implements the J1J2 model, constructs Hamiltonian to compute exact energies used as supervised lables
1. class QuantumJ1J2solver: Builds H, applies Pauli operators, computes expectation energy
- Methods: build hamiltonian, compute energy, get training data 

### Utils:
1. Data:
- Train val test split
- Set global seed
2. Features:
- Transform spin angles from angles or xyz into NN-readable features
- Angle features for MLP: (theta, phi) for each spin
  Concatenate theta and phi into single vector of dimension 2*n_spins, good for MLP
  Also can get  cartesian (xyz) to get dimension of 3*n_spins
- Angle features for CNN: Spins arranged in (B, 3, n_spins)
  Channel 0 = x components, Channel 1 = y components, Channel 2 = z components
  Each spin treated as a pixel with 3 channels 
3. Training:
- Train model, Validate model, Plotting curves

### Train:
1. train_mlp: full pipeline for MLP
2. train_cnn: full pipeline for CNN
Calls quantumJ1J2solver to generate dataset first, then extracts angle features, split dataset, builds model, trains with adam and saves metrics
Model evaluated with MLE (loss curves shown in plot)

### Run_experiments: To call and run the model
