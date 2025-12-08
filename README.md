# Melancholy LSTM:melancholy_LSTM

“This project explores how a small LSTM model learns the rhythm of a melancholy inner monologue.”


# 🍄 Mushroom Species Classification (using RBFN)
​This project aims to classify mushroom species using a Radial Basis Function Network (RBFN) implemented with PyTorch.
​💾 Dataset
​The dataset used in this project contains features related to mushroom species found in Bolu, Turkey, sourced from Kaggle.
​Dataset Name: Mushroom Species Found in Bolu
​Source: Kaggle - Eydanur Aydın
​File Name: mantar_veriseti.csv
​⚠️ Note: This dataset is the cleaned and preprocessed version of the original Kaggle data, tailored for this specific classification project.
​⚙️ Model and Methodology
​The classification is performed using a Radial Basis Function Network (RBFN) architecture, which differs from traditional Artificial Neural Networks (ANNs).
​Architecture:
​Input Layer: Feature Count (input_dim)
​Hidden Layer (RBF Kernel): 10 Centers (num_centers=10)
​Output Layer: 3 Classes (output_dim=3)
​Kernel Function: Gaussian Kernel (e^{-\beta ||\mathbf{x} - \mathbf{c}||^2})
​Training:
​Loss Function: nn.CrossEntropyLoss
​Optimization: optim.Adam(lr=0.01)
​The RBF centers (\mathbf{C}) and the \beta parameter are defined as learnable parameters (nn.Parameter), optimized along with the weights and biases of the linear output layer.
​💻 Required Libraries
​The fundamental libraries required to run this project are:
​torch (PyTorch)
​pandas
​scikit-learn
