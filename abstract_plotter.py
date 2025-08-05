import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from file
df = pd.read_csv("models/genpot1_dnn/loss_dnn.csv")

# Filter to first 100 data points
initial_100 = df[df['epoch'] < 60]  # or df.iloc[:100]

# Plot the initial 100 losses
plt.figure(figsize=(10, 6))
plt.plot(initial_100['epoch'], initial_100['loss'], label='Initial 100 Epochs', color='darkgreen')

# # Optional: highlight first 10 epochs
# early_epochs = initial_100[initial_100['epoch'] <= 10]
# plt.plot(early_epochs['epoch'], early_epochs['loss'], label='First 10 Epochs', color='red', linewidth=2)
# plt.scatter(early_epochs['epoch'], early_epochs['loss'], color='red')

# Labels and styling
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Trend in First 100 Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("genpot1.png")
