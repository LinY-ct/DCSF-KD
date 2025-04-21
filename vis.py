import matplotlib.pyplot as plt

# Data for the first plot (alpha)
alpha_values = [3, 5, 8, 10, 13]
mAP_values_alpha = [40.8, 41.0, 41.0, 41.3, 41.1]

# Data for the second plot (beta)
beta_values = [5, 7, 10, 15, 20]
mAP_values_beta = [40.0, 40.2, 40.8, 40.7, 40.4]

# Create the plot for alpha
plt.figure(figsize=(12, 6))  # Adjust the size as needed

# First subplot for alpha
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(alpha_values, mAP_values_alpha, marker='v', linestyle='--', color='blue', label='Alpha')
plt.title('ACT')
plt.xlabel('Alpha')
plt.ylabel('mAP')
plt.grid(True)
plt.legend()

# Second subplot for beta
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(beta_values, mAP_values_beta, marker='^', linestyle='--', color='red', label='Beta')
plt.title('CWT')
plt.xlabel('Beta')
plt.ylabel('mAP')
plt.grid(True)
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()
