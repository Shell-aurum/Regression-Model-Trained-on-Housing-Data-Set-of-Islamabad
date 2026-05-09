import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('zameen_islamabad.csv')

# outliers removal using IQR method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for 'Price (Cr)' and 'Area (Marla)'
df = remove_outliers_iqr(df, 'Price (Cr)')
df = remove_outliers_iqr(df, 'Area (Marla)')
print(f"Dataset size after outlier removal: {df.shape}")

np.random.seed(42) 
indices = np.random.permutation(df.shape[0])
train_size = int(0.8 * df.shape[0])
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Using .copy() to avoid Pandas SettingWithCopyWarnings later
df_train = df.iloc[train_indices].copy()
df_test = df.iloc[test_indices].copy()
print(f"Training set size: {df_train.shape}")
print(f"Test set size: {df_test.shape}")

# Compute the mean price for each location
location_means = df_train.groupby('Location')['Price (Cr)'].mean()
# Map each location to its mean price
df_train['Location'] = df_train['Location'].map(location_means)
df_test['Location'] = df_test['Location'].map(location_means).fillna(location_means.mean())  # Handle unseen locations in test set

# Create Input Matrices [Location, Area, Bedroom, Bathroom]
X_train = df_train.drop('Price (Cr)', axis=1).values 
Y_train = df_train['Price (Cr)'].values.reshape(-1, 1) 
X_test = df_test.drop('Price (Cr)', axis=1).values
Y_test = df_test['Price (Cr)'].values.reshape(-1, 1)


#  Save training statistics BEFORE modifying X_train
X_train_mean = X_train.mean(axis=0)
X_train_std = np.std(X_train, axis=0)
Y_train_mean = Y_train.mean()
Y_train_std = Y_train.std()

#  Normalize Training Data using saved stats
X_train = (X_train - X_train_mean) / X_train_std
Y_train = (Y_train - Y_train_mean) / Y_train_std
X_train = np.c_[np.ones(X_train.shape[0]), X_train] # Adding ones as intercept variable

#  Normalize Test Data using SAVED TRAINING stats
X_test = (X_test - X_train_mean) / X_train_std
Y_test = (Y_test - Y_train_mean) / Y_train_std
X_test = np.c_[np.ones(X_test.shape[0]), X_test]


# Initialize weights (theta) [b, w1, w2, w3, w4]
np.random.seed(42)
theta = np.random.randn(X_train.shape[1], 1)

# Compute Mean Squared Error Function
def compute_cost(x, y, theta):
    m = len(x)
    predictions = x @ theta
    cost = (1 / (2 * m)) * np.sum(np.square(y - predictions)) 
    return cost

def Gradient_descent(x, y, theta, alpha, epochs):
    m = len(y)
    cost_history = []
    for i in range(epochs):
        predictions = x @ theta
        theta -= (alpha / m) * (x.T @ (predictions - y)) 
        cost = compute_cost(x, y, theta)
        cost_history.append(cost)
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost:.5f}")
    return theta, cost_history

# Training the Model
alpha = 0.01 
epochs = 1000
final_theta, cost_history = Gradient_descent(X_train, Y_train, theta, alpha, epochs)


# Evaluate on training set
predicted_y_train = X_train @ final_theta

# Denormalize BOTH predictions and actual targets
predicted_y_train = np.clip(predicted_y_train * Y_train_std + Y_train_mean, 0, None)
actual_y_train = Y_train * Y_train_std + Y_train_mean

r2_train = 1 - ((np.sum((actual_y_train - predicted_y_train)**2)) / (np.sum((actual_y_train - np.mean(actual_y_train))**2)))
rmse_train = np.sqrt(np.mean((actual_y_train - predicted_y_train)**2))

print("\nFinal parameters:\n", final_theta)
print(f"Training R² Score: {r2_train:.4f}")
print(f"Training RMSE: {rmse_train:.4f} Cr")

# Evaluate on test set
predicted_y_test = X_test @ final_theta

# Denormalize BOTH predictions and actual targets
predicted_y_test = np.clip(predicted_y_test * Y_train_std + Y_train_mean, 0, None)
actual_y_test = Y_test * Y_train_std + Y_train_mean

r2_test = 1 - ((np.sum((actual_y_test - predicted_y_test)**2)) / (np.sum((actual_y_test - np.mean(actual_y_test))**2)))
rmse_test = np.sqrt(np.mean((actual_y_test - predicted_y_test)**2))

print(f"Test R² Score: {r2_test:.4f}")
print(f"Test RMSE: {rmse_test:.4f} Cr")


# Plot Price (Cr) vs. Area (Marla) with regression line
area_train = df_train['Area (Marla)'].values
price_train = df_train['Price (Cr)'].values

# Create points for regression line with limited range
area_range = np.linspace(0, 30, 100)  
# Normalize area_range using the SAVED training stats (Area is index 1 of the original features)
area_normalized = (area_range - X_train_mean[1]) / X_train_std[1] 

# Reconstruct the feature matrix for the line
X_line = np.ones((100, X_train.shape[1]))
X_line[:, 1] = 0  # Location (mean normalized is 0)
X_line[:, 2] = area_normalized  # Area (Marla)
X_line[:, 3] = 0  # Bedrooms (mean normalized is 0)
X_line[:, 4] = 0  # Bathrooms (mean normalized is 0)

# Predict and DENORMALIZE the line so it matches the Y-axis scale
y_line_predictions = X_line @ final_theta
y_line_denormalized = np.clip(y_line_predictions * Y_train_std + Y_train_mean, 0, None)

# Create scatter plot and regression line
plt.scatter(area_train, price_train, alpha=0.5, label='Training Data', color='blue')
plt.plot(area_range, y_line_denormalized, color='red', label='Regression Line', linewidth=2)
plt.xlabel('Area (Marla)')
plt.ylabel('Price (Cr)')
plt.title('Price (Cr) vs. Area (Marla) with Regression Line')
plt.legend()
plt.ylim(0, 30)  
plt.show()

# Plot cost history
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Error Function (MSE/2)')
plt.title('Minimizing Error over iterations')
plt.show()
