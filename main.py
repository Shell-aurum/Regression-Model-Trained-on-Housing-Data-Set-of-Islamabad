import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('zameen_islamabad.csv')

# Function to remove outliers using IQR
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

# Train-test split using NumPy (80% train, 20% test)
np.random.seed(42)  # For reproducibility
indices = np.random.permutation(df.shape[0])
train_size = int(0.8 * df.shape[0])
train_indices = indices[:train_size]
test_indices = indices[train_size:]
df_train = df.iloc[train_indices]
df_test = df.iloc[test_indices]
print(f"Training set size: {df_train.shape}")
print(f"Test set size: {df_test.shape}")

# Compute the mean price for each location
location_means = df_train.groupby('Location')['Price (Cr)'].mean()
# Map each location to its mean price
df_train['Location'] = df_train['Location'].map(location_means)
df_test['Location'] = df_test['Location'].map(location_means).fillna(location_means.mean())  # Handle unseen locations in test set

X_train = df_train.drop('Price (Cr)', axis=1).values # It will create Input Matrix with independent variables [Location, Area, Bedroom, Bathroom]
Y_train = df_train['Price (Cr)'].values.reshape(-1, 1) # Output dependent variable(price CR) column matrix (Numpy array)
X_test = df_test.drop('Price (Cr)', axis=1).values
Y_test = df_test['Price (Cr)'].values.reshape(-1, 1)

# Normalizing Data
X_train = (X_train - X_train.mean(axis=0)) / np.std(X_train, axis=0)
Y_train_mean, Y_train_std = Y_train.mean(), Y_train.std()
Y_train = (Y_train - Y_train_mean) / Y_train_std
X_train = np.c_[np.ones(X_train.shape[0]), X_train] # Adding ones as intercept variable in input matrix (c in y = mx + c)

# Normalize test set using training set statistics
X_test = (X_test - X_train[:, 1:].mean(axis=0)) / np.std(X_train[:, 1:], axis=0)
Y_test = (Y_test - Y_train_mean) / Y_train_std
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Now the parameter Matrix containing all weights and biases theta = [b, w1, w2, w3, w4]
np.random.seed(42)
theta = np.random.randn(X_train.shape[1], 1)
# Compute Mean Squared Error Function
def compute_cost(x, y, theta):
    m = len(x)
    predictions = x@theta
    cost = (1 / (2 * m)) * np.sum(np.square(y - predictions)) # Error function which is to be minimized (MSE divide by 2 for math convenience)
    return cost
def Gradient_descent(x, y, theta, alpha, epochs):
    m = len(y)
    cost_history = []
    for i in range(epochs):
        predictions = x@theta
        theta -= (alpha / m) * (x.T @ (predictions - y)) # gradient descent formula
        cost = compute_cost(x, y, theta)
        cost_history.append(cost)
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost:.5f}")
    
    return theta, cost_history
# Training the Model
alpha = 0.01 # learning rate, used in gradient descent method
epochs = 1000
final_theta, cost_history = Gradient_descent(X_train, Y_train, theta, alpha, epochs)


# Evaluate on training set
predicted_y_train = X_train@final_theta
# Denormalize and clip test predictions
predicted_y_train = np.clip(predicted_y_train * Y_train_std + Y_train_mean, 0, None)
r2_train = 1 - ((np.sum((Y_train - predicted_y_train)**2)) / (np.sum((Y_train - np.mean(Y_train))**2)))
rmse_train = np.sqrt(np.mean((Y_train - predicted_y_train)**2))
print("Final parameters:\n", final_theta)
print(f"Training R² Score: {r2_train}")
print(f"Training RMSE: {rmse_train}")


# Evaluate on test set
predicted_y_test = X_test@final_theta
# Denormalize and clip test predictions
predicted_y_test = np.clip(predicted_y_test * Y_train_std + Y_train_mean, 0, None)
r2_test = 1 - ((np.sum((Y_test - predicted_y_test)**2)) / (np.sum((Y_test - np.mean(Y_test))**2)))
rmse_test = np.sqrt(np.mean((Y_test - predicted_y_test)**2))
print(f"Test R² Score: {r2_test}")
print(f"Test RMSE: {rmse_test}")

# Plot Price (Cr) vs. Area (Marla) with regression line
# Get original Area (Marla) and Price (Cr) from training set
area_train = df_train['Area (Marla)'].values
price_train = df_train['Price (Cr)'].values

# Create points for regression line with limited range
area_range = np.linspace(0, 30, 100)  # Limit to 25 Marla
# Normalize area_range and create input matrix with mean values for other features
area_normalized = (area_range - X_train[:, 2].mean()) / X_train[:, 2].std()  # Area is 2nd feature (index 1 in X_train[:, 1:])
X_line = np.ones((100, X_train.shape[1]))
X_line[:, 1] = df_train['Location'].mean() / X_train[:, 1].std()  # Location (mean price)
X_line[:, 2] = area_normalized  # Area (Marla)
X_line[:, 3] = X_train[:, 3].mean()  # Bedrooms
X_line[:, 4] = X_train[:, 4].mean()  # Bathrooms

y_line_normalized = X_line @ final_theta
# Create scatter plot and regression line
plt.scatter(area_train, price_train, alpha=0.5, label='Training Data', color='blue')
plt.plot(area_range, y_line_normalized, color='red', label='Regression Line')
plt.xlabel('Area (Marla)')
plt.ylabel('Price (Cr)')
plt.title('Price (Cr) vs. Area (Marla) with Regression Line')
plt.legend()
plt.ylim(0, 30)  # Limit y-axis to match data range
plt.show()

# Plot cost history
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Error Function (MSE/2)')
plt.title('Minimizing Error over iterations')
plt.show()