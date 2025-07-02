import pandas as pd
import matplotlib.pyplot as plt

# load and explore the data
data = pd.read_csv('data.csv')
print(data.describe())

plt.scatter(data.YearsExperience, data.Salary)
plt.show()

# function isn't used because it's already included in the gradient descent function
def loss_function(m, b, points):
    total_error = 0
    # the loop is the "sum" (sigma)
    for i in range(len(points)):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        total_error += (y - (m * x + b)) ** 2
    return total_error / len(points)

def gradient_descent(points, m_now, b_now, learning_rate): 
    m_gradient = 0
    b_gradient = 0

    N = len(points)
    
    # calculate the gradients
    for i in range(N):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
    
        m_gradient += -(2/N) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/N) * (y - (m_now * x + b_now))
    
    # update the parameters
    m = m_now - m_gradient * learning_rate
    b = b_now - b_gradient * learning_rate
    return m, b

# starting values
m = 0
b = 0
learning_rate = 0.01
epochs = 50

for i in range(epochs):
    m, b = gradient_descent(data, m, b, learning_rate)
    if i % 10 == 0:  # print every 10 epochs
        print(f'Epoch {i}: m = {m}, b = {b}')

print(f'Final values: m = {m}, b = {b}')

# test 1 results
m1, b1 = m, b


# test 2
m = 0
b = 0
learning_rate = 0.001
epochs = 100

for i in range(epochs):
    m, b = gradient_descent(data, m, b, learning_rate)
    if i % 10 == 0:  # print every 10 epochs
        print(f'Epoch {i}: m = {m}, b = {b}')

print(f'Final values: m = {m}, b = {b}')

# test 2 results
m2, b2 = m, b

# Plot both lines on the same graph
plt.scatter(data.YearsExperience, data.Salary, color='blue', label='Data')
plt.plot(list(range(0,12)), [m1 * x + b1 for x in range(0,12)], color='red', label='LR=0.1, Epochs=50')
plt.plot(list(range(0,12)), [m2 * x + b2 for x in range(0,12)], color='green', label='LR=0.01, Epochs=100')
plt.legend()
plt.show()