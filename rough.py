# import numpy as np

# # Example array
# arr = np.array([[1, 2, 3],
#                 [4, 5, 6],
#                 [7, 8, 9]])

# # Extract the 0th column
# zeroth_column = arr[:, 0]

# # Add a new column at the beginning with the same values as the 0th column
# new_column = zeroth_column.reshape(-1, 1)

# # Concatenate the new column with the original array
# arr = np.concatenate((new_column, arr), axis=0)
# #arr = np.concatenate((arr,zeroth_column), axis=1)

# print(arr)
import numpy as np

# # Example array
grid_world = np.array([[1, 2, 3],
                 [4, 5, 6],
                [7, 8, 9]])

# # Extract the 0th row
# zeroth_row = arr[0, :]

# # Add a new row at the beginning with the same values as the 0th row
# new_row = zeroth_row.reshape(1, -1)

# # Concatenate the new row with the original array
# arr = np.concatenate((new_row, arr), axis=0)

# print(arr[-1])
# print(zeroth_row[-1])

grid_world = np.concatenate((grid_world[:,0].reshape(-1,1), grid_world, grid_world[:,-1].reshape(-1,1)), axis=1)
grid_world = np.concatenate((grid_world[0].reshape(1,-1), grid_world, grid_world[-1].reshape(1,-1)), axis = 0)
print(grid_world)