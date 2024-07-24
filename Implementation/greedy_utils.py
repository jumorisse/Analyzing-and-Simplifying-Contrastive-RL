import numpy as np

basic_action_grid = np.array([
                        [[-1.,-1.], [-1.,0.], [-1.,1.]],
                        [[0.,-1.], [0.,0.], [0.,1.]],
                        [[1.,-1.], [1.,0.], [1.,1.]],
                        ])

def scale_action_grid(action_grid, factor):
    """
    Scales action grid up, i.e. creates a more fine-grained grid with more cells.
    Scaling up is done by interpolating neighboring (8-neighbors) cells.
    Factor is the number of times interpolation is done.

    Parameters:
        action_grid: 3D numpy array of actions, shape (3,3,2)
        factor: int, number of times to interpolate neighboring cells
    
    Returns:
        3D numpy array of tuples, scaled up action grid of shape (action rows, action cols, 2)
    """
    for _ in range(factor):
        current_shape = action_grid.shape
        new_shape = (current_shape[0]+current_shape[0]-1, current_shape[1]+current_shape[1]-1, current_shape[2])
        new_action_grid = np.zeros(new_shape, dtype=np.float32)
        # fill new grid with values from old grid
        for i in range(current_shape[0]):
            for j in range(current_shape[1]):
                new_action_grid[2*i, 2*j] = action_grid[i,j]
        # interpolate values
        # in even rows including (0), odd cells are interpolated by their row neighbors
        for i in range(0, new_shape[0], 2):
            for j in range(1, new_shape[1], 2):
                new_action_grid[i,j] = (new_action_grid[i,j-1] + new_action_grid[i,j+1]) / 2
        # odd rows are completely interpolated by their row neighbors
        for i in range(1, new_shape[0], 2):
            for j in range(new_shape[1]):
                new_action_grid[i,j] = (new_action_grid[i-1,j] + new_action_grid[i+1,j]) / 2
        action_grid = new_action_grid    
    print("Scaled 3x3x2 grid to ", action_grid.shape)
    print("Grid contains", action_grid.size/2, "2d actions.")
    return action_grid

def get_action_grid(scale_factor):
    """
    Returns a scaled up action grid.

    Parameters:
        scale_factor: int, number of times to interpolate neighboring cells
    
    Returns:
        3D numpy array of tuples, scaled up action grid of shape (action rows, action cols, 2)
    """
    return scale_action_grid(basic_action_grid, scale_factor)