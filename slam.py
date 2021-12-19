## TODO: Complete the code to implement SLAM
import numpy as np

def initialize_constraints(N, num_landmarks, world_size):
    ''' This function takes in a number of time steps N, number of landmarks, and a world_size,
        and returns initialized constraint matrices, omega and xi.'''
    
    ## Recommended: Define and store the size (rows/cols) of the constraint matrix in a variable
    
    ## TODO: Define the constraint matrix, Omega, with two initial "strength" values
    ## for the initial x, y location of our robot
    constraint_dims = N + num_landmarks
    # (step+landmark, step+landmark, xy)
    omega = np.zeros((constraint_dims, constraint_dims, 2))
    omega[0, 0, :] = 1
    
    ## TODO: Define the constraint *vector*, xi
    ## you can assume that the robot starts out in the middle of the world with 100% confidence
    # (step+landmark, xy)
    xi = np.zeros((constraint_dims, 2))
    xi[0, :] = world_size // 2
    
    return omega, xi

## slam takes in 6 arguments and returns mu, 
## mu is the entire path traversed by a robot (all x,y poses) *and* all landmarks locations
def slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise):
    def _process_measurement(
        omega: np.ndarray,
        xi: np.ndarray,
        landmark: tuple[int, float, float],
        i: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process an individual measurement for SLAM

        Parameters
        ----------
        omega : np.ndarray
            omega matrix of dim=3 where the first two dims are the pose index
            followed by the landmark index
        xi : np.ndarray
        landmark : tuple[int, float, float]
            index, x, y of landmark observed
        i : int
            step index
        """
        # Unpack and calculate
        land_i, dx, dy = landmark
        land_i = N + land_i
        confidence = 1.0 / measurement_noise

        # Adjust omega
        omega[i, land_i, :] += -confidence
        omega[land_i, i, :] += -confidence
        omega[i, i, :] += confidence
        omega[land_i, land_i, :] += confidence

        # Adjust Xi
        xi[i, :] += (-dx, -dy)
        xi[land_i, :] += (dx, dy)
        return omega, xi

    def process_measurements(
        omega: np.ndarray,
        xi: np.ndarray,
        landmarks: list[tuple[int, float, float]],
        i: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process an all landmarks within a step of SLAM

        Parameters
        ----------
        omega : np.ndarray
            omega matrix of dim=3 where the first two dims are the pose index
            followed by the landmark index
        xi : np.ndarray
        landmarks : list[tuple[int, float, float]]
            list of observed landmarks
        i : int
            step index
        """
        for landmark in landmarks:
            omega, xi = _process_measurement(omega, xi, landmark, i)
        return omega, xi

    def process_move(
        omega: np.ndarray,
        xi: np.ndarray,
        move: tuple[float, float],
        i: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process an individual move for SLAM

        Parameters
        ----------
        omega : np.ndarray
            omega matrix of dim=3 where the first two dims are the pose index
            followed by the landmark index
        xi : np.ndarray
        move : tuple[float, float], list[float, float]
            x, y of move command to robot 
        i : int
            step index
        """
        # Unpack and calculate
        dx, dy = move
        next_i = i + 1
        confidence = 1.0 / motion_noise

        # Adjust omega
        omega[i, next_i, :] += -confidence
        omega[next_i, i, :] += -confidence
        omega[i, i, :] += confidence
        omega[next_i, next_i, :] += confidence

        # Adjust Xi
        xi[i, :] += (-dx, -dy)
        xi[next_i, :] += (dx, dy)
        return omega, xi

    
    ## TODO: Use your initialization to create constraint matrices, omega and xi
    omega, xi = initialize_constraints(N, num_landmarks, world_size)
    ## TODO: Iterate through each time step in the data
    ## get all the motion and measurement data as you iterate
    for step, entry in enumerate(data):
        landmarks, move = entry
        omega, xi = process_measurements(omega, xi, landmarks, step)
        omega, xi = process_move(omega, xi, move, step)


            
    ## TODO: update the constraint matrix/vector to account for all *measurements*
    ## this should be a series of additions that take into account the measurement noise
    # [ACM] See nested function - process_measurements & _process_measurement
    ## TODO: update the constraint matrix/vector to account for all *motion* and motion noise
    # [ACM] See nested function - process_move
    
    ## TODO: After iterating through all the data
    ## Compute the best estimate of poses and landmark positions
    ## using the formula, omega_inverse * Xi
    omega_inv = np.zeros_like(omega)
    mu = np.zeros_like(xi)
    dims = omega.shape[2]
    for i in range(dims):
        # [ACM] Not sure if the inverse of a 3D tensor results in the same as
        #       calculating the inverse of a 2D matrix so I will decompose
        omega_inv[:, :, i] = np.linalg.inv(np.matrix(omega[:, :, i]))
        mu[:, i] = omega_inv[:, :, i] @ xi[:, i]
        # @ needed since omega_inv is still an array
    
    return mu # return `mu`
