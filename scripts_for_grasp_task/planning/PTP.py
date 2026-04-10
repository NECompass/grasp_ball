import numpy as np

def PTP_quintic_interval(T: np.array, traj_start: np.array, traj_stop: np.array) -> list:
    """Interpolation from start to end configuration with a quintic spline

    Parameters
    ----------
    T: np.array((numsteps))
        Time vector for the interval
    
    traj_start: np.array((n))
        Start configuration
    
    traj_stop: np.array((n))
        End configuration
    
    Returns
    --------

    [Traj, dot_Traj, ddot_Traj]: list[np.array((n, numsteps))]
            
    """
    #Length of the interval
    dT = T[-1]-T[0]
    numsteps = len(T)
    #Shift the time interval -> starts at t=0
    T_new = T-np.ones((numsteps))*T[0]

    ###Workspace Start###
    ##Calculation of the coefficients
    k3 = 10
    k4 = -15
    k5 = 6
    tau = T_new/dT
    ##Calculate the path parameter s and its derivatives
    s = k3*tau**3 + k4*tau**4 + k5*tau**5
    dot_s = (3*k3*tau**2 + 4*k4*tau**3 + 5*k5*tau**4)/dT
    ddot_s = (6*k3*tau + 12*k4*tau**2 + 20*k5*tau**3)/dT**2
    ###Workspae End###

    #Initialise the trajectory
    size = len(traj_start)
    Traj = np.zeros((size, numsteps))
    dot_Traj = np.zeros((size, numsteps))
    ddot_Traj = np.zeros((size, numsteps))


    ###Workspace Start###
    ##Loop over all time steps to calculate the trajectory
    for i in range(numsteps):
        Traj[:,i] = traj_start + (traj_stop - traj_start)*s[i]
        dot_Traj[:,i] = (traj_stop - traj_start)*dot_s[i]
        ddot_Traj[:,i] = (traj_stop - traj_start)*ddot_s[i]
    ###Workspace End###
     
    return [Traj, dot_Traj, ddot_Traj]

def PTP_quintic(through_points: np.array, T_part: np.array, dt: float) -> list:
    """Planning of PTP movements between the interpolation points,
    Interpolation with quintic polynomials

    -numpoints: Number of interpolation points
    -dimension: Dimension of the interpolation points (e.g. positions in the workspace: dimension = 3)
    -totalsteps: Number of time steps of the entire trajectory

    Parameters
    ---------
    through_points: np.array((numpoints,dimension))
        interpolation points
    
    T_part: np.array((numpoints-1))
        Time for the individual intervals
    
    dt: float
        Cycle time

    Returns
    --------
    [T, Traj, dot_Traj, ddot_Traj]: list[np.array((totalsteps)), np.array((dimension,totalsteps)), ...]
        Trajectory: time vector T, positions Traj, velocities dot_Traj, accelerations ddot_Traj

    """    

    intervals = np.size(through_points,0) - 1 #Number of intervals
    size = np.size(through_points,1) #Number of points

    #Initialisation of the start time
    t_start = 0
    #Initialisation of the lists for saving the trajectory
    T_list = [] #Time vectors
    Trajectory_list = [] #Trajectory
    intervalsteps_list = [] #Number of time steps of the intervals

    #Loop over all intervals
    for i in range(intervals):
        t = np.arange(start = t_start, step=dt, stop=t_start + T_part[i]) #Time vector for interval
        traj_start = through_points[i,:] #Start value of the position
        traj_stop = through_points[i+1,:] #Target value of the position
        #Calculation of the trajectory
        traj = PTP_quintic_interval(t, traj_start, traj_stop)
        #Save the trajectory
        T_list.append(t)
        Trajectory_list.append(traj)
        intervalsteps_list.append(len(t))
        #Set the start time for the next interval
        t_start = t[-1]

    #Initialisation of the data matrices of the trajectory
    totalsteps = sum(intervalsteps_list) #Number of time steps
    T = np.empty(totalsteps) #Time vector
    Traj = np.empty((size,totalsteps)) #Positions
    dot_Traj = np.empty((size,totalsteps)) #Speeds
    ddot_Traj = np.empty((size,totalsteps)) #Accelerations

    #"Assembly" of the trajectory
    for i in range(intervals):
        T[sum(intervalsteps_list[:i]):sum(intervalsteps_list[:i+1])] = T_list[i]
        Traj[:,sum(intervalsteps_list[:i]):sum(intervalsteps_list[:i+1])] = Trajectory_list[i][0]
        dot_Traj[:,sum(intervalsteps_list[:i]):sum(intervalsteps_list[:i+1])] = Trajectory_list[i][1]
        ddot_Traj[:,sum(intervalsteps_list[:i]):sum(intervalsteps_list[:i+1])] = Trajectory_list[i][2]

    return [T, Traj, dot_Traj, ddot_Traj]
