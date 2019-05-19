import numpy as np


def runge_kutta_54(func, y0, t_start, t_stop, t_step, atol=1e-6, rtol=1e-3):
    """
        Perform the 4the order runga_kutta method for first order ODE integration.
    In:
        param: func -- The function describing the differential equation or the system
                        of first order ODEs to integrate. Must return a numpy array.
        param: y0   -- The initial conditions.
        param: t_start -- The time to start integration at.
        param: t_stop -- The time to stop integration at.
        param: t_step -- The initial step size to use. 
        param: steps -- The step size to use for integration.
        param: order -- The order of the algorithm to use.
    """
    
               
    # If true, solving a single ODE, else solving a system.
    if type(y0) is not np.ndarray: 
        y0 = np.array([y0]) # convert to array


    # Array with values to return, for both the integrated
    # values and the time stemps. The size is increased by a 
    # factor of 2 when needed.

    ret = np.zeros((int((t_stop-t_start)/t_step)+1, len(y0)))
    time = np.zeros(int((t_stop-t_start)/t_step)+1)
    
    # Set initial state.
    ret[0] = y0
    time[0] = t_start
    
    # Solve the ODE or the system of ODEs
    min_update_scale = 0.2
    max_update_scale = 10

    # Current time at the integration
    t_now = t_start 
    # Total amount of executed steps
    steps = 1  # skip zero
    
    # Current error
    error = 1.1
    y_next = 0
    
    while t_now <= t_stop:

        # Check if we need to expand the return arrays
        if steps >= ret.shape[0]:
            ret_old = ret.copy()
            ret = np.zeros((ret_old.shape[0]*2, ret_old.shape[1]))
            ret[0:steps] = ret_old

            time_old = time.copy()
            time = np.zeros(time_old.shape[0]*2)
            time[0:steps] = time_old
    
    
        # Get the value found at the previous step
        previous = ret[steps-1]     

        # Calculate the constants for the runge kutta method. TODO exact name 
        k1 = t_step*func(previous, t_now)            
        k2 = t_step*func(previous + (1/5)*k1, t_now + (1/5)*t_step)
        k3 = t_step*func(previous + (3/40)*k1 + (9/40)*k2, t_now + (3/10)*t_step)
        k4 = t_step*func(previous + (44/45)*k1 - (56/15)*k2 + (32/9)*k3, t_now + (4/5)*t_step)
        k5 = t_step*func(previous + (19372/6561)*k1 - (25360/2187)*k2 + (64448/6561)*k3 - (212/729)*k4, t_now + (8/9)*t_step)
        k6 = t_step*func(previous + (9017/3168)*k1 - (355/33)*k2 + (46732/5247)*k3 + (49/176)*k4 - (5103/18656)*k5, t_now + t_step)

        # Calculate the new value.
        y_next = previous + ( (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6)
        y_embedded = previous + ( (5179/57600)*k1 + (7571/16695)*k3 + (393/640)*k4-(92097/339200)*k5 + (187/2100)*k6 )
        
        # Calculate error
        delta = abs(y_embedded  - y_next)
        scale = atol+np.maximum(abs(previous),abs(y_next))*rtol
        error = np.sqrt(np.sum((delta/scale)**2)/len(k1))
                    
        # The factor used to calculate the new step size.
        update_scale = 0.9*(error)**(-0.2)
          
        # Make sure the factor is not to large or to small.
        if error == 0:
            update_scale = max_update_scale
        elif update_scale < min_update_scale:
            update_scale = min_update_scale
        elif update_scale > max_update_scale:
            update_scale = max_update_scale
            
        # Check if the current step should be accepted.
        if error > 1: # reject
            t_step*= min(update_scale, 1.0)
        else: # accept
            t_now += t_step
            t_step *= update_scale
        
            ret[steps] = y_next
            time[steps] = t_now
            steps += 1
       
          
    return ret[0:steps], time[0:steps]