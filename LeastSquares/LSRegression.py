def leastSquaresReg(filename, x_var, y_var):
    import numpy as np
    import pandas as pd
    #Read Data
    df = pd.read_csv(filename)
    #Create constant coeffs
    df["c"] = np.ones(len(df))
    #Solve for LS
    b = df[y_var]
    A = df[["c", x_var]]
    x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
    
    #Calculate
    predictions = np.array(x[0] + x[1]*df[x_var])
    SSE = sum((predictions - np.array(df[y_var]))**2)
    best_fit = {"intercept": x[0], "slope": x[1], "SSE":SSE}
    
    # Plot fitted line
    x_axis = np.linspace(start= min(df[x_var]), stop= max(df[x_var]), num=1000)
    y_axis = best_fit['intercept'] + best_fit['slope']*x_axis
    plt.plot(x_axis, y_axis,  color = "red")
    # Plot real data
    plt.scatter(x = df[x_var], y= df[y_var])

    return best_fit
