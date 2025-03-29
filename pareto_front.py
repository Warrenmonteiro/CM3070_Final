import numpy as np

def find_pareto_front(df):
    """
    Given a DataFrame with columns:
      - 'Vehicle_Wait_Time'
      - 'Emissions'
      - 'Ped_Safety'
      - 'Total_Cars'
    
    where lower Vehicle_Wait_Time, Emissions, and Ped_Safety are better, and
    higher Total_Cars is better, this function returns the rows on the Pareto front.
    """
    # Convert objective columns to numpy arrays
    wait = df['Vehicle_Wait_Time'].to_numpy()
    emissions = df['Emissions'].to_numpy()
    ped = df['Ped_Safety'].to_numpy()
    cars = df['Total_Cars'].to_numpy()
    
    n_points = len(df)
    is_dominated = np.full(n_points, False)  # Boolean mask; True if the point is dominated
    
    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                continue  # Skip comparing the same point
            
            # Check if point j dominates point i.
            # For objectives to be minimized, we use <=.
            # For objectives to be maximized, we use >=.
            if (wait[j] <= wait[i] and
                emissions[j] <= emissions[i] and
                ped[j] <= ped[i] and
                cars[j] >= cars[i]):
                # At least one objective must be strictly better:
                if (wait[j] < wait[i] or
                    emissions[j] < emissions[i] or
                    ped[j] < ped[i] or
                    cars[j] > cars[i]):
                    is_dominated[i] = True
                    break  # No need to check further; i is dominated.
    
    # Return only the nondominated (Pareto optimal) rows.
    return df[~is_dominated]

