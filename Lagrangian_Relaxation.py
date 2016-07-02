
#Loading the libraries for computation
import sys, pandas as pd, numpy as np





#Parameters
Node_Type, distance_cutoff, supply_points = int(sys.argv[-3]), int(sys.argv[-2]), int(sys.argv[-1])






def data_load(Node_Type):
    """
    :param Node_type: Which datasets
    :return: Load the datasets
    """
    if (Node_Type == 49):
        distance_49 = np.array(pd.read_csv('49 NodeDistanceData.txt')).reshape(Node_Type,Node_Type) # 49 Distance Node
        demand_49 = pd.read_table('49 NodeDemandData.txt', sep='\t', header=None) # 49 Demand Node
        demand_49 = demand_49.ix[:,0]
        return [distance_49, demand_49]

    elif (Node_Type == 88):
        distance_88 = np.array(pd.read_csv('88 NodeDistanceData.txt')).reshape(Node_Type,Node_Type) # 88 Distance Node
        demand_88 = pd.read_table('88 NodeDemandData.txt', sep='\t', header=None) # 49 Demand Node
        demand_88 = demand_88.ix[:,0]
        return [distance_88, demand_88]

    elif (Node_Type == 150):
        distance_150 = np.array(pd.read_csv('150 NodeDistanceData.txt')).reshape(Node_Type,Node_Type) # 150 Distance Node
        demand_150 = pd.read_table('150 NodeDemandData.txt', sep='\t', header=None) # 49 Demand Node
        demand_150 = demand_150.ix[:,0]
        return [distance_150, demand_150]

    else:
        return ['','']



def pre_process(dis_matrix, distance_cutoff):
    """
    This function will preprocess the distance.
    :param dis_matrix: a matrix with rows as node and columns as node and the entry is the distance
    :param Dc: the availabel distance that can be achieved
    :return pro_matrix: a matrix the entry where larger than the Dc will be assigned 0, otherwise 1.
    """
    m, n = dis_matrix.shape
    pro_matrix = np.zeros((m, n))
    pro_matrix[dis_matrix < distance_cutoff] = 1
    return pro_matrix





def lagrangian_relaxation(demand_data, distance_data, lambda_i, supply_points):
    """
    For performing one LR Iteration.
    :param demand_data: an array with demand for each node
    :param distance_data: an matrix with entry as distance between the nodes
    :param lambda_i: lagrangian multiplier for each node
    :return X_i as factory location, Z_i as demand satisfaction, upper bound and lower bound of the total demand
    """
    h_i, Z_i = demand_data, np.zeros(len(demand_data)) ## subproblem 1
    HL, P = h_i - lambda_i, supply_points
    top_p = sorted(HL, key = lambda x: -x)[:P]
    Z_i = [1 if h in top_p else 0 for h in HL]
    sub1_value = np.dot(Z_i, HL)

    a_ij_lambda_i = np.dot(distance_data, lambda_i) ## subproblem 2
    top_p = sorted(a_ij_lambda_i, key = lambda x: -x)[:P]

    X_i = [1 if A in top_p else 0 for A in a_ij_lambda_i] # X_i is the solution to subproblem 2
    sub2_value = np.dot(a_ij_lambda_i, X_i)

    upper_bound = sub1_value + sub2_value
    lower_bound = np.dot(h_i,Z_i)
    #print "upper_bound", upper_bound,"lower_bound", lower_bound

    return X_i, Z_i, upper_bound, lower_bound





def update_lambda(alpha, upper_bound, lower_bound, a_ij, X_i, Z_i, lambda_i):
    """
    This function returns the value of updated lambda using previous values
    :param alpha: constant
    :param a_ij: demand matrix that can be satisfied within the distance
    """
    t = alpha * (upper_bound - lower_bound)/ np.dot((np.dot(a_ij, X_i) - Z_i), (np.dot(a_ij, X_i) - Z_i).T)
    lambda_i = np.maximum(0, lambda_i - t*(np.dot(a_ij, X_i) - Z_i))
    return lambda_i





def iterator(demand, distance, supply_points = 10):
    """
    Getting started for all the different nodes
    :param demand: demand array
    :param distance: distance matrix
    """
    n, alpha = 1, 2.0
    lambda_i = np.mean(demand) + 0.5 * (demand - np.mean(demand))

    max_iterations, min_alpha = 500, 0.01 #To ensure convergence

    X_i, Z_i, upper_bound, lower_bound = lagrangian_relaxation(demand, distance, lambda_i, supply_points)
    best_lower_bound, best_upper_bound = lower_bound, upper_bound #Updating UB & LB after each iteration

    improvement_iteration = 0
    converged = False

    while not converged:
        X_i, Z_i, upper_bound, lower_bound = lagrangian_relaxation(demand, distance, lambda_i, supply_points)

        if upper_bound >= best_upper_bound:
            improvement_iteration += 1

        best_upper_bound = min(best_upper_bound, upper_bound)
        best_lower_bound = max(best_lower_bound, lower_bound)

        if best_upper_bound == best_lower_bound or n > max_iterations or alpha < min_alpha:
            return upper_bound, lower_bound

        elif improvement_iteration == 4:
            alpha = alpha/2.0

        else:
            lambda_i = update_lambda(alpha, upper_bound, lower_bound, distance, X_i, Z_i, lambda_i)

        n += 1
        if improvement_iteration >= max_iterations  or  improvement_iteration <= min_alpha:
            converged = True

        demand_met = np.dot(X_i, demand)

    return demand_met







# Running the main function
if __name__ == '__main__':

    print "******* Loading the dataset **********"
    distance, demand = data_load(Node_Type)

    print "******* Preprocessing the distance matrix **********"
    distance = pre_process(distance, distance_cutoff)

    print "******* Producing the Output **********"
    UB, LB = iterator(demand, distance, supply_points)
    print np.round(UB,0)
