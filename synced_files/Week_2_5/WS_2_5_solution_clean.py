# ----------------------------------------
import gurobipy as gp

# ----------------------------------------
# YOUR_CODE_HERE

# SOLUTION
# Project data
I = [90, 45, 78, 123, 48, 60]  # Environmental impact
P = [120, 65, 99, 110, 33, 99]  # Profit

# Minimum required profit
beta = 250
M = 100000

# Number of projects and types
num_projects = len(I)
num_type1_projects = 3
num_type2_projects = num_projects - num_type1_projects

# ----------------------------------------
# YOUR_CODE_HERE

# SOLUTION
model = gp.Model("Project_Selection")
x = model.addVars(num_projects, vtype=gp.GRB.BINARY, name="x")
model.setObjective(sum(I[i] * x[i] for i in range(num_projects)),
                   gp.GRB.MINIMIZE)
model.addConstr(x.sum() == 3, "Select_Projects")
model.addConstr((sum(x[i] for i in range(num_type2_projects, num_projects))
                 - sum(x[i] for i in range(num_type1_projects)) >= 0),
                 "Type_Constraint")
model.addConstr(sum(P[i] * x[i] for i in range(num_projects)) >= beta,
                "Minimum_Profit")
model.optimize()

# ----------------------------------------
# YOUR_CODE_HERE

# SOLUTION
print("Model structure:")        
# see the model that you have built in a nice why to interpret
model.display()  

if model.status == gp.GRB.OPTIMAL:
    print("Optimal Solution:")
    for i in range(num_projects):
        if x[i].x > 0.9:
            print(f"Project {i+1}: Selected")
else:
    print("No optimal solution found.")

    
print("Optimal Objective function Value", model.objVal)    

# ----------------------------------------
# YOUR_CODE_HERE

# SOLUTION
model2 = gp.Model("Project_Selection")
x = model2.addVars(num_projects, vtype=gp.GRB.BINARY, name="x")
model2.setObjective(sum(I[i] * x[i] for i in range(num_projects)),
                   gp.GRB.MINIMIZE)
model2.addConstr(x.sum() == 3, "Select_Projects")
model2.addConstr((sum(x[i] for i in range(num_type2_projects, num_projects))
                 - sum(x[i] for i in range(num_type1_projects)) >= 0),
                 "Type_Constraint")
model2.addConstr(sum(P[i] * x[i] for i in range(num_projects)) >= beta,
                "Minimum_Profit")
# added constraint
gamma = 130
model2.addConstr((sum(I[i] * x[i] for i in range(num_projects))
                 <= gamma * x[0]+ M * (1 - x[0])),
                 "Impact_Constraint") 



# ----------------------------------------
# YOUR_CODE_HERE

# SOLUTION
model2.optimize()

print("Model structure:")        
# see the model that you have built in a nice why to interpret
model2.display()  

# Display the solution
if model2.status == gp.GRB.OPTIMAL:
    print("Optimal Solution:")
    for i in range(num_projects):
        if x[i].x > 0.9:
            print(f"Project {i+1}: Selected")
else:
    print("No optimal solution found.")

    
print("Optimal Objective function Value", model2.objVal)   

