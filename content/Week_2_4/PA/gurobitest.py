import gurobipy
model = gurobipy.Model()
x = model.addVars(2001, name = 'x')
model.update()
model.optimize()