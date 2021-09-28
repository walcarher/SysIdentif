import cvxpy as cp
import numpy as np

x = cp.Variable(pos = True, name = "x")
y = cp.Variable(pos = True, name = "y")

objective_fn = x + y**2 # Cruiser-like function

constraints = [#x+y >= 3, # Plane with 135 degrees om the X-Y plane
               0.5*x**2 + 0.5*y**2 <= x + y**2, # Horizontal Plane constraint
               ]

objective = cp.Minimize(objective_fn)

prob = cp.Problem(objective,constraints)

print(constraints[0].is_dgp())

print("Problem is convex:", prob.is_dcp())
print("Problem can be solved by GP:", prob.is_dgp())

result = prob.solve()

print("Solution found:", prob.value)
print("Value for", x, "is:", x.value)
print("Value for", y, "is:", y.value)