from pulp import *

'''
In this exercise you are planning the production at a glass manufacturer. This manufacturer only produces wine and beer glasses:

there is a maximum production capacity of 60 hours
each batch of wine and beer glasses takes 6 and 5 hours respectively
the warehouse has a maximum capacity of 150 rack spaces
each batch of the wine and beer glasses takes 10 and 20 spaces respectively
the production equipment can only make full batches, no partial batches
Also, we only have orders for 6 batches of wine glasses. Therefore, we do not want to produce more than this. Each batch of the wine glasses earns a profit of $5 and the beer $4.5.

The objective is to maximize the profit for the manufacturer.
'''


# Initialize Class
model = LpProblem("Maximize Glass Co. Profits", LpMaximize)

# Define Decision Variables
wine = LpVariable('Wine', lowBound=0, upBound=None, cat='Integer')
beer = LpVariable('Beer', lowBound=0, upBound=None, cat='Integer')

# Define Objective Function
model += 5 * wine + 4.5 * beer

# Define Constraints
model += 6 * wine + 5 * beer <= 60
model += 10 * wine + 20 * beer <= 150
model += wine <= 6

# Solve Model
model.solve()
print("Produce {} batches of wine glasses".format(wine.varValue))
print("Produce {} batches of beer glasses".format(beer.varValue))

'''
Logistics planning problem
You are consulting for kitchen oven manufacturer helping to plan their logistics for next month. There are two warehouse 
locations (New York, and Atlanta), and four regional customer locations (East, South, Midwest, West). The expected 
demand next month for East it is 1,800, for South it is 1,200, for the Midwest it is 1,100, and for West it is 1000. 
The cost for shipping each of the warehouse locations to the regional customer's is listed in the table below. 
Your goal is to fulfill the regional demand at the lowest price.

Customer	New York	Atlanta
East	$211	$232
South	$232	$212
Midwest	$240	$230
West	$300	$280
Two Python dictionaries costs and var_dict have been created for you containing the costs and decision 
variables of the model. You can explore them in the console.

'''


# Initialize Model
model = LpProblem("Minimize Transportation Costs", LpMinimize)

# Build the lists and the demand dictionary
warehouse = ['New York', 'Atlanta']
customers = ['East', 'South', 'Midwest', 'West']
regional_demand = [1800, 1200, 1100, 1000]
demand = dict(zip(customers, regional_demand))

var_dict = LpVariable.dicts('Shipments_',
                            [(w, c) for w in warehouse for c in customers],
                            lowBound=0, cat='Integer')

costs = {('Atlanta', 'East'): 232,
 ('Atlanta', 'Midwest'): 230,
 ('Atlanta', 'South'): 212,
 ('Atlanta', 'West'): 280,
 ('New York', 'East'): 211,
 ('New York', 'Midwest'): 240,
 ('New York', 'South'): 232,
 ('New York', 'West'): 300}

# Define Objective
model += lpSum([costs[(w, c)] * var_dict[(w, c)] for c in customers for w in warehouse])

# For each customer, sum warehouse shipments and set equal to customer demand
for c in customers:
    model += lpSum([var_dict[(w, c)] for w in warehouse]) == demand[c]

model.solve()

'''
Scheduling workers problem
You are looking to hire workers to work in a warehouse. Each worker is expected to work 5 consecutive days and then 
have two days off. The chart below has the estimated number of workers you will need each day. You are looking to hire 
the minimum number of workers to handle the workload for each day.

Expected Workload

Day of Week	Employees Needed
0 = Monday	31
1 = Tuesday	45
2 = Wednesday	40
3 = Thursday	40
4 = Friday	48
5 = Saturday	30
6 = Sunday	25

'''

# The class has been initialize, and x, days, and objective function defined
model = LpProblem("Minimize Staffing", LpMinimize)
days = list(range(7))
x = LpVariable.dicts('staff_', days, lowBound=0, cat='Integer')
model += lpSum([x[i] for i in days])

# Define Constraints
model += x[0] + x[3] + x[4] + x[5] + x[6] >= 31
model += x[0] + x[1] + x[4] + x[5] + x[6] >= 45
model += x[0] + x[1] + x[2] + x[5] + x[6] >= 40
model += x[0] + x[1] + x[2] + x[3] + x[6] >= 40
model += x[0] + x[1] + x[2] + x[3] + x[4] >= 48
model += x[1] + x[2] + x[3] + x[4] + x[5] >= 30
model += x[2] + x[3] + x[4] + x[5] + x[6] >= 25

model.solve()



