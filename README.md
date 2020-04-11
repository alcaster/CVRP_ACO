# CVRP_ACO
## Capacitated Vehicle Routing Problem solved with Ant Colony Optimization

### Datasets: http://www.dca.fee.unicamp.br/projects/infobiosys/vrp/

Assumptions:
1. Graph is complete, there is possibility to reach every city from any city.
2. Demand of each city is <= max capacity of an ant.
3. Solution quality corresponds to total sum of weights of edges which were used in the path. 
4. Every city is traversed only once, except depot. 
5. Depot is always visited at the start, end, when there is no capacity for any other move. It is also possible to return to depot at will.    

Additional:
1. Zawsze opcja powrotu do depotu z probability jak do kazdego innego miasta
2. Jedyna metryka to totalna odległość
