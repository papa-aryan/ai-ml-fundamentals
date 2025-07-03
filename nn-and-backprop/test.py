from micrograd.engine import Value
from graphviz import Digraph

# even simpler example

x1 = Value(5.0)
x2 = Value(-2.0)
x3 = Value(3.0)

y = x1 * x2 + x3

print(f"Forward pass:")
print(f"x1 = {x1.data}")
print(f"x2 = {x2.data}")
print(f"x3 = {x3.data}")

print(f"y = x1 * x2 + x3 = {y.data}")

y.backward()

print(f"\nAfter backward pass:")
print(f"x1.grad = {x1.grad}") 
# so if x1 changes by 1, y changes by -2 (cause of the multiplication), therefore the gradient is -2 
print(f"x2.grad = {x2.grad}")
print(f"x3.grad = {x3.grad}")
print(f"y.grad = {y.grad}")

# example 2
print("\n\n\nExample 2:")

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db


def simple_viz(root, visited=None, depth=0):
    if visited is None:
        visited = set()
    
    if root in visited:
        return
    visited.add(root)
    
    indent = "  " * depth
    print(f"{indent}Value(data={root.data:.4f}, grad={root.grad:.4f})")
    
    if root._op:
        print(f"{indent}  └─ op: {root._op}")
    
    for child in root._prev:
        simple_viz(child, visited, depth + 1)

print("Computation Graph:")
simple_viz(g)