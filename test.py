# Print Sionna version
import sionna
print(f"Sionna version: {sionna.__version__}")

# Import required modules
from sionna.rt import Scene
import inspect

# Examine Scene class initialization parameters
print(f"Scene initialization parameters:")
print(inspect.signature(Scene.__init__))

# Check Scene class attributes
scene = Scene()
print("\nScene instance attributes:")
for attr in dir(scene):
    if not attr.startswith('__'):
        print(f"- {attr}")

# Check specifically for dtype-related attributes
dtype_attrs = [attr for attr in dir(scene) if 'dtype' in attr]
print("\nDtype-related attributes:")
print(dtype_attrs)

# Check the Scene's source code for frequency setter
print("\nFrequency setter method:")
try:
    print(inspect.getsource(Scene.__class__.frequency.fset))
except:
    print("Could not get source for frequency setter")