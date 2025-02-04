import sionna
import inspect

print("All available objects in sionna.channel:")
for name, obj in inspect.getmembers(sionna.channel):
    print(f"- {name}: {type(obj)}")
