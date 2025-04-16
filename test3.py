# test_sionna_rt_docs.py
import sionna.rt as rt
import inspect

print("🧠 Fetching available help for sionna.rt components...\n")

for name in dir(rt):
    if name.startswith("_"):
        continue  # Skip internal/private members

    obj = getattr(rt, name)
    if inspect.isclass(obj) or inspect.isfunction(obj):
        print(f"\n🔍 Help for: {name}")
        print("-" * 60)
        try:
            help(obj)
        except Exception as e:
            print(f"❌ Could not load help: {e}")
