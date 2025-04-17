import mitsuba
mitsuba.set_variant('cuda_ad_rgb')
bsdf = mitsuba.load_dict({'type': 'holder'})
print("Holder plugin loaded successfully" if bsdf else "Holder plugin not found")