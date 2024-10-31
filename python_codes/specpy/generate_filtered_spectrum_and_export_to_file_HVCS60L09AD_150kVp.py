# For compatibility with Python2
from __future__ import print_function, division, absolute_import
#

import spekpy as sp
print("\n** Script to generate a filtered spectrum and export to a file **\n")

# Generate unfiltered spectrum
s=sp.Spek(kvp=450,th=5)
# Filter the spectrum
s.filter('Cu',0.05).filter('Air',1000)
# Export (save) spectrum to file
spek_name = '240618_HVCS60L09AD_150kVp.txt'
s.export_spectrum(spek_name, comment='A demo spectrum export')
s.summarize(mode='full')


