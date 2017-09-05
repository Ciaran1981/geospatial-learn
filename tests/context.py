#Adds the project temporarily to the PATH value for this system, then imports
#from there

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import geospatial_learn
