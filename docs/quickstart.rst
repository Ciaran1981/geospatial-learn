.. _quickstart:

Quickstart
==========


Notes
---------

Be sure to replace the paths with paths to your own imagery/polygons!


Training and model creation
---------------------------

The following simple example uses the learning module to read in training from a shapefile and associated raster, then exhaustively grid search the model based on a default range of parameters. It is also possible to pass sklearn parameteter dicts to the create_model function. 

Bear in mind a large amount of training data and a lot of paramter combinations results in many model fits and lengthy grid search time! 

.. code-block:: python
   
   # Import the module required
   from geospatial_learn import learning
   
   # collect some training data
   trainShape = 'path/to/my/trainingShp.shp'
   inRas = 'path/to/my/rasterFile.shp'	

   # training collection, returning any rejects (invalid geometry - rej)
   # the 'Class' string is the title of the training label field attribute
   training, rej = learning.get_training(trainShape, inRas, 8, 'Class')
   
   # path to my model	
   model = 'path/to/my/model.gz'


   # 	
   results = learning.create_model(training, model, clf='rf', cv=3,
                                cores = 8, strat=True)

   
Classification 
---------------

The following code uses the learning module to classify an image based on the model made in the code above. 


.. code-block:: python

   from geospatial_learn import learning

   # no of bands in raster
   bands = 8

   # path to output map
   outMap = 'path/to/my/rasterFile'

   learning.classify_pixel_bloc(model, inRas, bands, outMap,  blocksize=256)


Polygon processing
------------------

Add attributes to a shapefile - perhaps with a view to classifying them later. 

The following calculates some geometric properties and pixel based statistics using functions from the shape module. 

.. code-block:: python

   from geospatial_learn.shape import shape_props, zonal_stats
   
   # path to polygon
   segShp = 'path/to/my/segmentShp.shp'
   
   # function to write 
   
   # Property of interest	
   prop = 'Eccentricity'

   # function
   shape_props(segShp, prop, inRas=None,  label_field='ID')

   # variables for function
   band = 1
   inRas = 'pth/to/myraster.tif'
   bandname = 'Blue'

   # function
   zonal_stats(segShp, inRas, band, bandname, stat = 'mean',
                write_stat=True, nodata_value=None)

To write multiple attributes a simple loop will suffice:

.. code-block:: python
   
   # shape props
   sProps = ['MajorAxisLength', 'Solidity']
   
   for prop in sProps:
      shape_props(segShp, prop, inRas=None,  label_field='ID')
   
   # zonal stats
   # please note that by using enumerate we assume the bandnames are ordered as the are in the image!
   bandnames = ['b', 'g', 'r', 'nir']


   # Please note we add 1 to the bnd index as python counts from zero
   for bnd,name in enumerate(bandnames):
      zonal_stats(segShp, inRas, bnd+1, name, stat = 'mean', write_stat = True)      


Train & then classify shapefile attributes
-----------------------------

In the previous example several attributes were calculated and written to a shapefile. The following example outlines how to train a ML model then classify these.
In this case the attributes are some of those calculated above

Training
--------

For training a model using shape attributes, an attribute containing the Class label (this can be done manually in any GIS) as well as feature attributes are required. We enter the column index of the Class label attribute. In this example it is column 1.

The remaining attributes are assumed to be features (here we are using the ones calculated in the above looped examples).   

.. code-block:: python

   # collect some training data

   train_col_number = 1

   training = path/to/my/model.gz


   get_training_shp(segShp, train_col_number, outFile = model)

The model is created in the same way as the image based method outlined earlier (see Training and model creation). After this the shapefile attributes are classified with the model as shown below and the results are written as a new attribute 'ClassRf'

.. code-block:: python

   attributes = ['b', 'g', 'r', 'nir','MajorAxisLength', 'Solidity']

   classify_object(model, segShe, attributes, field_name='ClassRf')
 

Sentinel 2 data
---------------

The following code will stack a set of Sentinel 2 (S2) bands into a single raster. The code uses the module 'geodata', which has a range of functions for manipulating raster data.
I have used a genuine S2 path here hence the extreme length of the string!

The function automatically names the stacked raster and saves it in the granule folder. 


.. code-block:: python

   from geospatial_learn import geodata

   path = '/path/to/S2A_MSIL1C_20161223T075332_N0204_R135_T36MYE_20161223T080853/S2A_MSIL2A_20161223T075332_N0204_R135_T36MYE_20161223T080853.SAFE/GRANULE/L2A_T36MYE_A007854_20161223T080853/'	

   outputPth = geodata.stack_S2(path)



