#!/home/ubuntu/anaconda3/bin/python


"""
Author: Ciaran Robb
Research Associate in Earth Observation
Centre for Landscape and Climate Research (CLCR)
Department of Geography, University of Leicester, University Road, Leicester, 
LE1 7RH, UK 

If you use code to publish work cite/acknowledge me and other lib authors as 
appropriate

Description
-----------
A series of tools for the download and preprocessing of data (mainly sentinel)




"""
import subprocess #import call #Popen, PIPE, STDOUT
import json
import glob2
import os
from tqdm import tqdm
#from sentinelsat import sentinel
try:
    from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
except:
    ImportError 
    print('using older version of sentinelsat')
    oldsat = True
    from sentinelsat.sentinel import SentinelAPI, get_coordinates

#import os
import gdal, ogr
#import select
#import shapefile
import numpy as np
#import xml.etree.ElementTree as etree
from more_itertools import unique_everseen
from shapely.wkt import loads
import xmltodict
#import wget
import csv
import datetime
import sys
from joblib import Parallel, delayed
from sentinelhub import download_safe_format
#from shapely.geometry import  mapping
#from shapely.geometry import Polygon

def sent2_query(user, passwd, geojsonfile, start_date, end_date, cloud = '100',
                output_folder=None, api = True):
    """A convenience function that wraps sentinelsat query & download although this is hardly necessary but I am lazy 
    
    Parameters
    ----------
    user : string
        username for esa hub
        
    passwd : string
        password for hub
        
    geojsonfile : string
        AOI polygon of interest
    
    start_date : string
        date of beginning of search
    
    end_date : string
        date of end of search
    
    output_folder : string
        where you intend to download the imagery
    
    cloud : string (optional)
        include a cloud filter in the search
    
    Notes
    -----
    
    I have found the sentinesat sometimes fails to download the second image,
    so I have written some code to avoid this - choose api = False for this
    
   
    """
##set up your copernicus username and password details, and copernicus download site... BE CAREFUL if you share this script with others though!
    api = SentinelAPI(user, passwd)

# NOWT WRONG WITH API - 
    if oldsat is True:
        footprint = get_coordinates(geojsonfile)
    else:
        footprint = geojson_to_wkt(read_geojson(geojsonfile))
    products = api.query(footprint,
                         ((start_date, end_date)), platformname="Sentinel-2",
                         cloudcoverpercentage = "[0 TO "+cloud+"]")#,producttype="GRD")
    products_df = api.to_dataframe(products)
    if  api is True and output_folder != None:

        api.download_all(directory_path=output_folder)
        

    else:        
        prods = np.arange(len(products))        
        # the api was proving flaky whereas the cmd line always works hence this
        # is alternate the download option
        if output_folder != None:
#            procList = []    
            for prod in prods:
                #os.chdir(output_folder)
                sceneID = products[prod]['id']
                cmd = ['sentinel', 'download','-p', output_folder,
                       user, passwd, sceneID]
                print(sceneID+' downloading')
                subprocess.call(cmd)
                
    
            #[p.wait() for p in procList]              
    return products_df, products

        
def sent1_query(user, passwd, geojsonfile, start_date, end_date,
                output_folder=None, api = True):
    """
    A convenience function that wraps sentinelsat query & download although 
    this is hardly necessary but I am lazy 
    
    Parameters
    ----------
    user : string
        username for esa hub
        
    passwd : string
        password for hub
        
    geojsonfile : string
        AOI polygon of interest
    
    start_date : string
        date of beginning of search
    
    end_date : string
        date of end of search
    
    output_folder : string
        where you intend to download the imagery
        
    Notes
    -----
    
    I have found the sentinesat sometimes fails to download the second image,
    so I have written some code to avoid this - choose api = False for this
    
    """

    api = SentinelAPI(user, passwd)

 
    if oldsat is True:
        footprint = get_coordinates(geojsonfile)
    else:
        footprint = geojson_to_wkt(read_geojson(geojsonfile))
    products = api.query(footprint,
                         ((start_date, end_date)),
                         platformname="Sentinel-1",
                         producttype="GRD" ,polarisationmode="VV, VH")
    products_df = api.to_dataframe(products)
    
    if  api is True and output_folder != None:

        api.download_all(directory_path=output_folder)
        

    else:        
        prods = np.arange(len(products))        
        # the api was proving flaky whereas the cmd line always works hence this
        # is alternate the download option
        if output_folder != None:
#            procList = []    
            for prod in prods:
                #os.chdir(output_folder)
                sceneID = products[prod]['id']
                cmd = ['sentinel', 'download','-p', output_folder,
                       user, passwd, sceneID]
                print(sceneID+' downloading')
                subprocess.call(cmd)
    return products_df, products

def sent2_google(scene, start_date, end_date,  outputcatalogs, 
                 cloudcover='100',):
    
    """ 
    Download S2 data from google. Adapted from a guys script into functional 
    form with some modifications
    
    Parameters
    ----------
    scene : string
        tileID (eg '36MYE')
    
    start_date : string 
                eg. '2016-12-23'
    
    end_date : string 
                eg. '2016-12-23'
                
    outputcatalogs : string
        destination folder for catalog that is searched for image
        
        output = destination folder
        
    Returns
    ------- 
    urlList : list
        a list of the image urls
        """

    
#    SENTINEL2_METADATA_URL = ('http://storage.googleapis.com/gcp-public'                     
#                                    '-data-sentinel-2/index.csv.gz')
    def _downloadMetadataFile(outputdir):
        url = ('http://storage.googleapis.com/gcp-public'                     
                                    '-data-sentinel-2/index.csv.gz')
        # This function downloads and unzips the catalogue files
        
        program = 'Sentinel'
        theZippedFile = os.path.join(outputdir, 'index_' + program + '.csv.gz')
        theFile = os.path.join(outputdir, 'index_' + program + '.csv')
        if not os.path.isfile(theZippedFile):
            print("Downloading Metadata file...")
            # download the file
            try:
                subprocess.call('curl ' + url + ' -o ' + theZippedFile, shell=True)
            except:
                print("Some error occurred when trying to download the Metadata file!")
        if not os.path.isfile(theFile):
            print("Unzipping Metadata file...")
            # unzip the file
            try:
                if sys.platform.startswith('win'):  # W32
                    subprocess.call('7z e -so ' + theZippedFile + ' > ' + theFile, shell=True)  # W32
                elif sys.platform.startswith('linux'):  # UNIX
                    subprocess.call(['gunzip', theZippedFile])
            except:
                print("Some error occurred when trying to unzip the Metadata file!")
        return theFile
        
    def _findS2InCollectionMetadata(collection_file, cc_limit, date_start, date_end, tile):
        # This function queries the sentinel2 index catalogue and retrieves an url for the best image found
        print("Searching for images in catalog...")
        cloudcoverlist = []
        cc_values = []
        with open(collection_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                year_acq = int(row['SENSING_TIME'][0:4])
                month_acq = int(row['SENSING_TIME'][5:7])
                day_acq = int(row['SENSING_TIME'][8:10])
                acqdate = datetime.datetime(year_acq, month_acq, day_acq)
                if row['MGRS_TILE'] == tile and float(row['CLOUD_COVER']) <= cc_limit and date_start < acqdate < date_end:
                    cloudcoverlist.append(row['CLOUD_COVER'] + '--' + row['BASE_URL'])
                    cc_values.append(float(row['CLOUD_COVER']))
                else:
                    url = ''
        urlList =[]
        for i in cloudcoverlist:
            if float(i.split('--')[0]) <= cc_limit:
                url = i.split('--')[1]
            if url != '':
                url = 'http://storage.googleapis.com/' + url.replace('gs://', '')
                urlList.append(url)
        return urlList
    
    
   # Main ---------------
    sentinel2_metadata_file = _downloadMetadataFile(outputcatalogs)
    cloudcover = float(cloudcover)
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    urlList = _findS2InCollectionMetadata(sentinel2_metadata_file,
                                     cloudcover, start_date,
                                     end_date, scene)
    
    return urlList
    #downloadS2FromGoogleCloud(url, output)
    
    
#    l1cList = glob2.glob(output+'/*L1C*.SAFE/GRANULE/*')
#    grans = np.arange(len(l1cList))
#    
#    for gran in grans:   
#        fld, pth = os.path.split(l1cList[gran])
#        if scene[2:6] in pth:
#            l1cList.pop(gran)
        
def _downloadS2FromGoogleCloud(url, outputdir):
        # this function collects the entire dir structure of the image files from
        # the manifest.safe file and builds the same structure in the output
        # location
        img = url.split("/")[len(url.split("/")) - 1]
        manifest = url + "/manifest.safe"
        destinationDir = os.path.join(outputdir, img)
        if not os.path.exists(destinationDir):
            os.makedirs(destinationDir)
        destinationManifestFile = os.path.join(destinationDir, "manifest.safe")
        subprocess.call('curl ' + manifest + ' -o ' + destinationManifestFile, shell=True)
        readManifestFile = open(destinationManifestFile)
        tempList = readManifestFile.read().split()
        for l in tempList:
            if l.find("href") >= 0:
                completeUrl = l[7:l.find("><") - 2]
                # building dir structure
                dirs = completeUrl.split("/")
                for d in range(0, len(dirs) - 1):
                    if dirs[d] != '':
                        destinationDir = os.path.join(destinationDir, dirs[d])
                        try:
                            os.makedirs(destinationDir)
                        except:
                            continue
                destinationDir = os.path.join(outputdir, img)
                # downloading files
                destinationFile = destinationDir + completeUrl
                try:
                    subprocess.call('curl ' + url + completeUrl + ' -o ' + destinationFile, shell=True)
                    #print(url + completeUrl + ' -o ' + destinationFile+' downloading')
                except:
                    continue        

def sent2_amazon(user, passwd, geojsonfile, start_date, end_date, output_folder, 
                 tile = None, cloud = '100'):
    """Query the ESA catalogue then download S2 from AWS with correct renaming of stuff
    

    Parameters
    ----------
    user : string
        username for esa hub
        
    passwd : string
        password for hub
        
    geojsonfile : string
        AOI polygon of interest
    
    start_date : string
        date of beginning of search
    
    end_date : string
        date of end of search
    
    output_folder : string
        where you intend to download the imagery
        
    tile : string
        S2 tile 
    
    cloud : string (optional)
        include a cloud filter in the search
    
    Notes:
    ------
        
    Credit to sentinelsat for the query aspect of this function, and 
    sentinelhub for the AWS aspect. 
    
    """
    
    
    # Examples of sentinehub usage:
    #download_safe_format('S2A_OPER_PRD_MSIL1C_PDMC_20160121T043931_R069_V20160103T171947_20160103T171947')
    #download_safe_format('S2A_MSIL1C_20170414T003551_N0204_R016_T54HVH_20170414T003551')
    #download_safe_format(tile=('T38TML','2015-12-19'), entire_product=True)
    #entire prod really mean whole tile in old format! Avoid!
    #download_safe_format(tile=('T54HVH','2017-04-14'))
    
    # Use sentinel sat to query  
    api = SentinelAPI(user, passwd)

    if oldsat is True:
        footprint = get_coordinates(geojsonfile)
    else:
        footprint = geojson_to_wkt(read_geojson(geojsonfile))
    products = api.query(footprint,
                         ((start_date, end_date)), platformname="Sentinel-2",
                         cloudcoverpercentage = "[0 TO "+cloud+"]")#,producttype="GRD")

    products_df = api.to_dataframe(products)

    # If using an aoi shape this is the option to follow at present until I
    # write a native function
    if tile is None:
        Parallel(n_jobs=-1,
                 verbose=2)(delayed(download_safe_format)(i,
                           folder = output_folder) for i in
                           products_df.identifier)
    # If the tile id is known then use this - likely handy for oldfmt
    else:


        # A kludge for now until I spend more than two mins writing this func
        dateList =[]
        for prod in products_df['ingestiondate']:
            date1 = prod.strftime('%Y-%m-%d')
            dateList.append(date1)


        Parallel(n_jobs=-1,
                 verbose=2)(delayed(download_safe_format)(tile=(tile,i),
                           folder = output_folder)
                           for i in dateList)
        
    
def sent_attributes(footprints):
    """
    Get a sorted list of tuples each containing the date and sceneID of S2 from
    a footprints geojsom produced by sentinelsat/sent1/2query
    
    Parameters
    ----------
        
    footprints : string
        path to geojson file
    
    Returns
    -------
    
    a list of attribute pairs 
        
    """
    shp = ogr.Open(footprints)    
    lyr = shp.GetLayer()
    noFeat = lyr.GetFeatureCount()+1
    attributes = list()
    
    for label in tqdm(range(1, noFeat)):
        feat = lyr.GetFeature(label)
        if feat == None:
            continue
        date = feat.GetField('date_beginposition')
        prod_Id = feat.GetField('product_id')
        attributes.append((date, prod_Id))
    attributes.sort()
    return attributes

def sent_latest(user, passwd, inShape, footprints, output_folder):
    """Download the latest scene defined by the downloaded geojson that intersects 
    the largest area of the AOI inShape
    The records are in order of date decending which helps here
    
    Parameters
    ----------
    user : string
        username for esa hub
        
    passwd : string
        password for hub
    
    inShape : string
        AOI polygon of interest
    
    footprints : string
        S2 footprints shape (OGR compatible)
    
    output_folder : string
        where you intend to download the imagery

    
    """
    
    #open files
    shpAOI = ogr.Open(inShape)
    shpFt = ogr.Open(footprints)
    #get layers
    lyrAOI = shpAOI.GetLayer()
    lyrFt = shpFt.GetLayer()
    #get AOI geometry in wkt form for shapely
    featAOI = lyrAOI.GetFeature(0)
    if featAOI== None:
        featAOI = lyrAOI.GetFeature(1)
    geomAOI = featAOI.GetGeometryRef()
    wktAOI = geomAOI.ExportToWkt()
    # shapely poly
    polyAOI = loads(wktAOI)
    
    #loop prep
    #lyrFt.GetFeatureCount
    #featFt = lyrFt.GetNextFeature()
    areaList = list()
    tileList = list()
    noFeat = lyrFt.GetFeatureCount()+1
    # loop through footprints and find the biggest and latest intersect
    for label in tqdm(range(1, noFeat)):
        featFt = lyrFt.GetFeature(label)
        geomFt = featFt.GetGeometryRef()
        ftWkt = geomFt.ExportToWkt()
        polyFt = loads(ftWkt)
        if polyFt.intersects(polyAOI) == True:
            tileId = featFt.GetField('product_id')
            interPoly = polyFt.intersection(polyAOI)
            tileList.append(tileId)
            areaList.append(interPoly.area)

    maxArea = np.array(areaList).max()
    marker = np.where(areaList==maxArea)[0]
    sceneID = tileList[marker]
    #return imageID    
    
    #lines below don't work
#    bash = 'ogr2ogr -f "ESRI Shapefile" '+inShape+'.shp "'+inShape+'.geojson"'
#    os.system(bash)
    api = SentinelAPI(user, passwd,)
#    shp = ogr.Open(inShape)
#    lyr = shp.GetLayer()
#    feat = lyr.GetFeature(1)
#    sceneID = feat.GetField("product_id")
    
    #r = shapefile.Reader(inShape)
    # Convieniently, the latest scene is always first!!!    #sceneID = r.records()[0][5]
    
    api.download_all(output_folder, show_progress=True,
                 max_rst_retries=10)
    
def _merge_images(folder, wildcard, mosaic):
    """ A function to merge rasters in a folder and subfolders such as those in
    the S2 file structure that uses the gdal_merge script"""

    fileList = glob2.glob(folder+'**/**/*'+wildcard)
    filenames = ' '.join(fileList)
    subprocess.call(['gdal_merge.py', '-of', 'Gtiff', '-o', mosaic, filenames])
    print('mosaic done')

def _get_S2_geoinfo(xmlFile, mode = 'L2A'):
    
    """ reads xml file associated with S2 data and pulls the relevant 
    geoinformation - internal fucntion  """
    
    
    # This opening method must be used to avoid errors
    xmlDoc = open(xmlFile)
    xmlPy = xmlDoc.read()
    
    xmlDict = xmltodict.parse(xmlPy)
    if mode != 'L2A':
        # It is L1C
        geomInfo = xmlDict['n1:Level-1C_Tile_ID']['n1:Geometric_Info']['Tile_Geocoding']
    else: #S2format == 'old':       
        geomInfo = xmlDict['n1:Level-2A_Tile_ID']['n1:Geometric_Info']['Tile_Geocoding']

        
    #TODO - it wouold be better to use xml2dict to make the code more 
    # understandable, as the indexing is all numerical at present
    #TODO - Add atmosphere info for Atcor in orfeo

    cs = geomInfo['HORIZONTAL_CS_CODE']
    cs_code = geomInfo['HORIZONTAL_CS_NAME']
    rows10 = geomInfo['Size'][0]['NROWS']
    cols10 = geomInfo['Size'][0]['NCOLS']
    
    rows20 =  geomInfo['Size'][1]['NROWS']
    cols20 =  geomInfo['Size'][1]['NCOLS']
     
    ulx10  = geomInfo['Geoposition'][0]['ULX']
    
    uly10  = geomInfo['Geoposition'][0]['ULY']
    
    ulx20  = geomInfo['Geoposition'][1]['ULX']
    
    uly20  = geomInfo['Geoposition'][0]['ULY']
       
    geoinfo = {'cs': cs, 'cs_code': cs_code, 'cols10': cols10, 'rows10': rows10,
               'cols20': cols20, 'rows20': rows20, 'ulx10': ulx10, 
               'uly10': uly10, 'ulx20': ulx20, 'uly20': uly20}
               
    return geoinfo

def get_intersect(folder, polygon, resolution=None):
    """Get intersect between rasters and AOI polygon
    
    Parameters
    ----------   
    folder : string
        the S2 tile folder containing the granules ending .SAFE
    polygon : string
        the AOI polygon (must be same crs as rasters)
        
    
    Notes
    -----
    gdal tile index is used as occasionally using raster info directly
    to compare geometry with polygons produced incorrect results, thus 
    there is longer processing time, but less likelyhood of errors
    """
    #paths = glob2.glob(folder+'/GRANULE/*/')
    # choice on resolution so search terms are correct
    if resolution == None:
        fileList = glob2.glob(folder+'/GRANULE/**/*.tif')
    if resolution == '20':
        fileList = glob2.glob(folder+'/GRANULE/**/*20m.tif')
        keyword = '_20'
    if resolution == '10':
        fileList = glob2.glob(folder+'/*/**/*10m.tif')
        keyword = '_10'
    fileList = list(unique_everseen(fileList))
    fileNames = str(fileList)
    fileNames=fileNames.replace(",", "")
    fileNames=fileNames.replace("'", "")
    fileNames=fileNames.replace("[", "")
    fileNames=fileNames.replace("]", "")
    #noPaths = np.arange(len(paths))
    #shapeList = list()
    
    vector = ogr.Open(polygon)
    layer = vector.GetLayer()
    feature = layer.GetFeature(0)
    if feature == None:
        feature= layer.GetFeature(1)
    vectorGeometry = feature.GetGeometryRef()
    #vectorWkt = vectorGeometry.ExportToWkt()
    #poly1 = loads(vectorWkt)
    imageList = list()
    outShp = folder+'/tile_index.shp'
    
    print('constructing granule tile index')
    cmd = 'gdaltindex ' + outShp + ' ' + fileNames
    os.system(cmd)
    print('tile index done')

    granuleIndex = ogr.Open(outShp)
    layerR = granuleIndex.GetLayer()
    records = np.arange(layerR.GetFeatureCount())
    deleteList = list()
    print('finding intersecting granules')
    for record in tqdm(records):
        
        featureR = layerR.GetFeature(record)
        if featureR == None:
            continue
        rasterGeometry = featureR.GetGeometryRef()
        #rasterWkt = rasterGeometry.ExportToWkt()
        if rasterGeometry.Intersect(vectorGeometry)==True:
            #if keyword in featureR.GetField('location')[-4:]:
            imageList.append(featureR.GetField('location'))
        else:
            deleteList.append(featureR.GetField('location'))
    # the left over granules
    deleteList = list(unique_everseen(deleteList))
    #the granules that intersect
    granuleList = list(unique_everseen(imageList))
    finalList = list()
    for item in range(0, len(granuleList)):
        finalList.append(os.path.dirname(os.path.dirname(granuleList[item])))
    deleteListF = list()
    
    for item in range(0, len(deleteList)):
        deleteListF.append(os.path.dirname(os.path.dirname(deleteList[item])))
        
    return finalList, deleteListF #, areaList
    
    

def _find_all(name, path):
    """ find all dirs with a specific name wildcard - internal fuction"""
    result = []
    for root, dirs, files in os.walk(path):
        if name in dirs:
            result.append(os.path.join(root, name))
    return result

def get_intersect_S2(folder, polygon, pixelSize=20):
    """ Get the S2 granules that intersect an area of interest, uses S2xml file
    to get granule coords - this function is quicker than get_intersect, but 
    more prone to errors
    
    Parameters
    --------------
    folder : string
        
    polygon : string
    
    pixelSize : int (optional)
    
    """
    
    #os.chdir(folder)
    paths = glob2.glob(folder+'/GRANULE/*/')
    fileList = glob2.glob(folder+'/GRANULE/**/*S2A_USER_MTD*.xml')
    vector = ogr.Open(polygon)
#    imageList = list
    xmls = np.arange(len(fileList))
    granuleList = list() 
    #areaList = list()
    #polyList=list()
    for im in tqdm(xmls):
        geoinfo = _get_S2_geoinfo(fileList[im])
        cols = int(geoinfo['cols20'])
        rows = int(geoinfo['rows20'])
        
        xLeft = int(geoinfo['ulx20'])
        yTop = int(geoinfo['uly20'])
        height = np.negative(pixelSize)
        xRight = int(xLeft+cols*pixelSize)
        yBottom = int(yTop-rows*height)
        
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(xLeft, yTop) # top left
        ring.AddPoint(xLeft, yBottom) # btm left
        ring.AddPoint(xRight, yTop) # top right
        ring.AddPoint(xRight, yBottom) #bottom right

#        topLeft = ((xLeft, yTop))
#        btmLeft = ((xLeft, yBottom))
#        topRight = ((xRight, yTop))
#        btmRight = ((xRight, yBottom))
        rasterGeometry = ogr.Geometry(ogr.wkbPolygon)
        rasterGeometry.AddGeometry(ring)
        
#        #reproject geometry to be same as raster
#        source = osr.SpatialReference()
#        source.ImportFromEPSG(4326)
#
#        target = osr.SpatialReference()
#        target.ImportFromEPSG(32736)

#        transform = osr.CoordinateTransformation(source, target)        
        
        
        
        # Get vector geometry
        layer = vector.GetLayer()
        feature = layer.GetFeature(0)
        if feature == None:
            feature = layer.GetFeature(1)
        vectorGeometry = feature.GetGeometryRef()
        #vectorWkt = vectorGeometry.ExportToWkt()
        #poly1 = loads(vectorWkt)
        #critical that points join in correct sequence
        #poly2 = Polygon([topLeft,topRight, btmRight, btmLeft])
        if rasterGeometry.Intersect(vectorGeometry)==True:
            #imageList.append(fileList[im])
#        if poly2.intersects(poly1) == True:
#            interPoly = poly2.intersection(poly1)
            granuleList.append(paths[im])
#            areaList.append(interPoly.area)
#            polyList.append(interPoly)
        #feature.Destroy()
        #del poly1, poly2
#            length = len(shapeDict['coordinates'][0])
#            coordList = list()
#            for coord in range(0, length):
#                coordList.append(j['coordinates'][0][coord])
#                intersectShp = Polygon(coordList)
    #maxArea = np.array(areaList).max()
    #marker = np.where(areaList==maxArea)[0]
    #imageID = granuleList[marker]
             
    return granuleList#, areaList

def unzip_S2_granules(folder, granules=None):
    
    """ Unzip the S2 granules dependent on a specific area in utm 
    granule - for the old cumbersome tile format 
        
    
    The function unzips only the tiles of interest to this area in this project
    
    Parameters
    ----------   
    folder : string
        a folder contain S2 tiles
    
    area : string 
        area of interest (optional)
    
    granules : string (optional) - recommended
        a list of granule UTM codes e.g ['36MYE', '36MZE']
    
    Notes
    -----
    This was written for the file format S2 imagery initally 
    came in from the ESA hub, which was enormous and impractical. 
    Fortunately this changed near end of 2016, in which case you can simply
    unzip with bash or whatever!

    """    
    fileList = glob2.glob(folder+'/*.zip*')

    
    # Now to unzip depending on options above using subprocess
    # The wildcard search terms are needed to ge all the correct file

    procList = []
    for file in tqdm(filez):
        # -o forces overwrite, otherwise the process hangs with a yes/no choice
        # only answerable from bash
        fld, fle = os.path.split(fileList[file])
        aux = fle[:-4]+'.SAFE/AUX_DATA'
        yip = '*'+aux+'*'
        
        if area is None:
            # TODO This is not good enough long term - a temp fix
            for granule in granules:
                cmd = ['unzip', '-o', fileList[file], yip,'*'+granule+'*',
                       '*DATASTRIP*', "*HTML*", "*S2A_OPER_MTD_SAFL1C_PDMC*",
                       "*INSPIRE*", "*rep_info*", "*manifest.safe*",  '-d',
                       fileList[file][:-4]]
                p = subprocess.Popen(cmd)
                procList.append(p)
        
        if area == 'mau':
#            wildcards = '"*'+aux+'*" "*MYE*" "*MZE*" "*DATASTRIP*" "*HTML*" \
#            "*S2A_OPER_MTD_SAFL1C_PDMC*" "*INSPIRE*" "*rep_info*" \
#            "*manifest.safe*"'
            cmd = ['unzip', '-o', fileList[file], yip,'*MYE*', '*MZE*',
                   '*DATASTRIP*', "*HTML*", "*S2A_OPER_MTD_SAFL1C_PDMC*",
                   "*INSPIRE*", "*rep_info*", "*manifest.safe*",  '-d',
                   fileList[file][:-4]]
            p = subprocess.Popen(cmd)
            procList.append(p)
            
        if area == 'laikipia':
            cmd = ['unzip', '-o', fileList[file], yip, "*NBA*", "*NCA*",
                   '*DATASTRIP*', "*HTML*", "*S2A_OPER_MTD_SAFL1C_PDMC*",
                   "*INSPIRE*", "*rep_info*", "*manifest.safe*",  '-d',
                   fileList[file][:-4]]
            p = subprocess.Popen(cmd)
            procList.append(p)
            
        if area == 'mtkenya':
            cmd = ['unzip', '-o', fileList[file], yip,  '*DATASTRIP*', 
                   "*HTML*", "*S2A_OPER_MTD_SAFL1C_PDMC*",
                   "*INSPIRE*", "*rep_info*", "*manifest.safe*",  '-d',
                   fileList[file][:-4]]
            p = subprocess.Popen(cmd)
            procList.append(p)
           
        if area == 'aberdare':
            cmd = ['unzip', '-o', fileList[file], yip, "*MBR*", "*MCR*",
                   '*DATASTRIP*', "*HTML*", "*S2A_OPER_MTD_SAFL1C_PDMC*",
                   "*INSPIRE*", "*rep_info*", "*manifest.safe*",  '-d',
                   fileList[file][:-4]]
            p = subprocess.Popen(cmd)
            procList.append(p)
        if area == 'northmau':
            cmd = ['unzip', '-o', fileList[file], yip, "*NYF*", "*NYG*",
                   '*DATASTRIP*', "*HTML*", "*S2A_OPER_MTD_SAFL1C_PDMC*",
                   "*INSPIRE*", "*rep_info*", "*manifest.safe*",  '-d',
                   fileList[file][:-4]]
            p = subprocess.Popen(cmd)
        if area == 'ngong':
            cmd = ['unzip', '-o', fileList[file], yip, "*MBQ*",  
                   '*DATASTRIP*', "*HTML*", "*S2A_OPER_MTD_SAFL1C_PDMC*",
                   "*INSPIRE*", "*rep_info*", "*manifest.safe*",  '-d',
                   fileList[file][:-4]]
            p = subprocess.Popen(cmd)
            procList.append(p)
            
    [p.wait() for p in procList]       
        #print(str(file)+' done')
    print('files extracted')




        


