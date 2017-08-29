#!/bin/bash

# A script to download S2 imagery from the amazon server
# Author Ciaran Robb, Research Associate in Earth Observation, CLCR, Uni of Leicester

# For some reason I get the 404 error when attempting recursive downloading.. eg

#  wget -nd -r -l 1 -e robots=off  http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/36/M/YE/2016/12/23/0

# hence the cludgy soloution below

# This is written with the intention of using python to pass the arguements below to this script.

# $1 = tile id
# $2 = date
# $3 = S2 file name
# $4 = granule name

mkdir $3.SAFE;

cd $3.SAFE
mkdir DATASTRIP AUX_DATA



#cd $2

# The product metadata is downloaded here


#wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/products/date/productID/filename
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/products/$2/$3/inspire.xml;
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/products/$2/$3/manifest.safe;
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/products/$2/$3/metadata.xml;
mv metadata.xml $3.xml
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/products/$2/$3/preview.png;
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/products/$2/$3/product.json;

wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/productInfo.json  
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/tileInfo.json

cd datastrip;

wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/products/$2/$3/datastrip/0/metadata.xml;

cd ..; 

cd $3

# http://sentinel-s2-l1c.s3.amazonaws.com/tiles/36/M/YE/2016/12/23/0/B01.jp2
#wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/date/0/filename

#wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/36/M/YE/2015/12/29/0/B01.jp2

mkdir GRANULE;

cd GRANULE;

mkdir $4cd ..

cd $4;

mkdir IMG_DATA QI_DATA AUX_DATA;

wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/metadata.xml;

cd IMG_DATA 
 
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B01.jp2 
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B02.jp2 
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B03.jp2  
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B03.jp2 
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B04.jp2  
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B05.jp2 
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B06.jp2  
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B07.jp2  
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B08.jp2 
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B09.jp2 
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B10.jp2 
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B11.jp2  
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B12.jp2  
wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B8A.jp2  

#wget http://sentinel-s2-l1c.s3-website.eu-central-1.amazonaws.com/tiles/$1/$2/0/B02.jp2


# http://sentinel-s2-l1c.s3.amazonaws.com/tiles/36/M/YE/2016/12/3/0/B01.jp2
