#!/usr/bin/env bash

# use this script in the destination folder.

mkdir PascalVOC2012
cd PascalVOC2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
rm VOCtrainval_11-May-2012.tar

wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug_Visualization.zip
wget http://cs.jhu.edu/~cxliu/data/list.zip
unzip SegmentationClassAug.zip
unzip SegmentationClassAug_Visualization.zip
unzip list.zip
rm SegmentationClassAug.zip
rm SegmentationClassAug_Visualization.zip
rm list.zip
