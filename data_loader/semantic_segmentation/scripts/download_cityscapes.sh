#!/usr/bin/env bash
global_path='../../../vision_datasets'
data_dir=$global_path'/cityscapes'

mkdir -p $data_dir
cd $data_dir

# enter user details
uname='' #
pass=''

wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username='$uname'&password='$pass'&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
# Uncomment if you want to download coarse
#wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=4
#wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=2

# Enter absolute path to the directory where the data is downloaded if you want to move the data to a different location
data_dir='/path/to/data'
mv gtCoarse.zip $data_dir
mv gtFine_trainvaltest.zip $data_dir
mv leftImg8bit_trainextra.zip $data_dir
mv leftImg8bit_trainvaltest.zip $data_dir

# Enter absolute path to the directory where the data is downloaded if you want to move the data to a different location
#unzip -q -o gtCoarse.zip # $data_dir/gtCoarse.zip
unzip -q -o $data_dir/gtFine_trainvaltest.zip
#unzip -q -o leftImg8bit_trainextra.zip # $data_dir/leftImg8bit_trainextra.zip
unzip -q -o $data_dir/leftImg8bit_trainvaltest.zip

#rm -rf gtCoarse.zip # $data_dir/gtCoarse.zip
rm -rf $data_dir/gtFine_trainvaltest.zip
#rm -rf leftImg8bit_trainextra.zip # $data_dir/leftImg8bit_trainextra.zip
rm -rf $data_dir/leftImg8bit_trainvaltest.zip
