# Download sinlge-cell data from https://github.com/rsinghlab/SCOT
wget https://github.com/rsinghlab/SCOT/archive/refs/heads/master.zip
unzip master.zip
mv SCOT-master/data/* .
rm -r SCOT-master
rm master.zip

# unzip CASP14 data
unzip CASP14.zip