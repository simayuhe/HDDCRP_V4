cd ./DynamicTimeWarping/
mex dpcore.c
cd ..
cd ./lightspeed-master/
install_lightspeed
cd ..
cd ./ZPclustering/
mex dist2aff.cpp
mex evrot.cpp
mex scale_dist.cpp
mex zero_diag.cpp
cd ..
