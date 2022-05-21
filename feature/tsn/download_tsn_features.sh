# Download TSN feature files, refer to https://github.com/salesforce/densecap#data-preparation for more details about feature extraction.
wget http://youcook2.eecs.umich.edu/static/dat/anet_densecap/training_feat_anet.tar.gz
wget http://youcook2.eecs.umich.edu/static/dat/anet_densecap/validation_feat_anet.tar.gz
wget http://youcook2.eecs.umich.edu/static/dat/anet_densecap/testing_feat_anet.tar.gz

tar -zxvf training_feat_anet.tar.gz
tar -zxvf validation_feat_anet.tar.gz
tar -zxvf testing_feat_anet.tar.gz
