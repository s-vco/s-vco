# CounterCurate: https://github.com/HanSolo9682/CounterCurate
git lfs install
git clone https://huggingface.co/datasets/nlphuji/flickr30k
cd flickr30k
unzip flickr30k-images.zip
cd ..
rm -rf flickr30k/.git
git clone https://huggingface.co/datasets/HanSolo9682/Flickr30k-Counterfactuals
cd Flickr30k-Counterfactuals
tar -xvf train_data.tar.gz
cd ..
rm -rf Flickr30k-Counterfactuals/.git
python prepare_countercurate_data.py

# FineCops-Ref: https://github.com/liujunzhuo/FineCops-Ref
mkdir finecops-ref
cd finecops-ref
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip
wget "https://figshare.com/ndownloader/articles/26048050?private_link=e323fe078924c8b36043" -O finecopsref_anno.zip
unzip finecopsref_anno.zip
tar -xvf neg_images.tgz
