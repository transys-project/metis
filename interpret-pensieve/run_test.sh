conda create -n metis python=3.7 -y
conda activate metis
pip install numpy==1.17.2 tensorflow==1.14.0 tflearn==0.3.2 scikit-learn==0.21.3 pydotplus==2.0.2 protobuf<4
sudo apt install graphviz -y
unzip cooked_traces.zip
mkdir decision_tree
python main.py 100