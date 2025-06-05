# I had to adjust this for MacOS

if [ ! -d "cifar-10-batches-py" ]; then
  curl -o cifar-10-python.tar.gz http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  tar -xzvf cifar-10-python.tar.gz
  rm cifar-10-python.tar.gz
fi