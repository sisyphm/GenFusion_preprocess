ENV_NAME="GenFusion"

echo "Creating conda environment: $ENV_NAME"
conda env create --file environment.yml || echo "Environment $ENV_NAME already exists or creation failed."

cd Reconstruction

echo "Installing simple-knn in $ENV_NAME environment..."
conda run -n $ENV_NAME --no-capture-output --live-stream bash -c "CC=gcc-9 CXX=g++-9 pip install submodules/simple-knn"

echo "Installing diff-surfel-rasterization in $ENV_NAME environment..."
conda run -n $ENV_NAME --no-capture-output --live-stream bash -c "CC=gcc-9 CXX=g++-9 pip install submodules/diff-surfel-rasterization"

echo "Setup script finished."