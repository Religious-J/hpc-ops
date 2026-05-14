set -xe

rm -rf cutlass 
rm -rf cutlass.git

git clone https://github.com/NVIDIA/cutlass.git cutlass.git

mkdir -p cutlass 
mv cutlass.git/include cutlass/

rm -rf cutlass.git
