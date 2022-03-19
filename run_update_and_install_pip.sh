#/usr/bin bash

rm -rf chk_tflite_mediapipe/__pycache__
rm -rf tflite_custom/tflite_runtime/__pycache__

echo "clean up previous tflite_custom"


echo "uninstall tflite-runtime and tflite-runtime-inp"
pip uninstall -y tflite-runtime-inp 
pip uninstall -y tflite-runtime
echo "tflite pip after uninstall"
pip list | grep tflite
echo ""

rm -rf tflite_custom
cp -r ~/prj_tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3 .
mv python3 tflite_custom

export PROJECT_NAME="tflite-runtime-inp"
export PACKAGE_VERSION="2.9.01"
pip install -e tflite_custom
echo "tflite pip after reinstall"
pip list | grep tflite
echo ""

echo "done"
