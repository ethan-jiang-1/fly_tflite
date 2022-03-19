#/usr/bin bash

rm -rf chk_tflite_mediapipe/__pycache__
rm -rf tflite_custom/tflite_runtime/__pycache__

export PROJECT_NAME="tflite-runtime-inp"
export PACKAGE_VERSION="2.9.01"
pip install -e tflite_custom
echo "tflite pip after reinstall"
pip list | grep tflite
echo ""

echo "done"
