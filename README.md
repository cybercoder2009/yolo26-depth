## Dependencies
#### ffmpeg7+
```
sudo apt update
sudo apt install -y pkg-config libavcodec-dev libavformat-dev libavutil-dev libavdevice-dev libavfilter-dev libswscale-dev libswresample-dev
```
#### ubuntu 24.04 cuda 13 runtime
```text
ultralytics-inference 0.0.43
        └── ort 2.0.0-rc.13
                └── CUDA 13 Runtime
                    ├── CUDA Runtime
                    ├── cuBLAS 13
                    └── cuDNN 9
```

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install \
    cuda-cudart-13-0 \
    libcublas-13-0 \
    libcudnn9-cuda-13
```
```bash
echo '/usr/local/cuda-13.0/targets/x86_64-linux/lib' \
    | sudo tee /etc/ld.so.conf.d/cuda-13-0.conf
sudo ldconfig
# verify
ldconfig -p | grep -E \
'libcublasLt.so.13|libcublas.so.13|libcudart.so.13'
```