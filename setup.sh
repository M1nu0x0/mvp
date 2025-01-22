#!/bin/bash

conda create -n mvp2 python=3.12 -y
conda activate mvp2

echo "패키지 목록 업데이트 중..."
sudo apt update

echo "필수 라이브러리 설치 중..."
sudo apt install -y libfreetype6-dev libpng-dev
sudo apt autoremove

# libstdc++.so.6 백업 및 심볼릭 링크 생성
lib_path=~/anaconda3/envs/mvp2/lib/libstdc++.so.6
backup_path=${lib_path}.bak

if [ -f "$lib_path" ]; then
    echo "Backing up $lib_path to $backup_path..."
    mv $lib_path $backup_path
else
    echo "$lib_path not found, skipping backup."
fi

echo "Creating symbolic link for libstdc++.so.6..."
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $lib_path

python ./lib/models/ops/setup.py build install
