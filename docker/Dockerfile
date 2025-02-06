FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
ARG NVIDIA_VISIBLE_DEVICES=all
# 패키지 업데이트 및 필수 라이브러리 설치
RUN apt-get update && apt-get install -y \
    git python3 python3-pip libfreetype6-dev libpng-dev wget 

# 프로젝트 클론 및 의존성 설치
RUN git clone https://github.com/M1nu0x0/mvp.git && \
    cd mvp && \
    pip3 install -r requirements2.txt

# # 모델 컴파일 실행
# RUN python3 mvp/lib/models/ops/setup.py build install

# 모델 다운로드를 위한 디렉터리 생성 및 gdown 설치
RUN mkdir -p mvp/models && \
    cd mvp/models && \
    pip3 install gdown && \
    gdown --id 1DfdZfUtN7mjCR0wSND_pAdKU5E6y12cw

# 데이터 디렉터리 생성 및 데이터 클론
RUN mkdir -p mvp/data && \
    cd mvp/data && \
    git clone https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox.git

# 작업 디렉토리 설정
WORKDIR /mvp