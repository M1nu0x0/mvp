# Docker setup
## Docker Image Build
```bash
chmod +x mvp/docker/build.sh
./docker/build.sh
```

## Docker Container 
```bash
docker run -itd -rm --gpus all -n mvp mvp 
```

## setup
1. Enter docker 
```bash
docker exec -it mvp /bin/bash
```

2. Installation
```bash
 cd ./models/ops
sh ./make.sh
```

3. Download Dataset
```bash
cd mvp/data
./scripts/getData.sh 171204_pose1
```

# Model
Backbone
- FOCUS/models/backbone/pose_resnet50_panoptic.pth.tar

MvP
- FOCUS/models/mvp/model_best_5view.pth.tar

# Dataset
## Panoptic
### Train
- 160422_ultimatum1
- 160224_haggling1
- 160226_haggling1
- 161202_haggling1
- 160906_ian1
- 160906_ian2
- 160906_ian3
- 160906_band1
- 160906_band2

### Validation
- 160906_pizza1
- 160422_haggling1
- 160906_ian5
- 160906_band4

## AISL
### Train & Validation
- FOCUS/240705_1300
  - This dataset may located with panoptic datasets.
