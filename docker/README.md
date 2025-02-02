# Docker setup
----
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