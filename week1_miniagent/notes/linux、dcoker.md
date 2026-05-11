# Linux系统指令及docker容器创建指令

---

## Linux系统指令

### 查看当前目录内容

`ls` 查看当前终端地址存在的文件

`ls -a`  查看当前终端地址存在的文件(包含不可见文件)

`ls -R 目录名` 查看该目录下包含的所有文件结构

### 当前所在位置

`pwd` 当前终端所在的路径位置

### 创建目录

`mkdir 目录名`  创建目录

`mkdir -p A/{AA/AAA,AB,AC},B,C` 创建一个树状目录

`rm 文件名` 删除文件

`rmdir 目录` 删除目录

`rm -rf 目录` 删除非空目录

### 查看目录大小

`du -sh /home/LJR/Mucus_Project ` 查看该目录大小

### 复制目录

| **操作场景**                         | **命令写法**                              |
| ------------------------------------------ | ----------------------------------------------- |
| **复制目录 A 到 B**                  | `cp -r /path/to/A /path/to/B`                 |
| **复制目录内容（不含目录本身）**     | `cp -r /path/to/A/. /path/to/B`               |
| **保持属性复制（权限、时间等）**     | `cp -a /path/to/A /path/to/B`                 |
| **交互式复制（覆盖前询问）**         | `cp -ri /path/to/A /path/to/B`                |
| **可以看到正在复制那些文件**         | `cp -rv /path/to/A/. /path/to/B`              |
| **显示具体的百分比、速度和剩余时间** | `rsync -av --progress /path/to/A/ /path/to/B` |

若权限不够，前面加sudo

## Dockers 容器创建指令

### docker其他指令

`docker ps` 查看运行的docker容器

`docker ps -a`  查看创建过的容器（包括停止的）

`docker images` 或者 `docker image ls` 查看存在的镜像

`docker rm 容器ID` 删除容器

`docker rm 镜像名` 删除镜像

`docker start 容器名` 启动容器

`docker exec -it 容器名 /bin/bash` 打开容器

### 容器的创建流程

##### 创建镜像首先要创建dockerfile文件，来规定镜像的配置

```
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV nnUNet_raw=/data/nnUNet_raw
ENV nnUNet_preprocessed=/data/nnUNet_preprocessed
ENV nnUNet_results=/data/nnUNet_results


# basic system dependencies
RUN apt update && apt install -y \
    python3 \
    python3-pip \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python environment
RUN python3 -m pip install --upgrade pip

# install nnU-Net v2
RUN pip install nnunetv2

# set workspace dir
WORKDIR /workspace

# Default to entering bash (convenient for debugging/training)
CMD ["/bin/bash"]
```

* FROM   指定要进⾏扩展的基础镜像；
* ENV   指定运⾏容器时使⽤到的环境变量；
* RUN  创建镜像时熬运行的指定命令；
* COPY <主机路径> <镜像路径> 复制主机路径的文件到容器镜像中；
* EXPOSE 《PORT》 指定镜像要暴露的端口；
* USER 《user-or-uid》 后续指令运行时使用的默认用户；
* CMD 镜像创建的容器运行时默认执行的命令

##### 根据创建的dockerfile文件来创建镜像环境

`docker build <Dockerfile所在目录> -t <name:tag>` -t 来给镜像命名及起标签

如果已经在docker目录中，则可以使用 `docker build -t <name:tag> .`

##### 根据镜像来创建容器

```
docker run \
  --mount type=bind,source=/home/LJR/nnunet/data,target=/NNuet/data \
  --mount type=bind,source=/home/LJR/nnunet/NNuet,target=/NNuet \
  --gpus all\
  --name nnunet_dev \
  -it nnunet:260106
```

`docker run --gpus all --name 容器名称 -v 宿主机路径：容器映射的路径 -it 使用的镜像`

使用-v来将宿主机路径挂载到容器中，如果宿主机没有该目录则创建该目录

也可以使用--mount 来完成挂载，如果主机目录不存在则会报错

##### 容器的使用

创建容器之后，容器的base环境即为项目环境，可以直接在base安装项目所需的项目；

容器即相当于一个虚拟机，可以将代码直接放在容器中，或者通过-v 来将代码映射进来




##### 容器创建指令

```
docker run \
  --gpus all \
  --shm-size=32g \
  --memory=64g \
  --memory-swap=64g \
  --name mucusAlg-LJR \
  -it \
  -v /home/LJR/Mucus_project/demo_mucusAlgorithms:/workspace \
  -v /home/LJR/Mucus_project/dataset:/data \
  -p 5024:5024 \
  LJR/mucus_dev:base \
  /bin/bash
```
