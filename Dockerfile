# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/tensorflow:1.12.0-cuda9.0-py3

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /home/wcy

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /home/wcy

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]