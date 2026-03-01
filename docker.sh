docker run -it \
  --name tf \
  --network host \
  --ipc=host \
  --privileged \
  \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  --device=/dev/devmm_svm \
  \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /usr/local/openmpi:/usr/local/openmpi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
  \
  -v /docker/tf:/docker/tf \
  -w /docker/tf \
  \
  leo:cann8.5.0-910b-ubuntu22.04-py3.11-pt2.9.0