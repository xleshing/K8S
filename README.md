# K8s Guide

---
### 關閉swap
```
sudo vim /etc/fstab ，sudo vim /etc/fstab
```
### 將```/swap.img      none    swap    sw      0       0``` 注釋

---
### 下載kubectl, kubeadm, kubelet

---
### 啟用ip4轉發
```
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.ipv4.ip_forward = 1
EOF
```
---
### 下載docker

---
### 下載container runtime

### 產生預設配置
```
sudo mkdir -p /etc/containerd
sudo containerd config default > /etc/containerd/config.toml
```
### 調整 cgroup 驅動
```
sudo vim /etc/containerd/config.toml
```
```
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
SystemCgroup = false -> true
```
```
sudo systemctl restart containerd
```
> Node到此即可
### init k8s
```
sudo kubeadm init --cri-socket unix:///var/run/containerd/containerd.sock
```
---
### 下載calico

---
second scheduler:


apiVersion: kubescheduler.config.k8s.io/v1beta2 -> apiVersion: kubescheduler.config.k8s.io/v1
