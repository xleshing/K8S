# K8s on Ubuntu Guide

---
### 關閉swap
```
swapon --show
```
> 確認有沒有啟用 swap如果，沒有任何輸出，代表目前系統沒有啟用 swap
```
sudo vim /etc/fstab 
```
將/swapfile所在行注釋
>Swap 在 /etc/fstab 裡的樣子會像這樣： \
```UUID=xxxx-xxxx    none    swap    sw    0   0``` \
或是： \
```/dev/sdX#   none    swap    sw    0   0```
---
### 啟用ipv4轉發
暫時
```
sudo sysctl -w net.ipv4.ip_forward=1
```
永久
```
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.ipv4.ip_forward = 1
EOF
```
---
### 下載[kubectl, kubeadm, kubelet](https://kubernetes.io/docs/setup/)
---
### 下載[docker](https://docs.docker.com/engine/install/ubuntu/)
> docker在下載時會自動下載配套版本的containerd，如果有特殊需求再另外下載``` sudo apt install containerd ```
---
### 提升權限
```
sudo usermod -aG docker icanlab
```
>重新登錄去更新權限狀態
---
### 產生containerd預設配置
```
sudo mkdir -p /etc/containerd
sudo containerd config default | sudo tee /etc/containerd/config.toml
```
### 調整containerd 的 cgroup 驅動
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
> Node到此即可(sudo hostnamectl set-hostname new-hostname)
### init k8s
```
sudo kubeadm init --cri-socket unix:///var/run/containerd/containerd.sock --pod-network-cidr=<ip/mask> --control-plane-endpoint <ip>
```
---
### 下載[calico](https://docs.tigera.io/calico/latest/getting-started/kubernetes/quickstart)
---
second scheduler:


apiVersion: kubescheduler.config.k8s.io/v1beta2 -> apiVersion: kubescheduler.config.k8s.io/v1
