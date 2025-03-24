# K8s Guide
### 啟用ip4轉發
```
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.ipv4.ip_forward = 1
EOF
```

---
### 下載kubectl, kubeadm, kubelet

---
### 下載docker

---
### 下載container runtime
```
sudo apt-get install -y containerd.io(--version 1.6.x)
```
### 產生預設配置
```
sudo mkdir -p /etc/containerd
sudo containerd config default > /etc/containerd/config.toml
```
### 調整 cgroup 驅動
```
/etc/containerd/config.toml -> SystemCgroup = true
```
> Node到此即可
---
### 下載calico

---
second scheduler:


apiVersion: kubescheduler.config.k8s.io/v1beta2 -> apiVersion: kubescheduler.config.k8s.io/v1
