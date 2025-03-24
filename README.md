master:

cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.ipv4.ip_forward = 1
EOF
---
###下載kubectl, kubeadm, kubelet

---
###把docker下載加入本機
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
###下載docker

---
###下載container runtime
sudo apt-get install -y containerd.io(--version 1.6.x)
###產生預設配置
sudo mkdir -p /etc/containerd
sudo containerd config default > /etc/containerd/config.toml
###調整 cgroup 驅動
/etc/containerd/config.toml -> SystemCgroup = true
---
###下載calico
---
second scheduler:

apiVersion: kubescheduler.config.k8s.io/v1beta2 -> apiVersion: kubescheduler.config.k8s.io/v1
