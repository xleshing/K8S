master:

cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.ipv4.ip_forward = 1
EOF

sudo sysctl --system 

kubectl, kubeadm, kubelet

sudo mkdir -p /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

sudo apt-get install -y containerd.io(--version 1.6.x)

sudo mkdir -p /etc/containerd

sudo containerd config default > /etc/containerd/config.toml

/etc/containerd/config.toml -> SystemCgroup = true

calico

second scheduler:

apiVersion: kubescheduler.config.k8s.io/v1beta2 -> apiVersion: kubescheduler.config.k8s.io/v1
