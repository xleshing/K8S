#!/usr/bin/python3
from ansible.module_utils.basic import AnsibleModule
import os
import re

def run(module, cmd_list):
    rc, out, err = module.run_command(cmd_list)
    if rc != 0:
        module.fail_json(msg=f"Command failed: {' '.join(cmd_list)}", rc=rc, stderr=err)
    return out


def main():
    module = AnsibleModule(
        argument_spec={
            'pod_cidr':          {'type': 'str', 'required': True},
            'control_endpoint':  {'type': 'str', 'required': True},
            'kube_user':         {'type': 'str', 'required': True}, 
        },
        supports_check_mode=False
    )

    pod_cidr = module.params['pod_cidr']
    endpoint = module.params['control_endpoint']

    # 1. Disable swap
    run(module, ['swapoff', '-a'])
    run(module, ['sed', '-i.bak', '-E', r's@^([^#].*\s+swap\s+)@#\1@', '/etc/fstab'])

    # 2. Enable IPv4 forwarding
    sysctl_conf = '/etc/sysctl.d/k8s.conf'
    run(module, ['bash', '-c', f"echo 'net.ipv4.ip_forward = 1' > {sysctl_conf}"])
    run(module, ['sysctl', '--system'])

    # config containerd
    run(module, ['mkdir', '-p', '/etc/containerd'])
    run(module, ['bash', '-c', 'containerd config default | tee /etc/containerd/config.toml'])
    run(module, [ 
      'sed', '-i.bak',
      r's/SystemdCgroup\s*=\s*false/SystemdCgroup = true/',
      '/etc/containerd/config.toml'
    ])
    run(module, ['systemctl','restart','containerd'])

    # 5. kubeadm init
    init_cmd = [
        'kubeadm','init',
        f'--control-plane-endpoint={endpoint}',
        f'--pod-network-cidr={pod_cidr}',
        '--cri-socket=unix:///run/containerd/containerd.sock'
    ]
    rc, out, err = module.run_command(init_cmd)
    if rc != 0:
        module.fail_json(msg='kubeadm init failed', rc=rc, stderr=err)

    # save output file
    with open('kubeadm-init.out','w') as f:
        f.write(out)

    # 6. Setup kubeconfig
    kube_user = module.params['kube_user']
    home_dir  = f"/home/{kube_user}"
    kubeconfig = os.path.join(home_dir, '.kube', 'config')
    kube_dir   = os.path.dirname(kubeconfig)

    run(module, ['/bin/mkdir', '-p', kube_dir])
    run(module, ['/bin/cp', '/etc/kubernetes/admin.conf', kubeconfig])
    run(module, ['/bin/chown', f'{kube_user}:{kube_user}', kubeconfig]) 
    
    # extract join info
    lines = out.splitlines()
    full = ' '.join(lines[-2:]).replace('\\', '').replace('\t', ' ')
    m = re.search(r'kubeadm join\s+(\S+)\s+--token\s+(\S+)\s+--discovery-token-ca-cert-hash\s+(\S+)', full)
    if not m:
        module.fail_json(msg='join command error', stdout=out)
    join_endpoint, token, hashval = m.group(1), m.group(2), m.group(3)

    module.exit_json(
        changed=True,
        join_endpoint=join_endpoint,
        token=token,
        discovery_token_hash=hashval
    )

if __name__ == '__main__':
    main()
