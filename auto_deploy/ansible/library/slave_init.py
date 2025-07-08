#!/usr/bin/python3
from ansible.module_utils.basic import AnsibleModule
import os


def run(module, cmd_list):
    rc, out, err = module.run_command(cmd_list)
    if rc != 0:
        module.fail_json(msg=f"Command failed: {' '.join(cmd_list)}", rc=rc, stderr=err)
    return out


def main():
    module = AnsibleModule(
        argument_spec={
            # 完整的 kubeadm join 指令字符串
            'join_endpoint': {'type': 'str', 'required': True},
            'token': {'type': 'str', 'required': True},
            'discovery_token_hash': {'type': 'str', 'required': True},
        },
        supports_check_mode=False
    )

    join_endpoint = module.params['join_endpoint']
    token = module.params['token']
    discovery_token_hash = module.params['discovery_token_hash']

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

    # 5. Execute kubeadm join
    cmd = [
        'kubeadm', 'join', join_endpoint,
        '--token', token,
        '--discovery-token-ca-cert-hash', discovery_token_hash
    ]
    rc, out, err = module.run_command(cmd)
    if rc != 0:
        module.fail_json(msg='kubeadm join failed', rc=rc, stderr=err)

    module.exit_json(changed=True, stdout=out)

if __name__ == '__main__':
    main()
