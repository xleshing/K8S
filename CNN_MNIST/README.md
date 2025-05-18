flowchart TD
    subgraph Client
        A[Researcher<br/>main.py] -- REST --> B[/Layer&nbsp;Controller/]
    end

    subgraph Kubernetes Cluster
        B -- K8s API --> API[(API&nbsp;Server)]

        B .. creates .. L0[Layer&nbsp;Service&nbsp;Pod 0]
        B .. creates .. LN[Layer&nbsp;Service&nbsp;Pod N]
        B -->|/forward| L0
        L0 -->|/forward| L1[...]
        LN -->|/forward (最後輸出)| B

        %% Back-prop arrows (逆向)
        B -->|/backward| LN
        LN -->|/backward| L1
        L1 -->|/backward| L0
        L0 -->|梯度回傳| B

        %% Persistence
        B -- save/load --> PVC[(PersistentVolume)]
    end

    classDef c1 fill:#ffd,stroke:#333;
    class B,L0,LN,L1 c1;
