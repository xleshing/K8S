## Architecture Diagram

```mermaid
flowchart TD
    %% ===== Client =====
    subgraph Client
        A[Researcher&nbsp;main.py] -- REST --> B[/Layer&nbsp;Controller/]
    end

    %% ===== Cluster =====
    subgraph Kubernetes&nbsp;Cluster
        %% Controller → K8s API
        B -- K8s&nbsp;API --> API[(API&nbsp;Server)]

        %% Controller 動態產生 Layer Pods
        B ..>|creates| L0[Layer&nbsp;Pod&nbsp;0]
        B ..>|creates| LN[Layer&nbsp;Pod&nbsp;N]

        %% ===== Forward pass =====
        B -->|/forward| L0
        L0 -->|/forward| Lmid[⋯ other layers ⋯]
        Lmid -->|/forward| LN
        LN -->|output| B

        %% ===== Back-prop =====
        B <--|/backward| LN
        LN <--|/backward| Lmid
        Lmid <--|/backward| L0
        L0 <--|grad| B

        %% ===== Persistence =====
        B -- save/load --> PVC[(Persistent&nbsp;Volume)]
    end

    classDef ctl fill:#ffd,stroke:#333;
    class B,L0,Lmid,LN ctl;
