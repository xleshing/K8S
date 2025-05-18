## Architecture Diagram

```mermaid
flowchart TD
    %% ===== Client =====
    subgraph Client
        A[Researcher main.py] -- REST --> B[/Layer Controller/]
    end

    %% ===== Cluster =====
    subgraph Kubernetes Cluster
        B -- K8s API --> API[(API Server)]

        B .. creates .. L0[Layer Service Pod 0]
        B .. creates .. LN[Layer Service Pod N]
        B -->|/forward| L0
        L0 -->|/forward| L1[…]
        LN -->|/forward (final)| B

        %% Back-prop
        B -->|/backward| LN
        LN -->|/backward| L1
        L1 -->|/backward| L0
        L0 -->|grad→| B

        %% Persistence
        B -- save/load --> PVC[(PersistentVolume)]
    end

    classDef c1 fill:#ffd,stroke:#333;
    class B,L0,LN,L1 c1;
