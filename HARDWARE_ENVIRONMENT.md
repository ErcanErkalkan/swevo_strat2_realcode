# Production hardware environment

Author-confirmed production workstation used for the reported optimization campaign:

- **System:** Dell Vostro 3888
- **CPU:** Intel Core i5-10400 — 6 physical cores / 12 threads, 2.90 GHz base frequency
- **RAM:** 16 GB — 2×8 GB, operating at 2400 MHz
- **Graphics:** Intel UHD Graphics 630 (integrated; not used by the CPU-based optimization search)
- **Operating system:** Windows 11 Enterprise 64-bit, build 26200
- **Storage:** 256 GB SK hynix NVMe SSD + 480 GB Toshiba TR200 SATA SSD
- **BIOS:** Dell 2.35.1
- **Motherboard:** Dell 0RM5DR

The seed-level run ledger does not embed an immutable per-run hardware fingerprint, so this is campaign-level environment provenance rather than per-run machine attestation. Runtime comparisons are interpreted within this documented workstation and tier-matched wall-clock protocol.

The study did not use hardware power counters. The reported `compute_wh` field remains a runtime × assumed-tier-power model proxy, not measured electrical energy.
