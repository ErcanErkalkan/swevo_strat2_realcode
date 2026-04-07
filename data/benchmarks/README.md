Place public benchmark instance files here to enable real-file parsing.

Supported discovery paths:
- data/benchmarks/solomon/C101.txt
- data/benchmarks/homberger/<instance>.txt
- data/benchmarks/li_lim/<instance>.txt

Current parser support:
- Solomon-like whitespace tables with columns:
  CUST NO. / XCOORD / YCOORD / DEMAND / READY TIME / DUE DATE / SERVICE TIME

If no file is found for a manifest row, the suite falls back to deterministic synthetic generation.
