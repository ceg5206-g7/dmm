`trace_v1_cannon.json` is a real schema-1 trace captured before the DSL migration,
using runtime revision `d3733ad`. It executed n=4, P=4, c=1 with the serial
executor, FP32 A=`arange(16).reshape(4,4)/16`, and B=identity. The product was
checked against FP64 before serialization. The file retains its original tags,
regions, counts, peaks, input hashes and numerical errors. Tests replay its
saved counts without invoking an algorithm or loading input matrices.
