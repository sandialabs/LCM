whatis("LCM serial (within MPI rank) arch")

setenv("ARCH", "serial")
setenv("ARCH_STRING", "SERIAL")
setenv("ARCH_NAME", "Serial")

setenv("LCM_ENABLE_KOKKOS_EXAMPLES", "OFF")
setenv("LCM_PHALANX_INDEX_TYPE", "INT")
setenv("LCM_KOKKOS_DEVICE", "SERIAL")
setenv("LCM_TPETRA_INST_PTHREAD", "OFF")
setenv("LCM_ENABLE_HWLOC", "OFF")
