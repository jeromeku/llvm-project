rm -rf build && \
cmake -B build -S. -GNinja \
-DCMAKE_VERBOSE_MAKEFILE=1 \
-DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
--debug-output