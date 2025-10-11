// NOLINTBEGIN
#include "nanobind/nanobind.h"
#include "nanobind/nb_defs.h"

int add(int a, int b){
    return a + b;
}

NB_MODULE(my_ext, m) {
    m.def("add", &add);
}
// NOLINTEND