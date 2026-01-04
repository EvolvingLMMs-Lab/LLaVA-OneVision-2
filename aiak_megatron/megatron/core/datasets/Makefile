CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell python3 -m pybind11 --includes)

LIBNAME = helpers_cpp
LIBEXT = $(shell python3-config --extension-suffix)

OUT = $(LIBNAME)$(LIBEXT)
SRC = helpers.cpp

# Check if any compiled library exists
EXISTING_SO = $(wildcard helpers_cpp*.so)

default:
ifneq ($(EXISTING_SO),)
	@echo "Using existing compiled library: $(EXISTING_SO)"
else
	@$(MAKE) $(OUT)
endif

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
